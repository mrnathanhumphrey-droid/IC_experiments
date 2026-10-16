"""PROBE 5 — sequential-phase residual. Is the post-phase-2 residual deposit a usable prior for a LATER phase (memory)
or interference (debris)? Pre-reg PREREG_PROBE5_RESIDUAL_2026_06_23.md (LOCK fa350db0). Band ONLY.
childhood(400) -> phase-2 task-B (subj, 250) -> [residual] -> phase-3 task-C (TREC, 250). task-A retention after phase-3.
Arms (share the phase-2 end state): resid / proj (residual projected out of V) / randproj (curvature-matched random proj,
equal magnitude) / A (no-deposit baseline through same schedule). Reproduce-anchor: phase-2 B-A vs banked.
task-C = TREC (LOCKED, distinct from A=sst2 and B=subj, not in childhood stream); HARD-STOP if TREC fails to load.
"""
import os, time, json, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(100))   # floor go-forward: N>=100 (paired contrasts resid-proj/resid-randproj are floor-robust; abs levels demoted)
LR_DEPTH = 5e-5; STEPS_P2 = 250; STEPS_P3 = 250; LR_CHILD = 3e-5; STEPS_CHILD = 400; BATCH = 16; KPROJ = 128; N_FISHER = 200
HEARTBEAT = 25; _T0 = 0.0   # per-step monitor (matches floor_battery); _T0 set in main()
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",            "ck": OPT_C, "family": "OPT",    "anchor_BmA": 0.089},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt",  "ck": PY_C,  "family": "PYTHIA", "anchor_BmA": 0.073},
}
CELL = os.environ.get("CELL", ""); assert CELL in CELLS, f"CELL band-only; got {CELL!r}"
cfg = CELLS[CELL]; OUT = f"{G}/probe5_residual_{CELL}.json"


def _retry(fn, tries=5, wait=12):
    last = None
    for i in range(tries):
        try: return fn()
        except Exception as e: last = e; print(f"[hf] retry {i+1}: {str(e)[:80]}", flush=True); time.sleep(wait)
    raise last
def load_jsonB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]
def win(ds, k=900, w=4000):
    n = min(len(ds), w); return list(ds.shuffle(seed=0).select(range(n)))[: k]
def load_trec():
    """task-C: TREC coarse question-type (6-way). Distinct from sst2/subj; not in childhood stream.
    INFRA FIX 2026-06-25: HF datasets dropped script datasets, so CogComp/trec (trec.py) won't load. Use SetFit/TREC-QC
    (parquet mirror, SAME TREC data) with defensive column detection. Wrong schema -> caller HARD-STOPs (cheap VOID)."""
    from datasets import load_dataset
    L6 = ["abbreviation", "entity", "description", "human", "location", "number"]
    ds = _retry(lambda: load_dataset("SetFit/TREC-QC"))["train"]
    cols = ds.column_names; tcol = "text" if "text" in cols else cols[0]
    for lc in ("label_coarse_text", "label_text", "coarse_label_text"):  # prefer a text label
        if lc in cols:
            return [(f"{str(r[tcol]).strip()}\nType:", " " + str(r[lc]).strip().lower()) for r in win(ds)]
    for lc in ("label_coarse", "coarse_label", "label-coarse", "label"):  # else int coarse -> name
        if lc in cols:
            return [(f"{str(r[tcol]).strip()}\nType:", " " + L6[int(r[lc])]) for r in win(ds) if int(r[lc]) < 6]
    raise ValueError(f"TREC-QC columns not recognized: {cols}")
def load_hf(key):
    from datasets import load_dataset
    LD = lambda *a, **k: _retry(lambda: load_dataset(*a, **k))
    if key == "emotion":
        return [(f"{r['text'].strip()}\nEmotion:", " " + ["sadness","joy","love","anger","fear","surprise"][r["label"]]) for r in win(LD("dair-ai/emotion")["train"])]
    if key == "dbpedia":
        L = ["company","school","artist","athlete","politics","transport","building","nature","village","animal","plant","album","film","book"]
        return [(f"{r['content'].strip()[:400]}\nCategory:", " " + L[r["label"]]) for r in win(LD("fancyzhx/dbpedia_14")["train"])]
    if key == "yahoo":
        L = ["society","science","health","education","computers","sports","business","entertainment","family","politics"]
        return [(f"{(r.get('question_title') or '').strip()}\nTopic:", " " + L[r["topic"]]) for r in win(LD("community-datasets/yahoo_answers_topics")["train"]) if r.get("question_title")]
    if key == "newscat":
        return [(f"{(r.get('headline') or '').strip()}\nCategory:", " " + (r.get("category") or "").strip().lower()) for r in win(LD("heegyu/news-category-dataset")["train"]) if r.get("headline") and r.get("category")]
    raise ValueError(key)
def b_pairs(tok, items):
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"][:256]; c = tok(co, add_special_tokens=False)["input_ids"][:256]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    L = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)
def sst_rows(split, k, seed=0):
    from datasets import load_dataset
    ds = _retry(lambda: load_dataset("stanfordnlp/sst2"))[split]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))[:k]]
def fisher_taskA(model, params, names, tok):
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float32) for n in names}
    for s, l in sst_rows("train", N_FISHER, 0):
        inp, lb, am = b_pairs(tok, [(f"{s}\nSentiment:", LAB[l])]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: g = params[n].grad.detach().float(); F[n] += g.T @ g
    return F
def eval_sst(model, tok, evalset, posid, negid):
    model.eval(); correct = 0; n = 0
    with torch.no_grad():
        for i in range(0, len(evalset), 32):
            chunk = evalset[i:i + 32]; seqs = [tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"][:256] for s, _ in chunk]
            L = max(len(x) for x in seqs); pad = tok.pad_token_id
            inp = np.full((len(seqs), L), pad, np.int64); pos = np.zeros(len(seqs), np.int64)
            for j, sq in enumerate(seqs): inp[j, :len(sq)] = sq; pos[j] = len(sq) - 1
            lg = model(input_ids=torch.tensor(inp).to(dev)).logits
            last = lg[torch.arange(len(seqs)), torch.tensor(pos).to(dev)]
            pred = (last[:, posid] > last[:, negid]).long().cpu().numpy()
            for j, (_, lab) in enumerate(chunk): correct += int(pred[j] == lab); n += 1
    return correct / max(n, 1)
def childhood(model, tok, stream, steps, lr, seed, Vproj=None, carriers=None, hb=""):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    tasks = list(stream); params = dict(model.named_parameters()); losses = []
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        if Vproj is not None:
            for n in carriers: gd = params[n].grad; gd.sub_(((gd.double() @ Vproj[n].double()) @ Vproj[n].double().T).to(gd.dtype))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.item()))
        if hb and (step % HEARTBEAT == 0 or step == steps - 1):
            print(f"[hb] {hb} childhood {step+1}/{steps} loss={float(loss):.3f} ({time.time()-_T0:.0f}s)", flush=True)
    del opt; torch.cuda.empty_cache(); return float(np.mean(losses[:20])), float(np.mean(losses[-20:]))
def train_phase(model, tok, items, steps, lr, seed, hb=""):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if hb and (step % HEARTBEAT == 0 or step == steps - 1):
            print(f"[hb] {hb} {step+1}/{steps} loss={float(loss):.3f} ({time.time()-_T0:.0f}s)", flush=True)
    del opt; torch.cuda.empty_cache()
def labels_of(items): return sorted(set(c for _, c in items))


def main():
    global _T0
    t0 = time.time(); _T0 = t0; print(f"[P5:{CELL}] {cfg['model']} sequential residual (phase-3 task-C=TREC), N={len(SEEDS)}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    try:
        taskC = load_trec(); assert len(taskC) > 100
    except Exception as e:
        json.dump({"cell": CELL, "VOID": True, "reason": f"TREC load failed: {str(e)[:120]}"}, open(OUT, "w"), indent=2)
        print(f"[P5:{CELL}] HARD-STOP: TREC (task-C) failed to load -> VOID, no run. {e}", flush=True); return
    print(f"[P5:{CELL}] task-C TREC loaded ({len(taskC)} ex)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    NC = len(carriers); posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[P5:{CELL}] w_A acc={pre_sst:.3f}; {NC} carriers ({time.time()-t0:.0f}s)", flush=True)
    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); V[n] = vv[:, -KPROJ:].flip(dims=[1]).contiguous(); del FA[n]
    grng = torch.Generator(device="cpu").manual_seed(12345); S_rand = {}
    for n in carriers:
        dout = params[n].shape[0]; cin = torch.randn(KPROJ, 1, generator=grng).double(); vin = (V[n].cpu() @ cin); vin = (vin / vin.norm()).to(dev)
        vout = torch.randn(dout, 1, generator=grng).double().to(dev); vout = vout / vout.norm(); Sr = (vout @ vin.T); S_rand[n] = Sr / (Sr.norm() + 1e-30)
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
    ho_tr = load_jsonB("medium_subj")[:-200]
    stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]}
    stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}

    retB_p2 = []; retD_p2 = []; resid_mag = []
    r_resid = []; r_proj = []; r_randproj = []; r_A = []
    def snap(): return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ns = len(SEEDS)
    for si, sd in enumerate(SEEDS):
        lab = f"s{si+1}/{ns}"
        # --- deposit + phase-2 (task-B) -> phase-2 end state w2 (residual present) ---
        model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, hb=f"{lab} childB")
        train_phase(model, tok, ho_tr, STEPS_P2, LR_DEPTH, sd, hb=f"{lab} p2")
        retB_p2.append(eval_sst(model, tok, sst_eval, posid, negid)); w2 = snap()
        Pc = dict(model.named_parameters())
        # residual deposit direction per carrier (P_V(w2-w_A) unit) + magnitude
        S_res = {}; rmag = 0.0
        for n in carriers:
            df = (Pc[n].detach().double() - ckpt_sd[n].to(dev).double()); dv = (df @ V[n]) @ V[n].T
            m = float(dv.norm()); rmag += m; S_res[n] = (dv / (m + 1e-30))
        resid_mag.append(rmag / NC)
        # arm resid: phase-3 from w2
        train_phase(model, tok, taskC, STEPS_P3, LR_DEPTH, sd, hb=f"{lab} p3resid"); r_resid.append(eval_sst(model, tok, sst_eval, posid, negid))
        # arm proj: remove the in-V residual deposit from w2, then phase-3
        model.load_state_dict(w2)
        with torch.no_grad():
            Pp = dict(model.named_parameters())
            for n in carriers:
                df = (Pp[n].detach().double() - ckpt_sd[n].to(dev).double()); dv = (df @ V[n]) @ V[n].T
                Pp[n].sub_(dv.to(Pp[n].dtype))
        train_phase(model, tok, taskC, STEPS_P3, LR_DEPTH, sd, hb=f"{lab} p3proj"); r_proj.append(eval_sst(model, tok, sst_eval, posid, negid))
        # arm randproj: remove an equal-magnitude curvature-matched random projection from w2, then phase-3
        model.load_state_dict(w2)
        with torch.no_grad():
            Pr = dict(model.named_parameters())
            for n in carriers:
                df = (Pr[n].detach().double() - ckpt_sd[n].to(dev).double()); dv = (df @ V[n]) @ V[n].T
                Pr[n].sub_((float(dv.norm()) * S_rand[n]).to(Pr[n].dtype))
        train_phase(model, tok, taskC, STEPS_P3, LR_DEPTH, sd, hb=f"{lab} p3rand"); r_randproj.append(eval_sst(model, tok, sst_eval, posid, negid))
        # arm A: no-deposit baseline through phase-2 + phase-3
        model.load_state_dict(ckpt_sd); train_phase(model, tok, ho_tr, STEPS_P2, LR_DEPTH, sd, hb=f"{lab} A_p2"); train_phase(model, tok, taskC, STEPS_P3, LR_DEPTH, sd, hb=f"{lab} A_p3")
        r_A.append(eval_sst(model, tok, sst_eval, posid, negid))
        # phase-2 B-D anchor: D arm = deposit projected out during childhood, then phase-2 only
        model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, Vproj=V, carriers=carriers, hb=f"{lab} anchorD_child")
        train_phase(model, tok, ho_tr, STEPS_P2, LR_DEPTH, sd, hb=f"{lab} anchorD_p2"); retD_p2.append(eval_sst(model, tok, sst_eval, posid, negid))
        rp = float(np.mean(np.array(r_resid) - np.array(r_proj))); rr = float(np.mean(np.array(r_resid) - np.array(r_randproj)))
        print(f"[seed] {lab} DONE  resid={r_resid[-1]:.3f} proj={r_proj[-1]:.3f} rand={r_randproj[-1]:.3f}  run(resid-proj)={rp:+.4f} run(resid-rand)={rr:+.4f}  ({time.time()-_T0:.0f}s)", flush=True)

    ns = len(SEEDS); me = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)]
    Bp2 = np.array(retB_p2); Dp2 = np.array(retD_p2); BmA_p2 = me((Bp2 - np.array(r_A)).tolist()) if False else None
    # phase-2 anchor uses B vs D (both at phase-2 end); B-A_p2 would need a phase-2-only A — use B-D as the dispositive anchor proxy + report B level
    BmD_p2 = me((Bp2 - Dp2).tolist())
    resid = np.array(r_resid); proj = np.array(r_proj); rand = np.array(r_randproj); A3 = np.array(r_A)
    rep = {"cell": CELL, "family": cfg["family"], "model": cfg["model"], "n_carriers": NC, "n_seeds": ns, "w_A_acc": pre_sst,
           "VOID": False, "phase2_retB": me(Bp2), "phase2_retD": me(Dp2), "phase2_B_minus_D": BmD_p2,
           "anchor_BmD_expected": (0.055 if cfg["family"] == "OPT" else -0.006),
           "residual_mag_mean": me(resid_mag),
           "phase3_resid": me(resid), "phase3_proj": me(proj), "phase3_randproj": me(rand), "phase3_A": me(A3),
           "resid_minus_proj": me((resid - proj).tolist()), "resid_minus_randproj": me((resid - rand).tolist()),
           "MEMORY_if": "resid>proj AND resid>randproj", "DEBRIS_if": "resid<=proj or resid~=randproj"}
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"\n[P5:{CELL}] === phase-2 B-D={BmD_p2[0]:+.3f} (exp {rep['anchor_BmD_expected']:+.3f}); retB_p2={rep['phase2_retB'][0]:.3f} ===", flush=True)
    print(f"   phase-3 task-A retention: resid={rep['phase3_resid'][0]:.3f} proj={rep['phase3_proj'][0]:.3f} randproj={rep['phase3_randproj'][0]:.3f} A={rep['phase3_A'][0]:.3f}", flush=True)
    print(f"   resid-proj={rep['resid_minus_proj'][0]:+.3f}  resid-randproj={rep['resid_minus_randproj'][0]:+.3f}  (MEMORY if both >0; DEBRIS if <=0)", flush=True)
    print(f"[P5:{CELL}] saved {OUT} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
