"""PROBE 6 — displacement-absorption. Pre-reg PREREG_PROBE6_DISPLACEMENT_2026_06_25.md (LOCK 185910cd). Band ONLY.
Does a load-bearing childhood REDUCE the in-V displacement of task-A coords caused by phase-2?
Per seed, paired off the same w_A:  NoCH: w_A -> phase-2(250).   CH: w_A -> childhood(400) -> phase-2(250).
Damage quantity (from Probe-5): D = mean_carrier ||P_V(w_end - w_A)||.  ret = SST-2 (task-A) acc at w_end.
Contrasts (paired, floor-robust):  dD = D_noch - D_ch (>0 = absorption);  dret = ret_ch - ret_noch (>0 = load-bearing);
                                   r = corr(dD_seed, dret_seed) (mediation: less displacement -> more retention).
HARD-STOP (VOID) if datasets fail to load or the w_A reproduce-anchor is off-band.
"""
import os, time, json, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(100))   # floor go-forward: N>=100; paired dD/dret cancel the ~0.02 floor
LR_DEPTH = 5e-5; STEPS_P2 = 250; LR_CHILD = 3e-5; STEPS_CHILD = 400; BATCH = 16; KPROJ = 128; N_FISHER = 200
HEARTBEAT = 25; _T0 = 0.0   # per-step monitor (matches floor_battery/probe5); _T0 set in main()
# w_A reproduce-anchor (VOID if off): floor battery / probe5 values
ANCHOR_WA = {"OPT": 0.823, "PYTHIA": 0.753}; ANCHOR_TOL = 0.02
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",            "ck": OPT_C, "family": "OPT"},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt",  "ck": PY_C,  "family": "PYTHIA"},
}
CELL = os.environ.get("CELL", ""); assert CELL in CELLS, f"CELL band-only; got {CELL!r}"
cfg = CELLS[CELL]; OUT = f"{G}/probe6_displacement_{CELL}.json"


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
def childhood(model, tok, stream, steps, lr, seed, hb=""):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    tasks = list(stream); params = dict(model.named_parameters()); losses = []
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
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


def disp(params, ckpt_sd, V, carriers):
    """Per-carrier displacement of task-A coords: in-V ||P_V(w_end-w_A)|| and out-of-V ||P_perp||, averaged over carriers."""
    dv_sum = 0.0; perp_sum = 0.0
    with torch.no_grad():
        for n in carriers:
            df = (params[n].detach().double() - ckpt_sd[n].to(dev).double())
            dv = (df @ V[n]) @ V[n].T
            m_in = float(dv.norm()); m_tot = float(df.norm())
            dv_sum += m_in; perp_sum += float(np.sqrt(max(m_tot * m_tot - m_in * m_in, 0.0)))
    nc = len(carriers); return dv_sum / nc, perp_sum / nc


def main():
    global _T0
    t0 = time.time(); _T0 = t0
    print(f"[P6:{CELL}] {cfg['model']} displacement-absorption (NoCH vs CH), N={len(SEEDS)}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    NC = len(carriers); posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[P6:{CELL}] w_A acc={pre_sst:.3f}; {NC} carriers ({time.time()-t0:.0f}s)", flush=True)
    # validity gate: w_A reproduce-anchor
    exp = ANCHOR_WA[cfg["family"]]
    if abs(pre_sst - exp) > ANCHOR_TOL:
        json.dump({"cell": CELL, "VOID": True, "reason": f"w_A anchor off: {pre_sst:.3f} vs {exp:.3f}+-{ANCHOR_TOL}"}, open(OUT, "w"), indent=2)
        print(f"[P6:{CELL}] HARD-STOP: w_A anchor off-band -> VOID, no run.", flush=True); return
    # Fisher V (task-A curvature, top-128) — identical to probe5
    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); V[n] = vv[:, -KPROJ:].flip(dims=[1]).contiguous(); del FA[n]
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
    # datasets — HARD-STOP (VOID) on any load failure, no fabricated run
    try:
        ho_tr = load_jsonB("medium_subj")[:-200]; assert len(ho_tr) > 100
        stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]}
        stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}
        for sk, (it, _) in stream.items(): assert len(it) > 50, sk
    except Exception as e:
        json.dump({"cell": CELL, "VOID": True, "reason": f"dataset load failed: {str(e)[:120]}"}, open(OUT, "w"), indent=2)
        print(f"[P6:{CELL}] HARD-STOP: childhood/phase-2 datasets failed to load -> VOID, no run. {e}", flush=True); return
    print(f"[P6:{CELL}] datasets loaded (phase-2 subj {len(ho_tr)}; childhood {len(stream)} streams)", flush=True)

    D_noch = []; D_ch = []; Dp_noch = []; Dp_ch = []; ret_noch = []; ret_ch = []
    ns = len(SEEDS)
    for si, sd in enumerate(SEEDS):
        lab = f"s{si+1}/{ns}"
        # --- NoCH: w_A -> phase-2 ---
        model.load_state_dict(ckpt_sd)
        train_phase(model, tok, ho_tr, STEPS_P2, LR_DEPTH, sd, hb=f"{lab} NoCH_p2")
        din, dperp = disp(dict(model.named_parameters()), ckpt_sd, V, carriers)
        D_noch.append(din); Dp_noch.append(dperp); ret_noch.append(eval_sst(model, tok, sst_eval, posid, negid))
        # --- CH: w_A -> childhood -> phase-2 ---
        model.load_state_dict(ckpt_sd)
        childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, hb=f"{lab} CH_child")
        train_phase(model, tok, ho_tr, STEPS_P2, LR_DEPTH, sd, hb=f"{lab} CH_p2")
        din, dperp = disp(dict(model.named_parameters()), ckpt_sd, V, carriers)
        D_ch.append(din); Dp_ch.append(dperp); ret_ch.append(eval_sst(model, tok, sst_eval, posid, negid))
        # running paired contrasts
        dD = float(np.mean(np.array(D_noch) - np.array(D_ch))); drt = float(np.mean(np.array(ret_ch) - np.array(ret_noch)))
        print(f"[seed] {lab} DONE  D_noch={D_noch[-1]:.4f} D_ch={D_ch[-1]:.4f} ret_noch={ret_noch[-1]:.3f} ret_ch={ret_ch[-1]:.3f}  run(dD)={dD:+.4f} run(dret)={drt:+.4f}  ({time.time()-_T0:.0f}s)", flush=True)

    me = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)]
    Dn = np.array(D_noch); Dc = np.array(D_ch); Rn = np.array(ret_noch); Rc = np.array(ret_ch)
    dD_seed = (Dn - Dc); dret_seed = (Rc - Rn)
    r = float(np.corrcoef(dD_seed, dret_seed)[0, 1]) if ns > 2 and dD_seed.std() > 0 and dret_seed.std() > 0 else 0.0
    rep = {"cell": CELL, "family": cfg["family"], "model": cfg["model"], "n_carriers": NC, "n_seeds": ns, "w_A_acc": pre_sst, "VOID": False,
           "D_noch": me(D_noch), "D_ch": me(D_ch), "delta_D": me(dD_seed.tolist()),
           "Dperp_noch": me(Dp_noch), "Dperp_ch": me(Dp_ch),
           "ret_noch": me(ret_noch), "ret_ch": me(ret_ch), "delta_ret": me(dret_seed.tolist()),
           "corr_dD_dret": r, "phase2_eroded": bool(np.mean(Rn) < pre_sst - 0.02),
           "H1_absorption_if": "delta_D>0 (>=2sigma) AND delta_ret>0 (>=2sigma) AND corr>0",
           "H2_mediation_if": "corr_dD_dret>0", "SURPRISE_if": "delta_ret>0 (>=2sigma) but delta_D~=0"}
    json.dump(rep, open(OUT, "w"), indent=2)
    dDm, dDe = rep["delta_D"]; dRm, dRe = rep["delta_ret"]
    print(f"\n[P6:{CELL}] === dD={dDm:+.4f}+-{dDe:.4f} ({dDm/max(dDe,1e-9):+.1f}sigma)  dret={dRm:+.4f}+-{dRe:.4f} ({dRm/max(dRe,1e-9):+.1f}sigma)  r={r:+.3f} ===", flush=True)
    print(f"   D_noch={rep['D_noch'][0]:.4f} D_ch={rep['D_ch'][0]:.4f}  ret_noch={rep['ret_noch'][0]:.3f} ret_ch={rep['ret_ch'][0]:.3f}  phase2_eroded={rep['phase2_eroded']}", flush=True)
    print(f"[P6:{CELL}] saved {OUT} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
