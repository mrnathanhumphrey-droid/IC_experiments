"""PROBE 11 v2 — specificity with a REALIZED-path alttask, 3-cell. PREREG_PROBE11v2_REALIZEDALT_3CELL (LOCK 332e7029).
Supersedes Probe-11 v1 (VOID-soft: BLOOM's one-shot alttask SATURATED, childhood destructive). Two fixes vs 8v4 (0112d69b):
 (1) alttask = REALIZED 250-step TREC in-V direction (unit per carrier * magp), symmetric with phase-2 (kills saturation +
     construction asymmetry); (2) LR_GRID extended down [5e-6..3e-5] so BLOOM gets a non-destructive childhood (OPT/Pythia
     picks unchanged). Everything else = 8v4. S = N(phase2) - N(alttask); SPECIFIC if S>0(>=2s) & N_phase2>N_randtop16.
"""
import os, time, json, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(100))
LR_GRID = [float(x) for x in os.environ.get("LR_GRID", "5e-6,7e-6,1e-5,1.5e-5,2e-5,3e-5").split(",")]  # v3: env-overridable (gentle re-run)
K_CAL = 20; NONDESTRUCT_TOL = 0.05; CAL_MARGIN = 0.015
LR_DEPTH = 5e-5; STEPS_P2 = 250; STEPS_ALT = 250; STEPS_CHILD = 400; BATCH = 16; KPROJ = 128; TOPK = 16; N_FISHER = 200; K_RAND = 5
EPS = 1e-6; HEARTBEAT = 25; _T0 = 0.0
ANCHOR_WA = {"OPT": 0.823, "PYTHIA": 0.753, "BLOOM": 0.83}; ANCHOR_TOL = 0.02
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
BLOOM_C = ["self_attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",            "ck": OPT_C,   "family": "OPT"},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt",  "ck": PY_C,    "family": "PYTHIA"},
    "bloom560m":  {"model": "bigscience/bloom-560m",   "ckpt": "bloom560m_sst2_wA.pt",          "ck": BLOOM_C, "family": "BLOOM"},
    "opt13b":     {"model": "facebook/opt-1.3b",        "ckpt": "opt13b_sst2_wA.pt",             "ck": OPT_C,   "family": "OPT", "anchor": 0.93},  # Probe-12 scale
}
CELL = os.environ.get("CELL", ""); assert CELL in CELLS, f"CELL band-only; got {CELL!r}"
cfg = CELLS[CELL]; OUT = f"{G}/probe11_realizedalt_{CELL}.json"


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
    from datasets import load_dataset
    L6 = ["abbreviation", "entity", "description", "human", "location", "number"]
    ds = _retry(lambda: load_dataset("SetFit/TREC-QC"))["train"]; cols = ds.column_names; tcol = "text" if "text" in cols else cols[0]
    for lc in ("label_coarse_text", "label_text", "coarse_label_text"):
        if lc in cols: return [(f"{str(r[tcol]).strip()}\nType:", " " + str(r[lc]).strip().lower()) for r in win(ds)]
    for lc in ("label_coarse", "coarse_label", "label-coarse", "label"):
        if lc in cols: return [(f"{str(r[tcol]).strip()}\nType:", " " + L6[int(r[lc])]) for r in win(ds) if int(r[lc]) < 6]
    raise ValueError(f"TREC-QC columns: {cols}")
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
    tasks = list(stream)
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if hb and (step % HEARTBEAT == 0 or step == steps - 1):
            print(f"[hb] {hb} childhood {step+1}/{steps} loss={float(loss):.3f} ({time.time()-_T0:.0f}s)", flush=True)
    del opt; torch.cuda.empty_cache()
def train_phase(model, tok, items, steps, lr, seed, hb=""):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for step in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt; torch.cuda.empty_cache()
def labels_of(items): return sorted(set(c for _, c in items))
def indisp(cur, anchor_sd, V, carriers):
    """in-V displacement per carrier + per-carrier magnitude."""
    dV = {}; mag = {}
    with torch.no_grad():
        for n in carriers:
            d = (cur[n].detach().double() - anchor_sd[n].to(dev).double()); dv = (d @ V[n]) @ V[n].T; dV[n] = dv; mag[n] = float(dv.norm())
    return dV, mag


def main():
    global _T0
    t0 = time.time(); _T0 = t0
    print(f"[P11:{CELL}] {cfg['model']} REALIZED-alt specificity, N={len(SEEDS)} STEPS_ALT={STEPS_ALT} grid={LR_GRID}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    NC = len(carriers); posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[P11:{CELL}] w_A acc={pre_sst:.3f}; {NC} carriers ({time.time()-t0:.0f}s)", flush=True)
    exp_anchor = cfg.get("anchor", ANCHOR_WA[cfg["family"]])   # per-cell anchor override (opt13b=0.93)
    if abs(pre_sst - exp_anchor) > ANCHOR_TOL:
        json.dump({"cell": CELL, "VOID": True, "reason": f"w_A anchor off {pre_sst:.3f} vs {exp_anchor}"}, open(OUT, "w"), indent=2)
        print(f"[P11:{CELL}] HARD-STOP w_A off -> VOID", flush=True); return
    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); V[n] = vv[:, -KPROJ:].flip(dims=[1]).contiguous(); del FA[n]
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
    try:
        ho_tr = load_jsonB("medium_subj")[:-200]; assert len(ho_tr) > 100
        taskC = load_trec(); assert len(taskC) > 100
        stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]}
        stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}
        for sk, (it, _) in stream.items(): assert len(it) > 50, sk
    except Exception as e:
        json.dump({"cell": CELL, "VOID": True, "reason": f"dataset load failed: {str(e)[:120]}"}, open(OUT, "w"), indent=2)
        print(f"[P11:{CELL}] HARD-STOP datasets -> VOID. {e}", flush=True); return
    print(f"[P11:{CELL}] datasets loaded (phase-2 subj {len(ho_tr)}; alttask TREC {len(taskC)}; childhood {len(stream)})", flush=True)

    # ---------- STAGE 1: calibration (margin + K_CAL, extended-down grid) ----------
    thr = pre_sst - NONDESTRUCT_TOL; cal = {}
    print(f"[P11:{CELL}] STAGE-1 calibration; target r_cal>={thr+CAL_MARGIN:.3f}", flush=True)
    for lr in LR_GRID:
        accs = []
        for sd in range(K_CAL):
            model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, lr, sd); accs.append(eval_sst(model, tok, sst_eval, posid, negid))
        cal[lr] = float(np.mean(accs)); print(f"[cal] lr={lr:.1e} r_cal={cal[lr]:.4f} ({time.time()-_T0:.0f}s)", flush=True)
    q_m = [lr for lr in LR_GRID if cal[lr] >= thr + CAL_MARGIN]; q_b = [lr for lr in LR_GRID if cal[lr] >= thr]
    tight = False; no_nd = False
    if q_m: chosen_lr = max(q_m)
    elif q_b: chosen_lr = max(q_b); tight = True
    else: chosen_lr = min(LR_GRID); no_nd = True
    print(f"[P11:{CELL}] STAGE-1 DONE -> chosen_lr={chosen_lr:.1e} tight={tight} no_nd={no_nd}", flush=True)
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()

    # ---------- STAGE 2: realized-alt specificity ----------
    def snap(): return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    def perturb_eval(anchor_sd, fn):
        model.load_state_dict(anchor_sd)
        with torch.no_grad():
            p = dict(model.named_parameters())
            for n in carriers: p[n].add_(fn(n).to(p[n].dtype))
        return eval_sst(model, tok, sst_eval, posid, negid)
    def measure(anchor_sd, sd, hb):
        model.load_state_dict(anchor_sd); acc0 = eval_sst(model, tok, sst_eval, posid, negid)
        # phase-2 realized (subj) -> dVp, magp
        model.load_state_dict(anchor_sd); train_phase(model, tok, ho_tr, STEPS_P2, LR_DEPTH, sd, hb=hb)
        dVp, magp = indisp(dict(model.named_parameters()), anchor_sd, V, carriers)
        # alttask REALIZED (TREC) -> dVa (unit per carrier, applied at magp)
        model.load_state_dict(anchor_sd); train_phase(model, tok, taskC, STEPS_ALT, LR_DEPTH, sd + 90000)
        dVa, _ = indisp(dict(model.named_parameters()), anchor_sd, V, carriers)
        dphase2 = acc0 - perturb_eval(anchor_sd, lambda n: dVp[n])
        dalt = acc0 - perturb_eval(anchor_sd, lambda n: dVa[n] / (float(dVa[n].norm()) + 1e-30) * magp[n])
        V16 = {n: V[n][:, :TOPK].contiguous() for n in carriers}; drk = []
        for k in range(K_RAND):
            gen = torch.Generator(device="cpu").manual_seed(100000 * sd + 777 * k + 13)
            raws = {n: torch.randn(dict(model.named_parameters())[n].shape, generator=gen).double().to(dev) for n in carriers}
            def rfn(n, raws=raws):
                rv = (raws[n] @ V16[n]) @ V16[n].T; return rv / (float(rv.norm()) + 1e-30) * magp[n]
            drk.append(acc0 - perturb_eval(anchor_sd, rfn))
        model.load_state_dict(anchor_sd)
        return float(acc0), float(dphase2), float(dalt), float(np.mean(drk))

    dp2_n = []; dp2_c = []; dal_n = []; dal_c = []; drk_n = []; drk_c = []; retA_ch = []; ns = len(SEEDS)
    for si, sd in enumerate(SEEDS):
        lab = f"s{si+1}/{ns}"
        _, p2, al, rk = measure(ckpt_sd, sd, hb=f"{lab} NoCH_p2"); dp2_n.append(p2); dal_n.append(al); drk_n.append(rk)
        model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, chosen_lr, sd, hb=f"{lab} CH_child")
        ch_sd = snap(); retA_ch.append(eval_sst(model, tok, sst_eval, posid, negid))
        _, p2, al, rk = measure(ch_sd, sd, hb=f"{lab} CH_p2"); dp2_c.append(p2); dal_c.append(al); drk_c.append(rk)
        Np = float(np.mean(np.array(dp2_n) - np.array(dp2_c))); Na = float(np.mean(np.array(dal_n) - np.array(dal_c))); S = Np - Na
        print(f"[seed] {lab} dp2_n={dp2_n[-1]:.3f} dp2_c={dp2_c[-1]:.3f} dalt_n={dal_n[-1]:.3f} dalt_c={dal_c[-1]:.3f} drk_n={drk_n[-1]:.3f} retA={retA_ch[-1]:.3f} run(Np)={Np:+.4f} run(Na)={Na:+.4f} run(S)={S:+.4f} ({time.time()-_T0:.0f}s)", flush=True)

    me = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)]; a = np.array
    N_phase2 = me((a(dp2_n) - a(dp2_c)).tolist()); N_alt = me((a(dal_n) - a(dal_c)).tolist()); N_rk = me((a(drk_n) - a(drk_c)).tolist())
    S = me(((a(dp2_n) - a(dp2_c)) - (a(dal_n) - a(dal_c))).tolist())
    nondestruct = bool(np.mean(retA_ch) >= pre_sst - NONDESTRUCT_TOL)
    alt_saturated = bool(np.std(dal_n, ddof=1) < 1e-6)   # v1 failure mode: dalt_n zero variance
    bite_ok = bool(np.mean(dp2_n) > 0.03 and np.mean(dal_n) > 0.03)
    rep = {"cell": CELL, "family": cfg["family"], "model": cfg["model"], "n_carriers": NC, "n_seeds": ns, "w_A_acc": pre_sst, "VOID": False,
           "calibration_curve": {f"{k:.1e}": v for k, v in cal.items()}, "chosen_lr": chosen_lr, "tight_dose": tight, "no_nondestructive_point": no_nd,
           "ret_anchor_ch": me(retA_ch), "childhood_nondestructive": nondestruct, "alt_saturated": alt_saturated, "null_dead": (not bite_ok),
           "dphase2_noch": me(dp2_n), "dphase2_ch": me(dp2_c), "dalttask_noch": me(dal_n), "dalttask_ch": me(dal_c),
           "drandtop16_noch": me(drk_n), "drandtop16_ch": me(drk_c),
           "N_phase2": N_phase2, "N_alttask": N_alt, "N_randtop16": N_rk, "S_specificity": S,
           "SPECIFIC_if": "S>0(>=2s) AND N_phase2>N_randtop16", "GENERIC_if": "S~0 AND N_phase2~=N_alttask>N_randtop16"}
    json.dump(rep, open(OUT, "w"), indent=2)
    sg = lambda k: rep[k][0] / max(rep[k][1], 1e-9)
    print(f"[P11:{CELL}] chosen_lr={chosen_lr:.1e} nondestruct={nondestruct} alt_saturated={alt_saturated} bite={bite_ok} "
          f"(dp2_n={rep['dphase2_noch'][0]:.3f} dalt_n={rep['dalttask_noch'][0]:.3f}+-{rep['dalttask_noch'][1]:.3f} drk_n={rep['drandtop16_noch'][0]:.3f})", flush=True)
    print(f"[P11:{CELL}] N_phase2={N_phase2[0]:+.4f}({sg('N_phase2'):+.1f}s) N_alttask={N_alt[0]:+.4f}({sg('N_alttask'):+.1f}s) "
          f"N_randtop16={N_rk[0]:+.4f}({sg('N_randtop16'):+.1f}s) S={S[0]:+.4f}({sg('S_specificity'):+.1f}s) saved {OUT} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
