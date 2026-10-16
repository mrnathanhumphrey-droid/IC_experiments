"""PROBE 9b — Omega via the PHASE-2 LEVER (geometric overlap Omega vs Ramasesh-semantic sigma_sem, HEAD-TO-HEAD).
PREREG_PROBE9b_PHASE2_LEVER_2026_06_28 (LOCK 869ce94e). Successor to Probe-9 (screen showed Omega NOT dialable by
childhood-stream choice -> VERDICT_PROBE9_SCREEN). The pivot: Omega is set by the INTERFERENCE direction = the phase-2 task,
so sweep Omega by varying phase-2 (FIXED childhood=mix5, FIXED task-A=SST-2). task-A<->phase-2 semantic similarity (the
Ramasesh axis) now varies WITH phase-2, so sigma_sem is the DIRECT competing axis -> partials are a real geometric-vs-semantic test.

V = top-128 task-A Fisher @ w_A. fp32. Two modes (env MODE):
  screen  : Stage-0 abort-gate. childhood subspace built ONCE; per phase-2 task d, Omega(d)=principal-angle overlap
            (cos2_min + mean) between childhood-displacement subspace and phase-2(d)-from-w_A subspace, in V. Also
            calibrates the FIXED-childhood non-destructive lr (margin rule, K_CAL). -> screen JSON (span + abort-gate).
  measure : Stage-1 for ONE phase-2 task d (env PHASE2). Paired N=100 protection P(d)=forget_NoCH(d)-forget_CH(d)
            (Probe-7 Delta-forget form, REALIZED training — no perturbation estimator) + sigma_sem(d) CKA + Omega(d)
            (self-contained triple). childhood lr from env CHILD_LR (fallback: self-calibrate). -> measure JSON.
Omega cross-Gram is memory-light. HARD-STOP (VOID) on dataset/anchor fail; VOID-soft on tripped non-destruct gate.
"""
import os, time, json, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(100)); M_OMEGA = 12; N_SEM = 256
LR_GRID = [1e-5, 1.5e-5, 2e-5, 3e-5]; K_CAL = 20; NONDESTRUCT_TOL = 0.05; CAL_MARGIN = 0.015; SCREEN_LR = 3e-5
LR_DEPTH = 5e-5; STEPS_P2 = 250; STEPS_CHILD = 400; BATCH = 16; KPROJ = 128; N_FISHER = 200
EPS = 1e-6; HEARTBEAT = 25; _T0 = 0.0
ANCHOR_WA = {"OPT": 0.823, "PYTHIA": 0.753}; ANCHOR_TOL = 0.02
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",            "ck": OPT_C, "family": "OPT"},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt",  "ck": PY_C,  "family": "PYTHIA"},
}
# FIXED childhood (the protective stream) — held constant across the whole probe.
CHILD_KEYS = ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]   # = mix5
# locked phase-2 pool (the Omega lever): vary the INTERFERENCE task. names -> phase-2 keys.
PHASE2_POOL = {
    "subj": ["subj"], "agnews": ["agnews"], "dbpedia": ["dbpedia"], "yahoo": ["yahoo"],
    "newscat": ["newscat"], "emotion": ["emotion"], "trec": ["trec"],
    "subj+trec": ["subj", "trec"], "agnews+yahoo": ["agnews", "yahoo"],   # combos to thicken the ladder (>=7 needed)
}
CELL = os.environ.get("CELL", ""); assert CELL in CELLS, f"CELL band-only; got {CELL!r}"
MODE = os.environ.get("MODE", "screen"); assert MODE in ("screen", "measure")
PHASE2 = os.environ.get("PHASE2", "subj")
cfg = CELLS[CELL]; OUT = f"{G}/probe9b_{MODE}_{CELL}" + (f"_{PHASE2}" if MODE == "measure" else "") + ".json"


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
    if key == "agnews": return load_jsonB("low_agnews")
    raise ValueError(key)
def load_phase2(keys):
    """Phase-2 (interference) items for a pool entry — concat of its keys. subj/trec routed specially."""
    out = []
    for k in keys:
        if k == "subj": out += load_jsonB("medium_subj")[:-200]
        elif k == "trec": out += load_trec()
        else: out += load_hf(k)
    return out
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
    tasks = list(stream); losses = []
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.item()))
        if hb and (step % HEARTBEAT == 0 or step == steps - 1):
            print(f"[hb] {hb} childhood {step+1}/{steps} loss={float(loss):.3f} ({time.time()-_T0:.0f}s)", flush=True)
    del opt; torch.cuda.empty_cache(); return float(np.mean(losses[-20:]))
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


# ---- MATH (dry-run-tested; omega identical to probe9) ----
def vcoeff_flat(params, anchor_sd, V, carriers):
    """Flattened in-V displacement in V-coordinates: cat_carrier vec((w-w_A)@V). 1-D tensor on dev."""
    parts = []
    with torch.no_grad():
        for n in carriers:
            d = (params[n].detach().double() - anchor_sd[n].to(dev).double()); parts.append((d @ V[n]).reshape(-1))
    return torch.cat(parts)

def omega(Xc, Xp):
    """Principal-angle overlap between span(Xc) and span(Xp). Xc:(P,Mc), Xp:(P,Mp). Returns (cos2_min, cos2_mean)."""
    def inv_sqrt(Gm):
        w, U = torch.linalg.eigh(Gm); w = torch.clamp(w, min=1e-10); return (U * w.rsqrt()) @ U.T
    Gcc = Xc.T @ Xc; Gpp = Xp.T @ Xp; Gcp = Xc.T @ Xp
    Mx = inv_sqrt(Gcc) @ Gcp @ inv_sqrt(Gpp)
    s = torch.clamp(torch.linalg.svdvals(Mx), 0.0, 1.0); c2 = s * s
    return float(c2.max()), float(c2.mean())

def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True); Y = Y - Y.mean(0, keepdim=True)
    xy = (X.T @ Y).pow(2).sum(); xx = (X.T @ X).norm(); yy = (Y.T @ Y).norm()
    return float(xy / (xx * yy + 1e-12))

def hidden_reps(model, tok, texts, n=N_SEM):
    model.eval(); reps = []
    with torch.no_grad():
        for i in range(0, min(len(texts), n), 16):
            enc = tok(texts[i:i + 16], return_tensors="pt", padding=True, truncation=True, max_length=64).to(dev)
            h = model(**enc, output_hidden_states=True).hidden_states[-1]
            m = enc["attention_mask"].unsqueeze(-1).float(); reps.append(((h * m).sum(1) / m.sum(1)).float().cpu())
    return torch.cat(reps)


def build_childhood_subspace(model, tok, ckpt_sd, child_stream, V, carriers, lr):
    """M_OMEGA realizations of the FIXED childhood displacement, in V-coords. Returns Xc:(P,M_OMEGA)."""
    cols = []
    for i in range(M_OMEGA):
        model.load_state_dict(ckpt_sd); childhood(model, tok, child_stream, STEPS_CHILD, lr, i,
                                                  hb=(f"childsub {i+1}/{M_OMEGA}" if i % 4 == 0 else ""))
        cols.append(vcoeff_flat(dict(model.named_parameters()), ckpt_sd, V, carriers))
    return torch.stack(cols, 1)

def phase2_subspace(model, tok, ckpt_sd, items, V, carriers, hb_name):
    """M_OMEGA realizations of phase-2(d) displacement from w_A, in V-coords. Returns Xp:(P,M_OMEGA)."""
    cols = []
    for i in range(M_OMEGA):
        model.load_state_dict(ckpt_sd); train_phase(model, tok, items, STEPS_P2, LR_DEPTH, i,
                                                     hb=(f"{hb_name} p2sub {i+1}/{M_OMEGA}" if i % 4 == 0 else ""))
        cols.append(vcoeff_flat(dict(model.named_parameters()), ckpt_sd, V, carriers))
    return torch.stack(cols, 1)

def calibrate_child_lr(model, tok, ckpt_sd, child_stream, sst_eval, posid, negid, pre_sst):
    """Per-cell non-destructive childhood lr via the margin rule (K_CAL seeds). Returns (chosen_lr, cal, tight, no_nd)."""
    thr = pre_sst - NONDESTRUCT_TOL; cal = {}
    for lr in LR_GRID:
        accs = []
        for sd in range(K_CAL):
            model.load_state_dict(ckpt_sd); childhood(model, tok, child_stream, STEPS_CHILD, lr, sd)
            accs.append(eval_sst(model, tok, sst_eval, posid, negid))
        cal[lr] = float(np.mean(accs)); print(f"[cal] child lr={lr:.1e} r_cal={cal[lr]:.4f} ({time.time()-_T0:.0f}s)", flush=True)
    q_m = [lr for lr in LR_GRID if cal[lr] >= thr + CAL_MARGIN]; q_b = [lr for lr in LR_GRID if cal[lr] >= thr]
    chosen = max(q_m) if q_m else (max(q_b) if q_b else min(LR_GRID)); tight = (not q_m) and bool(q_b); no_nd = not q_b
    return chosen, cal, tight, no_nd


def main():
    global _T0
    t0 = time.time(); _T0 = t0
    print(f"[P9b:{CELL}:{MODE}] {cfg['model']} phase-2-lever MODE={MODE} PHASE2={PHASE2 if MODE=='measure' else 'POOL'} ({time.ctime()})", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    NC = len(carriers); posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[P9b:{CELL}] w_A acc={pre_sst:.3f}; {NC} carriers ({time.time()-t0:.0f}s)", flush=True)
    if abs(pre_sst - ANCHOR_WA[cfg["family"]]) > ANCHOR_TOL:
        json.dump({"cell": CELL, "mode": MODE, "VOID": True, "reason": f"w_A anchor off {pre_sst:.3f}"}, open(OUT, "w"), indent=2)
        print(f"[P9b:{CELL}] HARD-STOP w_A off -> VOID", flush=True); return
    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); V[n] = vv[:, -KPROJ:].flip(dims=[1]).contiguous(); del FA[n]
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()

    # datasets: childhood streams + phase-2 pool (or single phase-2 for measure)
    try:
        child_raw = {k: load_hf(k) for k in CHILD_KEYS}
        for k, it in child_raw.items(): assert len(it) > 50, k
        child_stream = {k: (child_raw[k], labels_of(child_raw[k])) for k in CHILD_KEYS}
        if MODE == "screen":
            p2_items = {name: load_phase2(keys) for name, keys in PHASE2_POOL.items()}
        else:
            assert PHASE2 in PHASE2_POOL, f"PHASE2 {PHASE2} not in pool"
            p2_items = {PHASE2: load_phase2(PHASE2_POOL[PHASE2])}
        for name, it in p2_items.items(): assert len(it) > 100, name
    except Exception as e:
        json.dump({"cell": CELL, "mode": MODE, "VOID": True, "reason": f"dataset load failed: {str(e)[:120]}"}, open(OUT, "w"), indent=2)
        print(f"[P9b:{CELL}] HARD-STOP datasets -> VOID. {e}", flush=True); return
    print(f"[P9b:{CELL}] datasets loaded (child { {k: len(v) for k,v in child_raw.items()} }; phase2 { {k: len(v) for k,v in p2_items.items()} })", flush=True)

    # ===== MODE screen: childhood subspace ONCE, Omega(d) for each phase-2 task + child lr calibration =====
    if MODE == "screen":
        Xc = build_childhood_subspace(model, tok, ckpt_sd, child_stream, V, carriers, SCREEN_LR)
        print(f"[omega] childhood subspace built (M={M_OMEGA}, dim={Xc.shape[0]}) ({time.time()-_T0:.0f}s)", flush=True)
        res = {}
        for name, items in p2_items.items():
            Xp = phase2_subspace(model, tok, ckpt_sd, items, V, carriers, name)
            c2min, c2mean = omega(Xc, Xp); del Xp; torch.cuda.empty_cache()
            res[name] = {"cos2_min": c2min, "cos2_mean": c2mean, "keys": PHASE2_POOL[name]}
            print(f"[omega] {name:14s} cos2_min={c2min:.4f} cos2_mean={c2mean:.4f} ({time.time()-_T0:.0f}s)", flush=True)
        names = list(res); vals = np.array([res[n]["cos2_min"] for n in names])
        span = float(vals.max() - vals.min())
        degenerate = [n for n in names if res[n]["cos2_min"] > 0.98 or res[n]["cos2_min"] < 1e-4]
        n_nondeg = len(names) - len(degenerate)
        ladder_healthy = bool(span >= 0.15 and n_nondeg >= 7)   # locked abort-gate: span>=0.15 AND >=7 non-degenerate points
        # FIXED-childhood non-destructive lr (margin rule) — persisted for the measure fan-out (pass via CHILD_LR env)
        chosen_lr, cal, tight, no_nd = calibrate_child_lr(model, tok, ckpt_sd, child_stream, sst_eval, posid, negid, pre_sst)
        model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
        rep = {"cell": CELL, "family": cfg["family"], "mode": "screen", "VOID": False, "M_omega": M_OMEGA, "screen_lr": SCREEN_LR,
               "omega": res, "ladder_all": names, "omega_span": span, "degenerate": degenerate, "n_nondegenerate": n_nondeg,
               "ladder_healthy": ladder_healthy, "child_keys": CHILD_KEYS,
               "chosen_child_lr": chosen_lr, "child_calibration": {f"{k:.1e}": v for k, v in cal.items()}, "child_tight": tight, "child_no_nondestructive": no_nd}
        json.dump(rep, open(OUT, "w"), indent=2)
        print(f"[P9b:{CELL}] SCREEN DONE span={span:.3f} healthy={ladder_healthy} n_nondeg={n_nondeg} degenerate={degenerate} chosen_child_lr={chosen_lr:.1e}", flush=True)
        print(f"[P9b:{CELL}] >>> CHILD_LR={chosen_lr:.1e} <<< (pass to measure)  omegas={ {n: round(res[n]['cos2_min'],3) for n in names} } saved {OUT} wall={time.time()-t0:.0f}s", flush=True)
        return

    # ===== MODE measure: paired protection P(d) + sigma_sem(d) + Omega(d) for ONE phase-2 task =====
    items = p2_items[PHASE2]
    # sigma_sem(d): CKA(frozen base on phase-2(d) inputs, task-A SST-2 inputs) = the Ramasesh axis
    p2_texts = [p for (p, _) in items][:N_SEM]
    taskA_texts = [f"{s}\nSentiment:" for (s, _) in sst_rows("train", N_SEM, 0)]
    Hp = hidden_reps(model, tok, p2_texts); Ha = hidden_reps(model, tok, taskA_texts)
    sigma_sem = linear_cka(Hp, Ha)
    print(f"[P9b:{CELL}:{PHASE2}] sigma_sem(CKA)={sigma_sem:.4f} ({time.time()-_T0:.0f}s)", flush=True)
    # Omega(d): self-contained (childhood subspace + phase-2(d) subspace, SCREEN_LR/LR_DEPTH) -> full triple
    Xc = build_childhood_subspace(model, tok, ckpt_sd, child_stream, V, carriers, SCREEN_LR)
    Xp = phase2_subspace(model, tok, ckpt_sd, items, V, carriers, PHASE2)
    omega_min, omega_mean = omega(Xc, Xp); del Xc, Xp; torch.cuda.empty_cache()
    print(f"[P9b:{CELL}:{PHASE2}] Omega cos2_min={omega_min:.4f} cos2_mean={omega_mean:.4f} ({time.time()-_T0:.0f}s)", flush=True)
    # childhood lr: from env (set by orchestrator from screen) else self-calibrate (robust on fresh fan-out box)
    env_lr = os.environ.get("CHILD_LR", "")
    if env_lr:
        chosen_lr = float(env_lr); cal = None; tight = None; no_nd = None
        print(f"[P9b:{CELL}:{PHASE2}] CHILD_LR from env={chosen_lr:.1e}", flush=True)
    else:
        chosen_lr, cal, tight, no_nd = calibrate_child_lr(model, tok, ckpt_sd, child_stream, sst_eval, posid, negid, pre_sst)
        print(f"[P9b:{CELL}:{PHASE2}] self-calibrated CHILD_LR={chosen_lr:.1e} tight={tight} no_nd={no_nd}", flush=True)
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()

    # paired protection: per seed, NoCH (w_A -> phase-2) vs CH (w_A -> childhood -> phase-2). REALIZED forgetting.
    forget_n = []; forget_c = []; retA = []; ns = len(SEEDS)
    for si, sd in enumerate(SEEDS):
        lab = f"{PHASE2} s{si+1}/{ns}"
        # NoCH arm
        model.load_state_dict(ckpt_sd); train_phase(model, tok, items, STEPS_P2, LR_DEPTH, sd, hb=f"{lab} NoCH")
        acc_n = eval_sst(model, tok, sst_eval, posid, negid); forget_n.append(pre_sst - acc_n)
        # CH arm
        model.load_state_dict(ckpt_sd); childhood(model, tok, child_stream, STEPS_CHILD, chosen_lr, sd, hb=f"{lab} CHc")
        ret = eval_sst(model, tok, sst_eval, posid, negid); retA.append(ret)
        ch_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ch_sd); train_phase(model, tok, items, STEPS_P2, LR_DEPTH, sd, hb=f"{lab} CH")
        acc_c = eval_sst(model, tok, sst_eval, posid, negid); forget_c.append(ret - acc_c)
        P = float(np.mean(np.array(forget_n) - np.array(forget_c)))
        print(f"[seed] {lab} forget_n={forget_n[-1]:.3f} forget_c={forget_c[-1]:.3f} retA={ret:.3f} run(P)={P:+.4f} ({time.time()-_T0:.0f}s)", flush=True)
    model.load_state_dict(ckpt_sd)

    me = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)]; a = np.array
    P = me((a(forget_n) - a(forget_c)).tolist())
    rep = {"cell": CELL, "family": cfg["family"], "mode": "measure", "phase2": PHASE2, "keys": PHASE2_POOL[PHASE2], "VOID": False,
           "n_seeds": ns, "w_A_acc": pre_sst, "chosen_child_lr": chosen_lr, "child_lr_from_env": bool(env_lr),
           "omega_cos2_min": omega_min, "omega_cos2_mean": omega_mean, "sigma_sem": sigma_sem,
           "ret_anchor_ch": me(retA), "childhood_nondestructive": bool(np.mean(retA) >= pre_sst - NONDESTRUCT_TOL),
           "forget_noch": me(forget_n), "forget_ch": me(forget_c), "P_protection": P}
    if cal is not None: rep["child_calibration"] = {f"{k:.1e}": v for k, v in cal.items()}; rep["child_tight"] = tight; rep["child_no_nondestructive"] = no_nd
    json.dump(rep, open(OUT, "w"), indent=2)
    sg = P[0] / max(P[1], 1e-9)
    print(f"[P9b:{CELL}:{PHASE2}] Omega={omega_min:.3f} sigma_sem={sigma_sem:.3f} forget_n={rep['forget_noch'][0]:.3f} forget_c={rep['forget_ch'][0]:.3f} "
          f"P={P[0]:+.4f}({sg:+.1f}s) nondestruct={rep['childhood_nondestructive']} saved {OUT} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
