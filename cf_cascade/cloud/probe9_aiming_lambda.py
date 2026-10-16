"""PROBE 9 — specificity aiming-axis (geometric overlap Omega vs semantic sigma_sem). PREREG_PROBE9_AIMING_AXIS_2026_06_28 (LOCK 11fe73b3).
Successor to Probe-8 v4. task-A=SST-2, phase-2=subj FIXED; only childhood stream varies. V=top-128 task-A Fisher @ w_A.
Two modes (env MODE):
  screen  : Stage-0. For each candidate childhood stream, Omega = principal-angle overlap (cos2 theta_min + mean cos2) between
            the M_OMEGA-seed childhood-displacement subspace and the (shared) phase-2-from-w_A subspace, in V. -> screen JSON.
  measure : Stage-1 for one STREAM. Per-stream non-destructive lr calibration (margin), then paired S=N(phase2)-N(alttask)
            over N seeds with REALIZED short-training directions (v5 relabel) + sigma_sem (linear CKA). -> measure JSON.
Omega cross-Gram is memory-light (Frobenius inner products additive over carriers). HARD-STOP (VOID) on dataset/anchor fail.
"""
import os, time, json, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(100)); M_OMEGA = 12; N_SEM = 256
LR_GRID = [1e-5, 1.5e-5, 2e-5, 3e-5]; K_CAL = 20; NONDESTRUCT_TOL = 0.05; CAL_MARGIN = 0.015; SCREEN_LR = 3e-5
LR_DEPTH = 5e-5; STEPS_P2 = 250; STEPS_ALT = 60; STEPS_CHILD = 400; BATCH = 16; KPROJ = 128; TOPK = 16; N_FISHER = 200; K_RAND = 5
EPS = 1e-6; HEARTBEAT = 25; _T0 = 0.0
ANCHOR_WA = {"OPT": 0.823, "PYTHIA": 0.753}; ANCHOR_TOL = 0.02
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",            "ck": OPT_C, "family": "OPT"},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt",  "ck": PY_C,  "family": "PYTHIA"},
}
# locked candidate pool (Stage-0) — names -> childhood stream keys
CANDIDATES = {
    "agnews": ["agnews"], "dbpedia": ["dbpedia"], "yahoo": ["yahoo"], "newscat": ["newscat"], "emotion": ["emotion"],
    "agnews+dbpedia": ["agnews", "dbpedia"], "yahoo+emotion": ["yahoo", "emotion"], "newscat+dbpedia": ["newscat", "dbpedia"],
    "mix5": ["agnews", "dbpedia", "yahoo", "newscat", "emotion"],
}
CELL = os.environ.get("CELL", ""); assert CELL in CELLS, f"CELL band-only; got {CELL!r}"
MODE = os.environ.get("MODE", "screen"); assert MODE in ("screen", "measure")
STREAM = os.environ.get("STREAM", "mix5")
cfg = CELLS[CELL]; OUT = f"{G}/probe9_{MODE}_{CELL}" + (f"_{STREAM}" if MODE == "measure" else "") + ".json"


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
def build_stream(keys): return {k: (load_hf(k), None) for k in keys} if False else {k: ((load_hf(k)), None) for k in keys}


# ---- NEW MATH (dry-run-tested) ----
def vcoeff_flat(params, anchor_sd, V, carriers):
    """Flattened in-V displacement in V-coordinates: cat_carrier vec((w-w_A)@V). Returns 1-D tensor on dev."""
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

def vproj_dir(params, anchor_sd, V, carriers):
    """Per-carrier in-V displacement (dout x cin) and its per-carrier norm."""
    out = {}; mag = {}
    with torch.no_grad():
        for n in carriers:
            d = (params[n].detach().double() - anchor_sd[n].to(dev).double()); dv = (d @ V[n]) @ V[n].T
            out[n] = dv; mag[n] = float(dv.norm())
    return out, mag

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


def main():
    global _T0
    t0 = time.time(); _T0 = t0
    print(f"[P9:{CELL}:{MODE}] {cfg['model']} aiming-axis MODE={MODE} STREAM={STREAM if MODE=='measure' else 'ALL'} ({time.ctime()})", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    NC = len(carriers); posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[P9:{CELL}] w_A acc={pre_sst:.3f}; {NC} carriers ({time.time()-t0:.0f}s)", flush=True)
    if abs(pre_sst - ANCHOR_WA[cfg["family"]]) > ANCHOR_TOL:
        json.dump({"cell": CELL, "mode": MODE, "VOID": True, "reason": f"w_A anchor off {pre_sst:.3f}"}, open(OUT, "w"), indent=2)
        print(f"[P9:{CELL}] HARD-STOP w_A off -> VOID", flush=True); return
    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); V[n] = vv[:, -KPROJ:].flip(dims=[1]).contiguous(); del FA[n]
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
    try:
        ho_tr = load_jsonB("medium_subj")[:-200]; assert len(ho_tr) > 100
        taskC = load_trec(); assert len(taskC) > 100
        allkeys = sorted(set(k for ks in CANDIDATES.values() for k in ks))
        streams_raw = {k: load_hf(k) for k in allkeys}
        for k, it in streams_raw.items(): assert len(it) > 50, k
    except Exception as e:
        json.dump({"cell": CELL, "mode": MODE, "VOID": True, "reason": f"dataset load failed: {str(e)[:120]}"}, open(OUT, "w"), indent=2)
        print(f"[P9:{CELL}] HARD-STOP datasets -> VOID. {e}", flush=True); return
    def mk_stream(keys): return {k: (streams_raw[k], labels_of(streams_raw[k])) for k in keys}
    print(f"[P9:{CELL}] datasets loaded (subj {len(ho_tr)}; TREC {len(taskC)}; streams {list(streams_raw)})", flush=True)

    # ===== MODE screen: Omega for each candidate =====
    if MODE == "screen":
        # shared phase-2-from-w_A subspace
        Xp_cols = []
        for i in range(M_OMEGA):
            model.load_state_dict(ckpt_sd); train_phase(model, tok, ho_tr, STEPS_P2, LR_DEPTH, i, hb=(f"p2 {i+1}/{M_OMEGA}" if i % 4 == 0 else ""))
            Xp_cols.append(vcoeff_flat(dict(model.named_parameters()), ckpt_sd, V, carriers))
        Xp = torch.stack(Xp_cols, 1); del Xp_cols; torch.cuda.empty_cache()
        print(f"[omega] phase-2 subspace built (M={M_OMEGA}, dim={Xp.shape[0]}) ({time.time()-_T0:.0f}s)", flush=True)
        res = {}
        for cname, keys in CANDIDATES.items():
            st = mk_stream(keys); Xc_cols = []
            for i in range(M_OMEGA):
                model.load_state_dict(ckpt_sd); childhood(model, tok, st, STEPS_CHILD, SCREEN_LR, i)
                Xc_cols.append(vcoeff_flat(dict(model.named_parameters()), ckpt_sd, V, carriers))
            Xc = torch.stack(Xc_cols, 1); c2min, c2mean = omega(Xc, Xp); del Xc_cols, Xc; torch.cuda.empty_cache()
            res[cname] = {"cos2_min": c2min, "cos2_mean": c2mean, "keys": keys}
            print(f"[omega] {cname:18s} cos2_min={c2min:.4f} cos2_mean={c2mean:.4f} ({time.time()-_T0:.0f}s)", flush=True)
        # MEASURE-ALL ladder (re-lock 2026-06-28): NO down-select. Stage-1 runs every candidate clearing the
        # non-destructive gate (checked in measure mode). Screen = abort-gate only: span + degeneracy.
        names = list(res); vals = np.array([res[n]["cos2_min"] for n in names])
        span = float(vals.max() - vals.min())
        degenerate = [n for n in names if res[n]["cos2_min"] > 0.98 or res[n]["cos2_min"] < 1e-4]
        ladder_healthy = bool(span >= 0.15 and len(degenerate) == 0)  # abort-gate: ladder must span Omega, no degenerate streams
        rep = {"cell": CELL, "mode": "screen", "VOID": False, "M_omega": M_OMEGA, "screen_lr": SCREEN_LR,
               "omega": res, "ladder_all": names, "omega_span": span, "degenerate": degenerate, "ladder_healthy": ladder_healthy}
        json.dump(rep, open(OUT, "w"), indent=2)
        print(f"[P9:{CELL}] SCREEN DONE span={span:.3f} healthy={ladder_healthy} degenerate={degenerate} omegas={ {n: round(res[n]['cos2_min'],3) for n in names} } saved {OUT} wall={time.time()-t0:.0f}s", flush=True)
        return

    # ===== MODE measure: full paired-S + sigma_sem for one STREAM =====
    assert STREAM in CANDIDATES, f"STREAM {STREAM} not in candidate pool"
    st = mk_stream(CANDIDATES[STREAM])
    # sigma_sem (CKA frozen base: stream inputs vs subj inputs)
    stream_texts = [p for k in CANDIDATES[STREAM] for (p, _) in streams_raw[k]][:N_SEM]
    subj_texts = [p for (p, _) in ho_tr][:N_SEM]
    Hs = hidden_reps(model, tok, stream_texts); Hsub = hidden_reps(model, tok, subj_texts)
    sigma_sem = linear_cka(Hs, Hsub)
    print(f"[P9:{CELL}:{STREAM}] sigma_sem(CKA)={sigma_sem:.4f} ({time.time()-_T0:.0f}s)", flush=True)
    # per-stream non-destructive lr calibration (margin rule)
    thr = pre_sst - NONDESTRUCT_TOL; cal = {}
    for lr in LR_GRID:
        accs = []
        for sd in range(K_CAL):
            model.load_state_dict(ckpt_sd); childhood(model, tok, st, STEPS_CHILD, lr, sd); accs.append(eval_sst(model, tok, sst_eval, posid, negid))
        cal[lr] = float(np.mean(accs)); print(f"[cal] {STREAM} lr={lr:.1e} r_cal={cal[lr]:.4f} ({time.time()-_T0:.0f}s)", flush=True)
    q_m = [lr for lr in LR_GRID if cal[lr] >= thr + CAL_MARGIN]; q_b = [lr for lr in LR_GRID if cal[lr] >= thr]
    chosen_lr = max(q_m) if q_m else (max(q_b) if q_b else min(LR_GRID)); tight = (not q_m) and bool(q_b); no_nd = not q_b
    print(f"[P9:{CELL}:{STREAM}] chosen_lr={chosen_lr:.1e} tight={tight} no_nd={no_nd}", flush=True)
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()

    def snap(): return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    def perturb_eval(anchor_sd, fn):
        model.load_state_dict(anchor_sd)
        with torch.no_grad():
            p = dict(model.named_parameters())
            for n in carriers: p[n].add_(fn(n).to(p[n].dtype))
        return eval_sst(model, tok, sst_eval, posid, negid)
    def measure(anchor_sd, sd, hb):
        model.load_state_dict(anchor_sd); acc0 = eval_sst(model, tok, sst_eval, posid, negid)
        train_phase(model, tok, ho_tr, STEPS_P2, LR_DEPTH, sd, hb=hb)               # realized phase-2
        dVp, magp = vproj_dir(dict(model.named_parameters()), anchor_sd, V, carriers)
        model.load_state_dict(anchor_sd); train_phase(model, tok, taskC, STEPS_ALT, LR_DEPTH, sd + 90000)  # realized alttask (short)
        dVa, _ = vproj_dir(dict(model.named_parameters()), anchor_sd, V, carriers)
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

    dp2n = []; dp2c = []; daln = []; dalc = []; drkn = []; drkc = []; retA = []
    ns = len(SEEDS)
    for si, sd in enumerate(SEEDS):
        lab = f"{STREAM} s{si+1}/{ns}"
        _, p2, al, rk = measure(ckpt_sd, sd, hb=f"{lab} NoCH"); dp2n.append(p2); daln.append(al); drkn.append(rk)
        model.load_state_dict(ckpt_sd); childhood(model, tok, st, STEPS_CHILD, chosen_lr, sd, hb=f"{lab} CHc")
        ch_sd = snap(); retA.append(eval_sst(model, tok, sst_eval, posid, negid))
        _, p2, al, rk = measure(ch_sd, sd, hb=f"{lab} CH"); dp2c.append(p2); dalc.append(al); drkc.append(rk)
        Np = float(np.mean(np.array(dp2n) - np.array(dp2c))); Na = float(np.mean(np.array(daln) - np.array(dalc)))
        print(f"[seed] {lab} dp2_n={dp2n[-1]:.3f} dp2_c={dp2c[-1]:.3f} dalt_n={daln[-1]:.3f} dalt_c={dalc[-1]:.3f} retA={retA[-1]:.3f} run(Nphase2)={Np:+.4f} run(Nalt)={Na:+.4f} run(S)={Np-Na:+.4f} ({time.time()-_T0:.0f}s)", flush=True)

    me = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)]; a = np.array
    Nph = me((a(dp2n) - a(dp2c)).tolist()); Nal = me((a(daln) - a(dalc)).tolist()); Nrk = me((a(drkn) - a(drkc)).tolist())
    S = me(((a(dp2n) - a(dp2c)) - (a(daln) - a(dalc))).tolist())
    rep = {"cell": CELL, "family": cfg["family"], "mode": "measure", "stream": STREAM, "keys": CANDIDATES[STREAM], "VOID": False,
           "n_seeds": ns, "w_A_acc": pre_sst, "sigma_sem": sigma_sem, "chosen_lr": chosen_lr, "tight_dose": tight, "no_nondestructive_point": no_nd,
           "ret_anchor_ch": me(retA), "childhood_nondestructive": bool(np.mean(retA) >= pre_sst - NONDESTRUCT_TOL),
           "null_dead": bool(not (np.mean(dp2n) > 0.03 and np.mean(daln) > 0.03)),
           "dphase2_noch": me(dp2n), "dphase2_ch": me(dp2c), "dalttask_noch": me(daln), "dalttask_ch": me(dalc),
           "drandtop16_noch": me(drkn), "drandtop16_ch": me(drkc),
           "N_phase2": Nph, "N_alttask": Nal, "N_randtop16": Nrk, "S_specificity": S, "calibration_curve": {f"{k:.1e}": v for k, v in cal.items()}}
    json.dump(rep, open(OUT, "w"), indent=2)
    sg = lambda k: rep[k][0] / max(rep[k][1], 1e-9)
    print(f"[P9:{CELL}:{STREAM}] sigma_sem={sigma_sem:.3f} chosen_lr={chosen_lr:.1e} nondestruct={rep['childhood_nondestructive']} null_dead={rep['null_dead']}", flush=True)
    print(f"[P9:{CELL}:{STREAM}] N_phase2={Nph[0]:+.4f}({sg('N_phase2'):+.1f}s) N_alttask={Nal[0]:+.4f}({sg('N_alttask'):+.1f}s) S={S[0]:+.4f}({sg('S_specificity'):+.1f}s) saved {OUT} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
