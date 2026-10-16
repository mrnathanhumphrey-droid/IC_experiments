"""RUNG-4 M3 ON OPT — characterize the load-bearing deposit-routing mechanism at MATCHED CONDITIONS.

The cross-arch dispositive (N=30) stands and is real-but-unexplained: OPT B-D=+0.055 (map carries ~61% of protection), Pythia
B-D=-0.006 (inert); antagonism died at matched w_A. This run asks HOW OPT's deposited map routes the phase-2 surge to protect —
NOT testing antagonism (dead), just characterizing the load-bearing mechanism on the architecture where it's load-bearing.

STATISTIC (locked, validated): resid_B - resid_D, per (layer,carrier), NEVER pooled. resid_X = fraction of arm-X's phase-2 in-V
surge energy that lands in M_resid = (deposit-PCs of B's phase-1 Δw) residualized against arm-A's generic surge dirs. The matched-D
contrast controls generic-childhood routing (D rides B's deposit-residual too; resid_B-resid_D = B riding its OWN deposit beyond D).

MATCHED-CONDITIONS TRIPWIRES:
  M-1 matched w_A: load opt350m_sst2_wA.pt (the +0.055 operating point), NOT retrain. Same regime (lr/steps/held-out=subj).
  M-2 matched N: N=30 paired A/B/D (7-seed voided 3x today).
  M-3 matched arms: A=no childhood, B=natural replay, D=replay w/ grad projected out of V (mapfrac confirmed ~0.014).
REPRODUCE-KNOWN ANCHOR (hard VOID if not met): B-A~+0.089, B-D~+0.055, D-A~+0.034(ns), mapfrac B/D ~0.219/0.014, A_forget~0.237.
  regime_ok = A_forget>=0.15 AND B-D>=0.03 (load-bearing reproduced). M-stability re-checked HERE (not imported from Pythia precheck).
VERDICT->ROWS (quarantine auto-positive without ALL four): (a) resid_B-D per (layer,carrier) distribution; (b) M seed-stable >chance
  at THIS w_A; (c) matched-D baseline resid_D-resid_A reported (if large, riding not B-specific); (d) within-carrier-type breakdown
  (q/k/v/out_proj, fc1, fc2 separately — pooling manufactured SPIRAL-5's false +0.29). >=4/6 carrier-types positive required.
CUT-1 SWAP CONTROL (inline, mandatory if M3 fires): re-measure resid_B-D against M_topcurv (top-K_M curvature PCs at w_A) residualized
  the same way. If lift vs M_topcurv ~= lift vs deposit-M, M3 is generic curvature-family routing NOT deposit-specific (killed Thread 3's
  false PAYS_STRICT). M3 positive counts ONLY if deposit-M lift EXCEEDS M_topcurv lift by a margin.
PRE-REG direction: predict resid_B-D > 0 (deposit-riding) IF deposit-routing is the mechanism; null if the +0.055 routes elsewhere.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"
MODEL = "facebook/opt-350m"; CKPT = f"{G}/opt350m_sst2_wA.pt"
OUT = f"{G}/opt_m3.json"; NPZ = f"{G}/opt_m3.npz"
CARRIER_KEYS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(30))
STEPS_DEPTH = 250; LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16
KPROJ = 128; N_FISHER = 200; K_M = 16; K_GEN = 16; CHANCE = K_M / KPROJ
FORCE_HELDOUT = "subj"; FORGET_GATE = 0.15; REPRO_BD = 0.03
DEPOSIT_MARGIN = 0.01  # deposit-M lift must exceed M_topcurv lift by this to count as deposit-specific


def _retry(fn, tries=5, wait=12, what="hf"):
    last = None
    for i in range(tries):
        try: return fn()
        except Exception as e:
            last = e; print(f"[{what}] retry {i+1}/{tries}: {str(e)[:120]}", flush=True); time.sleep(wait)
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
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    for s, l in sst_rows("train", N_FISHER, 0):
        inp, lb, am = b_pairs(tok, [(f"{s}\nSentiment:", LAB[l])]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: g = params[n].grad.detach().double(); F[n] += g.T @ g
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


def train_single(model, tok, items, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def childhood(model, tok, stream, steps, lr, seed, Vproj=None, carriers=None):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    tasks = list(stream); losses = []; params = dict(model.named_parameters())
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, labels = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); batch = [items[i] for i in idx]
        inp, lb, am = b_pairs(tok, batch)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        if Vproj is not None:
            for n in carriers: gd = params[n].grad; gd.sub_((gd @ Vproj[n]) @ Vproj[n].T)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.item()))
    return float(np.mean(losses[:20])), float(np.mean(losses[-20:]))


def map_subspace_frac(params, ckpt_sd, carriers, V):
    num = 0.0; den = 0.0
    with torch.no_grad():
        for n in carriers:
            dw = (params[n].detach() - ckpt_sd[n].to(dev)).double(); proj = dw @ V[n].double()
            num += float((proj * proj).sum()); den += float((dw * dw).sum())
    return num / max(den, 1e-12)


def map_dirs(wB_params, ckpt_sd, carriers, V):
    M = {}
    with torch.no_grad():
        for n in carriers:
            dwB = (wB_params[n].detach() - ckpt_sd[n].to(dev)).double(); coords = dwB @ V[n].double()
            U, S, Vh = torch.linalg.svd(coords, full_matrices=False); M[n] = Vh[:K_M].T.contiguous()
    return M


def surge_pcs(post, start_sd, carriers, V, k):
    Gd = {}
    with torch.no_grad():
        for n in carriers:
            dw = (post[n].detach() - start_sd[n].to(dev)).double(); coords = dw @ V[n].double()
            U, S, Vh = torch.linalg.svd(coords, full_matrices=False); Gd[n] = Vh[:k].T.contiguous()
    return Gd


def residualize(M, Gd, eps=1e-2):
    R = {}
    for n in M:
        P = M[n] - Gd[n] @ (Gd[n].T @ M[n])
        U, S, Vh = torch.linalg.svd(P, full_matrices=False)
        keep = int((S > eps * float(S.max())).sum().item()) if (S.numel() and float(S.max()) > 0) else 0
        R[n] = U[:, :keep]
    return R


def align_per_name(post, start_sd, carriers, V, Qresid, QresidTC):
    """per-carrier-name: fiv (M2), align_resid (M3 deposit), align_residTC (M3 swap)."""
    fiv = {}; ar = {}; arTC = {}
    with torch.no_grad():
        for n in carriers:
            dw = (post[n].detach() - start_sd[n].to(dev)).double(); coords = dw @ V[n].double()
            inV = float((coords * coords).sum()); tot = float((dw * dw).sum())
            fiv[n] = inV / max(tot, 1e-12)
            ar[n] = (float(((coords @ Qresid[n]) ** 2).sum()) / max(inV, 1e-12)) if Qresid[n].shape[1] > 0 else 0.0
            arTC[n] = (float(((coords @ QresidTC[n]) ** 2).sum()) / max(inV, 1e-12)) if QresidTC[n].shape[1] > 0 else 0.0
    return fiv, ar, arTC


def grass(A, B):
    k = min(A.shape[1], B.shape[1])
    return float((A.T @ B).pow(2).sum() / max(k, 1)) if k > 0 else 0.0


def labels_of(items): return sorted(set(c for _, c in items))
def first_ids(tok, lbls): return [tok(w, add_special_tokens=False)["input_ids"][0] for w in lbls]
def snap(model, carriers): p = dict(model.named_parameters()); return {n: p[n].detach().clone() for n in carriers}
def ctype(n):
    for c in CARRIER_KEYS:
        if c in n: return c
    return "?"


def main():
    t0 = time.time(); print("[M3] RUNG-4 M3 ON OPT — matched-conditions deposit-routing mechanism (N=30, pinned w_A)", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); print(f"[M3] loaded pinned w_A {CKPT}", flush=True)
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in CARRIER_KEYS) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0)
    ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[M3] w_A SST acc={pre_sst:.3f} (dispositive 0.823); {len(carriers)} carriers ({time.time()-t0:.0f}s)", flush=True)
    assert pre_sst > 0.68, f"w_A too weak ({pre_sst:.3f})"

    FA = fisher_taskA(model, params, carriers, tok); V = {}; M_topcurv = {}
    eye = torch.eye(KPROJ, dtype=torch.float64, device=dev)
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n]); V[n] = vv[:, -KPROJ:].flip(dims=[1]).float().contiguous()   # col0=highest λ
        M_topcurv[n] = eye[:, :K_M].clone()   # top-K_M curvature axes IN V-coordinate space [KPROJ,K_M] (swap control)
    model.load_state_dict(ckpt_sd); params = dict(model.named_parameters())
    print(f"[M3] V basis + M_topcurv (KPROJ={KPROJ}, K_M={K_M}) done ({time.time()-t0:.0f}s)", flush=True)

    heldout = FORCE_HELDOUT; ho_items = load_jsonB("medium_subj"); ho_tr = ho_items[:-200]
    stream_keys = ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]
    stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in stream_keys}
    stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}
    print(f"[M3] held-out=subj FORCED; stream={list(stream)} ({time.time()-t0:.0f}s)", flush=True)

    # per-arm per-name accumulators
    ret = {a: [] for a in ["A", "B", "D"]}
    fivN = {a: {n: [] for n in carriers} for a in ["A", "B", "D"]}
    arN = {a: {n: [] for n in carriers} for a in ["A", "B", "D"]}      # resid vs deposit-M
    arTCN = {a: {n: [] for n in carriers} for a in ["A", "B", "D"]}    # resid vs M_topcurv (swap)
    EX = {"mapfrac_B": [], "mapfrac_D": [], "childB_drop": [], "childD_drop": [], "Mstab": [], "resid_dim": []}
    Mprev = None
    for sd in SEEDS:
        # B childhood -> deposit + M
        model.load_state_dict(ckpt_sd); b0, b1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd)
        wB = snap(model, carriers); mfB = map_subspace_frac(dict(model.named_parameters()), ckpt_sd, carriers, V)
        Mcur = map_dirs(wB, ckpt_sd, carriers, V)
        # D childhood (map-stripped)
        model.load_state_dict(ckpt_sd); d0, d1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, Vproj=V, carriers=carriers)
        wD = snap(model, carriers); mfD = map_subspace_frac(dict(model.named_parameters()), ckpt_sd, carriers, V)
        if Mprev is not None: EX["Mstab"].append(float(np.mean([grass(Mprev[n], Mcur[n]) for n in carriers])))
        Mprev = Mcur
        EX["mapfrac_B"].append(mfB); EX["mapfrac_D"].append(mfD); EX["childB_drop"].append(b0 - b1); EX["childD_drop"].append(d0 - d1)
        # depth phase per arm
        starts = {"A": ckpt_sd, "B": {**ckpt_sd, **wB}, "D": {**ckpt_sd, **wD}}; posts = {}
        for a in ["A", "B", "D"]:
            model.load_state_dict(starts[a]); train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
            posts[a] = snap(model, carriers); ret[a].append(eval_sst(model, tok, sst_eval, posid, negid))
        # residualize deposit-M and M_topcurv against arm-A generic surge dirs
        G_A = surge_pcs(posts["A"], starts["A"], carriers, V, K_GEN)
        M_resid = residualize(Mcur, G_A); M_residTC = residualize(M_topcurv, G_A)
        EX["resid_dim"].append(float(np.mean([M_resid[n].shape[1] for n in carriers])))
        for a in ["A", "B", "D"]:
            fiv, ar, arTC = align_per_name(posts[a], starts[a], carriers, V, M_resid, M_residTC)
            for n in carriers: fivN[a][n].append(fiv[n]); arN[a][n].append(ar[n]); arTCN[a][n].append(arTC[n])
        if sd % 5 == 0 or sd == SEEDS[-1]:
            rB = np.mean([np.mean(arN["B"][n]) for n in carriers]); rD = np.mean([np.mean(arN["D"][n]) for n in carriers])
            print(f"[M3] seed {sd}: ret A/B/D={ret['A'][-1]:.3f}/{ret['B'][-1]:.3f}/{ret['D'][-1]:.3f} | resid B/D(meanCarr)={rB:.3f}/{rD:.3f} | mfB/D={mfB:.3f}/{mfD:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # ===== aggregate =====
    A = np.array(ret["A"]); B = np.array(ret["B"]); D = np.array(ret["D"])
    def me(x): x = np.array(x, float); return [float(x.mean()), float(x.std(ddof=1) / max(len(x), 1) ** 0.5)]
    # per-seed pooled-over-carriers-by-MEAN (equal carrier weight, NOT energy pooled): resid_X[seed] = mean_n align
    def perseed_meanCarr(arDict):
        return np.array([np.mean([arDict[n][s] for n in carriers]) for s in range(len(SEEDS))])
    residB = perseed_meanCarr(arN["B"]); residD = perseed_meanCarr(arN["D"]); residA = perseed_meanCarr(arN["A"])
    residB_TC = perseed_meanCarr(arTCN["B"]); residD_TC = perseed_meanCarr(arTCN["D"])
    rBD = residB - residD; rDA = residD - residA
    rBD_TC = residB_TC - residD_TC
    # per-carrier-TYPE resid_B-D (row d)
    bytype = {}
    for c in CARRIER_KEYS:
        cn = [n for n in carriers if ctype(n) == c]
        diffs = np.array([np.mean([arN["B"][n][s] - arN["D"][n][s] for n in cn]) for s in range(len(SEEDS))])
        bytype[c] = {"resid_B_minus_D": me(diffs.tolist()), "frac_seed_pos": float(np.mean(diffs > 0)), "n_layers": len(cn)}
    # per-(layer,carrier) distribution (row a): mean over seeds of (B-D) for each name
    perlayer = np.array([np.mean([arN["B"][n][s] - arN["D"][n][s] for s in range(len(SEEDS))]) for n in carriers])

    rep = {"tag": "OPT", "model": MODEL, "heldout": heldout, "pre_sst": pre_sst, "n_seeds": len(SEEDS),
           "retention": {"A": me(A), "B": me(B), "D": me(D), "B_minus_A": me((B - A).tolist()),
                         "B_minus_D": me((B - D).tolist()), "D_minus_A": me((D - A).tolist())},
           "guards": {"mapfrac_B": me(EX["mapfrac_B"]), "mapfrac_D": me(EX["mapfrac_D"]),
                      "childB_drop": me(EX["childB_drop"]), "childD_drop": me(EX["childD_drop"]),
                      "M_stability": me(EX["Mstab"]) if EX["Mstab"] else [0, 0], "resid_dim": me(EX["resid_dim"])},
           "M2_fracV": {"A": me([np.mean([fivN['A'][n][s] for n in carriers]) for s in range(len(SEEDS))]),
                        "B": me([np.mean([fivN['B'][n][s] for n in carriers]) for s in range(len(SEEDS))]),
                        "D": me([np.mean([fivN['D'][n][s] for n in carriers]) for s in range(len(SEEDS))])},
           "M3": {"resid_A": me(residA.tolist()), "resid_B": me(residB.tolist()), "resid_D": me(residD.tolist()),
                  "resid_B_minus_D": me(rBD.tolist()), "resid_D_minus_A": me(rDA.tolist()),
                  "swap_resid_B_minus_D_topcurv": me(rBD_TC.tolist()),
                  "deposit_specific_margin": float(np.mean(rBD) - np.mean(rBD_TC)),
                  "by_carrier_type": bytype, "frac_seed_pos_BD": float(np.mean(rBD > 0)),
                  "perlayer_BD_mean": float(perlayer.mean()), "perlayer_BD_fracpos": float(np.mean(perlayer > 0))}}
    np.savez(NPZ, A=A, B=B, D=D, residB=residB, residD=residD, residA=residA, residB_TC=residB_TC, residD_TC=residD_TC,
             perlayer_BD=perlayer, mapfrac_B=np.array(EX["mapfrac_B"]), mapfrac_D=np.array(EX["mapfrac_D"]),
             Mstab=np.array(EX["Mstab"]), carriers=np.array(carriers))
    try:
        A_forget = pre_sst - A.mean()
        ba_m, ba_se = me((B - A).tolist()); bd_m, bd_se = me((B - D).tolist()); da_m, da_se = me((D - A).tolist())
        mfB = np.mean(EX["mapfrac_B"]); mfD = np.mean(EX["mapfrac_D"]); mstab = np.mean(EX["Mstab"]) if EX["Mstab"] else 0.0
        rBD_m, rBD_se = me(rBD.tolist()); rDA_m = float(np.mean(rDA)); margin = float(np.mean(rBD) - np.mean(rBD_TC))
        # ROWS for the verdict->rows tripwire
        guard_mapfrac = (mfB > 0.05) and (mfD < 0.5 * mfB)
        guard_Mstab = mstab > CHANCE + 0.02
        regime_ok = (A_forget >= FORGET_GATE) and (bd_m >= REPRO_BD)
        n_types_pos = sum(1 for c in CARRIER_KEYS if bytype[c]["resid_B_minus_D"][0] > 0)
        row_d_ok = n_types_pos >= 4                       # not pooling-driven by one carrier type
        deposit_specific = margin > DEPOSIT_MARGIN        # swap control: deposit-M lift exceeds curvature lift
        stat_pos = (rBD_m > 2 * rBD_se) and (rBD_m > 0.01)
        rows_ok = guard_Mstab and guard_mapfrac and row_d_ok and (rDA_m < rBD_m + 0.05)  # (c): D-A not swamping B-D
        rep["rows"] = {"a_perlayer_fracpos": float(np.mean(perlayer > 0)), "b_Mstab": float(mstab), "b_Mstab_chance": CHANCE,
                       "c_resid_D_minus_A": rDA_m, "d_n_carrier_types_pos": n_types_pos,
                       "swap_margin_deposit_minus_topcurv": margin, "deposit_specific": bool(deposit_specific)}
        if not regime_ok:
            v = (f"VOID (dispositive regime NOT reproduced): A_forget={A_forget:.3f} (need>={FORGET_GATE}), B-D={bd_m:+.3f} "
                 f"(need>={REPRO_BD}). Cannot characterize a mechanism for an effect that didn't reproduce. resid_B-D={rBD_m:+.3f} UNANCHORED.")
            fires = "VOID"
        elif stat_pos and rows_ok and deposit_specific:
            v = (f"M3 DEPOSIT-ROUTING FIRES (corroborated): resid_B-D={rBD_m:+.3f}±{rBD_se:.3f}, deposit-specific (margin vs topcurv "
                 f"{margin:+.3f}>{DEPOSIT_MARGIN}), {n_types_pos}/6 carrier-types pos, M-stab {mstab:.3f}>{CHANCE+0.02:.3f}, matched-D baseline "
                 f"resid_D-A={rDA_m:+.3f}. B rides its OWN deposit beyond matched-D -> deposit-routing is the mechanism for OPT's load-bearing "
                 f"map (B-D={bd_m:+.3f}, B-A={ba_m:+.3f}). Explains ~the +0.055 half.")
            fires = "FIRES"
        elif stat_pos and not (rows_ok and deposit_specific):
            v = (f"M3 QUARANTINED (stat-positive but rows fail): resid_B-D={rBD_m:+.3f}±{rBD_se:.3f} BUT deposit_specific={deposit_specific} "
                 f"(swap margin {margin:+.3f}), carrier-types pos {n_types_pos}/6, M-stab {mstab:.3f}, resid_D-A={rDA_m:+.3f}. "
                 f"Positive does NOT survive the corroboration rows (curvature-family routing and/or pooling and/or generic-childhood "
                 f"confound) -> NOT deposit-specific mechanism. needs-corroboration, do not claim.")
            fires = "QUARANTINE"
        else:
            v = (f"M3 NULL: resid_B-D={rBD_m:+.3f}±{rBD_se:.3f} (n.s. or <0.01), {n_types_pos}/6 carrier-types pos, M-stab {mstab:.3f}, "
                 f"resid_D-A={rDA_m:+.3f}. B does NOT ride its deposit beyond matched-D -> deposit-routing is NOT the mechanism; OPT's "
                 f"+0.055 (B-D={bd_m:+.3f}) is real-but-unexplained even at the mechanism layer. Map routes the surge through something else.")
            fires = "NULL"
        rep["fires"] = fires; rep["verdict"] = v
        rep["anchor"] = {"A_forget": float(A_forget), "B_minus_A": ba_m, "B_minus_D": bd_m, "D_minus_A": da_m,
                         "mapfrac_B": float(mfB), "mapfrac_D": float(mfD), "regime_ok": bool(regime_ok),
                         "dispositive_ref": {"B_minus_A": 0.089, "B_minus_D": 0.055, "D_minus_A": 0.034, "A_forget": 0.237, "mapfrac": "0.219/0.014"}}
        print(f"\n[M3] === RUNG-4 M3 ON OPT VERDICT ({fires}) ===\n  {v}", flush=True)
        print(f"  ANCHOR (vs dispositive): B-A={ba_m:+.3f}(ref+0.089) B-D={bd_m:+.3f}(ref+0.055) D-A={da_m:+.3f}(ref+0.034) "
              f"A_forget={A_forget:.3f}(ref0.237) mapfrac B/D {mfB:.3f}/{mfD:.3f}(ref .219/.014) regime_ok={regime_ok}", flush=True)
        print(f"  M3 resid_B-D={rBD_m:+.3f}±{rBD_se:.3f} | swap(topcurv)={float(np.mean(rBD_TC)):+.3f} margin={margin:+.3f} | "
              f"resid_D-A={rDA_m:+.3f} | M-stab={mstab:.3f}(chance {CHANCE:.3f}) | frac_seed_pos={float(np.mean(rBD>0)):.2f}", flush=True)
        print(f"  ROWS: (a)perlayer fracpos={float(np.mean(perlayer>0)):.2f} (b)Mstab={mstab:.3f} (c)resid_D-A={rDA_m:+.3f} (d)types_pos={n_types_pos}/6 | deposit_specific={deposit_specific}", flush=True)
        for c in CARRIER_KEYS:
            bt = bytype[c]; print(f"    [{c:9s}] resid_B-D={bt['resid_B_minus_D'][0]:+.3f}±{bt['resid_B_minus_D'][1]:.3f} fracpos={bt['frac_seed_pos']:.2f} (n_layers {bt['n_layers']})", flush=True)
        print(f"  M2 fracV A/B/D = {rep['M2_fracV']['A'][0]:.3f}/{rep['M2_fracV']['B'][0]:.3f}/{rep['M2_fracV']['D'][0]:.3f}", flush=True)
    except Exception as e:
        rep["error"] = traceback.format_exc(); print(f"[M3] readout error (saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[M3] saved opt_m3.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
