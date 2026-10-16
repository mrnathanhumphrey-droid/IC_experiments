"""RUNG-4 / #2c-ON-OPT (+ Pythia cross-arch) — ATTEMPT 3. Why does non-destructive warm-up protect? 3 candidates, on a
REPRODUCED rung-3 subj regime (hard-stop anchor), with the attempt-2 fixes:

  M1 BASIN-FLATTENING — FLAGGED WEAK. Attempt-2 used HVP curvature; but w_D is OFF task-A's minimum (D trained in the complement),
     so its Hessian inflates regardless of flattening (curvD came out ~10x curvA = anti-flat = off-min artifact). A symmetric small-eps
     sharpness probe just recovers curvature (same off-min issue). ATTEMPT-3: LARGE-eps symmetric perturbation in V -> task-A loss rise
     (basin-WIDTH, responsiveness, comparable across loss values), at w_A and w_D, reported but NOT a verdict driver — flattening may be
     intrinsically inseparable from off-min displacement here (a methodology note, banked).
  M2 PRE-MOVED-CAPACITY — phase-2 surge redirected OUT of V into the exercised complement (frac_in_V: D&B < A). Directional on both
     substrates in attempt-2 = the consistent survivor.
  M3 MAP-DEPOSITION — surge stays in V & RIDES B's deposit. ATTEMPT-2 CPU pre-check: raw align_M confounded (generic V-concentration);
     residualizing M against arm-A's generic surge dirs + the MATCHED-D contrast is sound. STATISTIC = resid_B - resid_D (B rides its
     OWN deposit beyond what the matched no-deposit childhood D does). Pre-check on Pythia: resid_B-resid_D=+0.046 (7/7), M seed-stable
     0.613; BUT generic-childhood routing is real (resid_D-resid_A=+0.043) so the D-contrast is mandatory. CAVEAT (Pythia, map inert):
     B rides deposit MORE than D yet protects no better -> riding != mechanism alone; OPT must show riding that TRACKS protection.

HARD-STOP REPRODUCE ANCHOR (pre-committed): OPT must reproduce rung-3 — arm-A subj-forget >= FORGET_GATE AND B-D >= REPRO_BD — else the
mechanism read is VOID (regime not reproduced; cannot explain an effect that isn't there, the att-2 emotion-regime failure). Held-out is
FORCED to subj (no auto-pick — att-2 drifted to emotion via stronger w_A). HF-retry wrapper kept. Saves OPT w_A.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"
OUT = f"{G}/rung4_mechanism.json"; NPZ = f"{G}/rung4_mechanism.npz"
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(7))
STEPS_DEPTH = 250; LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16
KPROJ = 128; N_FISHER = 200; K_M = 16; K_GEN = 16; CHANCE = K_M / KPROJ
SHARP_BATCH = 16; N_SBATCH = 4; K_SHARP = 6; ALPHAS = [0.02, 0.05]
FORCE_HELDOUT = "subj"; FORGET_GATE = 0.15; REPRO_BD = 0.08  # rung-3 reproduce anchor (OPT)

SUBSTRATES = [
    {"tag": "OPT", "model": "facebook/opt-350m",
     "carrier_keys": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
     "ckpt": None, "steps_wa": 500, "lr_wa": 5e-5, "save_wa": f"{G}/opt350m_sst2_wA.pt", "anchor": True},
    {"tag": "PYTHIA", "model": "EleutherAI/pythia-410m",
     "carrier_keys": ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"],
     "ckpt": f"{G}/pythia410m_sst2_subsetmin.pt", "steps_wa": 0, "lr_wa": 5e-5, "save_wa": None, "anchor": False},
]


def _retry(fn, tries=5, wait=12, what="hf"):
    last = None
    for i in range(tries):
        try:
            return fn()
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


def eval_clf(model, tok, eval_items, label_ids):
    model.eval(); correct = 0; n = 0; lid = torch.tensor(label_ids).to(dev)
    with torch.no_grad():
        for i in range(0, len(eval_items), 32):
            chunk = eval_items[i:i + 32]; seqs = [tok(p, add_special_tokens=False)["input_ids"][:256] for p, _ in chunk]
            L = max(len(x) for x in seqs); pad = tok.pad_token_id
            inp = np.full((len(seqs), L), pad, np.int64); pos = np.zeros(len(seqs), np.int64)
            for j, sq in enumerate(seqs): inp[j, :len(sq)] = sq; pos[j] = len(sq) - 1
            lg = model(input_ids=torch.tensor(inp).to(dev)).logits
            last = lg[torch.arange(len(seqs)), torch.tensor(pos).to(dev)]
            predpos = last[:, lid].argmax(1).cpu().numpy()
            for j, (_, co) in enumerate(chunk):
                true_id = tok(co, add_special_tokens=False)["input_ids"][0]; correct += int(label_ids[predpos[j]] == true_id); n += 1
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


# ---- M1: off-min-robust LARGE-eps sharpness (basin-width responsiveness), FLAGGED WEAK ----
def _base_loss(model, packs):
    s = 0.0
    with torch.no_grad():
        for (inp, lb, am) in packs: s += float(model(input_ids=inp, attention_mask=am, labels=lb).loss)
    return s / len(packs)


def sharpness_in_V(model, tok, packs, carriers, V, n_dir, seed, alphas):
    params = dict(model.named_parameters())
    wnorm = (sum(float((params[n].double() ** 2).sum()) for n in carriers)) ** 0.5
    b0 = _base_loss(model, packs); rng = torch.Generator(device="cpu").manual_seed(8000 + seed)
    out = {}
    for alpha in alphas:
        eps = alpha * wnorm; deltas = []
        for d in range(n_dir):
            v = {}
            for n in carriers:
                r = torch.randn(params[n].shape, generator=rng).to(dev).double()
                Vn = V[n].double(); v[n] = (r @ Vn) @ Vn.T
            nrm = (sum(float((v[n] * v[n]).sum()) for n in carriers)) ** 0.5
            for n in carriers: v[n] = v[n] / (nrm + 1e-30)
            dl = 0.0
            for sgn in (1.0, -1.0):
                with torch.no_grad():
                    for n in carriers: params[n].add_((sgn * eps) * v[n].to(params[n].dtype))
                dl += _base_loss(model, packs) - b0
                with torch.no_grad():
                    for n in carriers: params[n].sub_((sgn * eps) * v[n].to(params[n].dtype))
            deltas.append(dl / 2.0)
        out[alpha] = [float(np.mean(deltas)), float(np.std(deltas) / max(len(deltas), 1) ** 0.5)]
    return out, float(b0)


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


def surge_destination(post, start_sd, carriers, V, M, M_rand):
    inV = 0.0; tot = 0.0; inM = 0.0; inMr = 0.0
    with torch.no_grad():
        for n in carriers:
            dw2 = (post[n].detach() - start_sd[n].to(dev)).double(); coords = dw2 @ V[n].double()
            inV += float((coords * coords).sum()); tot += float((dw2 * dw2).sum())
            inM += float(((coords @ M[n]) ** 2).sum()); inMr += float(((coords @ M_rand[n]) ** 2).sum())
    return inV / max(tot, 1e-12), inM / max(inV, 1e-12), inMr / max(inV, 1e-12)


def subspace_align(post, start_sd, carriers, V, Q):
    inV = 0.0; inQ = 0.0
    with torch.no_grad():
        for n in carriers:
            dw2 = (post[n].detach() - start_sd[n].to(dev)).double(); coords = dw2 @ V[n].double()
            inV += float((coords * coords).sum())
            if Q[n].shape[1] > 0: inQ += float(((coords @ Q[n]) ** 2).sum())
    return inQ / max(inV, 1e-12)


def grass(A, B):
    k = min(A.shape[1], B.shape[1])
    return float((A.T @ B).pow(2).sum() / max(k, 1)) if k > 0 else 0.0


def labels_of(items): return sorted(set(c for _, c in items))
def first_ids(tok, lbls): return [tok(w, add_special_tokens=False)["input_ids"][0] for w in lbls]
def snap(model, carriers): p = dict(model.named_parameters()); return {n: p[n].detach().clone() for n in carriers}


def run(cfg, t0):
    tag = cfg["tag"]; print(f"\n[{tag}] loading {cfg['model']} (eager)...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32, attn_implementation="eager").to(dev)
    if cfg["ckpt"]:
        model.load_state_dict(torch.load(cfg["ckpt"])); print(f"[{tag}] loaded w_A ckpt", flush=True)
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["carrier_keys"]) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0)
    if cfg["steps_wa"] > 0:
        print(f"[{tag}] training task-A (SST) {cfg['steps_wa']} steps...", flush=True)
        train_single(model, tok, [(f"{s}\nSentiment:", LAB[l]) for s, l in sst_rows("train", 2000, 7)], cfg["steps_wa"], cfg["lr_wa"], 0)
    ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[{tag}] w_A SST acc={pre_sst:.3f} ({time.time()-t0:.0f}s)", flush=True)
    assert pre_sst > 0.68, f"w_A too weak ({pre_sst:.3f})"
    if cfg["save_wa"]:
        torch.save(ckpt_sd, cfg["save_wa"]); print(f"[{tag}] saved w_A -> {cfg['save_wa']}", flush=True)

    FA = fisher_taskA(model, params, carriers, tok); V = {}; M_rand = {}
    rng = torch.Generator(device="cpu").manual_seed(99)
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n]); V[n] = vv[:, -KPROJ:].float().contiguous()
        rr = torch.randn(KPROJ, K_M, generator=rng).double().to(dev); q, _ = torch.linalg.qr(rr); M_rand[n] = q[:, :K_M]
    model.load_state_dict(ckpt_sd); params = dict(model.named_parameters())
    print(f"[{tag}] V basis ({len(carriers)} carriers, KPROJ={KPROJ}) done ({time.time()-t0:.0f}s)", flush=True)

    spacks = [b_pairs(tok, [(f"{s}\nSentiment:", LAB[l]) for s, l in sst_rows("train", SHARP_BATCH, 555 + i)]) for i in range(N_SBATCH)]
    sharpA, baseA = sharpness_in_V(model, tok, spacks, carriers, V, K_SHARP, 0, ALPHAS)
    print(f"[{tag}] M1 sharpness @ w_A (FLAGGED WEAK) {sharpA} baseLoss {baseA:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # held-out = FORCED subj (no auto-pick); report its forget for the anchor
    heldout = FORCE_HELDOUT; ho_items = load_jsonB("medium_subj") if heldout == "subj" else load_hf(heldout)
    ho_tr, ho_ev = ho_items[:-200], ho_items[-200:]; ho_lids = first_ids(tok, labels_of(ho_items))
    model.load_state_dict(ckpt_sd); train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, 0)
    precheck_forget = pre_sst - eval_sst(model, tok, sst_eval, posid, negid); model.load_state_dict(ckpt_sd)
    print(f"[{tag}] HELD-OUT={heldout} FORCED (precheck forget {precheck_forget:.3f})", flush=True)

    stream_keys = ["agnews", "dbpedia", "yahoo", "newscat", ("emotion" if heldout == "subj" else "subj")]
    stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else (load_jsonB("medium_subj") if sk == "subj" else load_hf(sk))) for sk in stream_keys}
    stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}

    R = {a: {"sstA": [], "frac_inV": [], "align_M": [], "align_rand": [], "align_resid": []} for a in ["A", "B", "D"]}
    EX = {"sharpD": [], "baseD": [], "mapfrac_B": [], "mapfrac_D": [], "childB_drop": [], "childD_drop": [], "Mstab": [], "resid_dim": []}
    Mprev = None
    for sd in SEEDS:
        model.load_state_dict(ckpt_sd); b0, b1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd)
        wB = snap(model, carriers); mfB = map_subspace_frac(dict(model.named_parameters()), ckpt_sd, carriers, V)
        Mcur = map_dirs(wB, ckpt_sd, carriers, V)
        model.load_state_dict(ckpt_sd); d0, d1 = childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd, Vproj=V, carriers=carriers)
        wD = snap(model, carriers); mfD = map_subspace_frac(dict(model.named_parameters()), ckpt_sd, carriers, V)
        shD, bD = sharpness_in_V(model, tok, spacks, carriers, V, K_SHARP, sd, ALPHAS)
        if Mprev is not None:
            cc = carriers[:6]; EX["Mstab"].append(float(np.mean([grass(Mprev[n], Mcur[n]) for n in cc])))
        Mprev = Mcur
        EX["sharpD"].append(shD[ALPHAS[-1]][0]); EX["baseD"].append(bD); EX["mapfrac_B"].append(mfB); EX["mapfrac_D"].append(mfD)
        EX["childB_drop"].append(b0 - b1); EX["childD_drop"].append(d0 - d1)

        starts = {"A": ckpt_sd, "B": {**ckpt_sd, **wB}, "D": {**ckpt_sd, **wD}}; posts = {}
        for a in ["A", "B", "D"]:
            model.load_state_dict(starts[a]); train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
            posts[a] = snap(model, carriers); R[a]["sstA"].append(eval_sst(model, tok, sst_eval, posid, negid))
        G_A = surge_pcs(posts["A"], starts["A"], carriers, V, K_GEN); M_resid = residualize(Mcur, G_A)
        EX["resid_dim"].append(float(np.mean([M_resid[n].shape[1] for n in carriers])))
        for a in ["A", "B", "D"]:
            fiv, am, ar = surge_destination(posts[a], starts[a], carriers, V, Mcur, M_rand)
            R[a]["frac_inV"].append(fiv); R[a]["align_M"].append(am); R[a]["align_rand"].append(ar)
            R[a]["align_resid"].append(subspace_align(posts[a], starts[a], carriers, V, M_resid))
        print(f"[{tag}] seed {sd}: ret A/B/D={R['A']['sstA'][-1]:.3f}/{R['B']['sstA'][-1]:.3f}/{R['D']['sstA'][-1]:.3f} | "
              f"sharpD={shD[ALPHAS[-1]][0]:.3f}(A {sharpA[ALPHAS[-1]][0]:.3f}) baseD {bD:.3f}(A {baseA:.3f}) | "
              f"fracV A/B/D={R['A']['frac_inV'][-1]:.3f}/{R['B']['frac_inV'][-1]:.3f}/{R['D']['frac_inV'][-1]:.3f} | "
              f"resid A/B/D={R['A']['align_resid'][-1]:.3f}/{R['B']['align_resid'][-1]:.3f}/{R['D']['align_resid'][-1]:.3f} | mfB/D={mfB:.3f}/{mfD:.3f} ({time.time()-t0:.0f}s)", flush=True)

    def me(x): x = np.array(x, float); return [float(x.mean()), float(x.std() / max(len(x), 1) ** 0.5)]
    aR = {a: np.array(R[a]["align_resid"]) for a in R}
    rep = {"tag": tag, "model": cfg["model"], "heldout": heldout, "pre_sst": pre_sst, "anchor_substrate": cfg["anchor"],
           "M1_sharpness": {"wA": sharpA, "wD_largeEps": me(EX["sharpD"]), "baseLoss_wA": baseA, "baseLoss_wD": me(EX["baseD"]), "note": "FLAGGED WEAK (off-min)"},
           "M2": {"frac_inV_A": me(R["A"]["frac_inV"]), "frac_inV_B": me(R["B"]["frac_inV"]), "frac_inV_D": me(R["D"]["frac_inV"])},
           "M3": {"resid_A": me(R["A"]["align_resid"]), "resid_B": me(R["B"]["align_resid"]), "resid_D": me(R["D"]["align_resid"]),
                  "resid_B_minus_D": me(aR["B"] - aR["D"]), "resid_D_minus_A": me(aR["D"] - aR["A"]),
                  "align_M_raw": {a: me(R[a]["align_M"]) for a in R}, "M_stability": me(EX["Mstab"]) if EX["Mstab"] else [0, 0], "resid_dim": me(EX["resid_dim"])},
           "retention": {"A": me(R["A"]["sstA"]), "B": me(R["B"]["sstA"]), "D": me(R["D"]["sstA"]),
                         "B_minus_D": me(np.array(R["B"]["sstA"]) - np.array(R["D"]["sstA"])),
                         "D_minus_A": me(np.array(R["D"]["sstA"]) - np.array(R["A"]["sstA"]))},
           "guards": {"mapfrac_B": me(EX["mapfrac_B"]), "mapfrac_D": me(EX["mapfrac_D"]), "childB_drop": me(EX["childB_drop"]), "childD_drop": me(EX["childD_drop"])}}
    try:
        A_forget = pre_sst - np.mean(R["A"]["sstA"])
        bmd_arr = np.array(R["B"]["sstA"]) - np.array(R["D"]["sstA"]); bmd = float(np.mean(bmd_arr))
        fA = np.mean(R["A"]["frac_inV"]); fB = np.mean(R["B"]["frac_inV"]); fD = np.mean(R["D"]["frac_inV"])
        rBD = aR["B"] - aR["D"]; rBD_m, rBD_se = me(rBD.tolist()); rDA_m = float(np.mean(aR["D"] - aR["A"]))
        mfB = np.mean(EX["mapfrac_B"]); mfD = np.mean(EX["mapfrac_D"]); mstab = np.mean(EX["Mstab"]) if EX["Mstab"] else 0.0
        guard_mapfrac = (mfB > 0.05) and (mfD < 0.5 * mfB); guard_Mstab = mstab > CHANCE + 0.02
        regime_ok = (A_forget >= FORGET_GATE); map_loadbearing = (bmd >= REPRO_BD)
        M2_fires = (fD < fA - 0.02) and (fB < fA - 0.02)
        M3_fires = (rBD_m > 2 * rBD_se) and (rBD_m > 0.01) and guard_Mstab and guard_mapfrac
        rep["fires"] = {"M2": bool(M2_fires), "M3_deposit_riding": bool(M3_fires), "regime_ok": bool(regime_ok),
                        "map_loadbearing": bool(map_loadbearing), "guard_mapfrac": bool(guard_mapfrac), "guard_Mstab": bool(guard_Mstab)}
        rep["anchor"] = {"A_forget": float(A_forget), "B_minus_D": bmd, "forget_gate": FORGET_GATE, "repro_bd": REPRO_BD}
        if cfg["anchor"] and not (regime_ok and map_loadbearing):
            v = (f"VOID (anchor substrate, rung-3 NOT reproduced): A_forget={A_forget:.3f} (need >={FORGET_GATE}), B-D={bmd:+.3f} (need >={REPRO_BD}). "
                 f"Cannot read mechanism where the protection effect is absent. M2 fracV A/B/D {fA:.3f}/{fB:.3f}/{fD:.3f}, resid_B-D={rBD_m:+.3f} reported but UNANCHORED.")
        else:
            tag_regime = "load-bearing map" if map_loadbearing else "map-inert"
            m3 = (f"M3 DEPOSIT-RIDING {'FIRES' if M3_fires else 'NULL'}: resid_B-resid_D={rBD_m:+.3f}±{rBD_se:.3f} "
                  f"(generic-childhood resid_D-A={rDA_m:+.3f}, M-stab {mstab:.3f})")
            m2 = f"M2 CAPACITY {'FIRES' if M2_fires else 'flat'}: fracV A/B/D {fA:.3f}/{fB:.3f}/{fD:.3f}"
            if M3_fires and map_loadbearing:
                v = (f"{tag_regime}: {m2}; {m3}. B rides its deposit beyond the matched no-deposit childhood AND the map is load-bearing "
                     f"(B-D={bmd:+.3f}) -> MAP-DEPOSITION supported as the mechanism for the map's contribution.")
            elif M3_fires and not map_loadbearing:
                v = (f"{tag_regime}: {m3} BUT map inert (B-D={bmd:+.3f}~0) -> deposit-riding EXISTS without protecting = riding != mechanism "
                     f"(the Pythia caveat confirmed). {m2}.")
            else:
                v = f"{tag_regime}: {m2}; {m3} -> map-deposition NOT supported; capacity is what survives. B-D={bmd:+.3f}."
        rep["verdict"] = v
        print(f"\n[{tag}] === RUNG-4 VERDICT ({tag}, anchor={cfg['anchor']}) ===\n  {v}", flush=True)
        print(f"  ANCHOR A_forget {A_forget:.3f} (gate {FORGET_GATE}) | B-D {bmd:+.3f} (gate {REPRO_BD}) | ret A/B/D {np.mean(R['A']['sstA']):.3f}/{np.mean(R['B']['sstA']):.3f}/{np.mean(R['D']['sstA']):.3f}", flush=True)
        print(f"  M1 sharpness wA {sharpA[ALPHAS[-1]][0]:.3f} -> wD {np.mean(EX['sharpD']):.3f} (baseLoss wA {baseA:.3f} wD {np.mean(EX['baseD']):.3f}) [WEAK/off-min]", flush=True)
        print(f"  M2 fracV {fA:.3f}/{fB:.3f}/{fD:.3f} | M3 resid_B-D {rBD_m:+.3f}±{rBD_se:.3f} (D-A {rDA_m:+.3f}, M-stab {mstab:.3f}) | mfB/D {mfB:.3f}/{mfD:.3f}", flush=True)
    except Exception as e:
        rep["error"] = traceback.format_exc(); print(f"[{tag}] readout error (saved): {e}", flush=True)
    arrays = {f"{tag}_{a}_{m}": np.array(R[a][m]) for a in R for m in R[a]}
    arrays.update({f"{tag}_{k}": np.array(EX[k]) for k in EX})
    return rep, arrays


def main():
    t0 = time.time(); print("[R4] RUNG-4 ATTEMPT 3 (force-subj + reproduce-anchor; M2 / M3=resid_B-D; M1 sharpness WEAK) on OPT + Pythia", flush=True)
    reps = {}; allarr = {}
    for cfg in SUBSTRATES:
        try:
            rep, arr = run(cfg, t0); reps[cfg["tag"]] = rep; allarr.update(arr)
        except Exception as e:
            reps[cfg["tag"]] = {"error": traceback.format_exc()}; print(f"[{cfg['tag']}] SUBSTRATE FAILED: {e}", flush=True)
    try:
        cross = "(incomplete)"
        if "fires" in reps.get("OPT", {}) and "fires" in reps.get("PYTHIA", {}):
            of = reps["OPT"]["fires"]; pf = reps["PYTHIA"]["fires"]
            if not (of["regime_ok"] and of["map_loadbearing"]):
                cross = "OPT anchor NOT reproduced -> cross-arch UNREADABLE (the load-bearing regime is the whole comparison)."
            elif of["M3_deposit_riding"] and not pf["M3_deposit_riding"]:
                cross = "CROSS-ARCH SPLIT: OPT deposit-riding (load-bearing) / Pythia none (inert) -> map-deposition is OPT's mechanism, explains rung-3 inversion."
            elif of["M3_deposit_riding"] and pf["M3_deposit_riding"]:
                cross = "BOTH ride deposit, but only OPT's protects (Pythia inert) -> riding is necessary-not-sufficient; protection needs riding + (no-antagonism?). Refines toggle target."
            else:
                cross = "OPT no deposit-riding either -> capacity universal; map's load-bearing 74% needs a 4th account."
        reps["cross_arch"] = cross; print(f"\n[R4] === CROSS-ARCH ===\n  {cross}", flush=True)
    except Exception as e:
        reps["cross_arch_error"] = str(e)
    np.savez(NPZ, **allarr); json.dump(reps, open(OUT, "w"), indent=2)
    print(f"[R4] saved rung4_mechanism.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
