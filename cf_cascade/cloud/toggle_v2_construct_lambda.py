"""ANTAGONISM TOGGLE v2 — PHASE 1: CONSTRUCT + VALIDATE the synthetic task-B family. NO ARMS FIRED here.

The design crux of the whole experiment. Build a synthetic classification task-B(c) whose NATURAL gradient at w_A carries a
prescribed cosine c to task-A's gradient WITHIN the shared subspace V, while holding SHARING (subspace occupancy) FIXED across c.
No gradient surgery — antagonism is a property of the DATA (the labels), so perturbation magnitude is natural & constant. This
fixes the prior toggle's confound (surgery angle ∝ |c−c_nat| made D−A collapse in lockstep with B−D).

CONSTRUCTION (mirror of the dispositive's objects):
  w_A = pythia410m_sst2_subsetmin.pt; carriers = qkv/h_to_4h/4h_to_h; V = top-128 task-A Fisher per carrier (the shared subspace).
  proj_A = task-A decision direction in final-hidden space = unembed(e_pos) − unembed(e_neg) (the SST-2 boundary).
  proj_orth = a FIXED random unit direction in final-hidden space, orthogonalized vs proj_A, SAME across all levels.
  For input pool (SST-2 train sentences) at w_A: z_A(x)=standardize(h_final·proj_A), z_orth(x)=standardize(h_final·proj_orth).
  y_synth(x; c) = 1[ c·z_A(x) + sqrt(1−c²)·z_orth(x) > 0 ]  -> binary, emitted as " positive"/" negative" (task-A's exact I/O).
  Only the proj_A coefficient (c) changes across levels; proj_orth component is fixed -> sharing held fixed, cosine varies.

VALIDATION (the gates; report all, declare PASS/FAIL — do NOT fire arms unless construction is sound):
  Sweep nominal c in {−0.5,−0.23,0.0,+0.20,+0.5}. For each:
   (a) ACHIEVED COSINE: cos(g_A^V, g_B^V(c)) where g^V = loss-gradient on carriers projected into V (the rung-2 antagonism measure
       in the shared subspace). Must be MONOTONE in nominal c and SPAN a useful range incl ~−0.23.
   (b) SHARING: frac of g_B(c) mass inside V (||g_B@V|| / ||g_B||) — must be ~CONSTANT across c (this is the construction crux).
   (c) LEARNABILITY: train B(c) briefly, synthetic-label acc must be similar & well-above-chance across c (comparable surges).
  RECOMMEND the three nominal-c that ACHIEVE +0.20 / 0.0 / −0.23 (answer-blind: from the cosine map, not any retention outcome).
  This is Phase 1 only. Arms (3 levels × A/B/D × 30 seeds) are a SEPARATE script, gated on this passing.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"
OUT = f"{G}/toggle_v2_construct.json"; NPZ = f"{G}/toggle_v2_construct.npz"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
STEPS_DEPTH = 250; LR_DEPTH = 5e-5; BATCH = 16
KPROJ = 128; N_FISHER = 200; N_GRAD = 400; N_POOL = 1200; ORTH_SEED = 7
C_SWEEP = [-0.5, -0.23, 0.0, 0.20, 0.5]; C_TARGETS = [0.20, 0.0, -0.23]


def sst_rows(split, k, seed=0):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")[split]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))[:k]]


def b_pairs(tok, items):
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"][:256]; c = tok(co, add_special_tokens=False)["input_ids"][:256]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    Lm = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), Lm), pad, np.int64); lb = np.full((len(seqs), Lm), -100, np.int64); am = np.zeros((len(seqs), Lm), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


def fisher_taskA(model, params, names, tok):
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    for s, l in sst_rows("train", N_FISHER, 0):
        inp, lb, am = b_pairs(tok, [(f"{s}\nSentiment:", LAB[l])]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: g = params[n].grad.detach().double(); F[n] += g.T @ g
    return F


def _accum_grad(model, tok, items, micro=16):
    """accumulate loss-gradient over micro-batches (gradient is linear -> sum into .grad). Avoids full-vocab-logits OOM."""
    model.zero_grad(set_to_none=True)
    for i in range(0, len(items), micro):
        inp, lb, am = b_pairs(tok, items[i:i + micro])
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()


def grad_in_V(model, tok, items, carriers, V):
    """loss-gradient on carriers projected into V -> flat vector (the shared-subspace gradient)."""
    _accum_grad(model, tok, items); params = dict(model.named_parameters())
    parts = [(params[n].grad.detach().double() @ V[n].double()).reshape(-1) for n in carriers]
    model.zero_grad(set_to_none=True)
    return torch.cat(parts)


def grad_full_and_V(model, tok, items, carriers, V):
    """return (||g in V|| / ||g|| over carriers, flat g_in_V) — for the SHARING guard + cosine."""
    _accum_grad(model, tok, items); params = dict(model.named_parameters())
    vparts = []; num = 0.0; den = 0.0
    for n in carriers:
        g = params[n].grad.detach().double(); pv = g @ V[n].double()
        vparts.append(pv.reshape(-1)); num += float((pv * pv).sum()); den += float((g * g).sum())
    model.zero_grad(set_to_none=True)
    return num / max(den, 1e-12), torch.cat(vparts)


def final_proj(model, tok, sents, dA, dOrth):
    """h_final at the decision position projected onto dA, dOrth. Returns (zA, zOrth) arrays."""
    pA, pO = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sents), 32):
            chunk = sents[i:i + 32]; seqs = [tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"][:256] for s in chunk]
            Lm = max(len(x) for x in seqs); pad = tok.pad_token_id
            inp = np.full((len(seqs), Lm), pad, np.int64); pos = np.zeros(len(seqs), np.int64)
            for j, sq in enumerate(seqs): inp[j, :len(sq)] = sq; pos[j] = len(sq) - 1
            out = model(input_ids=torch.tensor(inp).to(dev), output_hidden_states=True)
            h = out.hidden_states[-1][torch.arange(len(seqs)), torch.tensor(pos).to(dev)].double()
            pA.append((h @ dA).cpu().numpy()); pO.append((h @ dOrth).cpu().numpy())
    return np.concatenate(pA), np.concatenate(pO)


def synth_items(sents, zA, zO, c):
    a = c; b = float(np.sqrt(max(0.0, 1 - c * c)))
    score = a * zA + b * zO; y = (score > 0).astype(int)
    return [(f"{s}\nSentiment:", LAB[int(y[i])]) for i, s in enumerate(sents)], y


def eval_synth(model, tok, sents, y, posid, negid):
    model.eval(); correct = 0; n = 0
    with torch.no_grad():
        for i in range(0, len(sents), 32):
            chunk = sents[i:i + 32]; seqs = [tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"][:256] for s in chunk]
            Lm = max(len(x) for x in seqs); pad = tok.pad_token_id
            inp = np.full((len(seqs), Lm), pad, np.int64); pos = np.zeros(len(seqs), np.int64)
            for j, sq in enumerate(seqs): inp[j, :len(sq)] = sq; pos[j] = len(sq) - 1
            lg = model(input_ids=torch.tensor(inp).to(dev)).logits
            last = lg[torch.arange(len(seqs)), torch.tensor(pos).to(dev)]
            pred = (last[:, posid] > last[:, negid]).long().cpu().numpy()
            for j in range(len(chunk)): correct += int(pred[j] == y[i + j]); n += 1
    return correct / max(n, 1)


def train_single(model, tok, items, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def main():
    t0 = time.time(); print("[TGv2-C] CONSTRUCT+VALIDATE synthetic task-B family (Phase 1, no arms)", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); print(f"[TGv2-C] loaded w_A {CKPT}", flush=True)
    ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]

    # V = shared subspace (top-128 task-A Fisher per carrier)
    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n]); V[n] = vv[:, -KPROJ:].float().contiguous()
    model.load_state_dict(ckpt_sd)
    print(f"[TGv2-C] V basis done ({time.time()-t0:.0f}s)", flush=True)

    # decision direction proj_A = unembed(pos) - unembed(neg); proj_orth = fixed random, orthogonalized
    W = model.get_output_embeddings().weight.detach().double()  # [vocab, hidden]
    dA = (W[posid] - W[negid]); dA = dA / dA.norm()
    rng = torch.Generator().manual_seed(ORTH_SEED)
    r = torch.randn(dA.shape[0], generator=rng, dtype=torch.float64).to(dev)
    r = r - (r @ dA) * dA; dOrth = r / r.norm()
    print(f"[TGv2-C] proj_A·proj_orth = {float(dA @ dOrth):.2e} (should be ~0)", flush=True)

    # input pool + projections at w_A
    pool = [s for s, _ in sst_rows("train", N_POOL, 1)]
    zA, zO = final_proj(model, tok, pool, dA, dOrth)
    zA = (zA - zA.mean()) / (zA.std() + 1e-9); zO = (zO - zO.mean()) / (zO.std() + 1e-9)
    print(f"[TGv2-C] pool={len(pool)} zA/zO standardized; corr(zA,zO)={float(np.corrcoef(zA,zO)[0,1]):+.3f} ({time.time()-t0:.0f}s)", flush=True)

    # task-A gradient in V (fixed reference)
    sstit = [(f"{s}\nSentiment:", LAB[l]) for s, l in sst_rows("train", N_GRAD, 2)]
    gA = grad_in_V(model, tok, sstit, carriers, V); gAn = gA / gA.norm()

    # sweep nominal c -> achieved cosine, sharing, learnability
    rows = []
    for c in C_SWEEP:
        items, y = synth_items(pool, zA, zO, c)
        gi = [items[i] for i in np.random.default_rng(3).permutation(len(items))[:N_GRAD]]
        share, gB = grad_full_and_V(model, tok, gi, carriers, V)
        cosV = float((gB / gB.norm()) @ gAn)
        bal = float(y.mean())
        # learnability: brief train, synth-label acc
        model.load_state_dict(ckpt_sd)
        train_single(model, tok, items, STEPS_DEPTH, LR_DEPTH, 0)
        acc = eval_synth(model, tok, pool, y, posid, negid)
        model.load_state_dict(ckpt_sd)
        rows.append({"nominal_c": c, "achieved_cosV": cosV, "sharing_fracV": float(share), "label_balance": bal, "learn_acc": float(acc)})
        print(f"[TGv2-C] c={c:+.2f}: achieved cosV={cosV:+.3f} | sharing fracV={share:.3f} | bal={bal:.2f} | learn_acc={acc:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # recommend nominal-c achieving each target (nearest achieved cosine, answer-blind)
    cos_by_c = {r["nominal_c"]: r["achieved_cosV"] for r in rows}
    rec = {}
    for tgt in C_TARGETS:
        best = min(C_SWEEP, key=lambda c: abs(cos_by_c[c] - tgt)); rec[str(tgt)] = {"nominal_c": best, "achieved": cos_by_c[best]}

    # gates
    ach = [r["achieved_cosV"] for r in rows]
    monotone = all(ach[i] <= ach[i + 1] + 1e-6 for i in range(len(ach) - 1))
    spans_neg = min(ach) <= -0.10 and max(ach) >= 0.10
    shares = [r["sharing_fracV"] for r in rows]
    sharing_const = (max(shares) - min(shares)) <= 0.15 * (np.mean(shares) + 1e-9) or (max(shares) - min(shares)) <= 0.03
    learnable = all(r["learn_acc"] >= 0.65 for r in rows) and (max(r["learn_acc"] for r in rows) - min(r["learn_acc"] for r in rows) <= 0.20)
    gate_a = monotone and spans_neg
    gates = {"a_cosine_monotone_spans": bool(gate_a), "b_sharing_constant": bool(sharing_const), "c_learnable_similar": bool(learnable)}
    PASS = gate_a and sharing_const and learnable

    rep = {"phase": "construct", "model": MODEL, "rows": rows, "recommend_nominal_c": rec, "gates": gates, "PASS": bool(PASS),
           "proj_A": "unembed(pos)-unembed(neg)", "orth_seed": ORTH_SEED, "KPROJ": KPROJ, "n_pool": len(pool)}
    print(f"\n[TGv2-C] === CONSTRUCTION GATES ===", flush=True)
    print(f"  (a) cosine monotone & spans incl −: {gate_a}  (achieved {['%+.3f'%x for x in ach]})", flush=True)
    print(f"  (b) sharing constant across c: {sharing_const}  (fracV {['%.3f'%s for s in shares]})", flush=True)
    print(f"  (c) learnable & similar: {learnable}  (acc {['%.3f'%r['learn_acc'] for r in rows]})", flush=True)
    print(f"  RECOMMEND nominal-c -> targets: {rec}", flush=True)
    print(f"  PASS={PASS}  ({'fire arms at recommended nominal-c' if PASS else 'REDESIGN — do not fire arms'})", flush=True)
    np.savez(NPZ, nominal=np.array(C_SWEEP), achieved=np.array(ach), sharing=np.array(shares), zA=zA, zO=zO)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[TGv2-C] saved toggle_v2_construct.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
