"""Unified s-measurement (the seam object) — answer-blind, serves three lanes.

At the SST-2 task-A minimum, BEFORE any task-B training (answer-blind to forgetting),
measure how much each task-B's pull (its gradient) aligns with task-A's binding (stiff
curvature directions). One measured quantity = interior-s for the overlap axis + the force
lane's pull-vs-binding alignment + the curvature-alignment driving sigma_cpl's [D,H] curl.

Per the two-scale curvature (few stiff dirs in a flat sea), the LOAD-BEARING output is the
SHAPE across the four levels: does pull hit the stiff subspace in a graded way (smooth s
gradation -> sigma_cpl peak cleanly testable) or all-or-nothing (step-like -> flat peak is
substrate-structure, pre-flag for the physics lane)?

Binding basis = K-FAC per-matrix (FK3, gauge-free) on the 24 attention.dense, collected on
TASK-A (SST-2) data at the A-min. Pull = task-B weight-gradient at the A-min, rotated into
that basis: C = U_G^T g_W U_A; pull-energy[i,j]=C[i,j]^2, binding[i,j]=eigG[i]*eigA[j].
Alignment metrics per level (aggregated over 24 matrices):
  s_stiff   = fraction of pull-energy in the top-1% binding directions (hits the stiff few?)
  bind_felt = pull-weighted mean binding = sum(C^2*bind)/sum(C^2), normalized to [0,1] across levels
LEVELS (answer-blind, declared): self(SST-2)~s1 no-drive ref, null(permuted)~s0, AG-News(low),
SUBJ(medium), Yelp(high). s measured, NOT assigned; hand-order is a sanity check only.
"""
import os, json, time
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import numpy as np, torch

MODEL = "EleutherAI/pythia-410m"; CKPT = "D:/IC_experiments/cf_cascade/gateA/pythia410m_sst2_subsetmin.pt"
AX = "D:/IC_experiments/cf_cascade/gateA/overlap_axis"; OUT = "D:/IC_experiments/cf_cascade/gateA/s_measurement.json"
BATCH = 16; N_KFAC = 16; N_GRAD = 16; SUBSET = 2000; dev = "cuda"
LAB_SST = {0: " negative", 1: " positive"}; torch.manual_seed(0)


def sst_subset():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")["train"]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:SUBSET]]


def batch_from(tok, items):
    """items: list of (prompt_str, completion_str)."""
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"]; c = tok(co, add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    L = max(len(s) for s in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


def load_level(name):
    d = json.load(open(f"{AX}/{name}.json")); tmpl = d["prompt_template"]
    return [(tmpl.format(text=e["text"]), e["completion"]) for e in d["examples"]]


def main():
    t0 = time.time()
    print(f"[s] start; loading model...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.eval()
    model.load_state_dict(torch.load(CKPT))
    print(f"[s] model loaded ({time.time()-t0:.0f}s); loading data...", flush=True)
    sst = sst_subset()
    print(f"[s] data loaded ({time.time()-t0:.0f}s)", flush=True)
    sst_items = [(f"{s}\nSentiment:", LAB_SST[l]) for s, l in sst]

    # ---- K-FAC binding eigenbases on TASK-A (SST-2) at the A-min ----
    dense = {n: m for n, m in model.named_modules() if n.endswith("attention.dense")}
    A_acc = {n: torch.zeros(1024, 1024, device=dev) for n in dense}
    G_acc = {n: torch.zeros(1024, 1024, device=dev) for n in dense}
    cnt = {n: 0 for n in dense}; cur = {}
    H = []
    for n, m in dense.items():
        def fwd(mod, i, o, nn=n): cur[nn] = i[0].detach()
        def bwd(mod, gi, go, nn=n):
            x = cur[nn].reshape(-1, 1024); g = go[0].detach().reshape(-1, 1024)
            keep = g.abs().sum(1) > 0
            if keep.any(): A_acc[nn] += x[keep].T @ x[keep]; G_acc[nn] += g[keep].T @ g[keep]; cnt[nn] += int(keep.sum())
        H += [m.register_forward_hook(fwd), m.register_full_backward_hook(bwd)]
    gA = torch.Generator().manual_seed(7)
    for _ in range(N_KFAC):
        it = [sst_items[k] for k in torch.randperm(len(sst_items), generator=gA)[:BATCH].tolist()]
        inp, lb, am = batch_from(tok, it); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
    for h in H: h.remove()
    basis = {}
    for n in dense:
        A = (A_acc[n] / max(cnt[n], 1)).double(); G = (G_acc[n] / max(cnt[n], 1)).double()
        eA, UA = torch.linalg.eigh(A); eG, UG = torch.linalg.eigh(G)
        basis[n] = (eA.clamp(min=0).float(), UA.float(), eG.clamp(min=0).float(), UG.float())
    print(f"[s] K-FAC binding bases built on {len(dense)} matrices ({time.time()-t0:.0f}s)", flush=True)

    # ---- per-level pull alignment ----
    def pull_grad(items):
        model.zero_grad(set_to_none=True)
        gtot = {n: torch.zeros(1024, 1024, device=dev) for n in dense}
        g = torch.Generator().manual_seed(11)
        for _ in range(N_GRAD):
            it = [items[k] for k in torch.randperm(len(items), generator=g)[:BATCH].tolist()]
            inp, lb, am = batch_from(tok, it)
            model.zero_grad(set_to_none=True)
            (model(input_ids=inp, attention_mask=am, labels=lb).loss / N_GRAD).backward()
            for n, m in dense.items(): gtot[n] += m.weight.grad.detach()
        return gtot

    levels = {"self_sst2": sst_items, "null_permuted": load_level("null_sst2permuted"),
              "low_agnews": load_level("low_agnews"), "medium_subj": load_level("medium_subj"),
              "high_yelp": load_level("high_yelp")}
    results = {}
    for lvl, items in levels.items():
        gtot = pull_grad(items)
        stiff_num = stiff_den = bind_w = energy = 0.0
        for n in dense:
            eA, UA, eG, UG = basis[n]
            C = (UG.T @ gtot[n].double().float() @ UA)          # (G-space, A-space) coeffs
            C2 = (C * C)
            bind = torch.outer(eG, eA)                          # binding eigenvalue per (i,j)
            e = float(C2.sum())
            thr = torch.quantile(bind.flatten(), 0.99)
            stiff_num += float(C2[bind >= thr].sum()); stiff_den += e
            bind_w += float((C2 * bind).sum()); energy += e
        s_stiff = stiff_num / max(stiff_den, 1e-30)
        mean_bind_felt = bind_w / max(energy, 1e-30)
        results[lvl] = {"s_stiff_top1pct": s_stiff, "mean_binding_felt": mean_bind_felt, "pull_energy": energy}
        print(f"  [{lvl:14s}] s_stiff(top1%)={s_stiff:.4f}  mean_binding_felt={mean_bind_felt:.3e}  pull_energy={energy:.3e}", flush=True)

    # normalize bind_felt to [0,1] across the driven levels (exclude self=no-drive ref)
    driven = ["null_permuted", "low_agnews", "medium_subj", "high_yelp"]
    bf = np.array([results[l]["mean_binding_felt"] for l in driven])
    bf_norm = (bf - bf.min()) / (bf.max() - bf.min() + 1e-30)
    for l, v in zip(driven, bf_norm): results[l]["s_normalized"] = float(v)

    # shape: smooth vs step-like, on s_stiff across the ordered driven levels
    ss = np.array([results[l]["s_stiff_top1pct"] for l in driven])
    gaps = np.diff(np.sort(ss))
    rng = ss.max() - ss.min()
    # step-like if one gap dominates (>60% of total range); smooth if gaps comparable
    step_like = (rng > 0) and (gaps.max() / rng > 0.6)
    shape = "STEP_LIKE" if step_like else "SMOOTH"
    monotone_handorder = bool(np.all(np.diff([results[l]["s_stiff_top1pct"] for l in
                          ["null_permuted", "low_agnews", "medium_subj", "high_yelp"]]) >= -0.02))

    print(f"\n[s] === SHAPE OF MEASURED ALIGNMENT ACROSS LEVELS ===", flush=True)
    print(f"  s_stiff by level: " + "  ".join(f"{l.split('_')[0]}={results[l]['s_stiff_top1pct']:.3f}" for l in driven), flush=True)
    print(f"  s_normalized(bind_felt): " + "  ".join(f"{l.split('_')[0]}={results[l]['s_normalized']:.2f}" for l in driven), flush=True)
    print(f"  range={rng:.4f}  max-gap/range={gaps.max()/max(rng,1e-30):.2f}", flush=True)
    print(f"  SHAPE: {shape}  (monotone in hand-order AG<SUBJ<Yelp: {monotone_handorder})", flush=True)
    print(f"  self(no-drive) s_stiff={results['self_sst2']['s_stiff_top1pct']:.4f} pull_energy={results['self_sst2']['pull_energy']:.3e} (should be ~0 drive at the min)", flush=True)
    if shape == "SMOOTH":
        print(f"    -> SMOOTH: pull hits stiff subspace gradually -> sigma_cpl peak-location CLEANLY testable on SST-2", flush=True)
    else:
        print(f"    -> STEP-LIKE: pull-vs-stiff is ~binary -> flat sigma_cpl peak is substrate-structure (two-scale), PRE-FLAG to physics; may need continuum task A for a clean peak", flush=True)
    json.dump({"levels": results, "shape": shape, "range": float(rng),
               "max_gap_over_range": float(gaps.max() / max(rng, 1e-30)),
               "monotone_handorder": monotone_handorder, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[s] saved s_measurement.json  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
