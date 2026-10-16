"""
probe_fwd_plastic.py -- NEW-3c (PREREG lock e6086a1b): PLASTIC forward transfer. Full-FT backbone on B, then LEARN few-shot A on the
B-trained representation. FT(A<-B) = AUC_A(few-shot on theta_B feats) - AUC_A(few-shot on theta0 feats). Cosine-governed? (dissociates from
frozen NEW-3b head-warmstart). CIFAR + DTD. $0 local. ASCII-only prints.
"""
from __future__ import annotations
import os, sys, json, time, copy
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R
import probe_dirlaw_arch as A
import probe_transfer as XF                  # get_banks / train_head / auc_head / split_idx

DEV = A.DEV
CAPS = list(range(20)); PER_CLASS = R._i("PER_CLASS", 100)
NFEW_POS = R._i("NFEW_POS", 8); NFEW_NEG = R._i("NFEW_NEG", 24); NOTH = R._i("NOTH", 135)
FT_STEPS = R._i("FT_STEPS", 300); FT_LR = R._f("FT_LR", 5e-4); K = R._i("K", 30)
NSEED = R._i("NSEED", 3); D_PROJ = R._i("D_PROJ", 24); SEED = R._i("SEED", 0)
DATASETS = os.environ.get("DATASETS", "cifar,dtd").split(",")
OUT = HERE / "out" / "fwd_plastic.json"; LOCK = "e6086a1b"
torch.manual_seed(SEED); np.random.seed(SEED)


def fewshot_auc(feat_by_cap, splits, i, dim, seed):
    Xtr, ytr, Xte, yte = splits[i]
    h = XF.train_head(dim, Xtr, ytr, K, seed=seed)
    return XF.auc_head(h, Xte, yte)


def make_splits(feats_proj, caps, dim):
    # per cap: few-shot train (from its projected feats) + test; neg = pooled others (projected)
    rng = np.random.RandomState(0); splits = {}
    for c in caps:
        pos = feats_proj[c]; others = np.concatenate([feats_proj[o] for o in caps if o != c])
        ptr, pte = XF.split_idx(len(pos), f"p{c}"); negsel = rng.choice(len(others), min(NOTH*3, len(others)), replace=False)
        neg = others[negsel]; ntr, nte = XF.split_idx(len(neg), f"n{c}")
        Xtr = np.concatenate([pos[ptr][:NFEW_POS], neg[ntr][:NFEW_NEG]])
        ytr = np.concatenate([np.ones(min(NFEW_POS, ptr.sum())), np.zeros(min(NFEW_NEG, ntr.sum()))]).astype(np.float32)
        Xte = np.concatenate([pos[pte], neg[nte]]); yte = np.concatenate([np.ones(pte.sum()), np.zeros(nte.sum())]).astype(np.float32)
        splits[c] = (Xtr, ytr, Xte, yte)
    return splits


def run(name):
    t0 = time.time(); print(f"[FP] === {name} ===", flush=True)
    banks = XF.get_banks(name); caps = [c for c in CAPS if c in banks]; n = len(caps)
    base = A.build("resnet18"); base.eval()
    F0 = {c: A.feats(base, banks[c]) for c in caps}; dim0 = F0[caps[0]].shape[1]
    P = (np.random.RandomState(0).randn(dim0, D_PROJ) / np.sqrt(D_PROJ)).astype(np.float32); dim = D_PROJ
    proj = lambda D: {c: (D[c] @ P).astype(np.float32) for c in caps}
    d0 = {c: R.ddir(F0[c], np.concatenate([F0[o] for o in caps if o != c])) for c in caps}
    cosM = np.array([[float(d0[i] @ d0[j]) for j in caps] for i in caps])

    # cold few-shot on theta0
    sp0 = make_splits(proj(F0), caps, dim); cold = np.zeros(n)
    for s in range(NSEED):
        for ci, i in enumerate(caps): cold[ci] += fewshot_auc(None, sp0, i, dim, seed=9000*s + ci) / NSEED
    med_cold = float(np.median(cold)); headroom = bool(0.60 <= med_cold <= 0.90)
    print(f"  [{name}] cold few-shot AUC med={med_cold:.3f} headroom={headroom} ({time.time()-t0:.0f}s)", flush=True)

    # per perturber B: full-FT backbone, then few-shot each A on theta_B feats
    rng = np.random.RandomState(1); FTmat = np.zeros((n, n))
    for cj, j in enumerate(caps):
        others = torch.cat([banks[o] for o in caps if o != j]); osel = rng.choice(len(others), min(NOTH*3, len(others)), replace=False)
        Xb = torch.cat([banks[j], others[torch.tensor(osel)]]); yb = torch.cat([torch.ones(len(banks[j])), torch.zeros(len(osel))]).long()
        mm = copy.deepcopy(base); A.full_ft(mm, dim0, Xb, yb, FT_STEPS, FT_LR)
        FB = {c: A.feats(mm, banks[c]) for c in caps}; spB = make_splits(proj(FB), caps, dim)
        del mm; torch.cuda.empty_cache()
        for ci, i in enumerate(caps):
            if i == j: continue
            onB = np.mean([fewshot_auc(None, spB, i, dim, seed=9000*s + ci) for s in range(NSEED)])
            FTmat[ci, cj] = onB - cold[ci]
        print(f"    [{name}] B={j}: mean FT to others={np.mean([FTmat[ci,cj] for ci in range(n) if caps[ci]!=j]):+.4f} ({time.time()-t0:.0f}s)", flush=True)
    del base; torch.cuda.empty_cache()

    m = ~np.eye(n, dtype=bool); fv = FTmat[m]; cv = cosM[m]
    xfer_frac = float((fv >= 0.01).mean()); hurt_frac = float((fv <= -0.01).mean()); net = float(fv.mean())
    rho = float(R.spearman(cv, fv)); pear = float(np.corrcoef(cv, fv)[0, 1])
    ed = np.quantile(cv, np.linspace(0, 1, 9)); curve = []
    for b in range(8):
        mm = (cv >= ed[b]) & (cv <= ed[b+1] if b == 7 else cv < ed[b+1])
        if mm.sum(): curve.append((round(float((ed[b]+ed[b+1])/2), 3), round(float(fv[mm].mean()), 4)))
    p2 = bool(rho >= 0.30)
    print(f"  [{name}] headroom={headroom} | net_FT={net:+.4f} transfer_frac={xfer_frac:.2f} hurt_frac={hurt_frac:.2f} range[{fv.min():+.3f},{fv.max():+.3f}]", flush=True)
    print(f"  [{name}] rho(cos,FT)={rho:+.3f} pearson={pear:+.3f} | curve: {curve}", flush=True)
    print(f"  [{name}] 3c-P2(plastic fwd transfer cosine-gov)={p2} ({time.time()-t0:.0f}s)", flush=True)
    return {"n": n, "cold_auc_median": round(med_cold, 4), "headroom": headroom, "net_FT": round(net, 4),
            "transfer_frac": round(xfer_frac, 3), "hurt_frac": round(hurt_frac, 3),
            "FT_range": [round(float(fv.min()), 4), round(float(fv.max()), 4)], "rho_cos_FT": round(rho, 3),
            "pearson": round(pear, 3), "FT_vs_cos_curve": curve, "checks": {"P2_cosine_gov": p2}}


def main():
    t0 = time.time(); print(f"[FP] lock={LOCK} datasets={DATASETS} caps={len(CAPS)} FT={FT_STEPS}/{FT_LR} K={K} D_PROJ={D_PROJ} dev={DEV}", flush=True)
    results = {}
    for name in DATASETS:
        try:
            results[name] = run(name)
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:180]}"}
            print(f"  [{name}] ERROR: {type(e).__name__}: {str(e)[:180]}", flush=True)
        OUT.write_text(json.dumps({"config": {"caps": CAPS, "ft": [FT_STEPS, FT_LR], "few": [NFEW_POS, NFEW_NEG], "K": K,
                                   "d_proj": D_PROJ, "nseed": NSEED, "seed": SEED, "prereg_lock": LOCK}, "results": results}, indent=1))
    print("\n=== PLASTIC FORWARD TRANSFER (full-FT on B, few-shot A on theta_B; FT vs cos) ===", flush=True)
    print(f"  {'dataset':8}{'coldAUC':>8}{'net_FT':>9}{'xferFrac':>9}{'hurtFrac':>9}{'rho(cos,FT)':>12}   P2", flush=True)
    for nme, r in results.items():
        if "error" in r: print(f"  {nme:8} ERROR {r['error'][:50]}", flush=True); continue
        print(f"  {nme:8}{r['cold_auc_median']:>8.3f}{r['net_FT']:>9.4f}{r['transfer_frac']:>9.2f}{r['hurt_frac']:>9.2f}"
              f"{r['rho_cos_FT']:>12.3f}   {int(r['checks']['P2_cosine_gov'])}", flush=True)
    ok = [n for n in DATASETS if "error" not in results.get(n, {}) and results[n].get("headroom")]
    p2 = all(results[n]["checks"]["P2_cosine_gov"] for n in ok) if ok else False
    verdict = ("PLASTIC forward transfer IS cosine-governed -> at representation level cosine = general alignment law (transfer+forgetting)" if p2 and ok else
               "plastic forward transfer NOT cosine-governed -> cosine is purely a FORGETTING law at all levels" if ok else
               "no headroom -> retune")
    print(f"\n[FP] headroom={ok} P2_cosine_gov={p2} | VERDICT: {verdict}", flush=True)
    d = json.loads(OUT.read_text()); d["verdict"] = verdict; OUT.write_text(json.dumps(d, indent=1))
    print(f"[FP] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
