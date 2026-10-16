"""
probe_shape.py — NEW-1a''''-c (PREREG lock 44ba13f8): the SHAPE of the interference. Square directed forgetting matrix on CIFAR/DTD/GTSRB
(resnet18). Is it ASYMMETRIC off CIFAR? Real per-pair interaction? What shape (rank, pecking-order vs cyclic)? Does −cos capture it? $0 local.
"""
from __future__ import annotations
import os, sys, json, time, copy
from pathlib import Path
import numpy as np, torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R
import probe_dirlaw_arch as A               # build/prep/feats/theta/train_head1/head_auc/full_ft/DEV

DEV = A.DEV
CAPS = list(range(16)); NEG_BY = {"cifar": list(range(90, 100)), "dtd": list(range(30, 40)), "gtsrb": list(range(30, 40))}
FT_STEPS = R._i("FT_STEPS", 300); FT_LR = R._f("FT_LR", 5e-4); PER_CLASS = R._i("PER_CLASS", 100)
NJ = R._i("NJ", 128); SEED = R._i("SEED", 0); SIZE = 128
DATASETS = os.environ.get("DATASETS", "cifar,dtd,gtsrb").split(",")
ROOT = Path(r"C:\Users\Nate\AppData\Local\Temp\claude\c--As-Above-So-Below-Master\12e9afc0-72b4-4b00-81e3-2fc4d7b8a97f\scratchpad\tvdata")
OUT = HERE / "out" / "shape.json"; LOCK = "44ba13f8"
torch.manual_seed(SEED); np.random.seed(SEED)


def get_banks(name):
    needed = set(CAPS) | set(NEG_BY[name])
    if name == "cifar":
        b = R.load_banks(); return {c: b[c] for c in needed}
    import torchvision.datasets as tvd
    import torchvision.transforms.functional as TF
    ROOT.mkdir(parents=True, exist_ok=True)
    srcs = ([tvd.DTD(str(ROOT), split=s, partition=1, download=True) for s in ("train", "val", "test")] if name == "dtd"
            else [tvd.GTSRB(str(ROOT), split="train", download=True)])
    banks = {c: [] for c in needed}
    for ds in srcs:
        for i in range(len(ds)):
            img, lab = ds[i]; lab = int(lab)
            if lab in needed and len(banks[lab]) < PER_CLASS:
                banks[lab].append(TF.pil_to_tensor(TF.resize(img.convert("RGB"), [SIZE, SIZE])))
        if all(len(banks[c]) >= PER_CLASS for c in needed): break
    return {c: torch.stack(v) for c, v in banks.items() if len(v) >= 20}


def off(M):  # boolean off-diagonal mask
    n = M.shape[0]; return ~np.eye(n, dtype=bool)

def dcenter_off(M):
    n = M.shape[0]; m = off(M); D = M.copy().astype(float)
    rm = np.array([D[i, [j for j in range(n) if j != i]].mean() for i in range(n)])
    cm = np.array([D[[i for i in range(n) if i != j], j].mean() for j in range(n)])
    g = D[m].mean(); out = np.zeros_like(D)
    for i in range(n):
        for j in range(n):
            if i != j: out[i, j] = D[i, j] - rm[i] - cm[j] + g
    return out

def eff_rank(s): s2 = s**2; return float((s2.sum()**2)/(np.square(s2).sum()+1e-30))


def run(name):
    t0 = time.time(); print(f"[SHAPE] === {name} ===", flush=True)
    banks = get_banks(name); neg = NEG_BY[name]; Xneg_u8 = torch.cat([banks[c] for c in neg])
    caps = [c for c in CAPS if c in banks]; n = len(caps)
    base = A.build("resnet18"); base.eval(); dim = A.feats(base, banks[caps[0]][:2]).shape[1]
    Fneg0 = A.feats(base, Xneg_u8); heads, auc0 = {}, {}
    for c in caps:
        FA = A.feats(base, banks[c]); heads[c] = A.train_head1(FA, Fneg0, dim); auc0[c] = A.head_auc(heads[c], FA, Fneg0)
    print(f"  [{name}] n_caps={n} dim={dim} baseAUC={np.median(list(auc0.values())):.3f} ({time.time()-t0:.0f}s)", flush=True)

    theta0 = A.theta(base); dthetas = {}; F = np.zeros((n, n))
    for cj, j in enumerate(caps):
        Xb = torch.cat([banks[j], Xneg_u8]); yb = torch.cat([torch.ones(len(banks[j])), torch.zeros(len(Xneg_u8))]).long()
        mm = copy.deepcopy(base); A.full_ft(mm, dim, Xb, yb, FT_STEPS, FT_LR)
        dthetas[j] = (A.theta(mm) - theta0).float()
        Fneg1 = A.feats(mm, Xneg_u8)
        for ci, i in enumerate(caps): F[ci, cj] = (auc0[i] - A.head_auc(heads[i], A.feats(mm, banks[i]), Fneg1)) if i != j else 0.0
        del mm; torch.cuda.empty_cache()
        print(f"    perturb {j}: colFgt_med={np.median([F[ci,cj] for ci in range(n) if caps[ci]!=j]):+.3f} ({time.time()-t0:.0f}s)", flush=True)

    # Jacobian cos matrix
    COS = np.zeros((n, n)); Xn_g = A.prep(Xneg_u8[:NJ]).to(DEV)
    for ci, i in enumerate(caps):
        base.zero_grad(set_to_none=True)
        mrg = heads[i](base(A.prep(banks[i][:NJ]).to(DEV))).mean() - heads[i](base(Xn_g)).mean(); mrg.backward()
        g = torch.cat([pp.grad.detach().flatten() for pp in base.parameters()]).float().cpu(); gn = float(g.norm())
        for cj, j in enumerate(caps):
            COS[ci, cj] = float(g @ dthetas[j])/(gn*float(dthetas[j].norm())+1e-12) if i != j else 0.0
    base.zero_grad(set_to_none=True)

    m = off(F); offv = F[m]; med = float(np.median(offv)); inband = bool(0.03 <= med <= 0.45)
    S = (F + F.T)/2; K = (F - F.T)/2
    kratio = float(np.sqrt((K[m]**2).sum())/(np.sqrt((S[m]**2).sum())+1e-12))
    ij = [(i, j) for i in range(n) for j in range(n) if i < j]
    rho_sym = float(R.spearman(np.array([F[i, j] for i, j in ij]), np.array([F[j, i] for i, j in ij])))
    # interaction
    DC = dcenter_off(F); grand = offv.mean()
    inter_frac = float((DC[m]**2).sum()/(((F[m]-grand)**2).sum()+1e-12))
    s = np.linalg.svd(DC, compute_uv=False); s0 = float(s[0]); erank = eff_rank(s)
    rng = np.random.RandomState(0); null = []
    for _ in range(200):
        Sh = np.zeros((n, n)); vals = rng.permutation(DC[m]); Sh[m] = vals
        null.append(float(np.linalg.svd(Sh, compute_uv=False)[0]))
    null95 = float(np.percentile(null, 95)); real_struct = bool(s0 > null95)
    # dominance rank-1
    h = np.array([K[i, [j for j in range(n) if j != i]].mean() for i in range(n)])
    Khat = h[:, None] - h[None, :]; hier = float(1 - ((K[m]-Khat[m])**2).sum()/(((K[m])**2).sum()+1e-12))
    # -cos capture
    aligndc = float(R.spearman(dcenter_off(-COS)[m], DC[m]))

    p1 = bool(kratio >= 0.15 and rho_sym < 0.70); p2 = real_struct
    print(f"  [{name}] medFgt={med:+.3f} inband={inband} | ASYM ||K||/||S||={kratio:.3f} rho(Fij,Fji)={rho_sym:+.3f} | "
          f"INTER frac={inter_frac:.3f} s0={s0:.3f} null95={null95:.3f} real={real_struct} effrank={erank:.1f} | "
          f"dominance rank1={hier*100:.0f}% | align_dc(-cos,DC)={aligndc:+.3f} | P1={p1} P2={p2} ({time.time()-t0:.0f}s)", flush=True)
    del base; torch.cuda.empty_cache()
    return {"n": n, "median_forget": round(med, 4), "inband": inband, "asym_k_over_s": round(kratio, 3), "rho_sym": round(rho_sym, 3),
            "interaction_frac": round(inter_frac, 3), "dc_s0": round(s0, 3), "dc_null95": round(null95, 3), "interaction_real": p2,
            "dc_eff_rank": round(erank, 2), "dominance_rank1_pct": round(hier*100, 1), "align_dc_cos_vs_interaction": round(aligndc, 3),
            "checks": {"P1_asymmetric": p1, "P2_interaction_real": p2}}


def main():
    t0 = time.time(); print(f"[SHAPE] lock={LOCK} datasets={DATASETS} caps={len(CAPS)} dev={DEV}", flush=True)
    results = {}
    for name in DATASETS:
        try:
            results[name] = run(name)
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:180]}"}
            print(f"  [{name}] ERROR: {type(e).__name__}: {str(e)[:180]}", flush=True)
        OUT.write_text(json.dumps({"config": {"caps": CAPS, "ft": [FT_STEPS, FT_LR], "seed": SEED, "prereg_lock": LOCK},
                                   "results": results}, indent=1))
    print("\n=== SHAPE OF THE INTERFERENCE (resnet18, square directed matrix) ===", flush=True)
    print(f"  {'dataset':8}{'medFgt':>8}{'K/S':>7}{'rhoSym':>8}{'interFrac':>10}{'interReal':>10}{'effRank':>8}{'domRank1%':>10}{'cosCapt':>9}", flush=True)
    for nme, r in results.items():
        if "error" in r: print(f"  {nme:8} ERROR {r['error'][:50]}", flush=True); continue
        print(f"  {nme:8}{r['median_forget']:>8.3f}{r['asym_k_over_s']:>7.3f}{r['rho_sym']:>8.3f}{r['interaction_frac']:>10.3f}"
              f"{str(r['interaction_real']):>10}{r['dc_eff_rank']:>8.1f}{r['dominance_rank1_pct']:>9.0f}%{r['align_dc_cos_vs_interaction']:>9.3f}", flush=True)
    print(f"\n[SHAPE] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
