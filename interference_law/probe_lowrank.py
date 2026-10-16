"""
probe_lowrank.py — NEW-2 (PREREG lock 7c1f8288): is the capability interference matrix LOW-RANK and COMPLETABLE? Build N×N directed
forgetting matrix (CIFAR + DTD), then matrix-complete it (bias + rank-k soft-impute) from 50% observed → held-out R² vs rank k. $0 local.
"""
from __future__ import annotations
import os, sys, json, time, copy
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R
import probe_dirlaw_arch as A

DEV = A.DEV
CAPS = list(range(24)); NEG_BY = {"cifar": list(range(90, 100)), "dtd": list(range(30, 40))}
FT_STEPS = R._i("FT_STEPS", 300); FT_LR = R._f("FT_LR", 5e-4); PER_CLASS = R._i("PER_CLASS", 100)
SEED = R._i("SEED", 0); SIZE = 128; RANKS = [0, 1, 2, 3, 4, 6, 8]; MASK_SEEDS = 5; OBS_FRAC = 0.5
DATASETS = os.environ.get("DATASETS", "cifar,dtd").split(",")
ROOT = Path(r"C:\Users\Nate\AppData\Local\Temp\claude\c--As-Above-So-Below-Master\12e9afc0-72b4-4b00-81e3-2fc4d7b8a97f\scratchpad\tvdata")
OUT = HERE / "out" / "lowrank.json"; LOCK = "7c1f8288"
torch.manual_seed(SEED); np.random.seed(SEED)


def get_banks(name):
    needed = set(CAPS) | set(NEG_BY[name])
    if name == "cifar":
        b = R.load_banks(); return {c: b[c] for c in needed}
    import torchvision.datasets as tvd
    import torchvision.transforms.functional as TF
    ROOT.mkdir(parents=True, exist_ok=True)
    srcs = [tvd.DTD(str(ROOT), split=s, partition=1, download=True) for s in ("train", "val", "test")]
    banks = {c: [] for c in needed}
    for ds in srcs:
        for i in range(len(ds)):
            img, lab = ds[i]; lab = int(lab)
            if lab in needed and len(banks[lab]) < PER_CLASS:
                banks[lab].append(TF.pil_to_tensor(TF.resize(img.convert("RGB"), [SIZE, SIZE])))
        if all(len(banks[c]) >= PER_CLASS for c in needed): break
    return {c: torch.stack(v) for c, v in banks.items() if len(v) >= 20}


def build_matrix(name):
    banks = get_banks(name); neg = NEG_BY[name]; Xneg_u8 = torch.cat([banks[c] for c in neg])
    caps = [c for c in CAPS if c in banks]; n = len(caps)
    base = A.build("resnet18"); base.eval(); dim = A.feats(base, banks[caps[0]][:2]).shape[1]
    Fneg0 = A.feats(base, Xneg_u8); heads, auc0 = {}, {}
    for c in caps:
        FA = A.feats(base, banks[c]); heads[c] = A.train_head1(FA, Fneg0, dim); auc0[c] = A.head_auc(heads[c], FA, Fneg0)
    F = np.zeros((n, n))
    for cj, j in enumerate(caps):
        Xb = torch.cat([banks[j], Xneg_u8]); yb = torch.cat([torch.ones(len(banks[j])), torch.zeros(len(Xneg_u8))]).long()
        mm = copy.deepcopy(base); A.full_ft(mm, dim, Xb, yb, FT_STEPS, FT_LR); Fneg1 = A.feats(mm, Xneg_u8)
        for ci, i in enumerate(caps): F[ci, cj] = (auc0[i] - A.head_auc(heads[i], A.feats(mm, banks[i]), Fneg1)) if i != j else 0.0
        del mm; torch.cuda.empty_cache()
        print(f"    [{name}] perturb {j}: colFgt={np.median([F[ci,cj] for ci in range(n) if caps[ci]!=j]):+.3f}", flush=True)
    del base; torch.cuda.empty_cache()
    return F, caps


def fit_bias(F, obs, iters=60):
    n = F.shape[0]; a = np.zeros(n); b = np.zeros(n); mu = float(F[obs].mean())
    for _ in range(iters):
        for i in range(n):
            oj = obs[i];
            if oj.any(): a[i] = float((F[i][oj] - mu - b[oj]).mean())
        a -= a.mean()
        for j in range(n):
            oi = obs[:, j]
            if oi.any(): b[j] = float((F[oi, j] - mu - a[oi]).mean())
        b -= b.mean()
        mu = float((F[obs] - a[np.argwhere(obs)[:, 0]] - b[np.argwhere(obs)[:, 1]]).mean())
    return mu + a[:, None] + b[None, :]


def softimpute(Res, obs, k, iters=100):
    if k == 0: return np.zeros_like(Res)
    X = np.where(obs, Res, 0.0)
    for _ in range(iters):
        U, s, Vt = np.linalg.svd(X, full_matrices=False); s[k:] = 0
        Xlr = (U * s) @ Vt; X = np.where(obs, Res, Xlr)
    return Xlr


def eff_rank(s): s2 = s**2; return float((s2.sum()**2)/(np.square(s2).sum()+1e-30))
def dcenter(M):
    n = M.shape[0]; m = ~np.eye(n, dtype=bool); D = M.copy().astype(float)
    rm = np.array([D[i, [j for j in range(n) if j != i]].mean() for i in range(n)])
    cm = np.array([D[[i for i in range(n) if i != j], j].mean() for j in range(n)]); g = D[m].mean()
    out = np.zeros_like(D)
    for i in range(n):
        for j in range(n):
            if i != j: out[i, j] = D[i, j] - rm[i] - cm[j] + g
    return out


def analyze(name, F):
    n = F.shape[0]; offmask = ~np.eye(n, dtype=bool); offidx = np.argwhere(offmask)
    med = float(np.median(F[offmask])); inband = bool(0.03 <= med <= 0.45)
    # completion curve
    curve = {k: [] for k in RANKS}
    for ms in range(MASK_SEEDS):
        rng = np.random.RandomState(100 + ms); perm = rng.permutation(len(offidx))
        nobs = int(OBS_FRAC * len(offidx)); obs = np.zeros((n, n), bool); held = np.zeros((n, n), bool)
        for t in perm[:nobs]: obs[offidx[t][0], offidx[t][1]] = True
        for t in perm[nobs:]: held[offidx[t][0], offidx[t][1]] = True
        bias = fit_bias(F, obs); Res = F - bias
        ytrue = F[held]; sstot = float(((ytrue - ytrue.mean())**2).sum())
        for k in RANKS:
            Xlr = softimpute(Res, obs, k); pred = (bias + Xlr)[held]
            curve[k].append(1 - float(((ytrue - pred)**2).sum())/(sstot + 1e-12))
    r2 = {k: (round(float(np.mean(v)), 3), round(float(np.std(v)), 3)) for k, v in curve.items()}
    bestk = max(RANKS, key=lambda k: np.mean(curve[k])); bestr2 = float(np.mean(curve[bestk]))
    dR2 = bestr2 - float(np.mean(curve[0]))
    # interaction SVD
    DC = dcenter(F); s = np.linalg.svd(DC, compute_uv=False); vf = (s**2/(s**2).sum())
    er = eff_rank(s)
    print(f"  [{name}] medFgt={med:+.3f} inband={inband} | completion R2 by k: " +
          " ".join(f"k{k}={r2[k][0]:+.2f}" for k in RANKS) + f" | best k={bestk} R2={bestr2:.3f} dR2(vs k0)={dR2:+.3f}", flush=True)
    print(f"  [{name}] interaction SVD: effrank={er:.1f} top6 var%={[round(float(x)*100,1) for x in vf[:6]]}", flush=True)
    return {"n": n, "median_forget": round(med, 4), "inband": inband, "completion_r2_by_rank": {str(k): r2[k] for k in RANKS},
            "best_k": bestk, "best_r2": round(bestr2, 3), "dR2_interaction": round(dR2, 3),
            "interaction_eff_rank": round(er, 2), "interaction_var_top6": [round(float(x), 3) for x in vf[:6]]}


def main():
    t0 = time.time(); print(f"[LOWRANK] lock={LOCK} datasets={DATASETS} N={len(CAPS)} ranks={RANKS} obs={OBS_FRAC} dev={DEV}", flush=True)
    results = {}
    for name in DATASETS:
        try:
            print(f"[LOWRANK] === {name}: build matrix ===", flush=True)
            F, caps = build_matrix(name); results[name] = analyze(name, F); results[name]["caps"] = [int(c) for c in caps]
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:180]}"}
            print(f"  [{name}] ERROR: {type(e).__name__}: {str(e)[:180]}", flush=True)
        OUT.write_text(json.dumps({"config": {"caps": CAPS, "ft": [FT_STEPS, FT_LR], "obs_frac": OBS_FRAC, "mask_seeds": MASK_SEEDS,
                                   "ranks": RANKS, "seed": SEED, "prereg_lock": LOCK}, "results": results}, indent=1))
    print("\n=== INTERFERENCE MATRIX: LOW-RANK & COMPLETABLE? ===", flush=True)
    print(f"  {'dataset':8}{'medFgt':>8}{'R2_k0(bias)':>12}{'bestR2':>9}{'bestk':>6}{'dR2_inter':>10}{'effRank':>9}", flush=True)
    for nme, r in results.items():
        if "error" in r: print(f"  {nme:8} ERROR {r['error'][:50]}", flush=True); continue
        print(f"  {nme:8}{r['median_forget']:>8.3f}{r['completion_r2_by_rank']['0'][0]:>12.3f}{r['best_r2']:>9.3f}{r['best_k']:>6}"
              f"{r['dR2_interaction']:>10.3f}{r['interaction_eff_rank']:>9.1f}", flush=True)
    ok = [n for n in DATASETS if "error" not in results.get(n, {}) and results[n].get("inband") and results[n]["best_r2"] >= 0.50]
    p2 = [n for n in ok if results[n]["dR2_interaction"] >= 0.05]
    verdict = (f"COMPLETABLE on {ok}" + (f"; interaction adds beyond main-effects on {p2}" if p2 else "; via MAIN EFFECTS only (interaction ~noise)")
               if ok else "NOT completable (high-rank/idiosyncratic)")
    print(f"\n[LOWRANK] completable={ok} interaction-adds={p2} | VERDICT: {verdict}", flush=True)
    d = json.loads(OUT.read_text()); d["verdict"] = verdict; OUT.write_text(json.dumps(d, indent=1))
    print(f"[LOWRANK] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
