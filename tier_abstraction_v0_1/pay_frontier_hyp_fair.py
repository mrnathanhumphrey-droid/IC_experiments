"""Form 1 FAIR eval — confirmation run. Identical to pay_frontier_hyp.py EXCEPT the hyperbolic
prototype is the proper hyperbolic centroid (Einstein midpoint / Klein-model Lorentz-weighted
average, Khrulkov et al.) instead of the crude projected-Euclidean-centroid. Same deterministic
models (same seeds) -> Euclidean numbers reproduce exactly; only the hyperbolic prototype changes.

If hyperbolic STILL loses to Euclidean with this fair centroid -> Form-1 null is real (geometry
doesn't help on CIFAR-100's shallow tree) -> deep-tree run justified. If it flips -> it was the
eval, not the geometry.
"""
import argparse, json, os
import numpy as np
import torch

from pay_frontier_hyp import (HoldoutData, holdout_classes, train_armC_holdout, train_hyp_holdout,
                              poincare_np, boot_diff, FIT_POOLS)


def einstein_midpoint(X, eps=1e-9):
    """Hyperbolic centroid of Poincare-ball points X [n,K] -> [K]. Klein-model Lorentz-weighted mean."""
    x2 = (X ** 2).sum(-1, keepdims=True)
    Kc = 2 * X / (1 + x2)                               # Poincare -> Klein
    k2 = (Kc ** 2).sum(-1)
    gamma = 1.0 / np.sqrt(np.maximum(1 - k2, eps))      # Lorentz factors
    mk = (gamma[:, None] * Kc).sum(0) / np.maximum(gamma.sum(), eps)
    mk2 = (mk ** 2).sum()
    return mk / (1 + np.sqrt(np.maximum(1 - mk2, eps)))  # Klein -> Poincare


def ncm_super_fair(feat_fit, sup_fit, idx, feat_q, sup_q, metric):
    """Prototype super-classifier; hyperbolic uses Einstein-midpoint centroid + Poincare distance."""
    sf = sup_fit[idx]; ff = feat_fit[idx]
    protos, labs = [], []
    for s in range(20):
        m = sf == s
        if m.sum() == 0:
            continue
        pts = ff[m]
        protos.append(einstein_midpoint(pts) if metric == "hyperbolic" else pts.mean(0))
        labs.append(s)
    P = np.stack(protos); labs = np.array(labs)
    if metric == "euclidean":
        d = ((feat_q[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    else:
        d = poincare_np(feat_q[:, None, :], P[None, :, :])
    return (labs[d.argmin(1)] == sup_q).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    held = holdout_classes(15); data = HoldoutData(device, held)
    store = {p: {"zlow": [], "euc": [], "hyp": []} for p in FIT_POOLS}

    for seed in args.seeds:
        euc = train_armC_holdout(data, device, args.beta, seed, args.epochs)
        hyp = train_hyp_holdout(data, device, args.beta, seed, args.epochs)
        zl_e, _ = data.features(euc, data.train_in); zl_eq, zt_eq = data.features(euc, data.novel_query)
        _, zt_h = data.features(hyp, data.train_in); _, zt_hq = data.features(hyp, data.novel_query)
        zt_e = data.features(euc, data.train_in)[1]
        sup_fit = data.train_in["coarse"].cpu().numpy(); sup_q = data.novel_query["coarse"].cpu().numpy()
        rng = np.random.default_rng(seed)
        for p in FIT_POOLS:
            idx = rng.choice(zl_e.shape[0], size=min(p, zl_e.shape[0]), replace=False)
            store[p]["zlow"].append(ncm_super_fair(zl_e, sup_fit, idx, zl_eq, sup_q, "euclidean"))
            store[p]["euc"].append(ncm_super_fair(zt_e, sup_fit, idx, zt_eq, sup_q, "euclidean"))
            store[p]["hyp"].append(ncm_super_fair(zt_h, sup_fit, idx, zt_hq, sup_q, "hyperbolic"))
        print(f"[seed {seed}] done", flush=True)

    print("\n=== FORM 1 FAIR (Einstein-midpoint hyperbolic centroid) ===", flush=True)
    print(f"{'pool':>5} | {'zlow':>6} {'euc':>6} {'hyp':>6} | {'hyp-euc (fair)':>22} {'hyp-zlow':>22}")
    results = {}; confirm = False
    for p in FIT_POOLS:
        zlow = float(np.mean(store[p]["zlow"], axis=0).mean())
        euc = float(np.mean(store[p]["euc"], axis=0).mean())
        hyp = float(np.mean(store[p]["hyp"], axis=0).mean())
        dx, lx, hx = boot_diff(store[p]["hyp"], store[p]["euc"])
        dh, lh, hh = boot_diff(store[p]["hyp"], store[p]["zlow"])
        if (dx >= 0.02 and lx > 0 and p <= 1000) or (dh >= 0.02 and lh > 0 and p >= 4000):
            confirm = True
        results[p] = {"zlow": zlow, "euc": euc, "hyp": hyp, "hyp_minus_euc": [dx, lx, hx], "hyp_minus_zlow": [dh, lh, hh]}
        print(f"{p:>5} | {zlow:.3f} {euc:.3f} {hyp:.3f} | {dx:+.3f}[{lx:+.3f},{hx:+.3f}]  {dh:+.3f}[{lh:+.3f},{hh:+.3f}]")

    verdict = {"held_out_fine": held, "by_pool": results, "FORM1_FAIR_CONFIRMED": bool(confirm),
               "note": "Einstein-midpoint hyperbolic centroid; if NULL, shallow-tree null is real"}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "pay_frontier_form1_fair.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nFORM 1 FAIR {'CONFIRMED — geometry helps with proper centroid (was an eval artifact)' if confirm else 'NULL — hyperbolic loses even with a fair centroid; shallow-tree null is REAL'}", flush=True)


if __name__ == "__main__":
    main()
