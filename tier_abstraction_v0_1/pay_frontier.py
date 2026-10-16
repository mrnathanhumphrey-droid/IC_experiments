"""Axis 1 — data-scarcity sweep (the pay-frontier, v0.next). Reuses C5's held-out Arm C
harness; does NOT modify transfer_judge.py. Tests whether a low-data regime exists where the
compact 8-dim abstraction (z_trans) beats the 64-dim fine representation (z_low) on coarse
transfer to novel classes — the regime DPI says z_low can't dominate.

PAY GATE (pre-registered): "low-data regime exists" CONFIRMS iff
  superacc(z_trans) - superacc(z_low) >= +0.02 top-1, paired, 95% bootstrap CI excludes 0,
  at any super-classifier fit-pool <= 1000. Null if no pool size crosses over.
Also report the fine k-shot crossover (secondary) and the full curve.
"""
import argparse, json, os
import numpy as np

from transfer_judge import (HoldoutData, holdout_classes, train_armC_holdout,
                            super_clf_acc, ncm_fine_acc)
import torch

FIT_POOLS = [50, 200, 1000, 4000, 8000]
K_SHOTS = [1, 5, 10, 20]


def boot_ci(low_corr, trans_corr, B=2000, seed=12345):
    """low_corr/trans_corr: lists (over seeds) of per-query correctness arrays. Returns
    point estimate of paired diff (trans-low) and 95% CI via bootstrap over query."""
    low_mean = np.mean(low_corr, axis=0); trans_mean = np.mean(trans_corr, axis=0)
    point = float(trans_mean.mean() - low_mean.mean())
    nq = low_mean.shape[0]; rng = np.random.default_rng(seed); b = []
    for _ in range(B):
        idx = rng.integers(0, nq, nq)
        b.append(trans_mean[idx].mean() - low_mean[idx].mean())
    lo, hi = np.percentile(b, [2.5, 97.5])
    return point, float(lo), float(hi), float(low_mean.mean()), float(trans_mean.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    held = holdout_classes(15)
    data = HoldoutData(device, held)
    print(f"held-out fine ({len(held)}): {held}")

    # storage: per fit-pool -> list over seeds of per-query correctness
    pay = {p: {"low": [], "trans": []} for p in FIT_POOLS}
    fine = {k: {"low": [], "trans": []} for k in K_SHOTS}

    for seed in args.seeds:
        model = train_armC_holdout(data, device, args.beta, seed, args.epochs)
        zl_fit, zt_fit = data.features(model, data.train_in)
        sup_fit = data.train_in["coarse"].cpu().numpy()
        zl_q, zt_q = data.features(model, data.novel_query)
        zl_s, zt_s = data.features(model, data.novel_support)
        fine_q = data.novel_query["fine"].cpu().numpy(); sup_q = data.novel_query["coarse"].cpu().numpy()
        fine_s = data.novel_support["fine"].cpu().numpy()
        rng = np.random.default_rng(seed)

        for p in FIT_POOLS:
            idx = rng.choice(zl_fit.shape[0], size=min(p, zl_fit.shape[0]), replace=False)  # paired: same idx
            cl = super_clf_acc(zl_fit[idx], sup_fit[idx], zl_q, sup_q)
            ct = super_clf_acc(zt_fit[idx], sup_fit[idx], zt_q, sup_q)
            pay[p]["low"].append(cl); pay[p]["trans"].append(ct)
            print(f"[seed {seed}] super fit_pool={p:>5}  z_low={cl.mean():.4f}  z_trans={ct.mean():.4f}  diff={ct.mean()-cl.mean():+.4f}")

        for k in K_SHOTS:
            al, _ = ncm_fine_acc(zl_s, fine_s, zl_q, fine_q, held, k, args.episodes, rng)
            at, _ = ncm_fine_acc(zt_s, fine_s, zt_q, fine_q, held, k, args.episodes, rng)
            fine[k]["low"].append(al); fine[k]["trans"].append(at)

    # verdict: gate at any fit_pool <= 1000
    print("\n=== AXIS 1: data-scarcity super-transfer ===")
    pool_results = {}
    confirmed = False
    for p in FIT_POOLS:
        point, lo, hi, am_low, am_trans = boot_ci(pay[p]["low"], pay[p]["trans"])
        gate = (point >= 0.02) and (lo > 0)
        eligible = p <= 1000
        if gate and eligible:
            confirmed = True
        pool_results[p] = {"superacc_zlow": am_low, "superacc_ztrans": am_trans,
                           "paired_diff": point, "ci95": [lo, hi], "gate_pass": bool(gate), "eligible_<=1000": eligible}
        flag = "  <-- CROSSOVER" if (gate and eligible) else ("  (gate)" if gate else "")
        print(f"fit_pool={p:>5}  z_low={am_low:.4f}  z_trans={am_trans:.4f}  diff={point:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]{flag}")

    print("\n=== fine k-shot (secondary) ===")
    fine_results = {}
    for k in K_SHOTS:
        al = float(np.mean(fine[k]["low"])); at = float(np.mean(fine[k]["trans"]))
        fine_results[k] = {"z_low": al, "z_trans": at, "diff": at - al}
        print(f"k={k:>2}  z_low={al:.4f}  z_trans={at:.4f}  diff={at-al:+.4f}")

    verdict = {"held_out_fine": held, "fit_pools": pool_results, "fine_kshot": fine_results,
               "AXIS1_CONFIRMED": bool(confirmed), "gate": "z_trans-z_low>=+0.02 & CI>0 at fit_pool<=1000"}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "pay_frontier_axis1.json"), "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"\nAXIS 1 {'CONFIRMED — low-data regime of utility EXISTS' if confirmed else 'NULL — no crossover at any pool <= 1000'}")


if __name__ == "__main__":
    main()
