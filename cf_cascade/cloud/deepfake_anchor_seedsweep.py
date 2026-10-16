"""Anchor B driver — deepfake seed-robustness (PREREG_ANCHOR_DEEPFAKE_SEEDROBUST, LOCK a227f6a5).
COPY-NOT-EDIT: imports the physics_detector harness functions READ-ONLY (originals untouched), adds a SEED knob, the two
Omega-placement cosines (transparent defs), and writes ONLY to cf_cascade/gateA. Frozen DetectorCNN backbone -> 128-d head;
the substrate where reorganization is structurally impossible (head-only), so protection = suppression. ARM GPM=real
projection, GPM-RANDCTRL=random projection (suppression-only negative control). $0 local CPU.
Run:  SEED=0 python deepfake_anchor_seedsweep.py        # one seed (validate)
      python deepfake_anchor_seedsweep.py --sweep 0 1 2 3 4   # multi-seed
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
ORIG = Path("D:/physics_detector/v2/learned_detector")
sys.path.insert(0, str(ORIG))
import cl_gpm_diffusion as CG   # read-only reuse; its __main__ is guarded so import has no side effects
GATEA = Path("D:/IC_experiments/cf_cascade/gateA"); GATEA.mkdir(parents=True, exist_ok=True)


def cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def run_seed(seed):
    t0 = time.time(); print(f"\n===== SEED {seed} =====", flush=True)
    backbone = CG.load_backbone()
    reals, swap, diff = CG.build_items()
    swap = CG.subsample(swap, CG.SUBSAMPLE, seed); diff = CG.subsample(diff, CG.SUBSAMPLE, seed); reals = CG.subsample(reals, CG.SUBSAMPLE, seed)
    real_tr, real_te = CG.group_split(reals, 0.3, seed); swap_tr, swap_te = CG.group_split(swap, 0.3, seed); diff_tr, diff_te = CG.group_split(diff, 0.3, seed)
    Xr_tr, yr_tr = CG.extract(backbone, real_tr, "real_tr"); Xr_te, yr_te = CG.extract(backbone, real_te, "real_te")
    Xs_tr, ys_tr = CG.extract(backbone, swap_tr, "swap_tr"); Xs_te, ys_te = CG.extract(backbone, swap_te, "swap_te")
    Xd_tr, yd_tr = CG.extract(backbone, diff_tr, "diff_tr"); Xd_te, yd_te = CG.extract(backbone, diff_te, "diff_te")
    Xswap_tr = np.concatenate([Xs_tr, Xr_tr]); yswap_tr = np.concatenate([ys_tr, yr_tr])
    Xswap_te = np.concatenate([Xs_te, Xr_te]); yswap_te = np.concatenate([ys_te, yr_te])
    Xdiff_tr = np.concatenate([Xd_tr, Xr_tr]); ydiff_tr = np.concatenate([yd_tr, yr_tr])
    Xdiff_te = np.concatenate([Xd_te, Xr_te]); ydiff_te = np.concatenate([yd_te, yr_te])
    ev = lambda h: (CG.auc(CG.head_scores(h, Xswap_te), yswap_te), CG.auc(CG.head_scores(h, Xdiff_te), ydiff_te))
    base = CG.train_head(Xswap_tr, yswap_tr, seed=seed); base_sa, base_da = ev(base)
    base_state = {k: v.clone() for k, v in base.state_dict().items()}
    U_swap = CG.swap_memory_basis(Xs_tr, CG.PCA_K); U_rand = CG.random_orthonormal(128, CG.PCA_K, seed=seed)
    naive = CG.continue_train(base_state, Xdiff_tr, ydiff_tr, U=None, seed=seed); naive_sa, naive_da = ev(naive)
    gpm = CG.continue_train(base_state, Xdiff_tr, ydiff_tr, U=U_swap, seed=seed); gpm_sa, gpm_da = ev(gpm)
    randc = CG.continue_train(base_state, Xdiff_tr, ydiff_tr, U=U_rand, seed=seed); rand_sa, rand_da = ev(randc)
    # Omega-placement cosines (transparent defs): head-weight + class-mean-shift, swap vs diffusion
    diff_head = CG.train_head(Xdiff_tr, ydiff_tr, seed=seed)
    c_w = cos(base.fc.weight.detach().numpy(), diff_head.fc.weight.detach().numpy())
    real_mean = np.concatenate([Xr_tr, Xr_te]).mean(0)
    ms_swap = np.concatenate([Xs_tr, Xs_te]).mean(0) - real_mean; ms_diff = np.concatenate([Xd_tr, Xd_te]).mean(0) - real_mean
    c_ms = cos(ms_swap, ms_diff)
    rep = {"seed": seed, "base_swap_auc": base_sa, "base_diff_auc": base_da,
           "naive_swap_auc": naive_sa, "naive_diff_auc": naive_da, "gpm_swap_auc": gpm_sa, "gpm_diff_auc": gpm_da,
           "randctrl_swap_auc": rand_sa, "randctrl_diff_auc": rand_da,
           "swap_drop_naive": base_sa - naive_sa, "swap_drop_gpm": base_sa - gpm_sa, "swap_drop_randctrl": base_sa - rand_sa,
           "cos_w_swap_w_diff": c_w, "cos_meanshift_swap_diff": c_ms, "wall_s": round(time.time() - t0, 1)}
    (GATEA / f"anchor_deepfake_seed{seed}.json").write_text(json.dumps(rep, indent=1))
    print(f"  base_swap={base_sa:.3f} naive_swap={naive_sa:.3f} gpm_swap={gpm_sa:.3f} rand_swap={rand_sa:.3f} | gpm_diff={gpm_da:.3f} | cos_ms={c_ms:+.3f} cos_w={c_w:+.3f} ({rep['wall_s']}s)", flush=True)
    return rep


def main():
    if "--sweep" in sys.argv:
        seeds = [int(s) for s in sys.argv[sys.argv.index("--sweep") + 1:]]
    else:
        seeds = [int(os.environ.get("SEED", "0"))]
    reps = [run_seed(s) for s in seeds]
    if len(reps) > 1:
        agg = {}
        for k in ("base_swap_auc", "naive_swap_auc", "gpm_swap_auc", "randctrl_swap_auc", "gpm_diff_auc",
                  "swap_drop_naive", "swap_drop_gpm", "swap_drop_randctrl", "cos_meanshift_swap_diff", "cos_w_swap_w_diff"):
            v = np.array([r[k] for r in reps]); agg[k] = [float(v.mean()), float(v.std(ddof=1) / len(v) ** 0.5)]
        out = {"anchor": "deepfake_seedrobust", "lock": "a227f6a5", "substrate": "physics-detector frozen-backbone GPM (head-only, no-reorg)",
               "seeds": seeds, "n_seeds": len(seeds), "per_seed": reps, "aggregate_mean_se": agg,
               "reads": {"antagonistic_endpoint_firms": bool(np.mean([r["cos_meanshift_swap_diff"] for r in reps]) < -0.5),
                         "suppression_works": bool(np.mean([r["gpm_swap_auc"] for r in reps]) > np.mean([r["naive_swap_auc"] for r in reps])),
                         "randctrl_zero_protection": bool(np.mean([r["randctrl_swap_auc"] for r in reps]) <= np.mean([r["naive_swap_auc"] for r in reps]) + 0.03)}}
        (GATEA / "anchor_deepfake_seedrobust.json").write_text(json.dumps(out, indent=1))
        print(f"\n[AnchorB] AGG cos_ms={agg['cos_meanshift_swap_diff']} gpm_swap={agg['gpm_swap_auc']} naive_swap={agg['naive_swap_auc']} randctrl_swap={agg['randctrl_swap_auc']}", flush=True)
        print(f"[AnchorB] reads={out['reads']} -> {GATEA / 'anchor_deepfake_seedrobust.json'}", flush=True)


if __name__ == "__main__":
    main()
