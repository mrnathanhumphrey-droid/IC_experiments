"""
probe_phase_matched.py -- DIRECTION B, disruptiveness sub-map at MATCHED FORGETTING (isolates eta2_B-vs-rank from forget-magnitude).
The two single-recipe runs confounded at opposite ends (high-LR saturates high ranks, gentle-LR floors low ranks). Here: fix a MODERATE
LR and CALIBRATE steps PER RANK to hit a common target median forget (~0.10), then compare disruptiveness (eta2_B, within-B rho_cos,
drift->forget) across ranks at matched forgetting. Ranks {2,8,32} bracket the plasticity range. Reuses probe_representational_law +
probe_phase_rank machinery. Runs on Lambda GPU. ASCII-only prints.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R
import probe_phase_rank as PR   # eta2_by_group, drift

OUT = HERE / "out" / "phase_matched.json"
OUT.parent.mkdir(parents=True, exist_ok=True)
RANKS = [int(x) for x in os.environ.get("PM_RANKS", "2,8,32").split(",")]
LR = float(os.environ.get("PM_LR", "6e-3"))
TARGET = float(os.environ.get("PM_TARGET", "0.10"))
CAL_STEPS = int(os.environ.get("PM_CAL_STEPS", "800"))
N_CAL = int(os.environ.get("PM_NCAL", "4"))
N_B = int(os.environ.get("PM_NB", "8"))
MIN_STEPS = int(os.environ.get("PM_MIN", "400")); MAX_STEPS = int(os.environ.get("PM_MAX", "2600"))
SEED = int(os.environ.get("PM_SEED", "0"))
DEV = R.DEV
np.random.seed(SEED); torch.manual_seed(SEED)


def main():
    t0 = time.time()
    print(f"[PM] ranks={RANKS} LR={LR} target_forget={TARGET} cal_steps={CAL_STEPS} N_cal={N_CAL} N_B={N_B} dev={DEV}", flush=True)
    banks = R.load_banks(); print(f"[PM] loaded {len(banks)} banks ({time.time()-t0:.0f}s)", flush=True)
    Xneg_u8 = torch.cat([banks[c] for c in R.NEG_CLASSES])

    base = R.build_model(RANKS[0])
    Fneg0 = R.features(base, Xneg_u8)
    FA0 = {a: R.features(base, banks[a]) for a in R.A_CLASSES}
    accA0 = {a: R.probe_auc(FA0[a], Fneg0) for a in R.A_CLASSES}
    dA = {a: R.ddir(FA0[a], Fneg0) for a in R.A_CLASSES}
    print(f"[PM] base accA0 median={np.median(list(accA0.values())):.3f} ({time.time()-t0:.0f}s)", flush=True)

    rng = np.random.RandomState(1)
    BTASKS = [list(rng.choice(R.B_POOL, R.BCLS, replace=False)) for _ in range(N_B)]
    dB = {bi: R.ddir(np.concatenate([R.features(base, banks[c]) for c in bt]), Fneg0) for bi, bt in enumerate(BTASKS)}

    def run_btask(rank, bi, steps):
        """fine-tune LoRA rank on B-task bi for `steps`, return (per-A forgets list, drift)."""
        model = R.build_model(rank)
        R.finetune_B(model, banks, BTASKS[bi], steps, LR)
        Fneg1 = R.features(model, Xneg_u8); dr = PR.drift(Fneg0, Fneg1)
        fgs = [accA0[a] - R.probe_auc(R.features(model, banks[a]), Fneg1) for a in R.A_CLASSES]
        del model
        if DEV == "cuda": torch.cuda.empty_cache()
        return fgs, dr

    per_rank = {}
    for r in RANKS:
        # calibrate: probe forget at CAL_STEPS on N_CAL tasks, extrapolate steps to hit TARGET
        cal_f = []
        for bi in range(N_CAL):
            fgs, _ = run_btask(r, bi, CAL_STEPS); cal_f.append(np.median(fgs))
        f_probe = float(np.median(cal_f))
        needed = int(np.clip(CAL_STEPS * TARGET / max(f_probe, 0.004), MIN_STEPS, MAX_STEPS))
        print(f"  rank={r}: cal forget@{CAL_STEPS}={f_probe:.3f} -> matched steps={needed} ({time.time()-t0:.0f}s)", flush=True)
        # measure at matched steps over all N_B
        cos_all = []; fg_all = []; Bidx = []; perB_drift = {}
        for bi in range(N_B):
            fgs, dr = run_btask(r, bi, needed); perB_drift[bi] = dr
            for ai, a in enumerate(R.A_CLASSES):
                cos_all.append(float(dA[a] @ dB[bi])); fg_all.append(fgs[ai]); Bidx.append(bi)
        cos_all = np.array(cos_all); fg_all = np.array(fg_all); Bidx = np.array(Bidx)
        within = [R.spearman(cos_all[Bidx == bi], fg_all[Bidx == bi]) for bi in range(N_B) if (Bidx == bi).sum() > 5]
        eta2B = PR.eta2_by_group(fg_all, Bidx)
        perB_med = np.array([np.median(fg_all[Bidx == bi]) for bi in range(N_B)])
        rho_drift = R.spearman(np.array([perB_drift[bi] for bi in range(N_B)]), perB_med)
        med_f = float(np.median(fg_all))
        per_rank[r] = {"matched_steps": needed, "achieved_median_forget": round(med_f, 3),
                       "eta2_B": round(eta2B, 3), "within_B_rho_cos": round(float(np.mean(within)), 3),
                       "rho_drift_disrupt": round(rho_drift, 3),
                       "per_B_forget_spread": round(float(perB_med.max() - perB_med.min()), 3)}
        print(f"  rank={r:3d} MATCHED@{needed} | med_forget={med_f:.3f} | eta2_B={eta2B:.3f} within-B_rho={np.mean(within):+.3f} "
              f"rho_drift={rho_drift:+.3f} perB_spread={per_rank[r]['per_B_forget_spread']:.3f} ({time.time()-t0:.0f}s)", flush=True)
        OUT.write_text(json.dumps({"config": {"ranks": RANKS, "LR": LR, "target": TARGET, "N_B": N_B, "seed": SEED},
                                   "per_rank": per_rank}, indent=1))

    print("\n=== DIRECTION B: disruptiveness vs rank at MATCHED forgetting (~%.2f) ===" % TARGET, flush=True)
    print(f"  {'rank':>5} {'steps':>6} {'med_fgt':>8} {'eta2_B':>7} {'within-B':>9} {'rho_drift':>10} {'perB_spread':>12}", flush=True)
    for r in RANKS:
        p = per_rank[r]
        print(f"  {r:>5} {p['matched_steps']:>6} {p['achieved_median_forget']:>8.3f} {p['eta2_B']:>7.3f} {p['within_B_rho_cos']:>+9.3f} "
              f"{p['rho_drift_disrupt']:>+10.3f} {p['per_B_forget_spread']:>12.3f}", flush=True)
    es = [per_rank[r]["eta2_B"] for r in RANKS]
    trend = "RISES with rank" if es[-1] > es[0] + 0.05 else "FALLS with rank" if es[-1] < es[0] - 0.05 else "FLAT in rank"
    matched = all(abs(per_rank[r]["achieved_median_forget"] - TARGET) < 0.05 for r in RANKS)
    print(f"  forgetting matched across ranks: {matched} | disruptiveness (eta2_B) {trend}", flush=True)
    print(f"[PM] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
