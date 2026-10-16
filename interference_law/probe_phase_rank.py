"""
probe_phase_rank.py -- DIRECTION B: the frozen<->plastic PHASE TRANSITION, with LoRA rank as the ORDER PARAMETER.
The arc established the endpoints: frozen linear readout -> forget ~ -cos (rho -0.74, DIRECTION-governed); LoRA-rank-32 backbone
-> rho -0.12 (cosine dead, DISRUPTIVENESS-governed). The rank sweep BETWEEN them was explicitly deferred (VERDICT_REPRESENTATIONAL_LAW
line 43). This maps the crossover: as plasticity (LoRA rank) rises, does DIRECTION governance decay and DISRUPTIVENESS governance take over?

Reuses probe_representational_law machinery. Per rank r in {1,2,4,8,16,32}: fine-tune LoRA on each B-task, re-probe ALL A, collect
(cos, forget, B, feature-drift). Report per rank:
  DIRECTION axis      : rho(cos,forget) pooled + mean WITHIN-B partial rho(cos,forget|B)   [should decay with rank]
  DISRUPTIVENESS axis : eta2_B (frac of forget-variance from B-identity) + rho(per-B drift, per-B forget)  [should rise with rank]
The crossover rank = where |direction| gives way to disruptiveness. Runs on Lambda GPU. ASCII-only prints.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R   # build_model, finetune_B, features, probe_auc, ddir, lora_params, load_banks, spearman

OUT = HERE / "out" / "phase_rank.json"
OUT.parent.mkdir(parents=True, exist_ok=True)   # out/ may not exist on a fresh Lambda box
RANKS = [int(x) for x in os.environ.get("PR_RANKS", "1,2,4,8,16,32").split(",")]
N_BTASKS = int(os.environ.get("PR_NB", "8"))
B_STEPS = int(os.environ.get("PR_BSTEPS", "1000"))
B_LR = float(os.environ.get("PR_BLR", "4e-3"))
SEED = int(os.environ.get("PR_SEED", "0"))
DEV = R.DEV
np.random.seed(SEED); torch.manual_seed(SEED)


def eta2_by_group(vals, groups):
    # fraction of variance explained by group identity (between-group / total)
    vals = np.asarray(vals, float); groups = np.asarray(groups)
    gm = vals.mean(); ss_tot = ((vals - gm)**2).sum()
    ss_bet = 0.0
    for g in np.unique(groups):
        v = vals[groups == g]; ss_bet += len(v)*(v.mean()-gm)**2
    return float(ss_bet/(ss_tot+1e-12))


@torch.no_grad()
def drift(F0, F1):
    return float(np.mean(np.linalg.norm(F1 - F0, axis=1)))


def dtheta_norm(model):
    return float(np.sqrt(sum(float((p.detach()**2).sum()) for p in R.lora_params(model))))


def main():
    t0 = time.time()
    print(f"[PR] ranks={RANKS} N_B={N_BTASKS} B_steps={B_STEPS} B_lr={B_LR} dev={DEV}", flush=True)
    banks = R.load_banks(); print(f"[PR] loaded {len(banks)} banks ({time.time()-t0:.0f}s)", flush=True)
    Xneg_u8 = torch.cat([banks[c] for c in R.NEG_CLASSES])

    # base (frozen) features/directions -- rank-independent (LoRA init update = 0), compute ONCE
    base = R.build_model(RANKS[0])
    Fneg0 = R.features(base, Xneg_u8)
    FA0 = {a: R.features(base, banks[a]) for a in R.A_CLASSES}
    accA0 = {a: R.probe_auc(FA0[a], Fneg0) for a in R.A_CLASSES}
    dA = {a: R.ddir(FA0[a], Fneg0) for a in R.A_CLASSES}
    print(f"[PR] base accA0 median={np.median(list(accA0.values())):.3f} ({time.time()-t0:.0f}s)", flush=True)

    # fix the SAME B-tasks across ranks (fair comparison)
    rng = np.random.RandomState(1)
    BTASKS = [list(rng.choice(R.B_POOL, R.BCLS, replace=False)) for _ in range(N_BTASKS)]
    dB = {bi: R.ddir(np.concatenate([R.features(base, banks[c]) for c in bt]), Fneg0) for bi, bt in enumerate(BTASKS)}

    per_rank = {}
    for r in RANKS:
        cos_all = []; fg_all = []; Bidx = []; perB_drift = {}; perB_dth = {}
        for bi, bt in enumerate(BTASKS):
            model = R.build_model(r)
            R.finetune_B(model, banks, bt, B_STEPS, B_LR)
            Fneg1 = R.features(model, Xneg_u8)
            perB_drift[bi] = drift(Fneg0, Fneg1); perB_dth[bi] = dtheta_norm(model)
            for a in R.A_CLASSES:
                acc1 = R.probe_auc(R.features(model, banks[a]), Fneg1)
                cos_all.append(float(dA[a] @ dB[bi])); fg_all.append(accA0[a] - acc1); Bidx.append(bi)
            del model
            if DEV == "cuda": torch.cuda.empty_cache()
        cos_all = np.array(cos_all); fg_all = np.array(fg_all); Bidx = np.array(Bidx)
        # DIRECTION axis
        rho_pool = R.spearman(cos_all, fg_all)
        within = [R.spearman(cos_all[Bidx == bi], fg_all[Bidx == bi]) for bi in range(N_BTASKS) if (Bidx == bi).sum() > 5]
        rho_within = float(np.mean(within))
        # DISRUPTIVENESS axis
        eta2B = eta2_by_group(fg_all, Bidx)
        perB_med = np.array([np.median(fg_all[Bidx == bi]) for bi in range(N_BTASKS)])
        rho_drift = R.spearman(np.array([perB_drift[bi] for bi in range(N_BTASKS)]), perB_med)
        med_f = float(np.median(fg_all)); inband = bool(0.05 <= med_f <= 0.40)
        per_rank[r] = {"rho_cos_pool": round(rho_pool, 3), "rho_cos_within_B": round(rho_within, 3),
                       "eta2_B": round(eta2B, 3), "rho_drift_disrupt": round(rho_drift, 3),
                       "median_forget": round(med_f, 3), "inband": inband,
                       "mean_drift": round(float(np.mean(list(perB_drift.values()))), 3),
                       "mean_dtheta": round(float(np.mean(list(perB_dth.values()))), 3)}
        print(f"  rank={r:3d} | DIR rho_cos={rho_pool:+.3f} within-B={rho_within:+.3f} | DISRUPT eta2_B={eta2B:.3f} rho_drift={rho_drift:+.3f} "
              f"| med_forget={med_f:.3f} band={inband} drift={per_rank[r]['mean_drift']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        OUT.write_text(json.dumps({"config": {"ranks": RANKS, "N_B": N_BTASKS, "B_steps": B_STEPS, "B_lr": B_LR, "seed": SEED},
                                   "frozen_ref_rho": -0.74, "per_rank": per_rank}, indent=1))

    # crossover readout
    print("\n=== DIRECTION B: frozen<->plastic phase transition (order parameter = LoRA rank) ===", flush=True)
    print(f"  frozen readout reference (rank 0): rho(cos,forget) = -0.74 (DIRECTION-governed)", flush=True)
    print(f"  {'rank':>5} {'|rho_cos|':>9} {'within-B':>9} {'eta2_B':>7} {'rho_drift':>10} {'med_fgt':>8}", flush=True)
    for r in RANKS:
        p = per_rank[r]
        print(f"  {r:>5} {abs(p['rho_cos_pool']):>9.3f} {p['rho_cos_within_B']:>+9.3f} {p['eta2_B']:>7.3f} {p['rho_drift_disrupt']:>+10.3f} {p['median_forget']:>8.3f}", flush=True)
    meas = [r for r in RANKS if per_rank[r]["inband"]]
    print(f"  measurable (partial-forget) ranks: {meas}", flush=True)
    print(f"[PR] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
