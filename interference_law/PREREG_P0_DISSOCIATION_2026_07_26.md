# PREREG — P0: POWER the transfer/forgetting DISSOCIATION (frozen regime)

**Locked 2026-07-26.** Transfer-geometry thread, phase P0. $0 local, frozen resnet18, CIFAR + DTD, cached feats. Reuses `probe_forward.py` machinery (NEW-3b lock bb5bba73).

## Why (post-lit-review reframe)
Lit-review (2026-07-26, 3 agents) found the "two-axis / second-law" framing SCOOPED, but the CONTRARIAN wedge unscooped: prevailing 2025-26 view is COUPLING (transfer & forgetting = two signed ends of ONE cosine — TSV CVPR'25, NHB 2025, Geometry Conflict 2605.09608). NOBODY reports forward transfer DECOUPLED (rho~0.02) from the forgetting cosine. `out/forward.json` has that number at n=20/3-seed but NO reliability/CI/null. **The dissociation is only real if FT is a RELIABLE signal that is nonetheless cosine-blind** (Face-1b lesson: measurement noise manufactures false nulls). P0 settles that.

## Quantities (frozen readout regime; FT(i<-j) = AUC_A_i(from B_j-full-head init, few-shot) - AUC_A_i(cold few-shot))
1. **FT split-half reliability** rho_sb: split NSEED seeds odd/even -> two FT matrices -> Spearman across off-diag cells -> Spearman-Brown corrected. Bootstrap CI.
2. **rho(cos, FT_mean)** point estimate + bootstrap 95% CI (resample off-diag cell-pairs) + permutation null p (shuffle FT vs cos, 2000x).
3. Report per dataset (CIFAR, DTD), NSEED=12, n=20 caps.

## Pre-registered decision bands (commit BEFORE seeing numbers)
- **CONFIRMED (dissociation real -> proceed to P1):** FT reliability rho_sb >= 0.30 AND its boot CI lower > 0 (FT reproducibly rank-orders pairs) **AND** rho(cos,FT) boot 95% CI fully inside (-0.30, +0.30) (excludes coupling both signs).
- **REJECTED (reframe wrong -> STOP, reassess before spend):** rho(cos,FT) boot CI reaches beyond +/-0.30 on either dataset (coupling plausible; transfer may be cosine-mirror after all).
- **SAND (thread on noise -> de-noise FT first, or jump to plastic regime):** FT reliability rho_sb < 0.30 or CI includes 0 (FT mostly measurement noise; rho~0 trivial). Do NOT proceed to P1 descriptor bake-off on an unreliable target.

## Falsifier-first
The whole reframe can die here: if rho(cos,FT) CI includes +/-0.30, the contrarian wedge is unsupported = honest stop. If FT is unreliable, the frozen regime is the wrong substrate = pivot to plastic (NEW-3c) before spending. Either non-CONFIRM is decision-relevant and cheap ($0).

## Fixed knobs
NSEED=12, CAPS=20, K=30 few-shot, 8pos/24neg, STEPS_FULL=120, D_PROJ=24 (headroom, preserved backward cosine rho0.73 so forward null not a projection artifact), datasets cifar+dtd. Bootstrap B=2000, perm=2000. ASCII-only. Writes out/p0_dissociation.json.
