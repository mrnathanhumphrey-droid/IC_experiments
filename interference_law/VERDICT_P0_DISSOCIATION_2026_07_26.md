# VERDICT — P0: transfer/forgetting DISSOCIATION is POWERED + CONFIRMED (PREREG P0_2026_07_26). $0 local, frozen resnet18, D_PROJ=24, CIFAR+DTD, 12 seeds, boot=perm=2000.

## RESULT: OVERALL = CONFIRMED (both datasets). Forward transfer (frozen warm-start regime) is a HIGHLY RELIABLE signal (Spearman-Brown r_SB 0.95-0.98) that is STATISTICALLY BLIND to the forgetting cosine (rho ~0, boot 95% CI excludes +/-0.30 on BOTH datasets). The "sand" failure mode (FT = measurement noise -> trivial null) is decisively killed: FT reproducibly rank-orders task pairs. The COUPLING hypothesis (transfer = the constructive/opposite sign of the same cosine that governs forgetting) is REJECTED.

## The numbers (FT(i<-j) = AUC_A_i(from B_j-full-head init, few-shot) - AUC_A_i(cold few-shot); off-diag n(n-1) cells)
  dataset  coldAUC  FT-reliability r_SB [CI]     rho(cos,FT)  rho 95% CI        perm_p   BAND
  CIFAR     0.800    0.977 [0.969, 0.981]         +0.032       [-0.072, +0.130]  0.536    CONFIRMED
  DTD       0.809    0.945 [0.924, 0.961]         -0.012       [-0.115, +0.083]  0.816    CONFIRMED
  (xfer_frac ~0.27-0.31, hurt_frac ~0.49; FT range ~[-0.18,+0.11]; both signs of rho represented, both null.)

## Pre-registered decision bands (locked BEFORE numbers)
- CONFIRMED = r_SB>=0.30 & rel_CI_lo>0 (FT real) AND rho(cos,FT) boot 95% CI in (-0.30,+0.30) (excludes coupling). >>> MET on both.
- REJECTED = rho CI reaches beyond +/-0.30 (coupling plausible). Not triggered.
- SAND = FT reliability weak (null trivial). Not triggered (r_SB 0.95-0.98 >> 0.30).

## Why this matters (contrarian wedge, post lit-review)
Lit-review (2026-07-26) found the prevailing 2025-26 published position is COUPLING: transfer & interference are the two signed ends of ONE cosine/similarity axis (Task Singular Vectors CVPR'25 2412.00081 "constructive vs destructive proportional to cos of singular vectors"; Nature Human Behaviour 2025 "coupled via subspace alignment"; Geometry Conflict 2605.09608 "one covariance geometry does both"). P0 provides the empirical wedge NO paper reports: in the SAME feature space + SAME cosine descriptor, forgetting severity is cosine-governed (rho +0.55..+0.73, NEW-3) while forward transfer is cosine-AGNOSTIC (rho ~0, CI excludes +/-0.30) AND that transfer signal is highly reliable (r_SB 0.95-0.98, so the null is not noise). => forgetting and transfer are governed by DIFFERENT geometries. The coupling framing is empirically falsified here.

## Caveats / scope
- FROZEN readout regime: FT is a head-init WARM-START effect (B's trained head inits few-shot A), NOT plastic-backbone feature transfer. The plastic regime (B reshapes features, "NEW-3c") is untested and forgetting-confounded — a reviewer will ask if the dissociation survives there. That is a P3 generality question (see plan); if P1 descriptors come up null on frozen, plastic is the next substrate, not a kill.
- D_PROJ=24 headroom projection PRESERVED the backward cosine signal (rho 0.73), so the forward null is not a projection artifact.
- xfer net ~0 (help 27-31% vs hurt 49%): "reliable" means reproducibly rank-orders pairs, not "systematically beneficial."

## Bottom line
The dissociation is powered and bulletproof on two datasets: a reliable forward-transfer signal that the forgetting cosine cannot predict. This is the load-bearing wedge for the whole transfer-geometry thread. >>> PROCEED to P1: find geometric descriptor X (subspace principal angles / forward Jacobian alignment / feature-reuse effective-rank / A-vs-B CKA) that DOES predict this reliable FT, with LogME as the prior-to-lose (lit-review mandate) and pre-registered guards against the published CKA-fails (2601.03168) and eff-rank-fails (2602.20433) negatives.
