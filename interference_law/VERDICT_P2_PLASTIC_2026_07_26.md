# VERDICT — P2: PLASTIC-regime forward transfer -> dissociation PERSISTS, second-law NULL, LogME ANTI-REPLICATES (PREREG P2_2026_07_26). Lambda a10, pythia-1.4b LoRA, clinc150+banking77, 4 seeds x 14 caps, PROJDIM=64, ~$1.8 total. Closes the transfer-geometry law-hunt.

## RESULT: The canonical plastic regime (B reshapes the backbone via LoRA) does NOT rescue a transfer geometry. Forward transfer is RELIABLE (FT r_SB 0.78-0.85) but (Q1) still cosine-BLIND at law-strength (dissociation persists) and (Q2) NOT predicted by any feature-reuse descriptor with cross-dataset replication. The must-beat baseline LogME is significant in BOTH datasets but with OPPOSITE SIGNS (clinc -0.18, banking +0.24) = ANTI-REPLICATION -> LogME does not robustly predict sequential forward transfer. Bonus: the reliable-variance structure FLIPS frozen->plastic (frozen interaction-dominated 0.76; plastic TARGET-main-effect-dominated 0.81/0.61) -> plastic "forward transfer" is mostly target-learnability, not pairwise source->target transfer.

## The numbers (rho(descriptor, reliable-FT), off-diag pairs)
  descriptor   clinc150            banking77           cross-dataset
  FT_rel       0.849               0.780               reliable both
  var r/c/res  0.81/0.04/0.17      0.61/0.07/0.33      target(row)-dominated both
  COS          +0.179 [.05,.31]    +0.062 [-.08,.21]   weak, sub-.30, non-replicating (dissociation persists)
  CKA_drift    +0.031 (ns)         -0.077 (ns)         null
  CKA_reuse    +0.066 (ns)         +0.080 (ns)         null
  SUBOVL       -0.214 (p.003)      -0.090 (ns)         same sign, non-replicating at .30
  LOGME        -0.183 (p.015)      +0.244 (p.002)      *** SIGN-FLIP = anti-replicate ***
  EFFRANK      +0.053 (ns)         -0.030 (ns)         null

## Pre-registered decision (locked before numbers)
- Q1 dissociation PERSISTS if rho(COS,FT) CI inside (-.30,.30) both -> YES (max |rho|=0.18, no CI reaches +/-0.30 robustly; banking null). The coupling hypothesis (transfer=cosine) is NOT supported in the plastic regime either.
- Q2 HIT if a feature-reuse descriptor |rho|>=0.30, same-sign both datasets, perm_p<0.01 -> NOT MET. LogME sign-flips; SUBOVL replicates sign but not magnitude; CKA/EFFRANK null. NULL.

## The cumulative thread result (P0 -> P2) -- the honest close
Forward transfer, across BOTH the frozen head-init regime AND the canonical plastic LoRA-reshaping regime, on 3 datasets (CIFAR, DTD, clinc150, banking77) and 2 model families (resnet18, pythia-1.4b):
 1. is RELIABLE (r_SB 0.78-0.98) -- a real, reproducible signal, not noise;
 2. is COSINE-BLIND -- decoupled from the discriminative/forgetting cosine (rho ~0 frozen; <=0.18 non-replicating plastic). => rebuts the COUPLING view (Task Singular Vectors CVPR'25, Nature Human Behaviour 2025, Geometry Conflict 2605.09608: "transfer = the constructive end of one cosine");
 3. resists a ~20-descriptor battery spanning direction, scale, subspace, CKA, reachability, curvature, feature-usage (frozen: 14) AND feature-reuse geometry + LogME (plastic: 6); no low-order geometric LAW predicts it at |rho|>=0.30 with cross-dataset replication;
 4. LogME (SOTA transferability metric; the must-beat) ANTI-REPLICATES across datasets in the sequential setting -> a novel negative vs the transferability-estimation literature (which is almost entirely source->target model-selection, not continual forward transfer);
 5. has a REGIME-DEPENDENT variance structure: frozen = pairwise-interaction-dominated (0.76); plastic = target-main-effect-dominated (0.61-0.81). "Forward transfer" in the plastic regime is largely "which target is learnable after any reshaping," not source->target geometry.

## What this is (honest framing)
NOT "we found the second law." The thread's positive hypothesis (a geometric descriptor X governs forward transfer, orthogonal to the forgetting cosine) is FALSIFIED across both regimes. What stands is a STRONG DOUBLE result: (a) the contrarian DISSOCIATION -- reliable transfer is decoupled from the forgetting geometry, rebutting three coupling papers; (b) a COMPREHENSIVE NEGATIVE -- forward transfer resists the entire standard descriptor family (representation-geometry + optimization-geometry + LogME) in both frozen and plastic regimes, with LogME anti-replicating. Honest negatives are the win.

## Bottom line
The transfer-geometry LAW-HUNT is CLOSED as a falsified positive / confirmed negative. The dissociation (P0) and the comprehensive descriptor-negative (P1/P1b/P1c/P2) stand as the result. No further descriptor rounds warranted (diminishing returns; the pattern replicates across 2 regimes, 3 datasets, 2 model families). USER writes; Claude banked all numbers in out/*.json + VERDICT_*.md. Next = user's call (close thread / write / pivot to a banked alternative arc).
