# VERDICT — P1: descriptor BAKE-OFF => frozen forward transfer is GEOMETRICALLY STRUCTURELESS; only a weak NEGATIVE init-SCALE effect replicates (PREREG P1_2026_07_26). $0 local, frozen resnet18, D_PROJ=24, CIFAR+DTD, NSEED=10, boot=perm=2000, kpc=5.

## RESULT: NULL at law-strength (pre-registered -> MOVE TO PLASTIC REGIME). No descriptor reaches |rho|>=0.30 replicating cross-dataset. The DIRECTION family (cosine, subspace principal-angle overlap, linear CKA, zero-shot head-AUC) is null on BOTH datasets, confirming the P0 cosine-blindness EXTENDS to all low-order subspace/direction geometry. The ONLY robustly replicated signal is the B-init head NORM ||w_j|| = -0.22/-0.23 (both perm_p=0.000, same sign+magnitude cifar+dtd) -- a SCALE/conditioning effect, sub-law-strength, NEGATIVE (bigger B-init -> LESS transfer benefit). Frozen forward transfer has no rich geometry to find; it is weakly governed by init scale, exactly as NEW-3b suspected.

## The numbers (rho(descriptor, reliable-FT), off-diag n(n-1) cells)
  descriptor          family   cifar    dtd     replicated_HIT(>=.30 same-sign both)
  COS                 dir     +0.034  -0.014    no (null both -- confirms P0)
  X1_subspace         dir     +0.066  +0.015    no (null both)
  X4_cka              dir     -0.020  +0.018    no (null both; matches published CKA-fails 2601.03168)
  Xzs_auc             dir     +0.226  +0.116    no (weak+, non-replicating at .30)
  X0_wnorm            scale   -0.222  -0.231    NO at .30 but ONLY robust cross-dataset signal (perm_p 0.000 both)
  X3_headInAsub       scale   +0.111  +0.057    no
  X6_gradnorm         scale   -0.162  +0.022    no (sign-flips across datasets)
  X7_loss0            scale   -0.086  -0.092    no (weak, consistent sign)

## Pre-registered decision bands (locked before numbers)
- HIT = some X with |rho|>=0.30, CI excl 0, same-sign replication BOTH datasets, beats COS. -> NOT MET.
- NULL = no descriptor clears 0.30 with cross-dataset replication. -> MET. Pre-registered action: MOVE TO PLASTIC REGIME (feature-reuse geometry + LogME genuinely defined). Contrarian P0 result stands and is STRENGTHENED.
- Cross-dataset non-replication treated as null (X6 sign-flips -> correctly rejected).

## Reading it (mechanism)
1. DIRECTION-family collapse CONFIRMED: in a frozen linear-head regime, subspace overlap / CKA / zero-shot-AUC are ~monotone in the discriminative cosine, which P0 showed is null. All land null. Predicted, not surprising.
2. Frozen forward transfer is GEOMETRICALLY STRUCTURELESS: no direction/subspace descriptor predicts the reliable FT at law strength. This EXTENDS P0 (cosine-blind) to the full low-order geometry battery.
3. The only robust predictor is ||w_j|| (B-init head norm), rho ~ -0.23, replicated. NEGATIVE: a larger-norm B-head init HURTS few-shot A transfer (over-committed init the K=30 few-shot steps can't escape), independent of B's task identity/direction. This mechanistically confirms NEW-3b's "generic weight-scale/prior, alignment-independent" reading. It is a SCALE knob, not a transfer GEOMETRY.

## Why this is the right pivot (not a dead end)
The frozen regime shares ONE backbone across all source tasks -> "feature reuse" between A and B is confounded (features identical) and LogME is degenerate (source features identical -> constant over j). So the frozen regime CANNOT host a genuine feature-reuse transfer geometry by construction. P1's structurelessness is the pre-registered EVIDENCE that the real second-law hunt requires the PLASTIC regime, where B reshapes the backbone and feature reuse is a real, measurable, source-dependent quantity (and LogME/subspace-overlap/CKA become non-degenerate predictors).

## Bottom line
Frozen forward transfer: reliable (P0 r_SB 0.95-0.98), cosine-blind (P0), and now shown geometrically structureless (P1) -- weakly governed only by init scale (||w_j|| rho -0.23, replicated). The contrarian dissociation is stronger than ever. The transfer GEOMETRY, if it exists, lives in the PLASTIC regime. >>> NEXT: (a) cheap $0 tightening -- norm-control test (do unit-normalized B-inits kill FT's reliable variance? -> is frozen transfer PURELY scale?); then (b) plastic-regime forward-transfer build (LoRA reshapes backbone; feature-reuse geometry + LogME defined) = the real second-law hunt.
