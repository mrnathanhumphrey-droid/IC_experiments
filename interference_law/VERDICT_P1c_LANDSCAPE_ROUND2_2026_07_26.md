# VERDICT — P1c (round-2): the frozen-FT interaction RESISTS optimization-landscape descriptors too -> NULL -> pivot to plastic (2026-07-26). $0 local, frozen resnet18 D_PROJ=24, CIFAR+DTD, NSEED=10, boot=perm=2000. Strict bar |rho(resid)|>=0.30 & perm_p<0.01 & cross-dataset.

## RESULT: NULL (pre-registered -> PIVOT TO PLASTIC). None of the 6 optimization-landscape descriptors (reachability, Newton-decrement, score-spread, whitened-alignment, feature-usage-overlap, curvature-trace) predicts the proven interaction residual at law-strength on both datasets. Most consistent are Y3 score-spread (+0.12/+0.11) and Y5 feature-usage (-0.11/-0.09) -- same sign both datasets but far below 0.30. The dominant, reliable, scale-invariant pairwise transfer interaction (P1b) is orthogonal to optimization geometry as well as representation geometry.

## The numbers (rho(descriptor, interaction-residual), 10 seeds)
  descriptor       cifar    dtd     note
  Y1_reach        +0.055  +0.077   null
  Y2_newton       -0.063  +0.085   null (sign-flip)
  Y3_spread       +0.123  +0.108   weak, consistent sign (best), <<0.30
  Y4_whiten       +0.065  +0.008   null
  Y5_usage        -0.107  -0.092   weak, consistent sign, <<0.30
  Y6_curvtr       +0.110  -0.088   null (sign-flip)

## The cumulative negative (P0+P1+P1b+P1c) -- comprehensive
The reliable frozen forward-transfer signal (r_SB 0.94-0.98), dominated by a pairwise interaction (~76% reliable variance, comp-rel 1.00, scale-invariant), is now shown ORTHOGONAL to a 14-descriptor battery spanning:
  DIRECTION: cosine, subspace principal-angle overlap, linear CKA, zero-shot head-AUC, whitened-alignment
  SCALE:     ||w_j|| (=source main-effect only), grad-norm, initial-loss, score-spread
  SUBSPACE/REACH: head-in-A-subspace, reachable-gradient-span projection
  CURVATURE: Newton-decrement, curvature-trace, feature-usage-overlap
This is essentially the entire descriptor family underlying the transferability-estimation literature (LogME/CKA/subspace/separability). None captures the signal. => a striking, comprehensive negative: a reliable transfer interaction that defeats standard geometry+optimization descriptors.

## Interpretation (honest, two readings)
1. GENUINE PUZZLE: the interaction is governed by a higher-order / non-geometric quantity not yet conceived. Would be a real open problem.
2. REGIME-SPECIFIC ARTIFACT: reliable across few-shot GD seeds, but possibly specific to the frozen linear-head + AdamW + random-proj-24d setup (deterministic per-init x per-task optimizer dynamics), i.e. reliable but not a scientifically-generalizable "transfer geometry." The frozen head-init warm-start is NOT feature-reuse transfer.
Both readings point the SAME way: the canonical forward-transfer GEOMETRY (feature reuse, where subspace/CKA/LogME are non-degenerate) requires the PLASTIC regime (B reshapes the backbone). The frozen regime has been exhausted as a descriptor-hunt substrate.

## Standing package (publishable regardless of what plastic finds)
P0: transfer reliable + cosine-blind (rebuts coupling papers TSV/NHB/GeomConflict).
P1/P1b/P1c: transfer dominated by a reliable scale-invariant pairwise interaction that defeats the entire standard descriptor family. Novel vs BOTH the continual-learning coupling literature AND the transferability-estimation literature.

## Bottom line
Frozen regime EXHAUSTED: reliable transfer interaction, comprehensively un-captured (14 descriptors, 2 datasets). Pre-registered pivot fires. >>> NEXT = PLASTIC REGIME: B reshapes the backbone (LoRA / full-FT), forward transfer FT(A|B) measured through B-reshaped features; feature-reuse geometry (subspace overlap, CKA) + LogME become non-degenerate, source-dependent predictors. Heavier compute (per-source backbone fine-tune) -> scope + cost estimate + local-vs-Lambda decision BEFORE firing [[no_heavy_local_compute_without_greenlight]].
