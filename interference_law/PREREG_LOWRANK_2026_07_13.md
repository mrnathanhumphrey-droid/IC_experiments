# PREREG — NEW-2: is the capability interference matrix LOW-RANK and COMPLETABLE?

Author: arc (CF interference). Date: 2026-07-13. Status: PRE-REGISTERED, SHA256-lock before run. resnet18, square directed forgetting
matrix, CIFAR + DTD (hardened across datasets from the start). FULL fine-tune, deployed frozen-head metric. $0 local.

## The question
The directional-descriptor hunt was a null (1a''''-d: no geometry names the interaction). NEW-2 asks a structural question that does NOT
need a descriptor: is the N×N directed forgetting matrix LOW-RANK, and can you COMPLETE it — measure a fraction of entries and predict the
rest? DTD's interaction was low-rank (~4 eff-dim); if the full matrix is low-rank+completable, capabilities live on a low-dim manifold and
pairwise interference is predictable from a sample (a practical continual-learning tool). Tested on CIFAR AND DTD (learned the single-
substrate lesson).

## Design
CAPS = 24 classes (victim=perturber) per dataset; CIFAR (neg 90-99) + DTD (neg 30-39). resnet18 224px. F[i,j] = deployed forgetting of
victim i under full-FT on perturber j (i≠j; diagonal excluded). Decompose the FULL matrix into additive bias (μ + fragility_i + disrupt_j)
+ low-rank interaction. MATRIX COMPLETION: randomly mask 50% of off-diagonal entries (5 seeds); fit [bias + rank-k interaction] on the
OBSERVED half via soft-impute (iterative SVD); predict the held-out half; report held-out R² vs model rank k ∈ {0,1,2,3,4,6,8} (k=0 =
bias/main-effects only). Also SVD the double-centered interaction (effective rank, top-k variance).

## PRE-REGISTERED PREDICTIONS
NEW2-P1 — COMPLETABLE. With 50% of off-diagonal entries observed, a bias+rank-k model predicts the held-out 50% at held-out R² ≥ 0.50 (at
    its best k), on BOTH CIFAR and DTD. → interference matrices are low-dimensional, completable objects: measure half, infer the rest.
    FALSIFIED if best-k held-out R² < 0.30 on a dataset (matrix is high-rank / idiosyncratic / not completable).
NEW2-P2 — INTERACTION ADDS BEYOND MAIN EFFECTS. Adding low-rank interaction (best k) improves held-out R² over bias-only (k=0) by ΔR² ≥
    +0.05 on DTD (whose interaction is genuinely low-rank, per 1a''''-c). Report CIFAR (may be ≈0 — its interaction was diffuse). If ΔR²≈0
    on both, completability is JUST main effects (fragility×disrupt), and the "manifold" is 2 scalars/capability (still completable, trivially).
NEW2-P3 — LOW EFFECTIVE RANK. The double-centered interaction has effective rank (participation ratio) ≤ N/3 (≤8 for N=24) on both datasets
    — concentrated, not full-rank noise. Report the SVD variance spectrum (top-k %).

## Outcome grid (locked)
- P1 + P2: interference matrices are completable LOW-RANK objects with a real interaction manifold beyond main effects — a clean, practical,
  dataset-general structural result (predict unseen capability interference from a sample), independent of any geometric descriptor.
- P1 holds, P2 ≈ 0: matrices are completable but ONLY via main effects (fragility_i + disrupt_j) — the low-dim structure is 2 per-capability
  scalars; the per-pair interaction is not predictively useful (noise-like). Still a useful completability result, weaker manifold claim.
- P1 fails: interference is high-rank / per-pair idiosyncratic → no low-dim structure; completion doesn't work.

## Non-saturation
CIFAR + DTD non-saturated at these settings (medFgt ~0.23/0.26 in 1a''''-c/-d). Require each matrix median off-diag forget ∈ [0.03,0.45];
flag otherwise. GTSRB EXCLUDED (was saturated). Read rank/completion only on non-saturated matrices.

## Config (locked)
resnet18 IMAGENET1K_V1 224px, 24 caps + 10 neg per dataset, PER_CLASS 100, FULL FT wd 1e-4 300/5e-4, seed 0. Completion: soft-impute
(iterative rank-k SVD, ≤100 iters), 5 mask seeds at 50% observed, held-out R² = 1 − SS_res/SS_tot vs held-out mean. Metrics = completion-R²
vs-k curve + interaction SVD spectrum, per dataset. Single seed for the matrices (5 mask seeds for completion variance). Local; DTD downloaded.
