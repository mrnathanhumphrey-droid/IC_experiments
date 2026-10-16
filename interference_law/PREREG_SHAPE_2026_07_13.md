# PREREG — NEW-1a''''-c: the SHAPE of the interference — is it asymmetric off CIFAR, even if −cos doesn't fit?

Author: arc (CF interference). Date: 2026-07-13. Status: PRE-REGISTERED, SHA256-lock before run. resnet18, class-vs-neg, SQUARE directed
matrix. Datasets: CIFAR-100 (reference) + DTD + GTSRB. FULL fine-tune, deployed frozen-head metric. $0 local.

## The reframe (user)
The dataset switch (1a''''-b) killed the SPECIFIC predictor −cos(g_i,Δθ_p) off CIFAR. But that tested one HYPOTHESIS for the interaction's
shape, not whether the interference HAS directional shape at all. This probe characterizes the SHAPE of the directed forgetting matrix
model-free on DTD/GTSRB: is it ASYMMETRIC (like CIFAR's NEW-1: ‖K‖/‖S‖=0.285, ρ(F_ij,F_ji)=−0.02)? Is there REAL per-pair interaction
structure beyond fragility×disruptiveness? What shape (low-rank? pecking-order vs cyclic)? And how much does −cos capture, per dataset?

## Design (square directed matrix per dataset)
CAPS = classes 0..15 as BOTH victim and perturber (16×16 = 240 off-diagonal directed pairs). NEG disjoint (CIFAR 90-99; DTD/GTSRB 30-39).
resnet18 IMAGENET1K_V1, 224px. Per cap c: deployed frozen head w_c, Jacobian g_c=∂margin_c/∂θ; as perturber, full-FT (c vs neg) → Δθ_c.
F[i,j] = AUC_i(base) − AUC_i(after full-FT on j). S=(F+Fᵀ)/2, K=(F−Fᵀ)/2; DC = double-centered F (removes fragility row + disrupt col +
grand → pure interaction). cos[i,j] = cos(g_i, Δθ_j).

## PRE-REGISTERED PREDICTIONS
SHAPE-P1 — ASYMMETRY IS DATASET-GENERAL. On EACH of DTD and GTSRB: ‖K‖/‖S‖ ≥ 0.15 AND ρ(F_ij,F_ji) < 0.70. The interference is
    DIRECTIONAL (non-symmetric) on all datasets — asymmetry/fragility replicates even where the −cos predictor failed. (CIFAR reference
    must reproduce NEW-1: ‖K‖/‖S‖≈0.29, ρ≈0.) FALSIFIED on a dataset if ‖K‖/‖S‖ < 0.10 AND ρ ≥ 0.7 (matrix ~symmetric → interference is
    symmetric/magnitude-driven there).
SHAPE-P2 — THE INTERACTION IS REAL (not noise). The double-centered DC carries structure: its top singular value exceeds the 95th
    percentile of an entry-shuffled null (200 shuffles) on EACH dataset. → there IS a per-pair interaction shape beyond main effects,
    independent of whether −cos describes it. FALSIFIED if DC top-SV within the shuffle null (interaction = noise).
SHAPE-P3 — CHARACTERIZE THE SHAPE (descriptive, no threshold). Report per dataset: (a) interaction fraction ‖DC‖²/‖F−grand‖² (how much is
    interaction vs main effects); (b) effective rank of DC (participation ratio of singular values — low-rank vs diffuse); (c) rank-1
    dominance % (K ≈ h_i−h_j fit — transitive pecking-order vs cyclic residual, cf CIFAR NEW-1 45.8%). This is "the shape."
SHAPE-P4 — HOW MUCH DOES −cos CAPTURE? Report ρ(align_dc, DC) = does −cos(g_i,Δθ_j) align with the REAL interaction, per dataset. Expect
    positive on CIFAR (it was built there), weak/zero on DTD/GTSRB (1a''''-b). Quantifies −cos as a CIFAR-specific descriptor of a shape
    that may exist everywhere. If DTD/GTSRB have real interaction (P2) that −cos misses (P4≈0), the shape is real but −cos is not its law.

## Outcome grid (locked)
- P1+P2 hold on DTD/GTSRB, P4≈0 there: the interference IS directionally-shaped (asymmetric + real interaction) on all datasets, but −cos
  is a CIFAR-specific descriptor — the GENERAL shape is something else (SVD/dominance from P3 point at what). Reframes the downgrade: the
  directional STRUCTURE is general; our LAW for it was not.
- P1 holds, P2 fails off CIFAR: interference is asymmetric via MAIN EFFECTS (fragility≠disrupt) but has no real per-pair interaction off
  CIFAR → the "shape" is just row/col imbalance, no genuine pairwise directional law anywhere but CIFAR.
- P1 fails off CIFAR: interference is ~symmetric off CIFAR → even asymmetry is CIFAR-specific (stronger scope correction).

## Non-saturation
Each dataset's median off-diagonal forget ∈ [0.03, 0.45]; binary class-vs-neg full-FT 300/5e-4 (gate-passed on CIFAR/DTD/GTSRB earlier).
Flag any saturated matrix. Asymmetry/shape read only on non-saturated matrices.

## Config (locked)
resnet18 IMAGENET1K_V1 224px ImageNet-norm, fc→Identity+Flatten. 16 caps (victim=perturber) + 10 neg per dataset, PER_CLASS 100, FULL FT
wd 1e-4 300/5e-4, NJ 128, seed 0. Metrics = Frobenius asymmetry ratios + Spearman + SVD (participation-ratio eff-rank + shuffle null) +
rank-1 dominance fit. Single seed (the STRUCTURE is the claim). Local; DTD/GTSRB already downloaded.
