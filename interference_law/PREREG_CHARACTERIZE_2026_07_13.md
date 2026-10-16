# PREREG — NEW-1a''''-d: what DESCRIBES the low-rank interaction? Is it the FEATURE cosine (frozen law's variable), not the Jacobian?

Author: arc (CF interference). Date: 2026-07-13. Status: PRE-REGISTERED, SHA256-lock before run. resnet18, square directed matrix,
CIFAR + DTD. FULL fine-tune, deployed frozen-head metric. $0 local.

## The question the shape probe created
1a''''-c: the interference is directionally-shaped everywhere (asymmetric + real LOW-RANK interaction on DTD, ~4 eff-dims) but −cos(g_i,Δθ_j)
(weight-space Jacobian) describes ONLY CIFAR's interaction (+0.29), MISSES DTD's (+0.06). So a real compact shape exists off CIFAR that we
have not named. HYPOTHESIS: the general descriptor is the FEATURE-space discriminative cosine −cos(d_i,d_j) — the SAME variable as the
frozen 4-op law (#1) — and the weight-space Jacobian −cos was a CIFAR-coincidental proxy. Test which candidate fits the interaction on
BOTH datasets, and characterize the SVD subspace.

## Design (rebuild square matrices, compute all candidate descriptors)
CAPS = 16 classes (victim=perturber) per dataset; CIFAR (neg 90-99) + DTD (neg 30-39). resnet18 224px. Per cap c at BASE: feature
discriminative dir d_c = ddir(feat_c, feat_neg); margin-Jacobian g_c = ∂margin_c/∂θ; as perturber, Δθ_c from full-FT (c vs neg).
F[i,j] = deployed forgetting of victim i under perturber j. DC = double-centered F (the interaction). Candidate descriptor matrices (i≠j),
all compared to DC via their OWN double-centering (isolate interaction):
  FC[i,j] = −cos(d_i, d_j)        feature discriminative-direction cosine  (frozen-law variable)
  JC[i,j] = −cos(g_i, Δθ_j)       Jacobian–displacement cosine             (the CIFAR −cos law)
  GC[i,j] = −cos(g_i, g_j)        margin-gradient alignment
  DS[i,j] = −cos(μ_i, μ_j)        raw feature-centroid cosine (crude similarity control)

## PRE-REGISTERED PREDICTIONS
CHAR-P1 — FEATURE COSINE IS THE GENERAL DESCRIPTOR. On DTD, ρ(dcenter(FC), dcenter(F)) ≥ +0.20 AND it is the LARGEST |ρ| among the four
    candidates (beats JC's known +0.06). → DTD's real interaction is described by feature-space discriminative geometry, not the weight-
    space Jacobian. FALSIFIED if FC < +0.10 on DTD or another candidate clearly wins.
CHAR-P2 — FEATURE COSINE FITS CIFAR TOO (generality). On CIFAR, ρ(dcenter(FC), dcenter(F)) ≥ +0.20 as well. If CHAR-P1 ∧ CHAR-P2: the
    FEATURE cosine describes the plastic directional interaction on BOTH datasets → it is the general descriptor; the Jacobian −cos (CIFAR
    +0.29 / DTD +0.06) was a CIFAR-coincidental proxy that happens to align with the feature cosine on CIFAR only. Unifies the plastic
    interaction geometry with the FROZEN 4-op law's variable (−cos of discriminative directions).
CHAR-P3 — SUBSPACE CHARACTERIZATION (descriptive). Report per dataset: DC singular spectrum (top-4 σ, variance fraction), and for the top
    mode the victim-side (u_1) and perturber-side (v_1) class loadings — which capabilities drive / receive the dominant interaction mode,
    and whether u_1≈v_1 (a class that both gives and takes along the same axis) or they differ (giver≠taker structure).

## Outcome grid (locked)
- P1+P2: FEATURE cosine is the general interaction descriptor — the plastic directional interaction IS the frozen-law geometry (−cos of
  discriminative directions), dataset-general; the Jacobian −cos was CIFAR-specific noise-alignment. Clean unification, rehabilitates a
  GENERAL directional law (just in feature not weight space).
- P1 holds, P2 weak: feature cosine fits DTD but not CIFAR → each dataset has its own descriptor; no single general law (structure general,
  law not).
- P1 fails (no candidate fits DTD): the DTD low-rank interaction is described by NONE of these geometries → the shape is real but its
  descriptor is outside {feature, Jacobian, gradient, centroid} cosines → report the SVD subspace as the open object (P3), name it later.

## Non-saturation
CIFAR + DTD both non-saturated at 16×16 in 1a''''-c (medFgt 0.247 / 0.259). GTSRB EXCLUDED (was partly saturated, untrustworthy). Reuse
FT 300/5e-4. Read descriptors only on the (confirmed non-saturated) matrices.

## Config (locked)
resnet18 IMAGENET1K_V1 224px, 16 caps + 10 neg per dataset, PER_CLASS 100, FULL FT wd 1e-4 300/5e-4, NJ 128, seed 0. Metrics = Spearman of
double-centered descriptor-vs-F + SVD (σ, U/V loadings). Single seed (descriptor identity is the claim). Local; DTD downloaded.
