# PREREG — NEW-1b: is the interference asymmetry a PECKING ORDER or a CYCLE? (Hodge decomposition + split-half reality test)

Author: arc (CF interference). Date: 2026-07-13. Status: PRE-REGISTERED, SHA256-lock before run. resnet18, square directed forgetting
matrix, CIFAR + DTD. FULL fine-tune, deployed frozen-head metric. $0 local.

## The question
The interference matrix is robustly ASYMMETRIC (1a''''-c, all datasets). NEW-1's rank-1 dominance fit explained only ~46% of the
antisymmetry → ~54% "cyclic residual." Is that residual REAL non-transitive structure (capability rock-paper-scissors: A damages B > B
damages A, B>C, C>A) or measurement NOISE? NEW-2 showed the per-pair interaction is largely structureless, so this is the decisive test of
whether ANY of the antisymmetry beyond a pecking order is real.

## Method: Helmholtz-Hodge decomposition of the tournament
K = (F − Fᵀ)/2 (antisymmetric) decomposes ORTHOGONALLY on the complete graph into:
  GRADIENT (transitive pecking order): grad_ij = h_i − h_j, where h_i = mean_j K_ij = (fragility_i − disrupt_i)/2  [net-dominance potential]
  CURL (cyclic, non-transitive): curl_ij = K_ij − grad_ij  [nonzero triangle circulation → genuine cycles]
transitivity_frac = ‖grad‖²/‖K‖² ; cyclic_frac = ‖curl‖²/‖K‖²  (sum to 1). A pure pecking order → curl=0.

## Split-half REALITY test (the decisive design)
Split each class's images into two disjoint halves A/B. Build F TWICE independently (heads, full-FT perturbers, victim eval all from the
SAME half). → F_a, F_b are independent noisy estimates. Reproducibility across halves separates SIGNAL from NOISE:
  ρ(F_a, F_b) offdiag      = overall matrix reproducibility (sanity)
  ρ(h_a, h_b)              = does the net-DOMINANCE ranking reproduce? (positive control — predator/prey identity stable?)
  ρ(curl_a, curl_b) offdiag = does the CYCLIC structure reproduce? (THE test — real cycles vs noise)

## PRE-REGISTERED PREDICTIONS
1b-P1 — CYCLIC FRACTION IS SUBSTANTIAL (measured). ‖curl‖²/‖K‖² ∈ [0.4, 0.6] on both datasets (replicating NEW-1's ~54%). This is the
    APPARENT cyclicity; P3 tests whether it is real.
1b-P2 — DOMINANCE IS REAL (positive control). Split-half ρ(h_a, h_b) ≥ +0.50 on both datasets — capabilities have a STABLE net-dominance
    identity (predator vs prey), reproducing across independent samples. FALSIFIED < +0.30 (even dominance is noise → whole asymmetry
    unreliable at this precision).
1b-P3 — IS THE CYCLE REAL? (decisive). Split-half ρ(curl_a, curl_b):
    ≥ +0.30 on both → GENUINE non-transitive interference: real capability CYCLES (rock-paper-scissors) exist beyond the pecking order.
    < +0.15 → the cyclic residual is NOISE: the asymmetry is a (reproducible) transitive DOMINANCE + measurement noise, NO real cycles.
    [Honest prior: given NEW-2's structureless interaction, I lean toward curl being mostly NOISE (P3 < 0.15) — but genuinely uncertain;
    a real curl would be the more striking finding.]
1b-P4 — DATASET-GENERAL. Whatever P2/P3 return, it is consistent across CIFAR + DTD (both give the same qualitative answer).

## Outcome grid (locked)
- P2 holds, P3 real (≥.30): NON-TRANSITIVE interference is REAL — capability rock-paper-scissors. Striking, novel structural finding;
  reframes interference as not globally orderable.
- P2 holds, P3 noise (<.15): the asymmetry = a STABLE TRANSITIVE DOMINANCE order (net predator/prey via fragility−disrupt) + noise; the
  "cyclic residual" is measurement noise, no real cycles. Clean, consistent with NEW-2; interference IS globally orderable (a pecking order).
- P2 fails: even net-dominance doesn't reproduce → asymmetry unreliable at this measurement precision (report; would need more images/seeds).

## Non-saturation
CIFAR + DTD non-saturated (1a''''-d medFgt ~0.23/0.27). Split-half uses 50 img/class per half (PER_CLASS 100) → noisier F per half; the
reproducibility test ACCOUNTS for this (noise won't reproduce). Require each half's median off-diag forget ∈ [0.03, 0.55] (looser upper
bound; flag if a half saturates). GTSRB excluded (saturated).

## Config (locked)
resnet18 IMAGENET1K_V1 224px, 16 caps (victim=perturber) + 10 neg per dataset, PER_CLASS 100 split 50/50 into halves A/B, FULL FT wd 1e-4
300/5e-4, seed 0. Metrics = Hodge fractions + split-half Spearman (F, h, curl). Single backbone seed; the split-half IS the noise control.
Local; DTD downloaded.
