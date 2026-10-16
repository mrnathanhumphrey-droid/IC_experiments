# PREREG — NEW-3b: does FORWARD transfer exist? Aligned-B initialization for few-shot A. (the fair transfer test)

Author: arc (CF interference). Date: 2026-07-13. Status: PRE-REGISTERED, SHA256-lock before run. FROZEN resnet18 features, CIFAR + DTD. $0 local.

## The question NEW-3 left open
NEW-3 found NO backward transfer, but flagged the test as structurally biased (sequential head-retrain moves toward B, away from A). The
FAIR test of the phenomenon's positive half is FORWARD transfer: does starting from an aligned task-B head help you LEARN a data-limited
task A better than a cold start? Features are FROZEN (so B can only help via a better INIT/prior, not by changing features), A is FEW-SHOT
(so a prior can matter), and both inits get the SAME K training steps on A — isolating the init effect. If aligned B improves few-shot A
and anti-aligned B hurts, forward transfer exists and the signed cosine law is COMPLETE (forward=transfer, backward=forget).

## Design
Frozen resnet18 features (random-proj to headroom if needed). Per cap: task = class vs pooled-others; d = ddir; FULL train (for B heads),
FEW train (8 pos / 24 neg, for A few-shot), held-out test. For ordered pair (A=i, B=j):
  cold(i)    = AUC on i-test of a head trained from RANDOM init on A-few for K steps
  binit(i,j) = AUC on i-test of a head initialized from the B-FULL head hB[j], then trained on A-few for the SAME K steps
  FORWARD TRANSFER FT(i←j) = binit(i,j) − cold(i)      (FT>0 = B-init helped learn A; FT<0 = B-init hurt)
K moderate (init still matters, not converged). 3 seeds avg. Descriptor = cos(d_i, d_j).

## PRE-REGISTERED PREDICTIONS
3b-P1 — FORWARD TRANSFER EXISTS. With few-shot headroom (median cold AUC ∈ [0.60, 0.90]), ≥ 15% of ordered pairs show FT ≥ +0.01 (aligned-B
    init measurably improves few-shot A). FALSIFIED if < 5% (even forward transfer absent → phenomenon is genuinely one-sided).
3b-P2 — COSINE-GOVERNED. ρ(cos(d_i,d_j), FT) ≥ +0.30 — aligned B (high +cos) gives POSITIVE forward transfer, anti-aligned B (−cos) gives
    NEGATIVE init (worse than cold). The same cosine axis governs forward transfer as governs forgetting. FALSIFIED if ρ < +0.15.
3b-P3 — SIGNED / ZERO-CROSSING. The aligned end has POSITIVE mean FT (real transfer, not just "less bad"): the top cosine bin mean FT > 0.
    Report the FT-vs-cos curve + zero-crossing c₀. This is the transfer that the BACKWARD test (NEW-3, all-negative) could not exhibit.
3b-P4 — DATASET-GENERAL. 3b-P1 ∧ 3b-P2 on BOTH CIFAR and DTD.

## Outcome grid (locked)
- P1+P2+P3: FORWARD TRANSFER is REAL and cosine-governed → the POSITIVE half of the phenomenon exists in the forward direction; the signed
  cosine law is COMPLETE (aligned→forward transfer, anti→interference). Closes the transfer question: transfer lives forward, forgetting
  lives backward, one cosine axis governs both. Satisfying, honest close to the whole arc.
- P1 holds, P2 weak: forward transfer exists but is not cosine-governed (separate axis).
- P1 fails: NO forward transfer either → the phenomenon is genuinely ONE-SIDED even in the fair regime; frozen features + a prior task give
  no positive transfer. Also a clean, definitive close (the negative is real, not a design artifact).

## Non-saturation / headroom
Few-shot A (8 pos/24 neg) should give cold AUC in [0.60, 0.90] (room for a prior to help). If cold AUC > 0.92 (few-shot still saturates)
add random projection (D_PROJ) to lower it, per the NEW-3 headroom protocol. If < 0.55 (too noisy) ease the few-shot count. Read FT only in
the headroom band.

## Config (locked)
frozen resnet18 IMAGENET1K_V1 224px, victim=class vs pooled-others, D_PROJ configurable for headroom, few-shot 8 pos/24 neg, B-full head
45 pos/135 neg, K=30 A-few steps, 3 seeds, 20 caps per dataset (CIFAR + DTD), PER_CLASS 100. Metrics = FT distribution (transfer fraction) +
Spearman(cos, FT) + FT-vs-cos curve (zero-crossing). Local; DTD downloaded. Frozen readout regime.
