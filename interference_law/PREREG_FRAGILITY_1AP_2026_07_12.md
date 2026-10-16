# PREREG — NEW-1a': fragility = SENSITIVITY? The real instrument (Jacobian ‖∂margin/∂θ‖), + a first-order model of the matrix.

Author: arc (CF interference). Date: 2026-07-12. Status: PRE-REGISTERED, SHA256-lock before run. Substrate: CIFAR-100 / ResNet18,
FULL fine-tune, deployed frozen-head metric. $0 local. Reuses probe_fragility victims/perturbers.

## Why (1a left this open with a dead instrument)
NEW-1a found MARGIN PROTECTS (ρ(d′,fragility)=−0.51) but could NOT test the lead hypothesis — fragility = local SENSITIVITY of the
readout to backbone movement — because the random-perturbation instrument was degenerate (ε too small + isotropic-random barely moves
features → mean|Δmargin|/margin=0.002=noise, ρ−0.13 uninformative). 1a' brings the RIGHT instrument: the exact first-order sensitivity,
the Jacobian norm ‖∂margin_i/∂θ‖ (intrinsic, base-measured, non-circular, no ε). And tests whether a first-order LINEAR model g_i·Δθ_p
explains the directed forgetting matrix.

## Definitions (victim i, perturber p; victims 0..39, perturbers [50,55,60,65,70,75,80,85] vs neg, NEG 90-99 — SAME as 1a)
- margin_i(θ) = mean_x∈i head_i(f_θ(x)) − mean_x∈neg head_i(f_θ(x))   (head_i FROZEN at base)
- g_i = ∂margin_i/∂θ |_{θ0}   (autograd through the backbone, head fixed) — the intrinsic sensitivity vector of capability i
- sens_jac_i = ‖g_i‖   ← the REAL sensitivity instrument (replaces 1a's dead random-perturbation sens)
- Δθ_p = θ_p − θ0   (weight displacement from full-FT on perturber p)
- firstorder(i,p) = −(g_i · Δθ_p)   (first-order predicted margin DROP; positive = predicted forgetting)
- fragile_i = mean_p forget(i←p)   (target, as in 1a; d′_i = base discriminability, carried for the partial-correlation control)

## PRE-REGISTERED PREDICTIONS
1a'-P1 — SENSITIVITY IS THE MECHANISM. ρ(sens_jac_i, fragile_i) ≥ +0.50. The intrinsic margin-Jacobian norm predicts which capabilities
    forget → fragility = how sensitive a readout's margin is to backbone movement. FALSIFIED if < +0.30 → fragility is NOT margin-Jacobian
    sensitivity (then MARGIN/separation, not sensitivity, is the story, and 1a's d′−0.51 is the mechanism).
1a'-P2 — THE INSTRUMENT IS FIXED (vs 1a). sens_jac has real dynamic range (max/min ≥ 3) and |ρ(sens_jac, fragile)| substantially exceeds
    1a's dead isotropic sens (|ρ| 0.13): |ρ(sens_jac)| − 0.13 ≥ +0.20. Confirms the null in 1a was the instrument, not the hypothesis.
1a'-P3 — FIRST-ORDER MODEL OF THE MATRIX. firstorder(i,p) = −(g_i·Δθ_p) predicts actual forget(i←p) across all 40×8=320 (i,p) pairs:
    ρ ≥ +0.40. A first-order (linear-response) model of directed forgetting from base Jacobians + perturbation displacements. [Ambitious —
    Δθ_p is a large step (‖Δθ‖≈12), so the linear approx may under-predict; registered modestly. NULL if < +0.20 → forgetting is
    intrinsically nonlinear / not first-order-predictable from base.]
1a'-P4 — SENSITIVITY vs MARGIN: distinct or the same? Partial Spearman ρ(sens_jac, fragile | d′). If ≥ +0.30, sensitivity is a DISTINCT
    (deeper) mechanism beyond margin. If ≈ 0, sensitivity and margin are ONE story (high margin = low sensitivity). Also report ρ(sens_jac,
    d′) directly (expect negative: well-separated → less sensitive).

## Outcome grid (locked)
- 1a'-P1 + P4-distinct: fragility = margin-Jacobian SENSITIVITY, a mechanism deeper than raw separation — the clean mechanistic answer,
  with a cheap base-measurable estimator, and (if P3) a first-order predictive model of the whole matrix.
- 1a'-P1 holds but P4≈0: sensitivity and margin are the SAME variable (well-separated capabilities are intrinsically less sensitive) —
  fragility has ONE mechanism, viewable as either margin or sensitivity. Clean.
- 1a'-P1 fails: fragility is NOT first-order sensitivity → it is SEPARATION/structure (1a's margin) and/or higher-order — the sensitivity
  hypothesis is then properly refuted (good instrument), closing the loop 1a opened.

## Non-saturation / cost
fragile_i reused (1a median 0.184, partial band). g_i via autograd on 128 victim + 128 neg samples (head fixed). Store 8 Δθ_p vectors
(~358MB CPU); compute g_i one at a time, dot with the 8 Δθ_p, discard. Jacobian norm over ALL backbone params (total sensitivity; BN/early-
layer contribution noted as a caveat, not restricted). Single seed.

## Config (locked)
build_full ResNet18 IMAGENET1K_V1 fc=Identity, FULL FT, perturber FT 300/5e-4. 40 victims, 8 perturbers, seed 0. Metrics = Spearman +
partial Spearman (control d′ via rank-residualization) + the 320-pair first-order correlation. No production system, no network, pure local CIFAR.
