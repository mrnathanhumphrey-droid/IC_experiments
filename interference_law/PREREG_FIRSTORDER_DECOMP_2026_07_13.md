# PREREG — NEW-1a''': decompose the first-order model. Is ALIGNMENT the plastic-regime directional variable #6b/#7 missed?

Author: arc (CF interference). Date: 2026-07-13. Status: PRE-REGISTERED, SHA256-lock before run. Substrate: CIFAR-100 / ResNet18,
FULL fine-tune, deployed frozen-head metric. $0 local. Reuses probe_fragility_jac pipeline (recompute g_i, Δθ_p, forget — seed 0 deterministic).

## The question
1a' found the first-order term −(g_i·Δθ_p) predicts directed forgetting at ρ+0.45 — recovering directional structure that #6b (gradient
COSINE, null +0.02) and #7/#7b (static projections, null/wrong-sign) missed. The term factorizes:
    −(g_i·Δθ_p) = −‖g_i‖ · ‖Δθ_p‖ · cos(g_i, Δθ_p)
where ‖g_i‖ = victim margin-Jacobian norm (per-victim; ALONE wrong-signed with fragility, ρ−0.57), ‖Δθ_p‖ = perturber displacement
(per-perturber, ≈const 12–14), cos(g_i,Δθ_p) = ALIGNMENT of the perturbation with the victim's margin-gradient (per-PAIR, directional).
Which factor carries the +0.45? If ALIGNMENT does — and survives removing row(fragility)+col(disruptiveness) structure — then we have
FOUND the plastic-regime directional law: a perturbation forgets a capability to the extent it moves θ AGAINST that capability's margin-gradient.

## Definitions (victim i∈0..39, perturber p∈[50,55,60,65,70,75,80,85]; 320 pairs)
- forget(i,p)  = AUC_i(base) − AUC_i(after full-FT on p)     the directed forgetting matrix
- g_i = ∂margin_i/∂θ|θ0 ; gmag_i = ‖g_i‖ ; dmag_p = ‖Δθ_p‖ ; Δθ_p = θ_p − θ0
- align(i,p) = −cos(g_i, Δθ_p)   signed alignment (positive = perturbation reduces margin_i = predicted forgetting)
- fo(i,p) = −(g_i·Δθ_p) = gmag_i·dmag_p·align(i,p)   the full first-order term
- double-centering X_dc[i,p] = X[i,p] − rowmean_i − colmean_p + grandmean   (removes victim + perturber main effects → pure interaction)

## PRE-REGISTERED PREDICTIONS
1a'''-P1 — ALIGNMENT PREDICTS (pooled). ρ(align, forget) ≥ +0.30 across 320 pairs. Perturbations anti-aligned with a victim's margin-
    gradient cause more forgetting — a directional signal, magnitudes stripped. FALSIFIED < +0.15.
1a'''-P2 — GENUINE DIRECTIONAL INTERACTION (the #6b/#7 quarry — THE test). After double-centering BOTH forget and align (removing victim-
    fragility rows AND perturber-disruptiveness cols), the residual interaction correlation ρ(align_dc, forget_dc) ≥ +0.20. → alignment
    carries PAIR-SPECIFIC directional structure, not merely row×col magnitude. This is the plastic-regime directional law, if it holds.
    FALSIFIED < +0.10 → the +0.45 is mostly fragility×disruptiveness (magnitude) structure; no genuine per-pair directional law.
1a'''-P3 — ALIGNMENT SUPPLIES THE SIGN MAGNITUDE LACKS. ρ(gmag_i broadcast, forget) < 0 (raw victim magnitude is wrong-signed, replicating
    1a') WHILE ρ(align, forget) > 0 → alignment, not magnitude, provides the correct predictive direction; the full model's +0.45 CANNOT be
    a magnitude artifact. Report the full decomposition table: ρ(forget, ·) for {fo, align, gmag_broadcast, dmag_broadcast, align_dc}.

## Outcome grid (locked)
- P1 + P2: FOUND IT — the plastic-regime directional variable is ALIGNMENT of the perturbation with the victim's margin-Jacobian. The
  directional law #6b/#7 hunted exists, at the weight-Jacobian level, as a genuine per-pair interaction. Resolves the arc's central tension
  (frozen=direction, plastic=magnitude) into: plastic ALSO has a directional term — Jacobian-alignment — that static/cosine probes couldn't see.
- P2 fails (interaction null): the first-order +0.45 is fragility×disruptiveness (row×col magnitude) structure; NO per-pair directional law
  → confirms/ tightens the dichotomy (plastic forgetting is magnitude-organized even at first order). Also a clean, publishable resolution.
- P1 fails: alignment carries nothing → the +0.45 came from magnitude/row-col; first-order model is not directional. (Unlikely given
  gmag alone is wrong-signed yet fo is +0.45 — but registered.)

## Non-saturation / cost
forget reused (median ~0.18, partial band). Recompute g_i (autograd, 128+128 samples) + Δθ_p (8 full-FT) + forget — deterministic seed 0,
first-order +0.45 must reproduce as a sanity check. Store 8 Δθ_p (~358MB CPU); g_i one at a time. Jacobian over all backbone params.

## Config (locked)
build_full ResNet18 IMAGENET1K_V1 fc=Identity, FULL FT 300/5e-4, 40 victims, 8 perturbers, NJ=128, seed 0. Metrics = Spearman (pooled +
double-centered interaction) + the decomposition table. Single seed. No production system, no network, pure local CIFAR.
