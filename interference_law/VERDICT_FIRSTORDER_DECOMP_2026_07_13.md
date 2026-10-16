# VERDICT — NEW-1a''': FOUND the plastic-regime DIRECTIONAL LAW (PREREG 49633fa5, locked 2026-07-13). Landed, $0 local 98s, 40×8 = 320 pairs, non-saturated (median 0.171).

## RESULT: The first-order model's ρ+0.45 is carried ENTIRELY by ALIGNMENT, not magnitude. Directed forgetting ∝ −cos(g_i, Δθ_p): a perturbation forgets a capability to the extent its weight-displacement is anti-aligned with that capability's margin-Jacobian. A GENUINE per-pair directional interaction survives double-centering (ρ+0.30). This is the plastic-regime directional law that #6b (gradient cosine) and #7/#7b (feature projections) could not see — it lives in TINY weight-space cosine deviations (|cos|<0.015) from high-dim orthogonality.

## The decomposition (ρ with directed forget, 320 pairs)
  predictor                       ρ        reading
  fo_full = −(g_i·Δθ_p)         +0.448    reproduces 1a' (deterministic sanity ✓)
  align = −cos(g_i,Δθ_p)        +0.445    ← ALIGNMENT ALONE (magnitudes stripped) carries ALL of it. (P1 ✅ ≥+0.30)
  align_dcentered              +0.299    ← survives removing fragility(row)+disrupt(col) → GENUINE per-pair interaction. (P2 ✅ ≥+0.20)
  gmag_i broadcast             −0.297    victim magnitude ALONE wrong-signed (robust caps have big |g|). (P3 ✅)
  dmag_p broadcast             −0.186    perturber magnitude weakly wrong-signed
  mag_only (gmag·dmag)         −0.318    magnitudes-only model points the WRONG WAY entirely
  cos range across pairs       [−0.015, +0.014]   ← the signal lives in TINY deviations from orthogonality

## The law
forget(i←p) ∝ −cos(g_i, Δθ_p), where g_i = ∂margin_i/∂θ (victim's margin-preservation direction in weight space) and Δθ_p = the
perturber's accumulated weight-travel. A capability is forgotten when the perturbation moves θ AGAINST the direction that preserves its
decision margin. Magnitudes (‖g_i‖, ‖Δθ_p‖) give the WRONG sign; the pure alignment carries the prediction and a real interaction remains
(+0.30) after the fragility and disruptiveness main effects are removed.

## Why #6b and #7/#7b missed it (the resolution)
- #6b tested cos(g_A, g_B) = alignment of two TASK-LOSS gradients, and NORMALIZED it (cosine) → nulled (+0.02). Two differences here:
  (i) g_i is the MARGIN Jacobian ∂margin_i/∂θ (the readout-preservation direction), not a task-loss gradient; (ii) the partner is the
  ACCUMULATED displacement Δθ_p (where the perturber went), not an instantaneous gradient.
- #7/#7b projected FEATURE-space directions (d_A, w_A) → null/wrong-sign. The directional variable is in WEIGHT space, not feature space.
- The high-dim intuition ("everything is orthogonal") is TRUE in magnitude (|cos|<0.015) but the SMALL DEVIATIONS from orthogonality
  carry real, ranked signal (ρ+0.45). Normalized-cosine and large-range probes discard exactly this; the signed dot / rank keeps it.

## What this does to the arc's central tension
The frozen/plastic DICHOTOMY is REFINED, not overturned:
- FROZEN readout: forget ∝ −cos(d_A, d_B) — FEATURE-space direction, LARGE cosines. (4-op law.)
- PLASTIC backbone: forget(i←p) ≈ fragility(i) [victim main effect, ~margin/d′] + disruptiveness(p) [perturber main effect, ~‖Δθ‖] +
  −cos(g_i, Δθ_p) [directional INTERACTION, +0.30] — WEIGHT-space margin-Jacobian alignment, TINY cosines but real.
So BOTH regimes have a directional law; the plastic one is a weight-space Jacobian alignment invisible to the feature-space and
normalized-gradient-cosine probes that reported "direction dethroned." "Magnitude governs" (#1) captured the main effects; the directional
interaction was underneath, revealed by double-centering. The dichotomy was a measurement horizon, not an absence.

## Caveats (honest — this is a strong claim, keep it bounded)
Single seed; ResNet18; CIFAR-100; 40 victims / 8 perturbers (interaction dof (39)(7)=273). First-order/linear-response approx over a LARGE
step (‖Δθ‖≈12) — the +0.45 pooled / +0.30 interaction are a MEANINGFUL linear-response component, not a full nonlinear model. The
governing cosines are tiny (|cos|<0.015): the directional signal is real in RANK but small in magnitude — a seed-repeat + a larger
capability set would harden it (natural #1a''''). Jacobian over all backbone params (BN/early-layer unrestricted). Margin main-effect
(fragility, d′−0.61) and perturber main-effect still dominate the raw variance; the alignment is the INTERACTION on top, and it is genuine.

## Where this leaves the fragility line
"What is fragility" resolved pragmatically: fragility (victim main effect) ≈ margin/separation (d′ −0.61); the DIRECTIONAL part of
forgetting is the SEPARATE alignment interaction (this probe), not a victim property. The three-term model forget ≈ fragility ⊕
disruptiveness ⊕ alignment is the plastic-regime law. The sensitivity-as-fragility sub-question (fractional ‖g‖/margin) is now lower value —
the real directional variable was the pair alignment, not a per-victim sensitivity scalar. Arc-central tension RESOLVED.
