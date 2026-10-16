# VERDICT — NEW-1a': fragility = sensitivity? Jacobian instrument + first-order model (PREREG 6f5c2aa5, locked 2026-07-12). Landed, $0 local 87s, 40 victims × 8 perturbers, non-saturated (median fragility 0.179).

## RESULT: Mixed but PRODUCTIVE. The sensitivity instrument is now ALIVE (P2 ✅) but the RAW Jacobian norm predicts fragility with the WRONG SIGN (ρ−0.57, P1 ✗) — it is confounded by margin SCALE. The GEM: a first-order directed model −(g_i·Δθ_p) predicts the asymmetric forgetting matrix at ρ=+0.45 (P3 ✅). Margin/d′ robustly protects (−0.61). Fragility-as-sensitivity remains scale-tangled (fractional instrument untested); the matrix's linear-response structure is the real find.

## The numbers
  1a'-P1 sensitivity      ρ(sens_jac, fragility) = −0.573                 ✗ WRONG SIGN (predicted ≥+0.50). Scale-confounded (below).
  1a'-P2 instrument fixed ‖g‖ range max/min = 3.6 ; |ρ| 0.57 vs dead 0.13 ✅ instrument ALIVE (vs 1a's degenerate 0.002 / ρ0.13).
  1a'-P3 first-order model ρ(−(g_i·Δθ_p), forget(i←p)) = +0.448 (pear +0.415), 320 pairs   ✅ CONFIRMED (≥+0.40). The gem.
  1a'-P4 sens vs margin   ρ(sens_jac,d′)=+0.592 ; partial(sens_jac,frag|d′)=−0.328 ; d′ alone −0.614   entangled, same direction.

## Why P1 has the wrong sign (the scale confound)
sens_jac_i = ‖∂margin_i/∂θ‖ is the ABSOLUTE margin-gradient. But a capability with a big margin naturally has a big absolute gradient
(∂/∂θ of a larger quantity). ρ(sens_jac, d′) = +0.59 confirms it: high-Jacobian capabilities are the high-margin, WELL-SEPARATED, ROBUST
ones. So raw ‖g‖ tracks robustness, not fragility. What governs forgetting is FRACTIONAL sensitivity ‖∂margin/∂θ‖ / margin (relative
change), which I did NOT register. So — for the THIRD time on this sub-question — the sensitivity hypothesis is not cleanly tested: 1a's
ε was too small (dead), 1a''s raw norm is scale-confounded (wrong sign). The fractional instrument ‖g‖/margin is swing #3 (a new prereg).

## The gem: the directed matrix has FIRST-ORDER (linear-response) structure
−(g_i·Δθ_p) — the victim's margin-Jacobian dotted with the perturber's weight displacement — predicts actual forget(i←p) at ρ=+0.45
across 320 pairs. This is a real mechanistic model of the ASYMMETRIC matrix (it naturally gives g_i·Δθ_p ≠ g_j·Δθ_i). It matters
because it recovers DIRECTIONAL structure in plastic forgetting that earlier probes missed:
  - #6b: gradient COSINE cos(g_A,g_B) nulled (ρ+0.02) — normalized away the magnitudes.
  - #7/#7b: STATIC feature-space projections (d_A, w_A) nulled / wrong-signed.
  - 1a'-P3: the first-order WEIGHT-space term g_i·Δθ_p (direction × magnitude, true Jacobian × true displacement) WORKS (+0.45).
So plastic directed forgetting IS first-order predictable — but only with the true margin-Jacobian and the true displacement, combining
alignment and magnitude (not a cosine, not a static direction). A clean follow-up: DECOMPOSE P3's signal into ‖g_i‖ (wrong-signed alone),
‖Δθ_p‖ (≈const), and the alignment cos(g_i,Δθ_p) — which factor carries the +0.45?

## What stands (honest)
- MARGIN PROTECTS, replicated: d′ ρ−0.61 (1a gave −0.51). Robust capabilities are well-separated.
- The margin-Jacobian instrument is ALIVE (range 3.6×) — but raw ‖g‖ measures margin-scale-robustness, not fragility (wrong sign).
- FIRST-ORDER DIRECTED MODEL works: forget(i←p) ≈ −g_i·Δθ_p, ρ+0.45 — the asymmetric matrix has linear-response structure. ★ new.
- "Fragility = sensitivity" is STILL open (fractional instrument untested). Two instrument failures suggest DIMINISHING RETURNS on this
  exact sub-question; the yield migrated to the first-order model.

## Follow-ups (NEW preregs if pursued — NOT done)
1a'' : fractional sensitivity ‖∂margin/∂θ‖/margin as the fragility instrument (swing #3) — OR accept margin/d′ as the answer and stop.
1a''': DECOMPOSE the first-order model's +0.45 into magnitude vs alignment (which factor predicts directed forgetting?) — the richer thread.

## Caveats
Single seed; ResNet18; 40 victims / 8 perturbers; Jacobian over ALL backbone params (BN/early-layer contribution unrestricted — could be
the scale-confound's source, note). First-order over a large step (‖Δθ‖≈12) → +0.45 is a moderate linear approx to a nonlinear process;
honest as "meaningful linear-response component," not a full model. P1's wrong sign is a scale artifact, named. The two robust results
(margin protects; first-order model works) carry; the sensitivity mechanism is the open, twice-mis-instrumented sub-question.
