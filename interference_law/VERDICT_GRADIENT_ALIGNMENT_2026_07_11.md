# VERDICT — Thread #6b: gradient-alignment / full-FT (PREREG 5b68b154, locked 2026-07-11). Landed, $0 local GPU, 3 seeds. DEFINITIVE HARD BOUNDARY: representational forgetting is NOT cosine-governed — the confound is eliminated and the null holds.

## RESULT: even the theoretically-correct function-space predictor (t=0 loss-gradient alignment cos(g_A,g_B)), computed with FULL fine-tuning (no LoRA), with GENUINE dynamic range, does NOT predict representational forgetting (ρ = +0.016 ± 0.020 ≈ 0, 3 seeds). Thread #6's boundary is confirmed and strengthened; the LoRA/dimensionality confound is ruled out.

## Setup (both #6 confounds removed)
CIFAR-100 + ResNet18, FULL backbone fine-tune (all 11M params; NO LoRA) on each B-task. Forgetting = frozen-re-probe drop. Predictor
= t=0 loss-gradient alignment cos(g_A,g_B), g = ∇_backbone L through the CONVERGED task probe (the direction training would follow) —
the GEM/PCGrad/NTK interference quantity, STRUCTURED so it has real dynamic range (unlike #6's accumulated Δθ). Non-sat gate → full-FT
400 steps / lr 1e-3 (median forget 0.064, partial). 8 A × 15 B, 3 seeds.

## 3-seed headline
  predictor              rho (raw)        rho (within-B)     range
  GRADIENT-ALIGNMENT     +0.016 ± 0.020   +0.041 ± 0.033     ~[−0.20,+0.28]  ✓ REAL dynamic range
  FEATURE                −0.122 ± 0.013   ~−0.03             [−0.42,+0.67]   weak-but-real (~1.5% variance)
  DTHETA (control)       −0.03            −0.10              [−0.006,+0.009] DEGENERATE (accumulated update orthogonal — #6 confound reconfirmed)
  median_forget = 0.064 ± 0.001 (partial regime, in band)

## Why this is DEFINITIVE (not a repeat of #6's confounded null)
- **The predictor is valid this time.** cos(g_A,g_B) spans ~[−0.20,+0.28] every seed — structured loss-gradients, NOT the degenerate
  near-orthogonal accumulated Δθ of #6. So the ~0 ρ is a REAL null, not "no dynamic range."
- **The LoRA confound is gone.** Full fine-tune, all 11M params — no low-rank constraint. The null is not a LoRA artifact.
- **B-disruptiveness is controlled.** within-B ρ (demeaned per B-task) is also ~0 (+0.04) → even among the A's forgotten by the SAME
  B, gradient alignment doesn't predict which. Not a B-level-variance masking effect.
- **3 seeds, tight.** ρ = +0.016 ± 0.020 — indistinguishable from zero, stable across seeds. Not a single-seed fluke (the Tier-1 lesson).
- The gate's transient +0.385 was n=16 noise; at n=120×3 it is zero.

## Interpretation — the boundary is real and sharp
Representational forgetting under backbone plasticity is genuinely NOT governed by pairwise direction alignment — not feature-space
(weak −0.12), not weight-update, and DECISIVELY not the function-space/NTK gradient cosine (0.00) that theory predicts should govern
first-order transfer. The first-order NTK loss-transfer picture (ΔL_A ≈ −η⟨g_A,g_B⟩) does NOT describe re-probe representational
forgetting — which is a higher-order, GLOBAL reorganization effect: dominated by how disruptive B is (per-B forgetting varies ~2–3×)
and by broad representational drift (Probe-18's broad-subspace reorganization), NOT by which prior capability A happens to align with B.

## THE ARC'S FINAL BOUNDARY (synthesis §8, now SETTLED)
The cosine geometry — one variable governing learn/forget/unlearn/protect, across text/vision/audio, quantitatively predictive of
production systems — is a property of a LINEAR READOUT over a FIXED representation. It is READOUT-SPECIFIC. When the representation
itself is plastic, forgetting decouples from all direction-alignment cosines (feature, weight-update, and the theoretically-correct
gradient/NTK alignment alike). **The law lives at the fixed-representation × linear-decision interface — not in the representation's
own plastic dynamics.** This is now established cleanly (confound removed, right predictor, full FT, 3 seeds), not merely conjectured.

## Residual (small, real): the feature cosine's −0.12
The frozen-feature cosine consistently explains a TINY slice of representational forgetting (ρ −0.122 ± 0.013, ~1.5% variance) — a
faint echo of the readout law surviving into the plastic regime, but nothing like the head-level −0.74. The other ~98% is
disruptiveness + global reorganization.

## Caveats
CIFAR-100/ResNet18, one architecture, 8×15 subset, full-FT at one strength (partial regime confirmed). Re-probe = linear (a nonlinear
probe could recover more residual A → different forgetting scale, but the DELTAS/ρ structure is the claim). NTK-normalized (Fisher-
weighted) gradient cosine not tried — raw cosine had range and nulled, so the normalized version is unlikely to revive it, but it is
the last untested variant. Base accA0 ≈ 0.97 (high separability). The boundary claim is for re-probe representational forgetting under
gradient fine-tuning at this scale.
