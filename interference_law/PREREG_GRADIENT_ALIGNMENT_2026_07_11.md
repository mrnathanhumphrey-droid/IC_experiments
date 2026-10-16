# PRE-REG — Thread #6b: is representational forgetting EVER cosine-governed? Full fine-tune + gradient-alignment (function-space/NTK) cosine. 2026-07-11.

SHA256-lock before running. Thread #6 (LOCK 1bab16d2) found the frozen-FEATURE cosine barely predicts representational forgetting
(ρ≈−0.12) and the Δθ/Fisher cosines nulled BUT were CONFOUNDED (LoRA accumulated-update near-orthogonal in 650k-dim → no dynamic
range). This resolves the confound two ways at once: (1) FULL fine-tune (no low-rank constraint), and (2) the principled
function-space predictor — the t=0 loss-GRADIENT alignment cos(∇L_A, ∇L_B) (GEM/PCGrad/GPM's quantity), which is STRUCTURED (gradients
concentrate in task-relevant dirs) so it has real dynamic range where accumulated Δθ did not.

## Setup (removes the LoRA confound)
CIFAR-100 + ResNet18 (ImageNet-pretrained). **FULL backbone fine-tune** on each B-task (all params trainable; no LoRA). Forgetting =
frozen-re-probe drop (linear A-probe on features before vs after full-FT on B). Same 8 A-classes × 15 B-tasks subset + the same
efficient design (one B full-FT → re-probe all A). Non-saturation gate FIRST (full-FT can wipe everything): tune LR/steps so median
forget ∈ [0.05,0.40].

## Predictors (all at t=0 / base model, PRE-B → predictive, non-circular)
1. **GRADIENT-ALIGNMENT (the new clean one)** cos(g_A, g_B), g_A = ∇_backbone L_A(θ_base, head_A*) with head_A* the CONVERGED A-probe
   held fixed = the direction A-training would follow; g_B likewise (B 5-way head). FULL param space (~11M). The first-order/NTK
   interference measure. Also report an NTK/Fisher-normalized variant if the raw cosine lacks range.
2. **FEATURE** cos(d_A, d_B) in original φ — carry-over (does it predict FULL-FT forgetting? #6 said no for LoRA).
3. **Δθ (control)** cos(Δθ_A, Δθ_B) full-param accumulated update — expected STILL ~0 (confirms the #6 confound was update-noise, not
   the framework; the contrast with gradient-cos being ranged is the diagnostic).

## Pre-registered read (locked before running)
- **LAW REVIVES (function-space):** gradient-alignment ρ(cos, forget) < −0.4 with dynamic range (cos spans a real interval, not all
  ~0), and it PARTIALS-OUT per-B mean (the within-B A–B alignment still predicts after removing B-disruptiveness). → representational
  forgetting IS cosine-governed, in GRADIENT/FUNCTION space (not feature space). The 4-op law extends to the representation via the
  right cosine.
- **HARD BOUNDARY (clean null):** gradient-alignment has real dynamic range BUT ρ≈0 (and feature≈0, Δθ≈0) → representational forgetting
  is genuinely NOT direction-governed; it is dominated by B-disruptiveness + broad reorganization. The cosine law is definitively
  readout-specific. (Distinguished from #6's confounded null BY the gradient cosine having range this time.)
- **DEGENERATE (report, escalate):** if cos(g_A,g_B) is itself ~0 for ALL pairs (gradients orthogonal even though structured) → the
  full-param raw cosine is the wrong measure → try NTK-normalized / last-block-only gradient, report the degeneracy honestly.
- **CONTROL:** Δθ full-param expected ~0 (no range); if Δθ ALSO had range and predicted, that would revise the #6 interpretation.

## Confound controls (locked)
1. Saturation (4th+ time): non-sat gate to median forget ∈ [0.05,0.40] BEFORE the run.
2. Structured-gradient range: report the cos(g_A,g_B) distribution; a degenerate ~0-everywhere invalidates the predictor (escalate).
3. Trained-probe dependence: g computed through the CONVERGED task probe, held fixed (well-defined improvement direction).
4. B-disruptiveness dominates (#6 finding): report BOTH raw ρ and the per-B-demeaned (within-B) ρ — the within-B is the clean test of
   A–B alignment controlling for how disruptive B is.
5. Seeds: ≥3 seeds on the headline ρ (Tier-1 lesson).
6. Circularity: gradients + feature dirs at t=0; forgetting post-B.

## Compute / ops
Full-FT ResNet18 × 15 B-tasks + 8×15 gradient cosines + re-probes. Local-first ($0): full-FT is heavier than LoRA but ResNet18 is
11M — est ~20–40 min. Lambda a100 only if it overflows (cost-confirm first). `probe_gradient_alignment.py` (reuse #6 data/re-probe +
full-FT + full-param gradient extraction). py_compile + gate before the full run. Output `out/gradient_alignment.json`.

## Status
PREREG written. Next: SHA256-lock → build → non-sat gate (full-FT strength) → confirm cos(g_A,g_B) has range → full run → read
(gradient-cos ρ raw + within-B, vs feature, vs Δθ) → verdict (revive vs hard boundary) + bank.
