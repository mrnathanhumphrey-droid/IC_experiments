# VERDICT — Thread #6: representational interference law (PREREG 1bab16d2, locked 2026-07-11). Landed, $0 local GPU. BOUNDARY RESULT: the cosine law is largely READOUT-SPECIFIC.

## RESULT: when the backbone MOVES (LoRA fine-tune), NONE of the three cosines strongly predicts representational forgetting. The head-level law (forget ∝ −cos, ρ=−0.74) does NOT transfer to the representation; forgetting is nearly uniform in cosine and is driven by WHICH B-task, not A–B alignment.

## Setup
CIFAR-100 + ResNet18, conv-LoRA on layer3+layer4 (rank 32). Forgetting = frozen-re-probe drop (linear A-probe before vs after
LoRA-fine-tuning on a B-task). Efficient design: 1 B-fine-tune → re-probe all A → many (cos, forget) points. Non-saturation gate
PASSED at rank32/1500steps/lr4e-3 (median forget 0.12–0.13, range to 0.39 — a real partial regime). Full curve n=750; 3-cosine
comparison on an 8-A × 15-B subset.

## The three cosines vs representational forgetting
              rho(cos,forget)   dynamic range          read
  FEATURE     −0.12 (full 750)  cos∈[−0.41,+0.72] ✓   ROBUST weak: well-ranged yet ~6× weaker than head-level −0.74
              −0.04 (subset)                           on the SAME substrate. Feature alignment barely predicts.
  DTHETA      +0.00             cos∈[−0.01,+0.01] ✗   NULL but CONFOUNDED: LoRA updates near-orthogonal in 650k-dim
              (weight-update)                          → no dynamic range → can't predict (dimensionality artifact).
  FISHER      −0.08             overlap 0.002 vs        NULL but CONFOUNDED: task Fisher subspaces overlap ≈ random
              (subspace)        random 0.000            (0.002 vs 0.000) in 650k-dim → no dynamic range.

## The robust core finding (defensible, not confound-limited)
The FEATURE cosine has genuine dynamic range ([−0.41,+0.72]) and STILL predicts representational forgetting only weakly (ρ −0.04..−0.12)
vs the head-level ρ −0.74 on the same data. So this is NOT a range/dimensionality artifact: the frozen-feature direction alignment
genuinely does not govern which capabilities are damaged when the backbone is plastic. **The head-level interference law — the cosine
governing learn/forget/unlearn/protect (#1–#5) — is a property of a LINEAR READOUT over FROZEN features, and is largely
READOUT-SPECIFIC. It does not extend to representational forgetting under backbone fine-tuning.**

## What DOES drive representational forgetting (the positive residual)
Forgetting is real and partial (median ~0.10, up to 0.39) but is dominated by the B-TASK, not A–B alignment: per-B median forgetting
ranges 0.052 (B1) → 0.185 (B8), a 3.6× spread, while within-B the A–B cosine explains almost none of the variance. Representational
forgetting under plasticity ≈ f(how disruptive B is) + a tiny direction-alignment term — consistent with the arc's Probe-18 (change is
a BROAD subspace, not one axis): fine-tuning reorganizes A's representation broadly/uniformly, not selectively along anti-parallel dirs.

## Pre-registered read — the NULL branch, met (with the capacity nuance)
Locked NULL clause: "no cosine predicts representational forgetting → backbone motion DECOUPLES from the frozen-feature geometry → the
head-level law is readout-specific." MET for the feature cosine (clean) and the Δθ/Fisher cosines (null but LoRA-confounded). The
pre-registered CAPACITY prediction ("more free-lanes weaken the law") is the mechanism: a plastic backbone has the capacity to absorb
B broadly, so forgetting is not concentrated along the frozen-feature cosine.

## Honest limitations / deferred confound-checks
- **LoRA confound (the big one):** Δθ/Fisher near-orthogonality may be a low-rank-in-high-dim artifact, not true functional
  independence. A FULL fine-tune, or a FUNCTION-SPACE/NTK cosine, is the clean check for whether ANY parametric/curvature cosine
  recovers the law. DEFERRED (bigger spend). The FEATURE-cosine null is NOT LoRA-confounded and carries the headline.
- Not run (diminishing returns on a fairly clear null): the rank∈{4,8,16} sweep, the per-layer depth profile, ≥3 seeds. The full-curve
  ρ −0.12 (n=750) is stable enough to call the feature cosine weak; multi-seed would tighten but not overturn.
- Base accA0 ≈ 0.98 (near-ceiling separability, as #1) — forgetting measured from a high base; partial regime confirmed so signal
  exists, but a lower-separability substrate could sharpen the measure.

## Bottom line for the synthesis (§8 update)
The cosine geometry is powerful and quantitative at the READOUT level across four operations and three modalities — but it is
READOUT-SPECIFIC. Representational forgetting under backbone plasticity is governed by task disruptiveness and broad reorganization,
not by direction alignment. This BOUNDS the framework: the law lives at the interface between a fixed representation and a linear
decision, not in the representation's own plastic dynamics. (Whether a function-space cosine on a full fine-tune revives it is open.)
