# PRE-REG — Representational interference law: does the SAME cosine govern forgetting when the BACKBONE moves (not just a frozen-feature readout)? 2026-07-11.

SHA256-lock before any run. Head-level result (#1): cosine between discriminative directions in a FROZEN backbone predicts readout
forgetting (forget ≈ −0.445·cos+0.266, held-out R²≈0.5). Open question (§8 of the synthesis): when you FINE-TUNE THE BACKBONE on B,
does representational forgetting of A obey the same geometry? The feature space itself moves, so d_A/d_B are moving targets — this
tests whether the law is a readout artifact or a property of the representation. User-chosen: CIFAR-100 + ResNet18 LoRA, all three
candidate cosines.

## Substrate & efficient design
- CIFAR-100, ResNet18 (ImageNet-pretrained). Positives per class vs a SHARED negative pool (10 reserved classes). Same 50 held-in
  "A" classes as #1 → the representational curve is directly comparable to the head-level curve on the SAME data.
- **Backbone motion = LoRA** on the last two residual blocks (layer3+layer4), rank r=8 (conv-LoRA parallel low-rank path; earlier
  layers frozen). Bounds the update to a subspace and keeps each fine-tune cheap.
- **EFFICIENT PAIR GENERATION (the cost fix):** pick K=15 "B-tasks" (each = a held-out class-group binary/small-multiclass
  interference task). For each B: fine-tune the LoRA backbone on B ONCE; then re-probe ALL 50 held-in A classes before vs after.
  cos(d_A,d_B) varies across A → ~50 (cosine, forgetting) points per single fine-tune → ~750 points for ~15 backbone fine-tunes.

## Forgetting measure (isolates REPRESENTATION, not readout)
FROZEN-RE-PROBE DROP. For each A: train a linear probe (A-vs-shared-neg) on the ORIGINAL φ → acc_A0. After LoRA-fine-tuning the
backbone on B, train a FRESH linear probe on φ_B → acc_A1. **representational_forget(A|B) = acc_A0 − acc_A1.** Re-probing controls
for readout drift → measures whether the REPRESENTATION of A degraded. (Linear probe = standard; a too-powerful probe would recover
residual A info and under-estimate forgetting — fixed at linear, flagged.)

## The three candidate cosines (all measured at t=0, PRE-B → predictive, non-circular)
1. **FEATURE** cos(d_A, d_B) in the ORIGINAL φ — exactly #1's head-level cosine (free; the direct "does the frozen-feature geometry
   still predict?" test). FULL curve (all ~750 points).
2. **WEIGHT-UPDATE** cos(Δθ_A, Δθ_B) — realized LoRA deltas from fine-tuning A alone vs B alone (measure the REALIZED update, not the
   raw gradient, to absorb Adam scaling). SUBSET (~120 pairs: 8 A-classes × 15 B-tasks, A-deltas fine-tuned once each).
3. **FISHER-SUBSPACE overlap** — principal-angle overlap between task-A and task-B top-K Fisher subspaces (reuse the base arc's
   inV/inSwap machinery). SUBSET (same ~120 pairs).

## Pre-registered read (locked before running)
- **LAW HOLDS at the representation level:** representational_forget is monotone-decreasing in cosine (Spearman ρ < −0.4) for at
  least one cosine, with held-out R² > 0.20 (fit on half the points, predict the other half), SAME sign as the head-level law.
- **WHICH cosine wins** (a headline in itself): rank the three by held-out predictive R². Prediction — FEATURE and FISHER track
  forgetting; WEIGHT-UPDATE is weaker (basis-dependent). If functional cosines win over parametric → "representational interference
  is functional, not parametric" (bears on whether GPM-in-weight-space is the right tool).
- **HEAD-vs-REPRESENTATION comparison:** overlay this curve on #1's head-level curve (same substrate). Same slope sign = the geometry
  transfers across the frozen/plastic boundary; a shallower slope = capacity dilutes but doesn't erase it.
- **CAPACITY confound (explicit control):** sweep LoRA rank r∈{4,8,16} on one B-task. Prediction — larger r (more "free lanes")
  WEAKENS the cosine→forget relationship. Report the trend; a weaker law at high r is capacity-dependence, NOT the law being false.
- **NULL:** no cosine predicts representational forgetting (all ρ≈0 / R²<0.05) → backbone motion DECOUPLES from the frozen-feature
  geometry → the head-level law is readout-specific. Report honestly — an important boundary either way.

## Confound controls (locked)
1. **Circularity:** ALL cosines measured at t=0 (pre-B); forgetting measured post-B. No post-hoc cosine.
2. **Saturation (4th time — guard up front):** sweep B-fine-tune strength (LR/steps) so MEDIAN representational_forget ∈ [0.05,0.40]
   (partial). If no setting yields a partial band (backbone won't move / floors A), report as a design-limit and interpret.
3. **Capacity:** the r∈{4,8,16} sweep (above).
4. **Layer dependence:** re-probe at TWO depths (penultimate avgpool AND layer3 output) → a forgetting-by-depth readout, not a single
   number.
5. **Optimizer scaling:** cosine (2) uses the REALIZED Δθ, not raw gradients.
6. **Seeds:** ≥3 seeds on the headline curve (Tier-1 taught us single-seed lies); report mean±SE.
7. **Difficulty:** report base-acc_A0 distribution; keep it off the ceiling (reuse #1's non-saturation finding on this substrate).

## Compute / ops
Local-first ($0, my GPU): ~15 B-task LoRA fine-tunes + re-probes + the Δθ/Fisher subset + r-sweep ≈ 1–2h. Lambda a100 only if it
overflows (cost-confirm first). `probe_representational_law.py` (reuse #1's feature/probe machinery + a conv-LoRA wrapper on
ResNet18 + the base arc's Fisher-subspace overlap). py_compile + --smoke (2 B-tasks) before full. Output
`out/representational_law.json` (per-cosine curves, ρ, held-out R², head-vs-rep overlay, rank-sweep, depth profile).

## Status
PREREG written (plan of attack + confounds, per user). Design locked: CIFAR-100/ResNet18-LoRA, 3 cosines, frozen-re-probe forgetting,
efficient 1-fine-tune-many-A-reprobes, saturation+capacity+layer+seed guards. Next: SHA256-lock → build → --smoke → non-sat sweep →
full → read (which cosine, head-vs-rep, capacity trend) → verdict + bank.
