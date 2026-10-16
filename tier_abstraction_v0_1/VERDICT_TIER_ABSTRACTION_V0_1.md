# VERDICT — Tier-Abstraction Toy Experiment v0.1

**Date:** 2026-06-12
**Status:** COMPLETE. Pre-registered falsifier resolved: **H0 confirmed (H1 falsified).**
**Runs:** 48 = 8 β × 2 arms × 3 seeds, CIFAR-100, K=8, 50 epochs each.

## Question (pre-registered)

Does an IB-regularized translation module between two tiers produce *genuine tier
abstraction* (upper tier reorganizes information so abstract category structure becomes
accessible while fine detail is selectively discarded), or does it merely produce a
pooled/compressed copy of the lower tier?

## Result: the bottleneck compresses, it does not abstract.

Aggregated across 3 seeds (mean ± sd):

| arm | β | KL rate | **selectivity index** | sup←z_trans | fine←z_trans | CKA |
|---|---|---|---|---|---|---|
| A (λ=0, bottleneck alone) | 0.0 | 130.9 | −0.002 ± 0.001 | 0.500 | 0.386 | 0.83 |
| A | 0.001 | 36.1 | −0.001 ± 0.006 | 0.503 | 0.395 | 0.79 |
| A | 0.01 | 18.8 | −0.003 ± 0.003 | 0.509 | 0.400 | 0.75 |
| A | 0.1 | 6.3 | −0.001 ± 0.003 | 0.498 | 0.355 | 0.68 |
| A | 0.2 | 3.24 | +0.000 ± 0.006 | 0.436 | 0.267 | 0.61 |
| A | 0.3 | 1.72 | −0.011 ± 0.007 | 0.359 | 0.187 | 0.62 |
| A | 0.5 | 0.00 | −0.105 ± 0.016 | 0.086 | 0.026 | 0.09 |
| A | 1.0 | 0.00 | −0.112 ± 0.012 | 0.050 | 0.010 | — |
| B (λ=0.5, super head) | 0.0 | 121.9 | +0.005 ± 0.008 | 0.529 | 0.385 | 0.81 |
| B | 0.001 | 37.5 | +0.019 ± 0.003 | 0.537 | 0.390 | 0.77 |
| B | 0.01 | 19.9 | +0.023 ± 0.005 | 0.547 | 0.398 | 0.74 |
| B | 0.1 | 7.3 | +0.017 ± 0.003 | 0.541 | 0.369 | 0.66 |
| B | 0.2 | 4.40 | +0.018 ± 0.004 | 0.506 | 0.306 | 0.63 |
| B | 0.3 | 2.94 | +0.025 ± 0.004 | 0.454 | 0.245 | 0.58 |
| B | 0.5 | 1.34 | +0.002 ± 0.007 | 0.361 | 0.166 | 0.56 |
| B | 1.0 | 0.00 | −0.073 ± 0.012 | 0.175 | 0.060 | 0.18 |

Selectivity index = (within-superclass error fraction on z_trans) − (same on z_low).
Positive ⇒ z_trans's fine-class errors concentrate *within* superclass (the signature of
hierarchical abstraction). Chance within-super error fraction = 0.040.

## Reading the pre-registered discriminator

1. **The bottleneck is real and binds.** KL rate falls monotonically with β (131 → 0 nats);
   CKA(z_low, z_trans) falls 0.83 → 0.06. The network does **not** route around the
   bottleneck — that H0 failure mode is ruled out. The machinery works.

2. **Arm A — bottleneck alone produces ZERO abstraction.** Selectivity index is within
   ±0.011 of zero (and never significantly positive) across the entire functional β range
   (0 → 0.3), with tiny error bars. At extreme β it goes strongly negative as both tiers
   collapse together (uniform catastrophic degradation, not selective abstraction).

3. **Fine drops faster than super — but for the trivial reason.** As the bottleneck tightens
   (e.g. β=0.2, KL=3.2 nats), fine accuracy degrades more than super (retains ~65% vs ~85%).
   Measured by aggregate accuracy alone this *looks* like weak abstraction. It is not: the
   selectivity index stays at zero, meaning the lost fine information is discarded
   **without respect to superclass boundaries**. This is rate-distortion — the harder task
   (more bits) dies first under compression — not hierarchical reorganization. *The
   within-superclass confusion metric is what distinguishes these two explanations; aggregate
   accuracy cannot.*

4. **The only abstraction signal is supervision-driven, not bottleneck-driven.** Arm B shows
   a small but statistically real selectivity (~+0.02, e.g. +0.023 ± 0.005 at β=0.01, ~4σ).
   But it is **β-independent** — present at low β, not amplified by bottleneck pressure,
   fading by β=0.5. It tracks the presence of the superclass head (λ), not the IB rate.

## Conclusion

**H1 falsified at this dataset/scale/architecture.** An information bottleneck constrains
*how much* information passes, not *which* — and a fine-classification objective gives the
compressor no reason to organize its loss along the superclass hierarchy. Compression ≠
abstraction. The IB-regularized translation module, on its own, produces redundancy (a
compressed copy of the lower tier), exactly the H0 the design was built to detect.

This is an *informative* null, not a dead end:
- It rules out the cheap version of the IC tier-abstraction claim (bottleneck pressure alone).
- It localizes the missing ingredient: abstraction requires an architectural **reason** to
  allocate coarse structure to the upper tier — i.e. the **compartmentalization** piece
  (v0.2), which this result now justifies as *necessary*, not optional.
- The Arm B +0.02 flicker is the one pulse: directing the upper-tier objective at the
  hierarchy produces structured loss the pure bottleneck does not. Worth amplifying.

## Live threads (ranked by data support)

1. **v0.2 compartmentalization** — give the architecture a reason to allocate abstraction.
   The strongest read; this experiment is the evidence it's required.
2. **Superclass as PRIMARY upper objective** (not auxiliary). Cheap on this rig (swap heads).
   The Arm B flicker says this direction has a pulse.
3. **Discrete bottleneck** (VQ codebook) instead of continuous Gaussian — forces categorical
   commitment that smooth IB structurally avoids.
4. **A task where abstraction pays** (few-shot new fine classes within known supers;
   hierarchical reward). CIFAR-100 fine classification rewards no superclass abstraction.

Explicitly NOT a thread: "scale it to transformers/more data" (no mechanism, just hope —
scaling was pre-quarantined as a separate question); further β/K/λ tuning on this exact setup
(flat zero across four orders of magnitude of β — same knobs won't surprise).

## Artifacts
- `results/summary.csv` (48 rows), `results/sweet_spot.png` (per-arm curves vs β)
- `results/sweep_seed0.log`, `sweep_gapfill_seed0.log`, `sweep_seeds12.log`
- Per-run `results/train_*.json`, `results/eval_*.json`, `checkpoints/*.pt`
