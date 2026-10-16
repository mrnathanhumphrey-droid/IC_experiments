# VERDICT — Thread #2: Superclass-Primary Tier (Arm C)

**Date:** 2026-06-12
**Status:** COMPLETE. **POSITIVE RESULT** — genuine tier abstraction emerges.
**Follows:** v0.1 (H1 falsified — bottleneck alone does not abstract). See VERDICT_TIER_ABSTRACTION_V0_1.md.
**Runs:** 24 Arm-C = 8 β × 3 seeds, CIFAR-100, K=8, 50 epochs. Compared against v0.1 Arms A & B (48 runs).

## What changed vs v0.1

v0.1 wired both tiers' objectives onto z_trans (fine primary, super auxiliary). Arm C
**compartmentalizes the objectives across tiers**:

- fine label  ← head on **z_low**  (lower tier keeps fine detail; bypasses the bottleneck)
- super label ← head on **z_trans** (upper tier, through the VIB bottleneck, is the coarse adjudicator)
- `L = L_fine(z_low) + L_super(z_trans) + β·KL`

z_trans receives gradient ONLY from the superclass loss. Everything else (model size, K=8,
optimizer, schedule, data, probe protocol) is identical to v0.1.

## Result: selectivity goes from ~0 to +0.13, with a β sweet spot.

Selectivity index = within-super error fraction on z_trans − same on z_low (mean ± sd, n=3):

| β | KL | Arm A (bottleneck alone) | Arm B (super aux, shared rep) | **Arm C (objectives split)** |
|---|---|---|---|---|
| 0.0 | ~110 | −0.002 ± 0.001 | +0.005 ± 0.008 | **+0.090 ± 0.003** |
| 0.001 | ~35 | −0.001 ± 0.006 | +0.019 ± 0.003 | **+0.095 ± 0.002** |
| 0.01 | ~18 | −0.003 ± 0.003 | +0.023 ± 0.005 | **+0.112 ± 0.008** |
| **0.1** | ~5 | −0.001 ± 0.003 | +0.017 ± 0.003 | **+0.125 ± 0.003 ← peak** |
| 0.2 | ~3 | +0.000 ± 0.006 | +0.018 ± 0.004 | **+0.111 ± 0.013** |
| 0.3 | ~2 | −0.011 ± 0.007 | +0.025 ± 0.004 | **+0.088 ± 0.003** |
| 0.5 | ~1 | −0.105 ± 0.016 | +0.002 ± 0.007 | **+0.028 ± 0.015** |
| 1.0 | 0 | −0.112 ± 0.012 | −0.073 ± 0.012 | **+0.026 ± 0.013** |

At the peak (β=0.1) the effect is ~40σ from zero. The inverted-U is clean and seed-robust.

Supporting metrics at the sweet spot (β=0.1), all consistent with genuine tier abstraction:
- `fine←z_low` holds ~0.50 across ALL β — **lower tier keeps full fine detail** (bypasses bottleneck).
- `fine←z_trans` drops to 0.28 — **upper tier discarded fine structure the lower tier kept.**
- `sup←z_trans` holds ~0.58 — **superclass preserved while fine selectively lost.**
- Within-super confusion on z_trans climbs to ~0.41 vs z_low's ~0.28 — **the loss is structured along the hierarchy**, not uniform.
- CKA(z_low, z_trans) falls to 0.52 (vs 0.68 Arm A / 0.66 Arm B at β=0.1) — **the two tiers are measurably more distinct.**

## Interpretation (with the skeptical decomposition)

This is a positive result, but the credit splits two ways and it matters to be precise:

1. **Objective separation does most of the work (the load-bearing insight).** At β=0 — no
   bottleneck at all — Arm C already shows +0.090 selectivity, vs ~0 (Arm A) and +0.005
   (Arm B). The decisive difference between Arm B (+0.02) and Arm C (+0.09) is NOT the
   presence of superclass supervision (both have it) — it is that Arm B *also demanded fine
   from z_trans*, which fights abstraction, while Arm C routes fine off z_low so z_trans is
   free to specialize. **The rule is not "supervise the coarse label" — it is "don't demand
   coarse and fine from the same representation."** That is compartmentalization.

2. **The bottleneck is constructive — it sharpens the split.** Within Arm C, selectivity
   rises +0.090 (β=0) → +0.125 (β=0.1), a ~39% boost, then collapses past β=0.5 as the
   bottleneck strangles superclass itself. So IB pressure has a real, beneficial role — but
   only *because there is now a coarse objective for it to act on*. This is the inverted-U
   "sweet spot" the original v0.1 spec predicted; it was absent in v0.1 only because the
   objectives were not compartmentalized.

## Bottom line for the IC architecture

The refined, positively-supported thesis:

> **Abstraction does not emerge from compression. It emerges when objectives are
> compartmentalized across tiers — and the information bottleneck then sharpens the split,
> with an optimal pressure (here β≈0.1, KL≈5 nats).**

v0.1 ruled out the cheap version (bottleneck alone). Arm C is a minimal instance of the v0.2
compartmentalization idea, and it works: the upper tier becomes a qualitatively coarser
representation than the lower tier, measurably and seed-robustly. This justifies building the
full v0.2 — multiple compartmentalized lower-tier paths with restricted information flow — on
firm empirical ground rather than hope.

## Open questions banked for v0.2
- Does the effect grow with MORE compartmentalization (multiple blind lower-tier squads)?
- Discrete (VQ) bottleneck — does categorical commitment push selectivity past +0.13?
- ~~A task where the abstraction PAYS (few-shot new fine classes within known supers) — does
  the abstracted upper tier then improve downstream transfer, not just probe selectivity?~~
  **[ANSWERED — YES. Struck 2026-07-14 by staleness audit; see `D:/IC_experiments/STATUS_RECONCILIATION_2026_07_14.md`.]**
  The run exists and has sat unlinked: `results/run_6b_transfer_result.json` (2026-06-16) →
  `"verdict": "COMPARTMENTALIZATION-TRANSFERS"`, armC **0.336** vs armB **0.302**, paired_diff **+0.0349**,
  CI95 [0.0216, 0.0484], clears MDE 0.03. **Compartmentalization transfers downstream, not just probe-deep.**
  ⚠️ CROSS-ARC: this is the SAME run as `float_module/III_RESULT_6B_TRANSFER_2026_06_16.md` (identical numbers;
  lock `1f66aa5c`, prereg `bbb91aeb`) — it is banked THERE, under float_module's §6.B, not here. Neither arc linked
  the other, which is why this read as an open question for a month. float_module's write-up adds the decisive
  secondary: the observational **+0.457** was *entirely readout capacity* (matched-readout Δ = −0.0004).
- Sweet-spot β may shift with K and depth — worth a small (β, K) co-sweep for v0.2.

## Artifacts
- `train_armC.py`, `sweep_armC.py`; features/eval/train JSON for `armC_*` runs
- `results/summary.csv` (72 rows incl. all 3 arms), `results/sweet_spot.png`
- `results/sweep_armC_seed0.log`, `sweep_armC_seeds12.log`
