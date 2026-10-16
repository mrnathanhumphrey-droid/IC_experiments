# VERDICT — Form-3: Compositional vs Holistic abstraction. 2026-06-14.

**RESULT: REFUTE.** Factoring does not reliably buy combinatorial generalization at matched
capacity. Where compositional wins, it is mostly a better-classifier effect, not composition.

## Design (locked, PREREG_FORM3_COMPOSITIONAL_2026_06_13)
3DShapes (6 independent factors), 120k images. Two arms, identical trunk/VIB/optim, ONLY
factoring differs: COMPOSITIONAL (F modules dim d, one per factor) vs HOLISTIC (1 latent dim F*d,
all factors off it) — capacity matched. Hardened holdout: novel pairings of individually-seen
factor values, 5 RANDOM splits (not hand-picked), 3 seeds. Gate: comp-hol >= +3% top-1 (paired
CI>0) on held-out combos in >=3/5 splits. Secondary guard: gap must CONCENTRATE on held-out vs
in-distribution (else "better classifier", not composition).

## Result: 2/5 clear the gate -> REFUTE; secondary guard cuts it to 1/5 genuine
| split | axes | held-out gap [CI] | in-dist gap | read |
|---|---|---|---|---|
| 0 | wall-hue x scale | +0.030 [.021,.039] | +0.024 | mild/mixed (sub-gate) |
| 1 | shape x scale | +0.075 [.069,.082] WIN | +0.085 | classifier artifact (id>=ho) |
| 2 | scale x obj-hue | +0.052 [.044,.061] WIN | +0.015 | GENUINE composition (ho>>id) |
| 3 | wall-hue x orient | +0.016 [.007,.025] | +0.048 | anti-composition (id>ho) |
| 4 | floor-hue x scale | +0.00004 [-.008,.009] | +0.046 | anti-composition (id>ho) |

- Direction positive in 4/5 (CI>0 in 4/5) — compositional is not *null*, just sub-threshold.
- Applying the secondary guard: split 1's win is classifier-effect (id ~ ho); only split 2 shows
  composition-specific benefit (advantage concentrated on novel combos). 2 splits anti-compose.
- **Net: 1/5 genuine composition. REFUTE stands and is cleaner than the 2/5 headline.**

## What it means (consolidated with Form-1)
- FORM frontier: Form-1 (geometry) REFUTED + Form-3 (compositional) REFUTED. Two forms, both refute.
- **Earned conclusion: abstraction-that-pays is about WHAT YOU DROP (omission / objective-pricing),
  not HOW YOU ENCODE what survives (form).** Confirms the pricing-bridge spine; form is downstream
  of the drop decision, so no form-change reprices the drop. Predicted in advance; confirmed.

## Thin thread (banked, not chased)
Compositional benefit is factor-pair-DEPENDENT (split 2 genuine vs splits 3/4 anti). Possibly the
"objective semantic-distance" axis (Wilson) — compositional may help for distinct factor pairs and
hurt for entangled ones. 5 splits cannot resolve it; underpowered. Bank as a maybe, do not chase
off 5 points. The live spine lever remains objective-pricing GRANULARITY (un-run: objective-count
ablation 1 vs 2 vs 3 objectives x selectivity x semantic distance), NOT more form variants.
