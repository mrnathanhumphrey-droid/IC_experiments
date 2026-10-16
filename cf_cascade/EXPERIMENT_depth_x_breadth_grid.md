# EXPERIMENT — Depth × Breadth Grid: does matching application-breadth to a layer's natural resolution protect best?

**For:** ML training agent (GPU). **Status:** specced, ready to fire. Pythia-410M substrate. The RICHER sibling of the cascaded-broad experiment — cascaded-broad fixes breadth (always full stream) and varies only depth-ORDERING; this CROSSES depth-of-application × breadth-of-stream independently.

---

## THE IDEA (distinct from cascaded-broad)
Cascaded-broad keeps breadth constant (the whole stream) and varies WHEN/WHERE it's applied (depth-order). This experiment varies BOTH axes independently:
- **Depth-of-application:** which layer-band the phase-1 pass trains (early / mid / deep).
- **Breadth-of-stream:** how WIDE the stream applied at that band is (narrow = few task-kinds / near-task-A; wide = many task-kinds / full neighborhood).

The crossing asks: **does the right BREADTH depend on the DEPTH?** I.e. is there an interaction where matching the stream-width to the layer's natural resolution protects best?

## THE PRE-REGISTERED PREDICTION (baked in from SPIRAL-1 — commit before measuring)
SPIRAL-1 found the network NATURALLY runs **wide/separated early → narrow/collapsed deep** (early layers discriminate many task-kinds in high-dim; deep layers collapse onto the shared low-dim universal core). So the natural funnel-shape is wide-early, narrow-deep.

**Prediction: applying WIDE breadth at EARLY layers + NARROW breadth at DEEP layers (matching the natural funnel) protects task-A better than the mismatched assignments** (narrow-early/wide-deep, or uniform). The hypothesis: you protect best by *helping the network build the funnel-shape it naturally wants* — wide exposure where it's naturally wide (early), narrow where it's naturally narrow (deep). If the protection tracks the MATCH between application-breadth and the layer's natural resolution, then the funnel-geometry is something you can deliberately CONSTRUCT-to-protect (the inert→load-bearing flip), and the construction recipe is "match breadth to depth's natural resolution."

## THE GRID
Two factors crossed. Depth-band ∈ {early L0-7, mid L8-15, deep L16-23}. Breadth-of-stream ∈ {narrow (≈2-3 near-task-A kinds), wide (≈8+ kinds, full neighborhood)}. Plus baselines.

Core arms (the interaction test — apply a specific breadth at a specific band, freeze rest, master-none):

| arm | early band | mid band | deep band | tests |
|---|---|---|---|---|
| **MATCH (funnel)** | WIDE | (taper) | NARROW | the SPIRAL-1-matched recipe — wide-early→narrow-deep |
| **ANTI (inverted)** | NARROW | (taper) | WIDE | the inverted funnel — narrow-early→wide-deep |
| UNIFORM-WIDE | WIDE | WIDE | WIDE | breadth everywhere, no depth-matching (≈ #2's all-at-once but band-staged) |
| UNIFORM-NARROW | NARROW | NARROW | NARROW | narrow everywhere |
| A. no childhood | — | — | — | yank floor |

(If 3 bands × 2 breadths full-factorial is too heavy, the MATCH-vs-ANTI-vs-UNIFORM contrast is the minimal decisive set — fire that first, expand to full grid only if the interaction shows.)

## DECISIVE CONTRASTS
- **MATCH vs ANTI** — does wide-early/narrow-deep (funnel-matched) beat narrow-early/wide-deep (inverted)? **THE headline.** MATCH > ANTI → the breadth-must-match-depth's-natural-resolution interaction is real; you protect by building the funnel the network wants. MATCH ≈ ANTI → no depth×breadth interaction; the funnel-shape isn't a constructible protection recipe.
- **MATCH vs UNIFORM-WIDE** — does matching beat just-apply-wide-everywhere (#2's all-at-once)? MATCH > UNIFORM-WIDE → the depth-MATCHING adds value beyond just-more-breadth. MATCH ≈ UNIFORM-WIDE → matching is no better than uniform wide; breadth is breadth.
- **(all) vs A** — confirm any phase-1 protects over none.

## METRICS (two-sided sliver)
- task-A retention after depth-training the held-out task-B (vs pre-task-B baseline).
- task-B accuracy (coexist check — retain A AND learn B; matching mustn't starve task-B).

## PRE-COMMITTED VERDICT
- **MATCH > ANTI AND MATCH > UNIFORM-WIDE (clearing error bars)** → the depth×breadth INTERACTION is real and load-bearing: protection is maximized by matching application-breadth to the layer's natural resolution (wide-early/narrow-deep) → the funnel-geometry is a CONSTRUCTIBLE protection recipe, and SPIRAL-1's natural funnel-shape tells you how to build it. **The result that makes the depth-geometry load-bearing-as-construction** (the inert→load-bearing flip, depth×breadth version).
- **MATCH ≈ ANTI** → no interaction; the direction of breadth-vs-depth matching doesn't matter → the funnel-shape is not a constructible protection recipe (consistent with the inert observed funnel).
- **MATCH > ANTI BUT MATCH ≈ UNIFORM-WIDE** → matching beats inverted but not uniform-wide → it's "don't put narrow where wide should go" (avoiding the anti-pattern) rather than "matching adds value over just-wide" → weaker, partial interaction.
- **task-B starved in MATCH** → rigidity not slivering; doesn't support.

## ROBUSTNESS (carry every lesson)
- **~6-8 seeds**, ERROR-BAR SEPARATION for MATCH-vs-ANTI and MATCH-vs-UNIFORM (small expected effect — an interaction on top of #2's ~0.12).
- **Held-out task-B = subj** (real dynamic range, in-neighborhood, comparable to #2 and cascaded-broad).
- **Matched total compute across MATCH/ANTI/UNIFORM arms** — the difference is the DISTRIBUTION of breadth-width across bands, not total breadth steps. (This is subtle: "wide" and "narrow" must be compute-matched — narrow-fewer-tasks-more-steps vs wide-more-tasks-fewer-steps, or hold steps fixed and vary only stream-width. Declare the compute-matching scheme answer-blind; it's the construction crux — if MATCH simply gets more total breadth than ANTI, the contrast is confounded.)
- **Breadth-not-depth at every band** (master-none; stream loss not converged — even "narrow" must not master its few tasks).
- **Define narrow vs wide from #1's neighborhood map** — narrow = the in-neighborhood near-task-A kinds (topic/subj/emotion); wide = the full stream including farther kinds. Use the banked v3/#1 neighborhood structure to pick the stream compositions principledly, not arbitrarily.

## GUARDS
Arm A forgets; task-B learns in all arms; every band low-res (master-none); compute-matched across MATCH/ANTI/UNIFORM (the crux — a confounded compute-match voids the interaction read); per-seed distributions; align=0 / dataset-identity; save before readout; save phase-1 checkpoints for seed-branching.

## RELATIONSHIP TO CASCADED-BROAD AND #2c
- **vs cascaded-broad:** cascaded-broad fixes breadth (full stream) and tests depth-ORDERING (early→deep vs reverse). This fixes nothing and crosses depth × breadth-WIDTH, testing the INTERACTION (does the right width depend on the depth). Cascaded-broad is the simpler first swing; this is the full crossing. Run cascaded-broad first (cheaper, cleaner headline); run this if cascaded-broad shows depth-ordering matters (then the question becomes "and does width-matching matter too").
- **vs #2c:** standalone; if #2c returns pre-moved-capacity, a positive MATCH result is explained as each band pre-exercising a depth-appropriate-sized complement (wide-early exercises a big early complement, narrow-deep a small deep one — matching the natural dimensionality). Does not wait on #2c.

## USER ADDITIONS
