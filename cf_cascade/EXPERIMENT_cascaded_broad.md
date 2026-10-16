# EXPERIMENT — Cascaded-Broad Construction: does building the funnel in depth-order protect task-A better than all-at-once breadth?

**For:** ML training agent (GPU). **Status:** specced, ready to fire. Pythia-410M substrate (builds on SPIRAL-1's funnel). Adjacent-but-separate from #2c and the OPT ladder — runs parallel, no conflict.

---

## THE IDEA
SPIRAL-1 found the network NATURALLY collapses fine→coarse with depth (early layers = task-separated/high-dim, deep layers = collapsed onto the shared universal contested core) — but implicitly, as a side-effect of training. This experiment BUILDS that funnel deliberately: apply the breadth stream (whole stream, broad, master-none) at successive depth-bands — **broad@early → broad@mid → broad@deep, cascaded** — constructing the collapse layer-band by layer-band. Keep breadth at every stage (NOT one-task-per-depth — *broad* each time); what varies is WHICH depth-band the breadth pass trains, cascaded downward.

## WHY IT'S HIGH-LEVERAGE
SPIRAL-3/5 showed the OBSERVED funnel/spiral is real-but-INERT (doesn't drive seed-location or the carrier-split; all three structural levels of the map — flat #2b, funnel SPIRAL-3, spiral SPIRAL-5 — are inert for the mechanism). BUT #2b's lesson is that CONSTRUCTION has causal power where OBSERVATION doesn't (the observed map was inert for protection on Pythia; non-destructive CONSTRUCTION was what protected). Cascaded-broad is a construction protocol — the depth-axis version of the thing that actually worked. **If building the funnel in depth-order protects, the depth-structure flips from inert-observation to load-bearing-construction** — the same inert→load-bearing flip non-destructiveness pulled off in #2b.

## THE ARMS

| arm | phase-1 protocol | isolates |
|---|---|---|
| A. no childhood | none → depth-train held-out task-B | yank floor (reuse #2's arm A if seeds/setup match) |
| B. all-at-once broad | breadth stream applied to WHOLE network at once → depth-train task-B | #2's childhood (the non-cascaded baseline — the thing to beat) |
| **C. cascaded-broad (THE NEW ARM)** | breadth stream applied in depth-bands, cascaded: broad@early, THEN broad@mid, THEN broad@deep (matched total compute to B) → depth-train task-B | does DEPTH-ORDERED construction of the funnel beat all-at-once? |
| D. anti-cascade (control) | breadth stream cascaded in REVERSE depth-order: broad@deep → mid → early (matched compute) | is it the DEPTH-ORDER specifically, or just any cascading/staging? |

## DECISIVE CONTRASTS
- **C vs B** — does cascaded-broad-by-depth beat all-at-once broad? **THE headline.** C > B → depth-ordering of the breadth is load-bearing (building the funnel in depth-order matters, even though the observed funnel was inert). C ≈ B → depth-ordering adds nothing; breadth is breadth regardless of cascade.
- **C vs D** — is it *early→deep* order specifically (building the funnel the way the network naturally collapses), or would ANY staging (even reverse) do equally? C > D → the *direction* matters (early→deep, matching the natural fine→coarse collapse). C ≈ D → staging-not-direction.
- **B vs A** — confirms #2's result reproduces (all-at-once childhood protects over none).

## IMPLEMENTING THE DEPTH-BANDS
Partition the 24 layers into bands (e.g. early L0-7 / mid L8-15 / deep L16-23, or finer). "Broad@band" = train ONLY that band's parameters (freeze the rest) on the breadth stream, master-none (low-res, the established childhood regime — confirm stream-task loss does NOT converge, the breadth-not-depth guard). Cascade = sequentially unfreeze-and-broad-train each band in depth-order. **Matched total compute across B/C/D** (same total breadth steps, just distributed differently across bands/order) — the ONLY difference is HOW the breadth is distributed over depth and order, not how much.

## METRICS (the sliver, two-sided)
- task-A retention after depth-training task-B (vs pre-task-B baseline) — did task-A survive?
- task-B accuracy — did cascading cost task-B learning? (coexist = retain A AND learn B)

## PRE-COMMITTED VERDICT
- **C > B (clearing error bars) AND C > D** → cascaded-broad in depth-order protects better than all-at-once, AND the early→deep direction is what matters → **depth-ordered construction of the funnel is LOAD-BEARING.** The depth-structure flips from inert to causal-when-built. Highest-value outcome.
- **C > B BUT C ≈ D** → cascading helps but direction doesn't matter → it's STAGING (distributing breadth over bands) not depth-ordering per se. Still a result, weaker claim.
- **C ≈ B** → depth-distribution of breadth adds nothing over all-at-once → the funnel-construction is inert even when built; breadth is breadth however applied across depth. Clean negative — consistent with the inert observed funnel.
- **C > B, C > D, BUT task-B starved** → cascade retains A by refusing to learn B → rigidity, not slivering. Doesn't support the claim.

## ROBUSTNESS (carry every lesson)
- **~6-8 seeds**, judge C-vs-B and C-vs-D on ERROR-BAR SEPARATION not point estimates (effect likely small — #2's childhood effect was ~0.12; a cascade *improvement on top* is smaller).
- **Held-out depth task-B = subj** (forgot 0.263 in #2's arm A — real dynamic range; same task as #2 for direct comparison to the all-at-once arm).
- **In-neighborhood only** (subj is in-neighborhood per #1 — keeps it interpretable, same scope bound as #2).
- **Matched total compute B = C = D** (the whole point — only distribution-over-depth-and-order varies).
- **Breadth not depth at every band** — confirm stream-task loss does NOT converge in any band (else it's cascaded multi-task DEPTH training, not broad — the developmental claim needs master-none).
- **Reuse #2's A/B** if seeds/setup match (only C and D are new compute — cheaper, apples-to-apples).

## GUARDS
Arm A forgets (floor), task-B learns in all arms (phase-2 converges), every band confirmed low-res (master-none), per-seed distributions reported, align=0 / dataset-identity, save before readout. **Save the cascaded-childhood checkpoints** (C and D's phase-1) so the held-out-depth seeds branch without recomputing the cascade.

## RELATIONSHIP TO #2c (adjacent, separate — does NOT wait on it)
#2c asks WHY non-destructive warm-up protects (flattening vs pre-moved-capacity). This asks DOES depth-ordered construction of breadth protect better than all-at-once — a clean standalone contrast (C vs B vs D) that stands regardless of #2c's mechanism answer. If #2c later returns pre-moved-capacity, it would EXPLAIN a positive cascade result (each band pre-exercises its complement = depth-distributed sink); if basin-flattening, the cascade result stands on its own as a construction finding.

## USER ADDITIONS
