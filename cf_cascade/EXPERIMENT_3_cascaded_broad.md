# EXPERIMENT #3 — CASCADED-BROAD CONSTRUCTION: does building the funnel in depth-order protect better than all-at-once?

**For:** ML code agent / main lane (GPU). **Status: DONE 2026-06-20 (Lambda A100, 4 arms × 7 seeds, 21 min, ~$0.85) —
CASCADE INERT (clean negative).** User authorized the spiral agent to fire it; ran on A100 (instance auto-terminated).

## ★★★ RESULT (2026-06-20): CASCADE INERT — depth-ordered construction adds NOTHING over all-at-once
Retention A/B/C/D = **0.527 / 0.653 / 0.610 / 0.630** (pre 0.753). Paired (same-seed, 7 seeds):
- **C−B = −0.043 ± 0.034 (within 2SE, C>B only 3/7)** → cascaded-by-depth does NOT beat all-at-once broad; if anything
  nominally LOWER. **The headline contrast is null.**
- **C−D = −0.020 ± 0.051 (within 2SE)** → no direction effect (early→deep ≈ deep→early).
- **C−A = +0.083 (clears 2SE, 6/7)** and **D−A = +0.103 (clears)** → a cascaded childhood (either order) STILL protects over
  no-childhood — it's a non-destructive prior phase (#2b's general finding holds) — just not *better* than whole-net broad.
- GUARDS all pass: **B−A = +0.126 ± 0.024 reproduces #2 EXACTLY** (harness bit-faithful → arms valid); arm A forgets 0.227
  (range present, not VOID); task-B not starved (A/B/C/D 0.88/0.86/0.88/0.86); master-none held (childhood l1 B 0.79 / C 1.22 /
  D 1.57, all ≫ converged — breadth not depth; C/D higher l1 as expected from band-restricted under-training).

**VERDICT = the C≈B base-rate prior (clean negative).** Building the funnel band-by-band in depth-order is NOT load-bearing;
neither is the direction. The depth-structure is inert even when **constructed** in depth-order — so the #2b
construction-has-causal-power asymmetry does NOT extend to the depth-ORDERING of construction. **What protects is the
breadth / non-destructiveness itself (C−A, D−A, B−A all positive), NOT how it's distributed across depth or in what order —
breadth is breadth.** C is nominally the *lowest* childhood arm (0.610 < D 0.630 < B 0.653): distributing breadth into
depth-bands mildly *underperforms* whole-net (consistent with the flagged matched-compute caveat — band-restricted childhood
under-trains each band, ~133 vs 400 steps). **CLOSES "is ANY form of the depth-structure causal?" → NO** (inert observed:
SPIRAL-3/5; inert constructed: this). The mechanism is non-destructiveness + curvature + sharing, not the depth-geometry at any
level. Artifacts `gateA/dev_experiment3.{json,npz}` + `cloud/dev_experiment3_cascade_lambda.py` + `cloud/lambda_launch_dev3.py`
+ log `cloud/dev_experiment3_cascade_lambda.log`.

---
*(original spec below; launcher referenced is now run)*

**For:** ML code agent / main lane (GPU). **Status:** ran 2026-06-20 (see result above).
**Launcher:** `cloud/dev_experiment3_cascade_lambda.py` (syntax-checked; derived line-for-line from the working
`dev_experiment_lambda.py` (#2) — only new machinery is `childhood_cascade` + arms C/D). Adapt `cloud/lambda_launch_dev.py`
to point at it (same A100 pattern, finally-terminate). **~35-45 min / ~$1.5.**

## THE CLAIM UNDER TEST
SPIRAL-1 found the network naturally collapses fine→coarse with depth (early=task-separated, deep=universal contested core),
**implicitly** as a side-effect of training. SPIRAL-3/5 found that OBSERVED depth-geometry is **inert** (doesn't drive the
mechanism). But #2b's lesson: **construction has causal power where observation doesn't** (the observed neighborhood map was
inert, yet non-destructive *construction* protected). Cascaded-broad is the depth-axis construction protocol — build the
collapse band-by-band, in depth-order. If building the funnel early→deep protects task-A better than all-at-once broad, the
depth-structure flips from **inert-observation** to **load-bearing-construction**.

## ARMS (A,B reproduce #2 for paired comparison; C,D new)
| arm | phase-1 (childhood) | isolates |
|---|---|---|
| **A** no childhood | none | yank floor (#2: 0.527) |
| **B** all-at-once broad | breadth stream on WHOLE net, 400 steps | #2's childhood (the thing to beat; #2: 0.653) |
| **C** cascaded-broad *(NEW)* | breadth in depth-bands **early→mid→deep**, matched 400 steps total | does depth-ordered construction beat all-at-once? |
| **D** anti-cascade *(NEW)* | breadth in bands **reverse deep→mid→early**, matched 400 | is it the early→deep DIRECTION, or just any staging? |

Phase-2 (depth-train held-out **subj**, 250 steps, whole net) is **identical across all arms** — only phase-1 differs.

## DECISIVE CONTRASTS (paired, same-seed, judge on 2·SE separation)
- **C vs B** — does cascaded-by-depth beat all-at-once? **The headline.** C>B → depth-ordering is load-bearing (inert→causal).
- **C vs D** — early→deep specifically, or any staging? C>D → direction matters (build it the way the net collapses).
- **B vs A** — confirms #2 reproduces (all-at-once childhood protects over none).

## PRE-COMMITTED VERDICT
- C>B (2SE) AND C>D AND task-B not starved → **depth-ordered construction LOAD-BEARING** (highest-value: building the funnel the
  way the net naturally collapses it protects task-A; depth-structure flips inert→causal-when-built).
- C>B BUT C≈D → staging, not direction (weaker claim: distributed-breadth > all-at-once).
- C≈B → **construction inert too** (clean negative, consistent with SPIRAL-3/5's inert geometry; breadth is breadth however
  distributed across depth).
- C>B, C>D, BUT task-B starved → rigidity not slivering.

## DE-RISKING (done CPU-side against existing artifacts, before firing)
1. **#2 config recovered for exact reuse** (`dev_experiment.json`): heldout=subj, stream=[agnews,dbpedia,yahoo,newscat,emotion]
   (NON-sentiment, no task-A replay), STEPS_CHILD=400/LR 3e-5, STEPS_DEPTH=250/LR 5e-5, 7 seeds, pre_sst=0.753. Arm A=0.527,
   arm B=0.653 reproduce in the new launcher (all 4 arms run in ONE job for fully-paired stats — avoids cross-run drift; cheaper
   reuse-#2's-A/B alternative exists but single-job is cleaner).
2. **Band partition L0-7 / L8-15 / L16-23 validated against the natural collapse** (SPIRAL-1 curv-ctrl eff_rank 5.72→5.08→4.82,
   high→low) → early→deep cascade builds the funnel the way the net actually collapses it.
3. **Dynamic range confirmed**: subj forgets 0.263 in arm A (#2 precheck) — real range to detect protection. The launcher
   re-asserts arm A forgets ≥0.10 (VOID guard).

## DESIGN SUBTLETIES / HONEST FLAGS (read before interpreting)
- **Matched compute = same DATA/STEPS (400 batches), NOT same params-updated.** In B every step updates ALL params; in C/D each
  step updates only the current band (~1/3 params), so each band gets ~133 steps vs B's effective 400. This is *inherent* to
  "distribute breadth over depth." The **D control** (same band-distribution, reversed order) isolates ORDER from this
  distribution — so C vs D is the clean direction test even though C vs B carries the distribution difference.
- **B already builds the collapse.** The net collapses fine→coarse during ANY training, including B's whole-net broad. So C's
  end-state may not differ from B's in *geometry*; the test is whether the band-ordered **path** leaves task-A more robust.
- **Cascade = one band at a time** (others frozen), broad stream cycled continuously so each band sees breadth. Cumulative-
  unfreeze (band1, then band1+2, then band1+2+3) is a defensible variant — flagged, not implemented (one-at-a-time most cleanly
  "builds the collapse band-by-band" and makes C vs D clean).
- **Master-none guard:** report per-arm childhood loss l0→l1; with ~133 steps/band on a 5-task stream, cascade won't converge
  (stays breadth not depth). The launcher prints it; confirm C/D don't converge (else it's cascaded multi-task *depth*).
- Embeddings / final-LN / lm-head are frozen during the cascade phase-1 (not in any depth-band).

## HONEST PRIOR (not a verdict — stated so the run reads straight)
Given #2b (protection = non-destructiveness, not the map) and SPIRAL-3/5 (depth-geometry inert for the mechanism), the **base
rate favors C≈B** (construction-distribution adds nothing). The reason to run anyway is the #2b construction-has-causal-power
asymmetry: observation was inert, construction wasn't — cascaded-broad is the depth-axis construction, the one untried place the
depth-structure could still be load-bearing. A C>B result would be genuinely new; a C≈B cleanly closes "is ANY form of the
depth-structure causal" (no — neither observed nor constructed).

## RELATIONSHIP TO #2c (adjacent, independent — does NOT wait on it)
#2c asks *why* non-destructive warm-up protects (basin-flattening vs pre-moved-capacity). #3 is a standalone contrast (C vs B vs
D) that stands regardless. If #2c returns pre-moved-capacity, it would *explain* a positive cascade (each band pre-exercises its
complement = depth-distributed sink); if basin-flattening, a positive cascade stands on its own as a construction finding.

## GUARDS (mandatory, baked into the launcher)
arm A forgets ≥0.10 (VOID else) · task-B learns in all arms · every band confirmed master-none (loss doesn't converge) · 7 paired
seeds, per-seed arrays saved before readout · matched compute B=C=D · same heldout/stream/seeds as #2. Artifacts →
`gateA/dev_experiment3.{json,npz}`.
