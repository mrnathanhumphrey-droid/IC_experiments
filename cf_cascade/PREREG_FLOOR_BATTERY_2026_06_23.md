# PRE-REG — B−D reproducibility-floor battery: measure it, decompose it, set the fix. 2026-06-23. LOCK below.

**For:** code/compute agent. **Status: fire-ready.** A near-replication exposed that two nominally-identical runs of
the same checkpoint (`pythia410m_sst2_subsetmin.pt`) differ in B−D by **0.066** at matched A_forget, identical regime.
That is the run-to-run noise floor, and the program's headline effects (OPT B−A +0.080, B−D +0.055, k0 +0.034) sit at it.
This battery measures the floor PER CELL, decomposes it into named sources, and returns the fix. **Noise
characterization, not mechanism** — there is no exciting outcome; the deliverable is a number, its parts, and a seed
budget.

**PRE-FLIGHT (done 2026-06-23):** checked gateA/ — no variance/floor block outputs exist (taskavar_seed_* = per-layer
task-A seed variance, different object). Full battery to fire; nothing to read out.

**Cells (pre-committed):** OPT-350m AND pythia410m — BOTH (floor likely scales with per-seed SD, expected
heteroskedastic). Report per-cell; do NOT report a single global floor. opt1.3b added to Part B ONLY if cheap.

**Matched dispositive regime (exact):** lr_depth 5e-5 / 250, lr_child 3e-5 / 400, pinned w_A, held-out subj, V = top-128
task-A Fisher. arm B = childhood(full) → phase-2; arm D = childhood(grad projected out of V) → phase-2 (dispositive
method). B−D = retB − retD at matched A_forget. Reproduce each w_A acc (OPT 0.823, Pythia 0.753) before any block.

**Anchors (reproduce-known gate; treat as ONE prior draw, not ground truth):** pythia410m dispositive B−D = −0.006;
OPT-350m = +0.055.

## PART A — variance decomposition (4 blocks/cell)
Pre-commit: block spreads combined in quadrature (RSS) should reconstruct ~0.066. If not, an unnamed source exists —
FLAG it, do not force the fit.

1. **bf16 floor** — SAME 30 seeds, same env, bf16 (current path), 5× independent repeats. SD of B−D across the 5 = pure
   hardware non-determinism.
2. **fp32 floor** — same 30 seeds, same env, fp32 on loss+projection path (+ deterministic algorithms), 5×.
   Pre-committed ≈ 0; if materially >0, a determinism source beyond bf16 exists — name it. (Doubles as the FIX TEST.)
3. **seed-sampling** — 5 different draws of N=30 (fp32, same env). Pre-committed ≈ per-seed-SD/√30 ≈ 0.018.
4. **drift** — re-run the exact original dispositive config in today's env, original seeds if recoverable. Gap from the
   banked anchor = library/env drift across weeks. THE DANGEROUS ONE — if large, old banked numbers aren't comparable
   and need a date-stamped re-baseline.

**RSS reconstruction (pre-committed):** floor² ≈ bf16² + seed² + drift² (Block 2 ≈ 0 unless it surprises). Report
reconstructed vs observed 0.066, and the dominant source.

**Decision gate (pre-committed BEFORE looking):**
- bf16 dominates → fp32 accumulation on projection+loss path is the fix; floor collapses to seed-sampling. Cheapest.
- seed dominates → fix is more N (Part B); fp32 won't help.
- drift dominates → pin env (container + lockfile), re-baseline all live cells, flag old numbers not-comparable. Most
  expensive, most important to know.

## PART B — seed-curve
One cell (OPT-350m; +opt1.3b if cheap), fp32, run N=100 once. Bootstrap-subsample B−D SE at N=10/20/30/40/60/100.
Report N where SE crosses **0.02** (bar to resolve a +0.05 effect at ~2.5σ within-run). Sets the per-cell seed budget;
replaces the guessed N=30.

## Controls / discipline
- Reproduce-known anchor per cell before banking any block.
- Per-cell floor, never global. Heteroskedasticity is the expected case.
- Pre-commit the RSS expectation and decision gate BEFORE viewing results — maximally fishing-exposed (the "answer" is a
  variance you're explaining); lock the logic first (this doc).
- No mechanism reads. Every number is a noise estimate. Outputs only: per-cell floor, its decomposition, dominant
  source, recommended fix, SE-vs-N curve.
- STOP-on-missing → generate the artifact yourself, do not proxy. Fire own launchers, no pipe to head/tail (SIGPIPE
  orphans), finally-terminate, match orphans by printed instance_id.

**Ops:** all existing checkpoints, no new training. Blocks 1–2: 5×30×2 cells; Block 3: 5 draws×30×2; Block 4: 1×2;
Part B: 100×1–2. Bounded, ~a day. fp32 is slower per run — budget it.

**What it settles & downstream:** the per-cell floor, its composition, whether fp32 cheaply kills it or N must rise,
whether drift moved the baseline — and the seed budget every future B−D cell needs. Feeds the floor-aggregation desk
table (how many banked effects clear lower-CI > floor). Sets the significance threshold for the swap-gate OPT anchor,
the depth-ladder magnitudes, the dedup paired-differential redo, and gates whether Probes 3/4/5 fire at N=30 or higher.
**Until this lands, no new mechanism arms** — they'd inherit an uncharacterized floor.

Harness `cloud/floor_battery_lambda.py` + launcher `cloud/lambda_launch_generic.py floorbattery`. CELL ∈ {opt350m,
pythia410m}; BLOCK env selects bf16|fp32|seedsample|drift|seedcurve. LOCK below.
