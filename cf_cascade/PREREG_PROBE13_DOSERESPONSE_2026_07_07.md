# PRE-REG — Probe-13 (Hunt 3): is the cross-cell protection spread a DOSE artifact or a real architecture difference? 2026-07-07.

SHA256-lock before GPU. Follow-up to Probe-11's Hunt-1 finding: on the hardened metric the cells' N_phase2 spread
(OPT-350m +0.130, Pythia +0.106, BLOOM +0.015, OPT-1.3b +0.066) tracks their per-cell calibrated childhood DOSE almost
perfectly (OPT 1e-5, Pythia 3e-5, BLOOM 2e-6, OPT-1.3b 1e-6). Protection ∝ dose (and ∝ threat-damage, Hunt-1a r=0.81). So the
cross-architecture comparison is **dose-confounded** — BLOOM may look weak only because its non-destructive dose is 5–15×
gentler. This probe sweeps the childhood dose per cell and compares protection at **matched childhood-effect**.

## DV + x-axis
- **N_phase2** = childhood's reduction of task-A damage along the realized phase-2 (subj) interference direction, magnitude
  applied at magp. **N_random** = same for a top-16 random direction (matched-magnitude control). paired NoCH vs CH.
  (NO alttask — Hunt-1 showed S=N(phase2)−N(alttask) is a threat-magnitude artifact; the clean measure is N_phase2 vs N_random.)
- **x-axis = childhood-effect = retA_drop = w_A − retA_CH** (how much the childhood cost task-A). Comparable across
  architectures (unlike raw lr). Also record raw lr per dose point. NO calibration — dose is SET, not searched.

## Design
Per cell, sweep childhood lr over a per-cell set spanning gentle → non-destructive → mildly destructive (to trace the full
curve; we do NOT gate on non-destructive here — the retA_drop axis captures cost). N=50 paired seeds/dose, K_RAND=3.
NoCH arm (anchor=w_A) is dose-independent → computed ONCE per seed, shared across doses (efficiency). Incremental JSON dump
after each seed (crash-resilient — the v2 launcher-hang lesson).
Per-cell dose sets (env DOSES): span retA_drop ≈ 0 → ~0.15:
- opt350m:  2e-6, 5e-6, 1e-5, 2e-5, 3e-5
- pythia410m: 5e-6, 1e-5, 2e-5, 3e-5, 5e-5
- bloom560m: 3e-7, 1e-6, 2e-6, 3e-6, 7e-6
- opt13b:   1e-7, 3e-7, 1e-6, 3e-6, 1e-5

## Pre-registered read
Plot N_phase2 (and N_phase2 − N_random) vs retA_drop, per cell, overlaid.
- **DOSE-ARTIFACT:** at matched retA_drop, all cells (incl BLOOM) fall on ~one curve → the cross-cell spread was dose; the
  "directional concentration on the interference" story is ARCHITECTURE-GENERAL (strengthens it). BLOOM's Probe-11 weakness =
  its tiny non-destructive dose, not the arch.
- **REAL ARCH DIFFERENCE:** at matched retA_drop, cells separate (e.g. BLOOM gives less N_phase2 per unit task-A cost) →
  genuine architecture difference in protection efficiency. Quantify the gap.
- **Scale read (OPT-350m vs OPT-1.3b overlaid):** do the two OPT curves coincide (concentration scale-invariant) or does 1.3b
  need more childhood-cost for the same protection (scale-fragile, the DSI hint)?
- Report the crossover/slope (ΔN_phase2 / Δ retA_drop) per cell; note the destructive-region points (retA_drop > 0.05) as
  measured-on-damaged-baseline (interpret with care, but they trace the curve).

## Scope (fire plan)
- **Targeted (recommended first):** BLOOM + OPT-350m reference — directly answers "is BLOOM dose or arch." 2 instances.
- **Full:** all 4 cells (adds Pythia + the OPT-350m/1.3b scale overlay). 4 instances.

## Ops
Harness `cloud/probe13_doseresponse_lambda.py` (probe11 spine; drop alttask + calibration; dose loop; shared NoCH; incremental
dump). Launcher: probe13 <cell> with DOSES/NSEEDS pass-through. GPU_PREF excludes a10 (opt13b → h100 mem-safe). Hang-guard on
run-start; killswitch. Per-cell ≈ one instance ~4–5 h (NoCH shared across doses). Targeted ~$25–30, full ~$45–55.

## Status
NOT built/fired. Next: lock → build harness → CPU dry-run → fire (targeted first) → read N_phase2 vs retA_drop.
