# HANDOFF — DSI / active-astronomer probe program (pre-compact). 2026-06-22. Agent 2.

## ★ IN FLIGHT (check first after compact) ★
**DSI swap-gate + OPT anchor re-run, both band cells (~2hr, 4 arms A/B/Bswap/D × 30 seeds):**
- opt350m: task `bf77qubmm`, instance `45165963edd44bfd9df38496bf044149`
- pythia410m: task `bzamyfno4`
On landing → `gateA/dsi_opt350m.json` + `dsi_pythia410m.json` (+ `_*.npz` per-(arm,step,carrier)).
THEN run `python probe1_carrier.py` (per-carrier signature + swap read). Read order is PRE-COMMITTED below.

## STANDING RULES (carry verbatim)
- I RUN what we have; never defer/pass; USER coordinates. Fire ALL GPU to Lambda myself (key=ML-CF, `D:/physics_detector/.env`,
  ssh `ml_cf_ed25519`). Hard finally-terminate. Match orphans by PRINTED instance_id, never "newest/idle" (other lanes share the key).
- Launch ops: NO `| head`/`| tail` pipe on launchers (SIGPIPE orphans). STAGGER launches (6-at-once = 429; geombasis/dsi `_req`
  don't all retry 429). SSH-readiness probe must try/except TimeoutExpired (slow-accept instance crashed launchers — fixed in
  geombasis + dsi launchers; bd_feed/wa launchers hardened by other agent).
- Six tripwires bind. Matched-acc / reproduce-anchor GATES every read. Per-(layer,carrier) never pooled. Curvature-controlled (V at w_A).
- **REPRODUCIBILITY FLOOR ~0.066 (agent 1, measured).** Absolute B-D levels are floor-exposed; PAIRED differentials (same run/cell/seeds)
  largely cancel it. Bank on differentials, demote absolute magnitudes. Don't diagnose a "systematic" gap until ruling out the floor.

## PROGRAM = 7-probe DSI battery (`universe_metaphor_probes.md`). Probe 2 = OTHER AGENT (rank ablation + `basisinv` K_E lane).
MINE: Probes 1,3,4,5 + gated 6,7. Band = OPT-350m / Pythia-410m ONLY (mechanism dead at 1B).

## BANKED (done)
- **Scale de-confound SETTLED — collision not exodus** (`VERDICT_SCALE_DECONFOUND_2026_06_22.md`): deposit stays curvature-enriched
  at scale (EXCESS_e90 flat both families; rand-null~0 validates analytic null). Raw out-of-V rise = fixed-K/growing-d artifact.
- **G5 scope-limit** (`VERDICT_DSI_2026_06_22.md`): G5 family-organized/structural VALID (RMD-SRC PAYS_STRICT) but does NOT predict
  the dynamic per-step signature (Pythia corr(G5,deflect)=+0.41, wrong sign). Static-valid, dynamical-invalid. Leave logged.
- **Thread-1/2** (earlier): G5 the load-bearing family feature; dissociation irreducible to SHAPE → pretraining bundle {design ∪ recipe};
  length dominant recipe knob (training erodes protection, modest); `VERDICT_THREAD1_G5`, `VERDICT_THREAD2_ASPECT`.

## DSI HARNESS (`cloud/dsi_lambda.py` + `cloud/lambda_launch_dsi.py`, CELL=opt350m|pythia410m)
- Reproduces dispositive regime (lr_child 3e-5/400, lr_depth 5e-5/250, subj, N=30, V=top-128 Fisher). Anchors: OPT B-A/B-D
  +0.089/+0.055; Pythia +0.073/-0.006.
- TWO perturbation bugs found+fixed via anchor gate: (1) arm construction must be FULL-STATE (train phase-2 directly from full
  post-childhood model), NOT carriers-only reconstruct (broke arm D). (2) #4 damage-attrib left model in eval() → DROPPED (re-add
  non-perturbingly later). Clean harness: Pythia anchor PASSES (-0.014); OPT B-D +0.023 (INSIDE 0.066 floor → noise, demoted).
- Arms: A (no deposit) / B (real deposit) / **Bswap (SWAP control: curvature-matched random deposit, matched in-V magnitude)** / D (proj-out).
  Per-step per-carrier #1 cos(g,S_dep) #2 in-V frac #3 survival + gnorm (magnitude-match). #4 damage dropped.

## PRELIM mechanism (NOT banked until swap-gate) — `probe1_carrier.py` on the clean run
Deposit SURVIVES both families (no erase). OPT carrier-SPECIFIC: v/k/fc1 ABSORB, q/out/fc2 DEFLECT. Pythia carrier-UNIVERSAL hard
DEFLECT, inert. **Headline (provisional): load-bearingness is ABSORPTION not deflection** — Pythia deflects harder+uniform yet inert
(deflection adds nothing over non-destructive floor D-A +0.079); OPT load-bearing where carriers ABSORB (non-destructive sinks).

## ★ READ ORDER WHEN SWAP CELLS LAND (pre-committed) ★
1. **SWAP-GATE FIRST = bank point (floor-robust paired differential).** Per carrier, real-B inV vs Bswap inV.
   OPT absorbers (v/k/fc1) real-B != Bswap AND Pythia uniform deflect real-B ~= Bswap → absorb-is-load-bearing LOCKS
   (deposit-specific vs generic-curvature). The 0.066 floor can't fake a paired differential.
2. **OPT anchor magnitude SECOND, DEMOTED.** ~+0.023..+0.055 is inside the floor → do NOT read as systematic, do NOT diagnose
   arm-D, do NOT let it gate the mechanism. B-A + direction carry the dissociation. "at-floor supporting, leans on ladder+vacuums+swap".

## NEXT AFTER SWAP (in order)
- Bank (or kill) the absorb-is-load-bearing inversion on the swap-gate verdict → write to `VERDICT_DSI_2026_06_22.md`.
- Build+fire (validated harness, floor-aware, staggered, no-pipe): **Probe 3** dose-response (childhood {100,200,400,800}, HARD-STOP
  off-band), **Probe 4** disjoint dual deposits, **Probe 5** sequential residual (phase-3 task-C). Each: reproduce-anchor + swap + per-step.
- PARKED floor-aware: dedup×absorb convergence — PAIRED-DIFFERENTIAL form ONLY (magnitude form died at floor). Re-add #4 damage non-perturbingly.
- Probes 6/7 GATED (6: natural-cosine synthetic task-B, NO surgery; 7: two-sided discrimination falsifier or unfired).

## KEY FILES
Harness: `cloud/dsi_lambda.py`,`cloud/lambda_launch_dsi.py`. Analysis: `probe1_carrier.py`,`settle_basis.py`,`rmdsrc_geom.py`,
`aspect_grid.py`,`mediation_depth.py`. Verdicts: `VERDICT_DSI_`,`VERDICT_SCALE_DECONFOUND_`,`VERDICT_THREAD1_G5_`,`VERDICT_THREAD2_ASPECT_`(all _2026_06_22).
Other agent: `universe_metaphor_probes.md` (battery prereg), `basis_invariance_lambda.py`/`basisinv_*.json` (K_E lane), `vacuums_lambda.py`.
