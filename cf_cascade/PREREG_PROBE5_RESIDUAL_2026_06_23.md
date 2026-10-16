# PRE-REG — Probe 5: sequential-phase residual. 2026-06-23. LOCK below.

**Question (pre-committed):** after the phase-2 surge partially erodes the childhood deposit, is the surviving RESIDUAL a
usable prior for a LATER phase (memory) or interference (debris)?

**Band (pre-committed):** OPT-350m, Pythia-410m ONLY (live band; dead at 1B).

**Matched dispositive regime:** lr_child 3e-5 / 400, lr_phase2 5e-5 / 250 (task-B = subj), lr_phase3 5e-5 / 250
(task-C), N=30 paired seeds, V = top-128 task-A Fisher at w_A. Childhood stream = {agnews, dbpedia, yahoo, newscat,
emotion}.

**task-C (pre-committed, LOCKED):** TREC coarse question-type classification (6-way) — structurally distinct from
task-A (SST-2 sentiment) and task-B (subj), and NOT in the childhood stream (not replay). If TREC fails to load, the
probe HARD-STOPs (does not silently substitute).

**Phases:** childhood (deposit) → phase-2 task-B (subj, deposit partially erodes) → measure residual deposit
‖P_{S_dep}(w−w_A)‖ → phase-3 task-C. task-A retention measured AFTER phase-3.

**Arms (per seed, all share the SAME phase-2 endpoint state):**
- **resid**     = phase-3 from the phase-2 end state (residual present)                          → retA_phase3
- **proj**      = phase-3 from phase-2 end with residual deposit PROJECTED OUT of V              → retA_phase3
- **randproj**  = phase-3 from phase-2 end with a curvature-matched RANDOM projection of equal magnitude removed
                  (controls "any projection of that size hurts", not "the residual specifically helps")
- **A_phase3**  = no-deposit baseline carried through the same phase-2+phase-3 schedule          → retA_phase3 baseline

**Reads (per seed → mean ± SE):** retA after phase-3 for {resid, proj, randproj, A}; residual magnitude at phase-2 end;
phase-2 B−D (the reproduce-anchor); childhood_lossdrop.

**Reproduce-anchor (GATES the read):** phase-2 B−A must reproduce banked OPT +0.089 / Pythia +0.073 within 0.03 BEFORE
reading phase-3, else harness invalid.

**Pre-committed outcomes:**
- **Memory** → resid retA_phase3 > proj retA_phase3 AND resid > randproj (the eroded deposit still guides protection in a
  new phase, specifically — not just "removing anything hurts").
- **Debris** → resid ≤ proj (or resid ≈ randproj): residual is interference / inert, not a prior.

**Controls:** projecting the residual out is itself a perturbation → the randproj arm matches it with an equal-magnitude
curvature-matched random projection (so "residual helps" is not "any projection hurts"). task-C distinct from A and B
(no replay). Swap/random-proj on any phase-3 effect.

**Floor discipline:** resid−proj and resid−randproj are within-run paired (same seeds, same phase-2 state) → floor-robust;
demote absolute phase-3 retentions.

Harness `cloud/probe5_residual_lambda.py` + launcher `cloud/lambda_launch_generic.py probe5`. CELL ∈ {opt350m, pythia410m}.
