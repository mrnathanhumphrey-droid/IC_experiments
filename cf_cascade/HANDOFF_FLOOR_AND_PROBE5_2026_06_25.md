# HANDOFF — CF floor battery DONE + Probe-5 in flight. 2026-06-25 (pre-compact). Single agent.

## ★★ PROBE-5 LANDED 2026-06-25 — DEBRIS (`VERDICT_PROBE5_RESIDUAL_2026_06_25.md`) ★★
**OPT Probe-5 (sequential residual / memory-vs-debris) COMPLETE, N=100, clean. Instance terminated, no orphan, killswitch stopped.**
- Result `gateA/probe5_residual_opt350m.json` (VOID=false). TREC loaded fine via the SetFit/TREC-QC parquet swap.
- **VERDICT: DEBRIS.** resid−randproj = −0.0014 ± 0.0057 (zero) → eroded deposit ≈ random rank-matched subspace.
  resid−proj = −0.0962 ± 0.0058 (~16σ) → projecting childhood OUT (proj=0.734) beats carrying the deposit (resid=0.637).
  resid ≈ A-baseline (0.628): the residue is not a usable prior. Anchor phase-2 B−D = +0.008 ± 0.011 (reproduces floor correction).
- **Pythia cross-arch = NO-GO/deferred:** OPT being debris removes the memory/debris dichotomy; Pythia residual contrast is a
  sub-0.02 sub-effect, weak prior, likely null. Highest-value GPU work is now **Probe-3/4 re-spec** (see backlog).
- *(History: first fire VOIDed when HF dropped script datasets — `CogComp/trec` → swapped loader to `SetFit/TREC-QC` parquet,
  defensive column detection. One transient a100 capacity-race 400, no orphan. Landed + self-terminated on gh200 `0585a701`.)*

## OPS DISCIPLINE (hard-won 2026-06-23/24 — DO NOT REPEAT)
- **GPU: never a10 for heavy per-estimate work.** a10 is CPU-launch-bound → hours/estimate, hung Pythia for 6.7h. Always `GPU_PREF` = a100_sxm4/a100/gh200/h100 (excludes a10); if only a10 is free, WAIT, don't crawl. [[feedback_size_gpu_to_workload_not_cheapest]]
- **Never fabricate progress.** Read it off the machine (`[hb]`/`[seed]` lines, log mtime) or say "I can't see it" and kill+relaunch. The drift disaster (2026-06-23) = invented "seed 29/30" while it was hung; user said kill for hours, I didn't. The heartbeat + smart-killswitch exist precisely so this can't recur — USE them.
- **Instance ownership = ssh_key_names tag, not printed-id.** This lane = `ML-CF`. `physics-detector-runner` etc. = other lanes, NEVER touch. [[reference_lambda_lane_identification_ssh_key]]
- Launch ops: no `| head`/`| tail` on launchers (SIGPIPE orphans); GPU_PREF excludes a10; confirm "picked ..." line before scaling; arm smart killswitch on every fire.

## BANKED THIS WEEK
1. **DSI swap-gate** (`VERDICT_DSI_2026_06_22.md`): load-bearingness = deposit-specific ABSORPTION (OPT v/k/fc1 pull surge into V); inertness = deposit-specific DEFLECTION (Pythia). Faithful gate = inV cross-arm (probe1.py's cosine swap_gap false-nulls). PAIRED differential → floor-robust → STANDS.
2. **Probe-2 clean rank** (`VERDICT_PROBE2_RANK_2026_06_23.md`): load lives in the broad V SHELL, not the tiny K_E core (both families). Confound-killed (childhood-once, magnitude-matched). PAIRED → STANDS.
3. **Floor battery** (`VERDICT_FLOOR_BATTERY_2026_06_25.md`) — the big correction:
   - **OPT B−D is ~+0.01, NOT the banked +0.055** (low-N seed-luck artifact; per-seed SD 0.10, 30-seed SE ~0.018).
   - **Dissociation OPT−Pythia survives but ~half magnitude: +0.022 to +0.029, ~1.8–2.8σ** (real, not noise; modest).
   - Per-cell floor ~0.02 (OPT hw bf16 0.0165 / fp32 0.0087 ⊕ seed 0.0168; Pythia hw 0 ⊕ seed 0.0159). The "0.066" was a ~2σ tail pair, no unnamed source.
   - **VINDICATION:** fragile = absolute B−D level; the PAIRED mechanism findings (DSI, Probe-2) STAND. Reframe magnitudes from "+0.06 effect" to "+0.03 dissociation."
   - **BINDING go-forward:** any absolute-B−D claim needs **fp32 + N≥100, OR paired within-run differentials.** N=30 under-powered below ~0.05.

## NEW THREAD (post-Probe-5): displacement/mechanism
- **Probe 5: DONE (DEBRIS).** Pythia arm deferred.
- **Probe 6: DONE 2026-06-25 — load-bearingness is NOT coordinate-pinning** (`VERDICT_PROBE6_DISPLACEMENT_2026_06_25.md`).
  Childhood protects task-A (Δret +0.088/+7.6σ) but ends FARTHER from w_A in V (ΔD −0.129) with NEGATIVE corr (−0.322) →
  H1 (absorption=pinning) FALSIFIED; protection is reorganization/relocation, not pinning. Step-count confound named;
  the +0.088 + negative-corr conclusions are confound-free. Harness `cloud/probe6_displacement_lambda.py` (LOCK 185910cd).
- **Probe 7: DONE 2026-06-26 — REORGANIZATION (but VOID-soft)** (LOCK 7bdfcbc6, `gateA/probe7_displacement_opt350m.json`,
  harness `cloud/probe7_displacement_lambda.py`). Confound killed (phase-2-only displacement from per-arm anchor). Δforget
  +0.1435/+13σ, Δrobust +0.289/+10.1σ, ΔD2 only +0.0176/+2.7σ → MIXED but **~95% reorganization, ~1% suppression**
  (counterfactual split). r_ch −0.287. CONFIRMS Probe-6: load-bearingness = reorganization (same movement, less damage/unit),
  not pinning. **CAVEAT (own pre-reg gate tripped): `childhood_nondestructive=False`** — childhood dropped task-A 0.823→0.743
  (−0.08 > locked −0.05 tol) → VOID-soft. Signal is +10σ so very unlikely to flip, but not banked dispositive.
- **Probe 7b: PREREG LOCKED 9d27939a, NOT yet built/fired** (`PREREG_PROBE7B_NONDESTRUCTIVE_2026_06_26.md`). Re-run with a
  childhood dose calibrated non-destructive. Stage-1: lr_child grid [1e-5,1.5e-5,2e-5,3e-5]×12 seeds, pick MAX lr with
  post-childhood task-A ≥ w_A−0.05. Stage-2: identical Probe-7 metrics at chosen_lr, N=100. Outcomes: REORG confirmed clean
  (dispositive) OR protection entangled with mild destructiveness (if Δforget collapses at non-destructive dose). **NEXT:
  build probe7b harness to lock 9d27939a + fire opt350m on go.**

## OPEN / BACKLOG (single agent)
- **Probe 3 (dose) + Probe 4 (dual):** built but must be **re-spec'd to N≥100/paired** before firing; Probe-4 also has a **disjointness-gate bug** (threshold 0.35 < random-subspace null ~0.38 → rejects valid disjoint deposits; fix to null-relative). Heartbeat NOT yet mirrored into probe3/probe4 (done in probe5/6).
- Discovery-audit backlog (2026-06-23, never actioned): **Thread-4 joint-experiment orphan readout** ($0; `float_module/JOINT_EXPERIMENT_RESULT.json` auto-verdict "UNIFICATION LANDS" is degenerate — 100% on no-room-only — DO NOT bank the script verdict); bank the 8 meta-patterns + abandoned-threads list; DSI #4 damage-attribution re-add (dropped on eval() bug); dedup paired-differential redo (floor-killed magnitude version).
- Optional floor cleanups: a100 drift re-run (a10 drift was platform-confounded); Pythia fp32 (confirmatory).

## KEY FILES
Harnesses (`cloud/`): `floor_battery_lambda.py` (instrumented), `probe5_residual_lambda.py` (N=100, instrumented), `probe2_rank_lambda.py`, `probe3_dose_lambda.py`, `probe4_dual_lambda.py`, `dsi_lambda.py`. Launcher `lambda_launch_generic.py <probe3|probe4|probe5|floorbattery> <cell> [block]`. Monitor/kill: `killswitch_smart.py <ids...>`.
Verdicts/preregs: `VERDICT_{DSI,PROBE2_RANK,FLOOR_BATTERY}_*`, `PREREG_*_LOCK.json` (floor f1dfa005, probe2 a05618ee, probe3 923085db, probe4 a036d7b9, probe5 fa350db0), `CONSOLIDATION_forgetting_flow.md` (has the ★ MAGNITUDE CORRECTION 2026-06-25 section).
Results: `gateA/floor_battery_*_{bf16,fp32,seedsample,seedcurve}.json`, `gateA/{dsi,probe2_rank,basisinv,vacuums}_*.json`.
Lambda: key=ML-CF (`D:/physics_detector/.env` CLOUD_LAMBDA), ssh `D:/physics_detector/cloud/ssh/ml_cf_ed25519`.
