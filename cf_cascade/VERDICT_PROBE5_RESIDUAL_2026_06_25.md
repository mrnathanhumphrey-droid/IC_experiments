# VERDICT — Probe-5 sequential residual (memory-vs-debris). 2026-06-25.

Pre-reg `PREREG_PROBE5_RESIDUAL` (LOCK fa350db0). N=100 seeds, fp32, gh200 (ML-CF),
per-step heartbeat + smart killswitch. Result `gateA/probe5_residual_opt350m.json` (VOID=false).
This is the first probe fired under the binding go-forward rule (N≥100 + paired differentials)
from the floor battery — it satisfies both.

## The question
After task-A is deposited (arm B) and then eroded by phase-2 depth training, is the **eroded
deposit a usable prior** for a new task-A retraining (resid arm) — i.e. is there recoverable
*memory* in the residue — or is it **debris** (no better than a random subspace of equal rank)?

Paired arms from the SAME phase-2 end-state w2 (floor cancels):
- **resid** — run phase-3 from w2 (eroded deposit present), then measure task-A.
- **proj** — remove the in-V residual deposit `P_V(w2−w_A)` from w2 first — i.e. **snap the task-A-subspace
  coordinates back to the original task-A solution w_A** (everything outside V stays at w2) — then phase-3.
- **randproj** — remove an equal-magnitude curvature-matched RANDOM projection from w2, then phase-3.
- **A** — train task A fresh, no deposit (baseline).

Decision rule (pre-committed): **MEMORY** if resid > proj AND resid > randproj; **DEBRIS** if
resid ≤ proj or resid ≈ randproj.

## Result (OPT-350m, N=100) → DEBRIS, unambiguous
w_A anchor 0.823 (reproduced exactly). phase-2 B−D = **+0.008 ± 0.011** — reproduces the
floor-battery correction (expected-of-record was the stale +0.055), self-validating the run.

| arm | phase-3 task-A retention | SE |
|---|---|---|
| proj (task-A V-coords restored to w_A) | **0.7336** | 0.0034 |
| randproj (random rank-matched) | 0.6387 | 0.0064 |
| resid (eroded deposit) | 0.6373 | 0.0055 |
| A (fresh, no deposit) | 0.6282 | 0.0089 |

Paired contrasts:
- **resid − proj = −0.0962 ± 0.0058 (~16σ negative)**
- **resid − randproj = −0.0014 ± 0.0057 (statistically zero)**

Stable since ~n=45; the back half (n=50→100) did not move the means.

## Reading
1. **The eroded deposit ≈ a random projection of equal rank** (resid−randproj = −0.001). Whatever
   task-A structure survived phase-2 carries **no recoverable signal** beyond its dimensionality.
   It is debris, not memory.
2. **resid ≈ A baseline** (0.637 vs 0.628): carrying the eroded deposit forward buys essentially
   nothing over just retraining the task from scratch.
3. **What helps is RESTORATION, not reuse:** proj — snapping the task-A V-coordinates back to w_A —
   sits +0.10 above all others (resid−proj = −0.096, ~16σ). This says the task-A information lives in
   V and phase-2 *displaced* those coordinates away from w_A; resetting them recovers task-A. Note this
   is not a free mitigation — it presupposes you already hold w_A and V. Its value is mechanistic: the
   damage is **coordinate displacement in V**, and the surviving displaced deposit (resid) is debris,
   whereas the original w_A coordinates are what carry task-A.

## How this fits the program
- Consistent with **Probe-2** (load lives in the broad V SHELL, not a reusable core) — there is no
  compact reusable "memory core" to recover, so a residue probe finding debris is the expected
  complement, not a contradiction.
- Consistent with the **floor-battery reframe**: the absolute deposit level is fragile/small; the
  mechanism is about *interference geometry* (what to remove), not a durable stored prior.
- The load-bearingness story is **absorption/deflection of the childhood gradient** (DSI), NOT a
  persistent retrievable memory of task A. Probe-5 closes the "is the residue a prior?" thread: **no.**

## Cross-arch (Pythia-410m) — GO/NO-GO
The original cross-arch test ("OPT memory vs Pythia debris") is **moot**: OPT is already debris, so
there is no memory/debris dichotomy to dissociate. The only remaining contrast is narrow:
does inert Pythia go resid **below** randproj while load-bearing OPT sits **at** randproj?

- Expected effect size: ≤ ~0.02 (both arms near the random floor), i.e. **inside the ~0.02 per-cell
  floor** characterized by the battery. N=100 paired gives SE≈0.006, so it is *measurable* if the
  separation is real — but the prior that it exists is weak, and a null is the likely outcome.
- Cost: ~1 gh200 run (~$15–20, ~6h).

**Recommendation: NO-GO for now (or defer/low-priority).** Probe-5's load-bearing question is
answered for OPT (debris); the Pythia arm tests a sub-0.02 sub-effect with a weak prior. Spend the
next GPU cycle on the **Probe-3/4 re-spec** (higher-value: dose-response and dual-deposit, both need
N≥100/paired and Probe-4 needs the disjointness-gate fix) rather than a likely-null confirmatory
Pythia run. If Probe-3/4 surface something that hinges on residue-reuse, revisit Pythia Probe-5 then.

## Files
Result `gateA/probe5_residual_opt350m.json`. Harness `cloud/probe5_residual_lambda.py` (N=100,
instrumented; TREC loaded via `SetFit/TREC-QC` parquet mirror after HF dropped script datasets).
Prereg `PREREG_PROBE5_RESIDUAL` (LOCK fa350db0). Floor context
`VERDICT_FLOOR_BATTERY_2026_06_25.md`; related `VERDICT_PROBE2_RANK_2026_06_23.md`,
`VERDICT_DSI_2026_06_22.md`. Instance `0585a701` terminated clean (no orphan); cost ~$13.
