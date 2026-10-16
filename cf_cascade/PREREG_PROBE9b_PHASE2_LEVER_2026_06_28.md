# PRE-REG — Probe-9b: Ω sweep via the PHASE-2 lever (geometric vs Ramasesh-semantic, head-to-head). 2026-06-28.

To be SHA256-LOCKED (`PREREG_PROBE9b_PHASE2_LEVER_LOCK.json`). Successor to Probe-9 (screen showed Ω is NOT dialable by
childhood-stream choice — `VERDICT_PROBE9_SCREEN_2026_06_28`). Inherits the post-floor discipline + the Probe-9 machinery
(Ω cross-Gram, per-stream non-destructive calibration, CKA, paired estimators, abort-gate) verbatim where applicable.

## The pivot
Ω is set by the INTERFERENCE direction = the **phase-2 task**, not the childhood. So sweep Ω by varying phase-2 (FIXED
childhood + fixed task-A). This also means task-A↔phase-2 SEMANTIC similarity (the **Ramasesh** axis) now varies WITH the
phase-2 task — so σ_sem becomes the *direct competing axis*, and the partial-correlation horse-race S/P · Ω | σ_sem vs
S/P · σ_sem | Ω is now a HEAD-TO-HEAD geometric-vs-Ramasesh test (stronger than the moot childhood-lever version).

## Design (locked)
Cells: opt350m (w_A 0.823), pythia410m (w_A 0.753). FIXED: task-A=SST-2; childhood = mix5 at the per-cell non-destructive
lr (Probe-9 margin calibration, K_CAL=20). V = top-128 task-A Fisher @ w_A. fp32, N=100 paired.
**Vary phase-2 task d over a pool** (locked): {subj, agnews, dbpedia, yahoo, trec, emotion, newscat} + combos as needed to
span Ω (≥7 points). For each d:
- **Ω(d)** = cos²θ_min principal-angle overlap (cross-Gram, M_Ω=12) between the FIXED-childhood displacement subspace and
  the phase-2(d)-from-w_A displacement subspace, in V. (Childhood subspace computed ONCE; phase-2 subspace per d.)
- **DV = protection P(d)** = forget_NoCH(d) − forget_CH(d), paired, N=100 (Probe-7 Δforget form: childhood's reduction of
  task-A forgetting under phase-2 task d). Floor-robust paired differential.
- **σ_sem(d)** = linear CKA(frozen base on phase-2(d) inputs, task-A inputs) = task-A↔phase-2 similarity (Ramasesh axis).

## Stage-0 abort-gate (locked, same discipline)
Screen Ω(d) for the phase-2 pool per cell. Require span ≥0.15 and ≥7 non-degenerate points → else VOID-soft (the phase-2
lever ALSO fails to span → report and rethink the Ω quantity itself). If healthy → measure P(d) for all spanning d.

## Pre-registered predictions / verdict (locked)
- **P1 (Ω governs protection):** Spearman ρ(P, Ω) ≥ 0.70 (n≥7, crit ~0.71 at n=7 / 0.60 at n=9 — report exact), with a
  crossover **Ω\*** above which P>0 (reorganization available) and below which P≈0; Ω\* with 95% CI (bootstrap).
- **P2 (geometric vs Ramasesh-semantic, head-to-head):** partial ρ(P, Ω | σ_sem) significant AND ρ(P, σ_sem | Ω) not →
  protection regime is GEOMETRIC, not semantic similarity (beats Ramasesh on its own axis). Reverse → semantic. Both → mixed.
- **P3 (cross-arch):** OPT shows the Ω-governed crossover; Pythia flat/absent (architecture modulates the *availability* of
  reorganization even when Ω is matched) OR Pythia also Ω-governed (unify). Pre-committed.
- Verdict map: Ω-LAW-CONFIRMED / RAMASESH-WINS / MIXED / BOUNDARY (phase-2 lever also can't span → Ω quantity rethink).

## Three-substrate Ω law
Probe-9b = the continuous-Ω spine (OPT/Pythia, phase-2 swept) joining the two banked anchor points: Librarian (free-lane,
Ω 0.046, protection holds) and Deepfake (antagonistic cos −0.99, suppression-only, no reorg). Axis = Ω, not architecture →
robust to the n=2 criticism.

## Ops (inherited)
SHA256-lock before GPU. GPU_PREF excludes a10; killswitch on every fire; heartbeat; read off the machine. Two-phase:
MODE=screen (phase-2 Ω, one short instance/cell, abort-gate) → MODE=measure per spanning phase-2 task (fan out). Harness =
new variant `probe9b_phase2lever_lambda.py` (reuse probe9 spine; swap the loop from childhood-streams to phase-2-tasks; DV =
paired protection P(d); σ_sem = task-A↔phase-2 CKA). Build + CPU dry-run + abort-gate screen BEFORE the measure fan-out.

## Status
NOT built/fired. Next: build `probe9b_phase2lever_lambda.py` to this lock + dry-run + screen (abort-gate) on go.
