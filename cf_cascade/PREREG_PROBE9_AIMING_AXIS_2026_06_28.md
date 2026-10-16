# PRE-REG — Probe-9: specificity aiming-axis (geometric overlap vs semantic similarity). 2026-06-28.

Status: to be SHA256-LOCKED on write (`PREREG_PROBE9_AIMING_AXIS_LOCK.json`). NO GPU until explicit go/no-go after the
dry-run gate. Successor to Probe-8 v4 (`VERDICT_PROBE8_v4_SPECIFICITY_2026_06_28`, LOCK 0112d69b) — reuse its harness spine.
Post-floor discipline: paired within-run differentials are load-bearing; absolute levels read against the ~0.02 floor.

## Question
Probe-8 v4 found a specificity dissociation: OPT (load-bearing) aims its protection at the phase-2 interference (S>0),
Pythia (inert) is generic (S≈0). Is that keyed to ARCHITECTURE, or is "architecture" a mislabel for **geometric overlap
Ω** between what the childhood reorganizes and the phase-2 interference direction — with **semantic task-similarity σ_sem**
as the competing axis? Engages Steele (arXiv 2603.02224, F=α(1−cos²θ_min)+β subspace-geometry law) on his own quantity
θ_min, and separates from Ramasesh (arXiv 2007.07400, semantic ∩-shape) by holding task-A↔phase-2 fixed and putting σ_sem
in a partial-correlation horse-race against Ω.

## Cells & fixed factors (locked)
Cells: **opt350m** (w_A=0.823), **pythia410m** (w_A=0.753). **task-A=SST-2 and phase-2=subj held FIXED across all cells
and streams** (so the Ramasesh task-A↔phase-2 ∩-shape cannot contaminate; only childhood↔phase-2 varies). V = top-128
task-A Fisher dirs per carrier @ w_A. Carriers as in the arc (OPT q/k/v/out_proj,fc1,fc2; Pythia qkv,h_to_4h,4h_to_h).
fp32. N≥100 for the S statistic.

## Independent variable — geometric overlap Ω (locked definition)
Per cell, the **phase-2 interference subspace** is computed ONCE from M_Ω=12 seeds of phase-2(250)-from-w_A (the canonical
interference on the bare deposit); childhood subspace is per-stream from M_Ω=12 seeds of childhood-from-w_A. Both
displacements projected into V and expressed in V-coordinates (Δ@V per carrier). Overlap via **principal angles between the
two M_Ω-seed subspaces**, computed memory-light from the cross-Gram (Frobenius inner products are additive over carriers):
  G_cc[i,j]=Σ_carr ⟨Δc_i@V, Δc_j@V⟩_F ; G_pp, G_cp likewise. Principal cosines = singular values of
  G_cc^{-1/2} G_cp G_pp^{-1/2}.  **Report cos²θ_min = (max principal cosine)² (Steele's quantity) AND mean cos²θ over the
  spectrum (robustness).** Ω ≜ cos²θ_min for the primary S·Ω test; mean cos²θ reported alongside.

## Ladder construction (Stage-0 screen + locked selection rule)
Candidate childhood-stream pool (locked): the 5 singles {agnews, dbpedia, yahoo, newscat, emotion}, three pairs
{agnews+dbpedia, yahoo+emotion, newscat+dbpedia}, and the full-5 mix = **9 candidates**. Stage-0: measure Ω (cos²θ_min) for
each candidate per cell at M_Ω=12. **Selection rule (deterministic):** order candidates by Ω; pick the 5 whose Ω values are
nearest to the 5 evenly-spaced quantiles {0,0.25,0.5,0.75,1.0} of the observed candidate-Ω range (ties → more seeds /
lower lr_child). The 5 selected streams + their Ω are the locked ladder for Stage-1. (Lock the RULE; data picks the streams
— same pattern as Probe-7b calibration.)

## Competing IV — semantic similarity σ_sem (locked)
Per cell, per stream: **linear CKA** between mean-pooled last-hidden-states of the FROZEN pretrained base on a fixed
N_SEM=256 sample of the stream's inputs vs the subj (phase-2) inputs. Measured, NOT used for selection.

## Dependent variable — S (locked, paired, floor-robust)
S = N(phase2) − N(alttask), N(d) = mean_seed( dprop_NoCH(d) − dprop_CH(d) ), dprop(d)=acc_A(anchor)−acc_A(anchor+dir).
**Probe-8 v5 relabel — BOTH phase2 AND alttask directions are REALIZED short-training displacements** (alttask = realized
displacement of STEPS_ALT=60 steps training on TREC from the anchor, projected to V, magnitude-matched to ‖Δ_V(phase2)‖) —
kills the v4 one-shot-gradient asymmetry so Ω and S are commensurable. randtop16 retained as the curvature baseline. N=100,
fp32, paired. Per stream, per cell.

## Confound control — per-stream non-destructive recalibration (locked, gate-checked)
For EACH selected stream, calibrate lr_child (grid [1e-5,1.5e-5,2e-5,3e-5], K_CAL=20, margin 0.015 over thr=w_A−0.05; same
rule as v3/v4) so the childhood leaves post-childhood task-A within ±0.05 of w_A. **A stream-cell whose full-N post-childhood
retention falls outside the band is VOID-soft and re-run** (this is the exact confound that VOID-soft'd Probe-8 v1/v2 — do
not skip).

## Pre-registered predictions & verdict-shapes (locked BEFORE outcome)
- **P1 (axis is geometric):** for OPT, S is monotone increasing in Ω across the ladder. **Threshold: Spearman ρ(S,Ω) ≥ 0.70
  AND per-cell S significant (the high-Ω streams' S ≥ 3σ paired).** Predicted BOUNDARY: a finite **Ω\*** within the swept
  range below which OPT S is indistinguishable from 0 (S CI includes 0). Report Ω\* with 95% CI (bootstrap over seeds, 1000
  resamples; Ω\* = interpolated Ω where the S-vs-Ω fit's lower CI crosses 0).
- **P2 (architecture-vs-unification fork):** for Pythia, EITHER S flat in Ω (Spearman |ρ|<0.3, no stream S≥2σ → architecture
  is the right axis; the v4 dissociation STANDS) OR S∝Ω like OPT (ρ≥0.70 → "architecture" was a mislabel; dissociation
  unifies into overlap×absorption). Both pre-committed.
- **P3 (geometric vs semantic discrimination):** partial correlations across the pooled ladder. If **partial ρ(S,Ω | σ_sem)
  significant AND partial ρ(S,σ_sem | Ω) not** → aiming axis is GEOMETRIC, not semantic (separates from Ramasesh). Reverse →
  semantic. Both significant → mixed. All three pre-committed.

**Verdict mapping (RDM-SRC):**
- **RECOVERED-on-axis:** P1 holds AND Ω\* found → specificity mechanism explained; geometric aiming axis confirmed.
- **RE-AXIS RECOVERY:** P2 unifies (Pythia S∝Ω) → the OPT/Pythia dissociation collapses into overlap×(DSI-absorption).
- **BOUNDARY:** P1 fails for OPT (no monotone S·Ω) → geometric hypothesis falsified by commitment; go to re-decomp set.

## Bounded re-decomposition set + stopping rule (locked)
If P1 fails for OPT (geometric IV does not fire), run IN ORDER, stop after (2):
  (1) **σ_sem as the aiming axis** — Spearman ρ(S, σ_sem) with the same thresholds.
  (2) **absorption-weighted overlap** — Ω with per-carrier contributions weighted by the DSI deposit-absorption
      (`VERDICT_DSI`: OPT v/k/fc1 absorb +0.17/+0.09/+0.07, q/out/fc2≈0; Pythia all-deflect) — i.e. does overlap *in the
      absorbing carriers* predict S where raw Ω did not.
Exhaustion across {Ω, σ_sem, absorption-weighted Ω} = BOUNDARY with the structural cause named (RDM-SRC discipline).

## Validity gates (VOID / flag)
- w_A anchor reproduce (OPT 0.823±0.02, Pythia 0.753±0.02). VOID if off.
- per-stream childhood_nondestructive (full-N) within ±0.05 of w_A → else VOID-soft + recalibrate.
- bite gate: mean dprop_NoCH(phase2) AND dprop_NoCH(alttask) > 0.03 (realized dirs must damage). Else null_dead flag.
- datasets (SST-2, subj, TREC, 5 childhood streams) load — HARD-STOP (VOID) on failure.

## Ops (hard rules, arc §8)
SHA256-lock this PREREG before any GPU. `GPU_PREF="gpu_1x_a100_sxm4,gpu_1x_a100,gpu_1x_gh200,gpu_1x_h100_sxm5"` (a10
excluded; if only a10 free WAIT / use the capacity watcher). Arm `killswitch_smart.py` on every fire (10-min stall, 20h
backstop). Per-step heartbeat in the harness; `[hb]`/`[cal]`/`[omega]`/`[seed]` lines. Read progress OFF the machine.
Two-phase orchestration: **MODE=screen** (Stage-0 Ω over the 9 candidates, one short instance per cell) → apply the locked
selection rule → **MODE=measure STREAM=<s>** (Stage-1 full paired-S + σ_sem for one selected stream); fan out the 5×2
stream-cells as separate instances (each ~Probe-8v4-sized, ~3h) to stay well under the 20h backstop. Pull all JSONs to
gateA/. Analysis (Spearman, partials, Ω\*) is a local CPU step over the landed JSONs (no GPU).

## Cost
Stage-0 screen: ~9 candidates × (12 childhood + shared 12 phase-2) per cell ≈ ~1h/cell. Stage-1: 5 streams × 2 cells ×
(~3h) — fan out in parallel ≈ ~3–4h wall, ~$60–90 total. Analysis local/free.

## Deliverables / STOP GATE (this run)
1. This PREREG, SHA256-locked.  2. The lock JSON.  3. `cloud/probe9_aiming_lambda.py` built to match (screen+measure modes,
Ω cross-Gram, realized-alttask, CKA σ_sem, per-stream recalibration, paired-S) + a CPU dry-run exercising the Ω-measurement,
the paired-S estimator, and the per-stream retention gate on a tiny synthetic shard. THEN STOP, print the lock hash and the
dry-run output, and request go/no-go. Do NOT fire the dual-cell GPU run until confirmation in a follow-up.
