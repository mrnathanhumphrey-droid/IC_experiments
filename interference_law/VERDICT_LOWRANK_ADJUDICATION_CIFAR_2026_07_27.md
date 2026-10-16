# VERDICT — #4 Low-rank adjudication, VISION powered CIFAR+DTD (dichotomy + dispersion mechanism CONFIRMED, cross-dataset)

- **Date:** 2026-07-27
- **Pre-reg:** `PREREG_LOWRANK_ADJUDICATION_2026_07_27.md` (LOCK `b5c3929a`)
- **Artifacts:** CIFAR `out/lowrank_adjudication.json` + `out/lowrank_rows.jsonl`; DTD `out/lowrank_adjudication_dtd.json` + `out/lowrank_rows_dtd.jsonl`; logs `out/lowrank_adj_full.log`, `out/lowrank_adj_dtd.log`; script `lowrank_adjudication_lambda.py`; local smoke `probe_lowrank_dirsmoke.py` (`out/lowrank_dirsmoke.json`).
- **Compute:** Lambda a10, resnet18 full-FT, 3 seeds × N∈{6,12,24}, FT=(300, 5e-4). CIFAR wall 2945s, DTD 3846s. Arc ~$3.8 (incl 2 diagnostic smokes).

## Claim tested
Reconcile **2606.18024** "Catastrophic Forgetting is Low-Rank" (single-task loss in a low-rank kernel eigenspace) with the repo's `out/lowrank.json` (pairwise interference **interaction NOT low-rank completable**, eff-rank ~9, flat spectrum). Synthesis: **each task's forgetting operator is low-rank, but the map (source,target)→forgetting is high-rank because per-pair forgetting DIRECTIONS are DISPERSED.**

## Result (3 seeds, bootstrap 95% CI) — CIFAR
| N | per-task eff-rank | interaction eff-rank | pooled dir eff-rank | mean\|cos\| |
|---|---|---|---|---|
| 6  | 1.71 [1.59,1.78] | 2.14 [2.05,2.22] | 7.65  | 0.212 |
| 12 | 2.10 [1.91,2.29] | 4.70 [4.26,4.96] | 13.20 | 0.175 |
| 24 | **2.00 [1.94,2.09]** | **9.57 [9.18,9.90]** | 17.54 [16.85,18.41] | 0.166 |

Scaling slopes: per-task **+0.013 (flat)**, interaction **+0.412**. Haar per-task ceiling 21.97 (obs 2.0 = 11× below random); `perm_p_pt_below = 0.0`. interaction 9.57 **reproduces** `lowrank.json` (9.13).

## Result (3 seeds, bootstrap 95% CI) — DTD (cross-dataset replication)
| N | per-task eff-rank | interaction eff-rank | pooled dir eff-rank | mean\|cos\| |
|---|---|---|---|---|
| 6  | 1.89 [1.76,2.01] | 2.32 [1.76,3.04] | 8.09  | 0.215 |
| 12 | 2.15 [2.02,2.36] | 4.85 [4.57,5.09] | 15.55 | 0.163 |
| 24 | **2.50 [2.17,2.75]** | **9.31 [8.41,10.19]** | 20.88 [20.05,21.47] | 0.153 |

Scaling slopes: per-task **+0.033 (flat)**, interaction **+0.386**. Haar per-task ceiling 21.97 (obs 2.5 = ~9× below random); `perm_p_pt_below = 0.0`. interaction 9.31 **reproduces** `lowrank.json` DTD (8.45).

## Falsifier check (pre-registered) — BOTH CLEARED on BOTH datasets
- **KILL "no dichotomy"** (per-task ≥6 ≈ interaction): NOT fired. per-task ≪ interaction (CIFAR 4.8×, DTD 3.7×), non-overlapping CIs.
- **KILL "mechanism wrong"** (per-pair directions aligned): NOT fired. mean|cos| ~0.15-0.17, per-task ~9-11× below Haar, perm-p 0 both.
- Honest nuance: DTD per-task creeps 1.89→2.50 (slope +0.033 vs CIFAR +0.013) so it is "flat-ish" not perfectly flat, and the separation is tighter (3.7×). The qualitative signature (interaction grows ~linearly, per-task stays low & far below random) replicates decisively.

## Verdict
**CONFIRMED, cross-dataset (CIFAR + DTD, powered).** Both papers are right about different objects: single-task forgetting IS low-rank (~2, flat across N), and the interference map IS high-rank (~9.5) *because* the low-rank per-task subspaces disperse across pairs (pooled 17-21, mean|cos| 0.15-0.17). The **scaling signature** (per-task flat while interaction grows ~linearly, ~12× steeper) is the decisive new evidence that dispersion — not per-task complexity — drives the high-rank map. Reconciles **2606.18024** ("forgetting is low-rank" — true per-task) with the repo's high-rank interference (true for the map).

## Scope / open rungs (NOT yet done)
- **Plastic regime (2nd substrate):** LoRA/pythia — needs new direction-capture on the reshaping-backbone path (P2 cache stores scalars, not raw activations). Separate rung; the only remaining generality question.
- Frozen readout here = full-FT feature-space (pos−neg) separation shift; per-task ~2-2.5 is well above the trivial rank-1 a *purely* frozen linear readout would give, so the operator is non-degenerate.
- Frozen readout here = full-FT feature-space (pos−neg) separation shift; per-task ~2 is well above the trivial rank-1 the pre-reg feared for a *purely* frozen linear readout, so the operator is non-degenerate.

## Infra notes (Lambda CIFAR data-loading, cost the 2 smokes)
- torchvision CIFAR100 download on Lambda is pathologically slow (66% of 169MB in 1800s → watchdog). Use HF `datasets`.
- HF `datasets==2.21.0` image decode needs Pillow with `Image.ExifTags` (≥9.4); Ubuntu system Pillow 9.0.1 lacks it and a `Pillow<11` pin is *satisfied* by it → force `Pillow==10.4.0 --upgrade --user`. (Resolves the memory's "CIFAR HF/PIL bug".)
