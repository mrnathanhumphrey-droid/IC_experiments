# VERDICT — Break the n=1: ALiBi×multilingual "childhood erodes retention" is BLOOM-idiosyncratic

**Date:** 2026-07-26 · **Lane:** ML-CF (Lambda A10) · **Harness:** `cloud/probe13_doseresponse_lambda.py` + `build_wA_lambda.py`

## The finding under test (n=1)
Probe-13 metric **N_phase2** = (task-A damage from interference WITHOUT a childhood warmup) − (WITH childhood), projected into the task-A Fisher top-128 subspace. `+` = childhood **protects**; `−` = childhood **erodes**. Swept childhood-LR (dose).
Published result: **BLOOM-560m (ALiBi+multilingual) was the ONLY eroder** among 6 cells (Falcon-rw-1b ALiBi+English, XGLM-564m sinusoidal+multi, OPT/Pythia English all PROTECT). BLOOM = the unique ALiBi×multilingual corner ⇒ **n=1**. Erosion was thin (published min −0.032 ± 0.034 ≈ 0.9σ; signal = sign+slope, not tight z).

## The test
Add a **2nd ALiBi+multilingual family**: `inceptionai/jais-family-590m` (public/base, ALiBi via config `position_embedding_type: alibi`, Arabic+English, Inception lineage, GPT-2-style **Conv1D** carriers). Scale-matched to BLOOM-560m. Plus a **BLOOM control** re-run to validate the harness under the same pins/seeds.
Pre-registered read: **Jais goes negative at high dose → n=2 conjunction LAW. Jais stays ≥0 → BLOOM-IDIOSYNCRATIC, KILLED.**

## Result (N_phase2 ± SEM; BLOOM control N=16, JAIS N=17; 5h watchdog cut both at seed 16–17)

| dose | BLOOM published (N=25) | BLOOM control (this run) | JAIS |
|---|---|---|---|
| 3e-7 | +0.048 ± 0.027 | +0.015 ± 0.037 | +0.010 ± 0.004 |
| 1e-6 | +0.030 ± 0.024 | **−0.083 ± 0.039** (−2.1σ) | −0.0004 ± 0.0035 (~0σ) |
| 2e-6 | +0.024 ± 0.021 | −0.068 ± 0.039 | +0.0004 ± 0.0038 |
| 3e-6 | +0.008 ± 0.025 | −0.070 ± 0.035 (−2.0σ) | +0.007 ± 0.004 |
| 7e-6 | −0.032 ± 0.034 | −0.042 ± 0.038 | **+0.013 ± 0.004** (+3.3σ) |

Anchors both healthy and gate-passed: BLOOM w_A=0.853 (exp 0.8475), JAIS w_A=0.910 (exp 0.91).

## Interpretation
1. **BLOOM control reproduces the erosion SIGN** (harness faithful) — actually deeper/earlier-negative than published (min −0.083 vs −0.032). BLOOM's erosion **magnitude is seed/anchor-sensitive** (caveat), but the qualitative "childhood erodes at working doses" is robust (~2σ, negative at 4/5 doses).
2. **JAIS does NOT erode.** Hugs zero everywhere; +3.3σ **positive** (mild protection) at the high dose where BLOOM is −0.04. ~10× tighter error bars → decisive null-to-positive, not underpowered.
3. **Mechanistic tell:** `frac_child_in_V` (childhood update landing in task-A Fisher subspace) is 3–4× higher in BLOOM (0.28–0.45) than JAIS (0.08–0.10). BLOOM writes childhood into the task-A-carrying subspace → interference; JAIS structurally doesn't. **Erosion tracks subspace overlap — a property of BLOOM's geometry, not of ALiBi+multilingual.**

## VERDICT: KILLED (n=1 broken)
"ALiBi×multilingual childhood-erodes-retention" is **NOT a conjunction law** — it is **BLOOM-560m-idiosyncratic** (one checkpoint; likely representational, childhood/task-A share subspace). A second ALiBi+multilingual family does not reproduce it. Clean honest negative: the break-n=1 killed a would-be law before write-up.

**Phase 2 (Qwen2-1.5B RoPE+multi on A100) = MOOT, SKIPPED** — it was conditional on Jais eroding to prove conjunction-not-multilingual-alone; with Jais not eroding, no cell to disambiguate.

## Artifacts
- `out/probe13_doseresponse_bloom560m.json`, `out/probe13_doseresponse_jais590m.json` (this run)
- `gateA/probe13_doseresponse_bloom560m.json` (published control)
- `out/build_wA_{bloom560m,jais590m}.json` (anchors), `out/p13_{cell}.log`
- Ledger: `D:/IC_experiments/lambda_cost_ledger.csv` (both re-fires, ~$0.7 total incl. VOIDs)

## Harness fix banked (cost 3 VOIDs / ~$1 across attempts)
Anchor hand-off `ANC=$(cat …txt)` → `ANCHOR=$ANC` in the ssh driver produced empty repeatedly. The robust fix (not shipped to the launcher yet, but the reliable pattern going forward): **probe13 should read the anchor directly from `{G}/build_wA_{CELL}.json["anchor_acc"]` when the ANCHOR env is empty** — no shell variable hand-off. The winning re-fires passed the gate because the anchor arrived correctly; make the file-read fallback permanent before the next cell.
