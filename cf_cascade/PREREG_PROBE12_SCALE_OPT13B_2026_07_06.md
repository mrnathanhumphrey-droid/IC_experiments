# PRE-REG — Probe-12: does OPT's specificity survive at SCALE (350m → 1.3b)? 2026-07-06.

SHA256-lock before GPU. Scale replication of the specificity result on the SAME hardened metric as Probe-11 v2
(`PREREG_PROBE11v2_REALIZEDALT_3CELL`, LOCK 332e7029). OPT-350m is the SPECIFIC / load-bearing cell (8v4 S+0.122/10σ;
Probe-11 v2 re-baselines it). DSI (`VERDICT_DSI_2026_06_22`) hinted **load-bearingness may DIE at ~1B** — this directly
tests whether the *aimed* specificity is scale-robust or a small-model phenomenon. The single highest-value scale point.

## Cell (new)
opt13b = `facebook/opt-1.3b` (4× OPT-350m). **w_A built 2026-07-06** (`gateA/build_wA_opt13b.json`): SST-2 anchor **0.93**
(best @ step 1750), reload-confirmed 0.93, base 0.718. Ckpt `opt13b_sst2_wA.pt`. Carriers = OPT_C {q/k/v/out_proj, fc1, fc2}
(same 6 types as OPT-350m → pure scale comparison). Per-cell anchor 0.93 (NOT the family default 0.823).

## Design
Identical to Probe-11 v2: realized-path alttask (TREC 250 steps, in-V, unit×magp), phase-2=subj (realized 250), childhood=mix5,
V=top-128 task-A Fisher per carrier, N=100 paired, extended-down LR_GRID [5e-6…3e-5] margin calibration, S=N(phase2)−N(alttask),
N_randtop16 control. Estimator UNCHANGED — only the cell (and its anchor) differ.

## Pre-registered read
- **SPECIFIC (survives scale):** S > 0 (≥2σ) AND N_phase2 > N_randtop16 → aimed specificity is scale-robust; the capstone
  generalizes up. Strengthens the result substantially.
- **GENERIC (dies at scale):** S within ±2σ of 0 AND N_phase2 ≈ N_alttask > N_randtop16 → specificity is a small-model
  phenomenon; consistent with the DSI "load-bearingness dies at 1B" hint. Important negative — report loudly.
- **INTERMEDIATE:** partial scale decay (S positive but attenuated vs 350m).
- **Validity gates:** anchor 0.93±0.02; alttask NOT saturated (dalt_n SE>0); bite (dp2_n & dalt_n >0.03); childhood
  non-destructive (retA ≥ 0.88, i.e. w_A−0.05, else VOID-soft).

## Ops / notes
Harness = `probe11_realizedalt_lambda.py` + opt13b cell (per-cell anchor). Launcher: probe11 opt13b (ckpt in CKPT_MAP).
**Memory:** OPT-1.3b fp32 (~5.2GB) + Fisher (fc2 cin=8192 → ~6GB of Fisher matrices) + AdamW state — may be tight on a 40GB
a100; if it OOMs during Fisher (early), re-fire on h100/gh200 (80–96GB). ~$25–30, ~10–14 h (4× slower than 350m). GPU_PREF
excludes a10; killswitch; EARLY GATE on first `[seed]` (dalt_n variance + retA + no OOM).

## Status
w_A built (anchor 0.93). NOT fired. Next: lock → add opt13b to harness (per-cell anchor) → fire → early-gate → read S.
