# PRE-REG — Probe-11: does the 8v4 SPECIFIC/GENERIC dissociation replicate on a 3rd architecture (BLOOM-560m)? 2026-07-06.

SHA256-lock before GPU. Replication of `PREREG_PROBE8_v4_SPECIFICITY_2026_06_28` (LOCK 0112d69b) on an independent 3rd
architecture, to move the load-bearing↔inert axis from n=2 → n=3. Same locked estimator, new cell. No new mechanism — a
generalization test of the capstone (8v4: OPT SPECIFIC S+0.122/10σ, Pythia GENERIC S+0.014 n.s.).

## Cell (new)
bloom560m = `bigscience/bloom-560m`. **w_A built 2026-07-06** (`build_wA_lambda.py`, `gateA/build_wA_bloom560m.json`):
SST-2 fine-tune, anchor_acc **0.84** (best @ step 750), reload-confirmed 0.84, base 0.605. Ckpt `bloom560m_sst2_wA.pt`.
Independent family (BigScience, ALiBi, distinct corpus) — a legitimate 3rd point. Caveat: linear naming is Megatron-style
(fused qkv) like Pythia, but the architecture (ALiBi, norms, activations, training) is distinct.

## Design (identical to 8v4 where applicable)
task-A = SST-2 (anchor 0.84). phase-2 = subj (realized 250 steps). alttask = TREC (one-shot grad, biting null). childhood =
mix5 (agnews/dbpedia/yahoo/newscat/emotion). V = top-128 task-A Fisher per carrier. N=100 paired. Stage-1 margin calibration
(K_CAL=20, margin 0.015) → chosen_lr. Stage-2: S = N(phase2) − N(alttask); N(d) = dprop_NoCH(d) − dprop_CH(d), magnitude-
matched; plus N_randtop16 (random within top-16 Fisher). **Carriers (pre-committed) = Pythia-analog Megatron set:
{self_attention.query_key_value, mlp.dense_h_to_4h, mlp.dense_4h_to_h}** — mirrors Pythia PY_C (fused qkv + both MLP,
EXCLUDES the attention output dense) for maximal comparability to the other Megatron-style cell. ANCHOR_WA[BLOOM]=0.84±0.02.

## Pre-registered read (same map as 8v4)
- **SPECIFIC** (OPT-like): S > 0 (≥2σ) AND N_phase2 > N_randtop16 (≥2σ) — neutralizes the actual interference over a generic
  damaging direction.
- **GENERIC** (Pythia-like): S within ±2σ of 0 AND N_phase2 ≈ N_alttask > N_randtop16 — protects real-task directions equally.
- **INTERMEDIATE**: significant S but small, or partial — a point between OPT and Pythia (consistent with Probe-10's finding
  that the arch axis is GRADED, not binary).
Validity gates (VOID/VOID-soft): w_A anchor 0.84±0.02; bite (mean dp2_n & dalt_n > 0.03); childhood non-destructive.

## Weak prior (not a hard commit)
BLOOM's Megatron-style linear structure resembles Pythia → a GENERIC outcome would be unsurprising; but ALiBi + distinct
training leave it open. Classify by the map, not by the prior.

## Verdict
3-point map on the load-bearing↔inert axis. SPECIFIC → strengthens "specific is not unique to OPT" (specific n=2);
GENERIC → strengthens "inert/generic is a real class" (generic n=2); INTERMEDIATE → the axis is continuous (best-supported by
the graded 8v4/8v3/Probe-10 picture). Either way n=2→3 on the architecture axis, the #1 external-review gap.

## Ops
Reuse `probe8v4_specificity_lambda.py` with bloom560m added to CELLS/ANCHOR_WA + BLOOM_C carriers (estimator UNCHANGED = the
locked 8v4 measurement). Launcher: probe8v4 bloom560m (ckpt in CKPT_MAP, JSONS low_agnews+medium_subj). GPU_PREF excludes a10;
killswitch; EARLY GATE on the first `[P8v4:bloom560m] w_A acc=... N carriers` line (confirm carriers resolve >0 + anchor in-band)
before committing the full N=100 run. Single cell, ~$15–20, ~5–7 h.

## Status
w_A built (anchor 0.84). NOT fired. Next: lock → add BLOOM to harness → fire → early-gate → read S.
