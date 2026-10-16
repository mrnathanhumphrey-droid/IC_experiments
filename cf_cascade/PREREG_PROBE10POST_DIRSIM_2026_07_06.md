# PRE-REG — Probe-10 post-hoc: is the operative similarity task-FORMAT or interference-DIRECTION? 2026-07-06.

Small, pre-committed post-hoc from `PREREG_PROBE10_SPECIFICITY_SIMILARITY` (LOCK c83884a7) — descriptive, not gated.
SHA256-lock before GPU. Probe-10 found the "same-family (by format)" middle unreliable (dbpedia a WEAK sibling: it protects
the yahoo direction *less* than the unrelated affect task). Hypothesis: the similarity that governs aimed protection S is the
overlap of the tasks' realized INTERFERENCE DIRECTIONS in V, not their surface task-format (topic vs affect).

## Compute (OPT only, cheap; no childhood, no paired seeds)
Load OPT w_A (SST-2, anchor 0.823). V = top-128 task-A Fisher per carrier (same as Probe-10). For each task X in
{agnews, dbpedia, yahoo, emotion, trec, subj}: realize X's 250-step in-V displacement from w_A over K=8 seeds (LR_DEPTH=5e-5),
flatten the in-V coordinates (d@V per carrier), unit-normalize → per-seed direction. **meandir[X]** = normalized mean of the K
unit directions; **self-consistency[X]** = mean pairwise cosine among the K (reliability check). Emit the cosine **overlap
matrix** cos(meandir[X], meandir[Y]) for all pairs, and specifically the columns for the two probe directions {agnews, yahoo}.

## Analysis (local, after)
For each probe T ∈ {agnews, yahoo} and each swapped-slot childhood task X ∈ {agnews, dbpedia, yahoo, emotion}, take
direction-overlap(X, T) = cos(meandir[X], meandir[T]) and pair it with the Probe-10 OPT specificity S(childhood=X, probe=T).
- **Format axis:** topic-family membership relative to T (agnews/dbpedia/yahoo = topic = 1; emotion = affect = 0).
- **Direction axis:** direction-overlap(X, T) (continuous).
Report Spearman ρ(S, direction-overlap) vs ρ(S, format) across the 4 childhoods (×2 probes, and pooled n=8).

## Pre-registered read (descriptive)
- **DIRECTION-WINS** if ρ(S, direction-overlap) > ρ(S, format) AND dbpedia's overlap with yahoo is LOW (below agnews→yahoo),
  explaining the weak-sibling anomaly geometrically.
- **FORMAT-WINS** if format tracks S at least as well and dbpedia→yahoo overlap is high (anomaly unexplained by direction).
- Either way: descriptive; does not change the Probe-10 verdict, only explains the middle-rung noise.

## Ops
Harness `cloud/probe10post_dirsim_lambda.py` (reuse Probe-10 loaders + V + train_phase; add realize-direction + cosine matrix).
CPU dry-run (cosine math + wiring). One short OPT instance (~12k train steps, ~$2, ~30 min). GPU_PREF excludes a10; killswitch.

## Status
NOT built/fired. Next: lock → build → dry-run → fire OPT.
