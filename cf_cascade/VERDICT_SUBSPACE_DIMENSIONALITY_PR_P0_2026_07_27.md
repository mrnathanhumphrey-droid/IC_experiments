# VERDICT — #3 Subspace dimensionality (PR predicts causal dimensionality): PRE-REGISTERED NEGATIVE (2 models)

- **Date:** 2026-07-27
- **Pre-reg:** `PREREG_SUBSPACE_DIMENSIONALITY_PR_2026_07_27.md` (LOCK `10d04137`); plan `EXEC_PLAN_SUBSPACE_DIMENSIONALITY_PR_2026_07_27.md`
- **Artifacts:** `gateA/pr_steering_gemma-2-2b.json`, `gateA/pr_steering_qwen.json`; harness `pr_steering_lambda.py`; launcher `cloud/lambda_launch_pr.py`; logs `cloud/pr_control_gemma.log`, `cloud/pr_control_qwen.log`.
- **Compute:** Lambda a10, activations-only (no training). Gemma-2-2B + Qwen-2.5-1.5B, 12 concepts × N=48 contrastive pairs, α=4. #3 arc ~$1.1 (7 runs incl 4 harness smokes).

## Hypothesis (locked)
Participation ratio `PR = (Σσ²)²/Σσ⁴` of a concept's contrastive difference matrix D predicts its **causal dimensionality** — # directions you must manipulate to control the concept. PR≈1 ⇒ single steerable axis; PR≫1 ⇒ broad subspace.

## Method (final, after 4 harness iterations documented below)
Per concept: harvest last-token residual acts for matched pos/neg prompts → D; SVD → spectral PR. Analysis layer picked by **causal ablatable signal** among top-separability candidates. Causal dimensionality = **participation ratio of the per-direction individual-steering effects** {|e_i|} (steer each SVD direction at equal magnitude, measure logit-diff response). **Random-direction CONTROL:** same-magnitude steering along directions orthogonal to the concept subspace; floor-correct at rand p95 → `causal_PR_signal`, `n_signal_dirs`.

## Result — PRIMARY NULL, replicated across 2 model families
| model | control concept/rand | ρ(specPR, causal_PR_signal) | perm_p | ρ(specPR, n_signal) |
|---|---|---|---|---|
| Gemma-2-2B | **1.91** (clean, not confounded) | **−0.371** | 0.235 (ns) | −0.007 |
| Qwen-2.5-1.5B | 1.47 (borderline) | **+0.021** | 0.956 (ns) | −0.014 |

- Spectral PR (range 3–14 across concepts) shows **no significant correlation** with causal dimensionality on either model, and the **signs disagree** (−0.37 vs +0.02) → no consistent relationship. HIT bar (|ρ|≥0.30 & perm_p<0.01 & control-clean) **not met**.
- **CONTROL** confirms concept directions are behaviorally distinguishable from random (Gemma 1.9×; Qwen 1.5× borderline) → on Gemma the null is not mere metric failure.

## Secondary observation — concepts are causally BROAD (Gemma-clean, Qwen-weak)
Even floor-corrected, **9–17 signal-carrying directions per concept**, and **no single direction carries >20%** of the causal effect (top1_causal_frac 0.07–0.20 on Gemma) — no concept is a single steerable axis, regardless of spectral PR. Clean on Gemma; on Qwen several concepts barely beat the random floor so the "broad" claim is weaker there.

## Verdict
**Pre-registered NEGATIVE.** A spectral scalar (PR of the contrastive difference matrix) does NOT predict a concept's causal dimensionality — replicated as a null across Gemma-2-2B and Qwen-2.5-1.5B. Separately (Gemma-clean), concepts are causally broad, not single axes → lands on the **Wollschläger 2502.17420 / Teo 2507.20333 "concept cone / high-dim subspace" side** of the refusal-geometry debate and against the **Arditi 2406.11717 single-direction** pole — with the novel addition that the *spectral* structure does not forecast the *causal* breadth. [[feedback_leave_no_stone_unturned_research_ethos]]

## Caveats (state plainly)
- n=12 concepts, **lexical-antonym family** (sentiment/size/temperature/…); refusal dropped (poor fit for a next-token logit-diff metric — its signal is in generation). Complex/multi-faceted concepts untested.
- Single α=4; **Qwen control borderline (1.47)** → Qwen causal signal is weak, so the Qwen leg is a weaker replication of the null (it agrees there's no PR→causal signal, but its causal metric is closer to generic-sensitivity).
- Consistent with the $0 cached smoke-gate (OPT-350m Probe-18/geom_basis: protected subspace broad, Ke90=14) — concepts broad; the spectral predictor is what's dead.

## Harness iteration log (methodology, for reuse)
1. Centered-D SVD bug → mean-diff outside the centered subspace → k* uncomputable. Fix: uncentered D.
2. Mean-diff steering k* saturates at 1 (1-D steer). Fix: ablation-rank.
3. Ablation curves overshoot non-monotonically (SVD-order ≠ causal-order). Fix: per-direction individual steering → causal participation ratio (monotone).
4. Equal-magnitude steering inflates breadth via generic sensitivity. Fix: random-direction control + floor-correction.
⚠️ Gemma-2-2b gated → HF_TOKEN in physics_detector/.env; datasets needs Pillow≥9.4 (system 9.0.1 too old, `Pillow<11` satisfied-by-it → force `Pillow==10.4.0`).

## NEXT (optional, if resumed)
Powered n (more concepts, esp. multi-faceted: refusal via generation-metric, sycophancy, persona) to resolve the Gemma −0.37 hint; sweep α; or P2 robustness/reversibility arm (was gated on a P0 HIT — moot now).
