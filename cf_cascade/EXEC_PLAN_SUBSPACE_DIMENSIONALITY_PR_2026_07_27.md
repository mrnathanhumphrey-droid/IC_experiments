# EXEC PLAN — #3 Subspace dimensionality (PR predicts steerable-axis-vs-broad-subspace)

- **Date:** 2026-07-27
- **Pre-reg:** `PREREG_SUBSPACE_DIMENSIONALITY_PR_2026_07_27.md` (LOCK `10d04137`)
- **Status:** DRAFT plan — awaiting sign-off. Nothing fired.
- **Smoke gate already passed ($0, cached):** OPT-350m Probe-18 protected subspace is broad — `geom_basis` Ke50=2 / Ke90=14 / Ke95=26 of 1024; sibling probe16 Fisher PR (eff_rank) 2.35–6.72 ≫1. Positively parameterizes Probe-18's "broad by elimination."

## Hypothesis (from lock)
A single spectral scalar — participation ratio `PR = (Σσ²)²/Σσ⁴` of the contrastive **difference-in-means matrix** `D∈ℝ^{N×d}` for a concept — predicts:
(i) **causal effective steering rank** `k*` = # top singular dirs of D to recover ≥90% of the full-D steering/ablation effect;
(ii) **robustness** (OOD steering transfer, ablation/jailbreak resistance);
(iii) **unlearning reversibility**.
`PR≈1` ⇒ single steerable axis; `PR≫1` ⇒ broad subspace.

## Substrate
- **Primary:** Gemma-2-2B (GemmaScope SAEs free → the SAE-latent-count baseline).
- **Replication:** Qwen-2.5-1.5B (Face-1c infra exists) → cross-model, guards Simpson's-paradox (2601.03168).
- Both fit local RTX 5070 Ti (bf16) for activation + steering work → **P0 is LOCAL $0**; only scale-up/reversibility may want Lambda.

## Concepts (contrastive prompt pairs)
- **P0 set (~12):** refusal (Arditi harmful/harmless), sentiment (pos/neg), sycophancy, formality, language-ID (en/fr), **day-of-week (Engels 2405.14860 — MUST read PR≫1, positive control)**, 3 factual toggles (true/false), truthfulness, verbosity, toxicity.
- **P1 expansion → ≥30:** add more factual toggles, styles, personas, task-format concepts.
- Each concept: N contrastive pairs (N≥64), harvest residual-stream acts at all layers → `D` (N×d).

## Per-concept measurement pipeline
1. **PR + stable rank** of `D` per layer (SVD). Pre-commit one analysis layer (mid-depth), frozen on smoke.
2. **Causal steering curve:** steer/ablate using ONLY top-k singular dirs of D; measure fraction of behavioral effect recovered vs k → `k*` (rank for 90%). Effect metric = concept-appropriate (refusal rate Δ, sentiment logit-diff, etc.).
3. **Robustness (P2):** OOD-prompt steering transfer (2407.12404); ablation/jailbreak resistance (2507.20333).
4. **Reversibility (P2):** small unlearn→relearn probe; does PR/k* predict recoverability. Baseline = Mean-PCA-distance (2505.16831).

## Phases + GATES (falsifier-first)
- **SMOKE (local $0, ~1hr):** 3 concepts × Gemma-2-2B × all-layers → validate harvest + steering + k* pipeline; confirm day-of-week PR≫1, refusal PR measured. Gate: pipeline sane, effect curves monotone.
- **P0 CORE (local $0, ~2-4hr): the make-or-break.** ~12 concepts × Gemma-2-2B. Compute `Spearman(PR, k*)` + perm-null + Haar-null band (QX1).
  - **HIT:** |ρ(PR,k*)| ≥ 0.30 (perm_p<0.01). → P1.
  - **NULL:** <0.30 → PR does not capture causal dimensionality → **KILL** (clean negative vs the safety-subspace framing). Stop.
- **P1 POWER (local or 1×a10, $0-5):** if HIT, ≥30 concepts × {Gemma-2-2B, Qwen-1.5B}. Bootstrap CI over concepts, perm-null, **per-model** (Simpson guard), **cross-model replication required**. Add must-beat baselines: top-1-dir effect, stable rank, SAE-latent count. PR must beat/complement them.
- **P2 ROBUSTNESS+REVERSIBILITY (1×a10, ~$5-10):** the (ii)/(iii) arms — the fork resolver.
  - PR predicts k* AND robustness/reversibility → **spectrum = safety** (headline, resolves refusal 1-vs-many).
  - PR predicts k* but NOT robustness/reversibility → **DOWNGRADE** to "spectrum measures dimensionality, not safety" (still publishable negative). Beat Mean-PCA-distance on reversibility.
- **P3 WRITEUP:** ⭐USER writes; Claude probes [[feedback_user_writes_i_probe]].

## Pre-registered landmines (published negatives to survive)
- CKA-fails-as-predictor (2601.03168) + eff-rank-doesn't-predict-finetuning (2602.20433): analogous negatives — PR must not be another dead spectral scalar.
- **Simpson's paradox (2601.03168):** analyze PER-MODEL; pooled sign can flip. Cross-model replication mandatory for HIT.
- Steering brittleness is dataset-dependent (2407.12404): control N-pair dataset; PR must beat "it's just the dataset."
- N-pair count biases PR: subsample-match N across concepts.
- Haar null band (QX1 finite-N) for PR/subspace-overlap significance.

## Cost / lane summary
SMOKE + P0 = **local $0** (5070 Ti). P1 $0-5. P2 ~$5-10. Total arc ~$10-15 IF it survives P0; **dies at P0 for ~$0 if PR is a dead scalar.**

## Standing floor (publishable regardless)
Even a P0 NULL = "a pre-registered spectral scalar fails to predict causal concept-steering dimensionality" — a clean negative against the safety-subspace literature, in the leave-no-stone ethos.

## Open decisions for sign-off
1. Substrate: Gemma-2-2B primary + Qwen-1.5B replication — OK, or swap/add (Llama-3.2-1B)?
2. P0 LOCAL on the 5070 Ti ($0) vs Lambda — recommend LOCAL.
3. Concept operationalization: OK to use the Arditi refusal set + Engels day-of-week control as anchors?
