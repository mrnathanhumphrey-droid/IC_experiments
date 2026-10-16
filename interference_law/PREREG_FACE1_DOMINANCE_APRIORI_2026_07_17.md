# PREREG — Face-1a: is the pecking order A-PRIORI computable? ($0-local, no fine-tuning)

**Date:** 2026-07-17 · **Arc:** Hodge pecking-order of interference (extends `VERDICT_CYCLIC`, `VERDICT_FRAGILITY`, `VERDICT_ASYMMETRY`).
**Ethos:** falsifier-first; priors stated to lose; pre-commit NUMBERS; $0 no-fine-tuning screen. Lambda arm deferred.

## Question
The dominance potential φ_i = (fragility_i − disruptiveness_i)/2 is the coordinate that DEFINES the transitive pecking order
(the Hodge gradient of the antisymmetric interference matrix). **Is φ_i predictable from FROZEN base-task statistics computed BEFORE
any fine-tuning?** If yes, the O(n²) interference grid collapses to an O(n) per-task scalar and the order is knowable a-priori.

## Data (all on disk; ZERO new fine-tuning)
- Target φ: `out/asymmetry_full.json:dominance_h` (12 caps {0,7,…,77}, CIFAR, seed 0). Components `fragile`, `disrupt` (same 12).
- Higher-powered arm: `out/fragility.json:fragile` + `predictors` (40 caps).
- Predictors recomputed via the EXACT `probe_fragility.py` code path (import `probe_representational_law` R, `probe_gradient_alignment` G),
  over `VICTIMS_EXT = list(range(40)) + [42,49,56,63,70,77]`, seed 0. Predictors: dprime, spread, munorm, wnorm (per-cap intrinsic); iso, sens (exploratory).

## PROVENANCE GUARD (config-hash equivalent)
Caps 0–39 consume the identical RNG stream as the original run → recomputed {dprime,spread,munorm,wnorm} for caps 0–39 MUST equal
`fragility.json` values to |Δ|<1e-4. **Assertion failure = extraction drift = ABORT** (do not report). This is the guard against a silent
feature-freezing mismatch masquerading as weak-but-CI-excludes-0 signal.

## Amendments folded in (per review 2026-07-17)
1. **No selection leakage.** Predictor trio is PRE-COMMITTED NOW (not best-subset): **{dprime, wnorm, spread}** — the top-3 by |ρ| from the
   fragility arc (`fragility.json:univariate_rho` = dprime −0.511 > wnorm −0.362 > spread +0.306), chosen from fragility data, NEVER from φ.
   Fixed across all LOO folds (no per-fold selection). Fold-nested selection reported ONLY as non-decisional robustness.
2. **d′ is the NAMED PRIMARY predictor** a-priori. The other five are EXPLORATORY with Bonferroni 99.2% CIs (α=0.05/6). A non-primary driver
   → exploratory finding that WAITS for the Lambda arm; it does NOT constitute a Face-1a pass.
3. **Null-calibrated threshold.** 1000 target-permutations → empirical null of LOO-R². Pass requires LOO-R² ≥ max(0.30, null 95th pct).
- Ridge (standardized, λ=1.0 pre-committed — no tuning). Bootstrap 10k (resample caps) for ρ CIs; permutation 10k for ρ p-values.

## Pre-committed decision rules
- **CARDINAL pass (φ magnitude predictable):** LOO-R²(φ; trio) ≥ max(0.30, perm-null 95th) AND ρ(d′,φ) 95% CI excludes 0.
- **ORDINAL pass (the ORDER is a-priori — answers the title):** Spearman(LOO-predicted φ, true φ) ≥ 0.50 AND its bootstrap CI excludes 0.
- **Reconfirm (n=40):** report ρ(d′,fragility)=−0.51 WITH bootstrap CI; predictability still fails if LOO-R²(trio) CI-upper < 0.30.

## Priors (stated to lose)
- CARDINAL: I bet FAIL (LOO-R²(φ) < 0.20). Reason: fragility alone already LOO-R²=0.08, and φ is measurement-noisy (split-half ρ 0.51 →
  attenuation biases toward null).
- ORDINAL: plausible PASS via d′ (well-separated capabilities resist forgetting → sit high in the order).
- Falsifier that proves me wrong = CARDINAL pass.

## Pre-committed HEADLINES (decided before results, so the write-up can't drift)
- ORDINAL pass (regardless of cardinal): **"YES — the pecking order is a-priori computable (ordinally)."** ← the real Face-1a answer.
- CARDINAL pass too: stronger — magnitude predictable, not just rank.
- BOTH fail: **"Full null — the order emerges only from interaction, not from static task properties."**
- Attenuation caveat: an ordinal null is confounded with φ's own unreliability (n=12, split-half 0.51); the Lambda arm (16 caps × 3 seeds ×
  CIFAR+DTD, SAVING h vectors) is the powered adjudicator.

## Output
`out/face1_dominance_apriori.json` + `VERDICT_FACE1_DOMINANCE_APRIORI_2026_07_17.md`. Cost: $0, CPU, minutes. Records sha256 of THIS file as lock.
