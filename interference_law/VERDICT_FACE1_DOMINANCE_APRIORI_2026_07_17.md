# VERDICT — Face-1a: is the pecking order A-PRIORI computable? — FULL NULL (screen)

**PREREG lock** b035c2ef · **2026-07-17** · `probe_face1_dominance_apriori.py` → `out/face1_dominance_apriori.json` · $0 local, no fine-tuning, wall 27s.

## Provenance guard: PASSED (airtight)
Recomputed predictors for caps 0–39 vs `fragility.json`: spread/munorm/dprime/wnorm all **max|Δ|≈5e-5, ρ=1.0000**. The RNG stream
aligned (extended cap list keeps 0–39 first), so even head-based dprime/wnorm bit-matched. Feature-freezing pipeline verified identical —
the "weak-but-CI-excludes-0 from extraction drift" failure mode is excluded.

## Result vs pre-committed thresholds
| arm | n | key stat | pass? |
|---|---|---|---|
| **A — fragility (reconfirm)** | 40 | ρ(d′,frag)=**−0.511**, boot 95% CI **[−0.741, −0.192]** (excludes 0); trio LOO-R²=**+0.205** vs perm-null95 **0.026** vs bar 0.30 | **predictable=NO** (real signal, below 0.30 bar) |
| **B — φ dominance potential** (on-target) | 12 | ρ(d′,φ)=−0.350, CI **[−0.727,+0.245]** (incl. 0), perm_p 0.27 | — |
| ↳ CARDINAL (φ magnitude) | 12 | trio LOO-R²=**−0.170** vs thresh 0.30 | **FAIL** |
| ↳ ORDINAL (the ORDER) | 12 | LOO rank-corr=**+0.077**, perm_p 0.81 | **FAIL** |
| B — fragility_i (n=12) | 12 | trio LOO-R²=+0.239 | weak, <0.30 |
| B — disruptiveness_j (n=12) | 12 | trio LOO-R²=**−0.523** | NO (interaction-born, as expected for ≈‖Δθ‖) |

## HEADLINE (pre-committed → cannot drift)
**FULL NULL — the pecking order emerges only from the fine-tuning interaction, not from static task properties.** φ is not a-priori
predictable, not even ordinally (rank-corr 0.08). The O(n²) interference grid does **not** collapse to an O(n) static per-task scalar.

## Priors: I bet cardinal-fail (WON) and ordinal-plausible-pass (LOST). The order is more emergent than I expected.

## What IS real (honest positives)
- **d′ → fragility is a genuine a-priori correlate** at n=40: ρ=−0.511, CI excludes 0, LOO-R² 0.205 is **~8× the permutation-null ceiling (0.026)** —
  above-chance but too weak (below the 0.30 usability bar) to be a stand-alone predictor. Well-separated capabilities do resist forgetting; that's
  a partial, not a controlling, effect. (This hardens `fragility.json`'s point LOO-R²=0.08 — the trio {dprime,wnorm,spread} beats the 6-predictor
  model by dropping overfit, but the conclusion "not structurally predictable" stands, now WITH a CI.)
- **Disruptiveness is purely interaction-born** (LOO-R² −0.52) — consistent with disruptiveness ≈ ‖Δθ‖ being a dynamics quantity, not static geometry.

## Caveat (why this is a SCREEN, not a powered verdict)
n=12 for φ, and φ is measurement-noisy (split-half ρ 0.51) → attenuation biases the ordinal test toward null; the ordinal null is confounded
with φ's own unreliability. BUT fragility (n=40, well-measured) also only reached 0.20 < 0.30, so static predictability is weak system-wide —
which makes the φ null credible rather than a pure power artifact.

## Powered adjudicator (the Lambda arm — deferred, ~1 GPU job)
Rebuild F at **16 caps × 3 seeds × {CIFAR, DTD}**, SAVE the per-cap h (φ) vectors, average φ across seeds to de-noise (split-half → de-attenuate).
Then re-run this exact regression. Decisive test: does a de-noised φ at n≥16 reveal the weak ordinal signal this screen's noise could hide, or does
the full-null hold? Pre-commit the same trio + thresholds. Only fire on user go.
