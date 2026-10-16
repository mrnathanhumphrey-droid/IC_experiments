# PREREG — Face-1c: does the pecking order (a-priori d′-law + transitivity) survive at LLM scale? (cheap first cut)

**Date:** 2026-07-17 · lock = sha256(this)[:8] · Lambda A100-first, ~4–6 GPU-hr, ~$8–12. Preceded by a ~$0.30 SMOKE self-check.
Extends the VISION result `VERDICT_FACE1B` (φ a-priori predictable via d′, pooled ρ(d′,φ)=−0.645; order transitive/acyclic on CIFAR+DTD).

## Question
Two things established on vision (ResNet18, n≤32) — do they hold on an LLM?
1. **A-priori d′-law:** is the dominance potential φ predictable from frozen base-task discriminability d′?
2. **Transitivity:** is the directed forgetting order acyclic (Hodge curl = noise) or do real cyclic (rock-paper-scissors) task-triples emerge at LLM scale?

## Design (faithful port of probe_cyclic/face1b to LLM activations)
- **Model:** Qwen2.5-1.5B (matches the in-context arc). Backbone frozen for activation extraction.
- **Capabilities:** banking77 intents (text — no PIL path). 16 caps (labels 0–15) + neg pool (labels 60–69). PER≈80/class (min 40).
- **Activation / readout:** mean-pooled final-layer hidden state (attention-masked) → linear victim probe per cap vs neg. d_c = normalize(mean(act|c) − mean(act|neg)).
- **Directed F[i←j]:** deploy frozen victim heads; **LoRA-SFT (r=16)** the backbone on binary (j vs neg), ~200 steps lr 1e-4; re-extract activations; F[i,j] = auc0_i − auc1_i. hodge(F) → φ, grad(dominance)/curl(cyclic) fractions.
- **3 seeds** (LoRA init + SGD) → seed-averaged φ (de-noise) AND cross-seed reproducibility of dominance-h vs curl (the transitivity test, mirroring vision split-half).
- **Predictors (a-priori, frozen acts):** dprime, spread, munorm, wnorm, iso — same defs as face1b. Primary = **d′**.

## Pre-committed (identical stats to face1b — no drift)
Trio {dprime,wnorm,spread}, d′ primary, ridge λ=1, LOO, bootstrap 10k CI, permutation 10k / null-1k.
- **A-priori-law PASS:** LOO-R²(φ; trio) ≥ max(0.30, perm-null95) AND/OR ordinal Spearman(LOO-pred φ, true) ≥ 0.50 (perm-p<0.05) AND ρ(d′,φ) CI excludes 0.
- **Transitive/acyclic:** cross-seed ρ(dominance-h) ≥ 0.50 AND cross-seed ρ(curl) < 0.15 (curl = noise). **Cyclic-REAL** if ρ(curl) ≥ 0.30.
- **Saturation guard:** median forgetting must land in [0.03, 0.55]; if it saturates (→chance) or floors (<0.03), the run is INCONCLUSIVE (regime tweak: LoRA rank/steps), not a result.

## Priors (stated to lose)
- A-priori d′-law: I expect it to WEAKEN vs vision (LLM directions are higher-dim, task overlap messier) but likely still ordinal-pass (~60% pass).
- Transitivity: I expect acyclic to HOLD (curl=noise) — but a genuine cyclic triple emerging at LLM scale would be the BIGGER, more surprising finding. Falsifier-of-note = ρ(curl) ≥ 0.30.

## Pre-committed HEADLINES
- Law-pass + acyclic: **"Pecking order generalizes to LLMs — d′ predicts it, order is transitive."**
- Law-fail + acyclic: **"Transitivity generalizes but the a-priori d′-law is vision-specific."**
- Cyclic-REAL: **"Rock-paper-scissors task-triples EXIST at LLM scale — the order is NOT globally transitive"** (the big result).
- Saturated: **"Inconclusive — regime not engineered; retry with LoRA/step tweak."**

## Output
`out/face1c_llm.json` + `VERDICT_FACE1C_*`. SMOKE self-check (2 caps/1 seed/30 steps) runs first in-instance; asserts pipeline lands + prints a forgetting number before the full grid. Cost logged.
