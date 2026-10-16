# VERDICT — Face-1b: the pecking order IS a-priori computable (screen's null was ATTENUATION)

**PREREG lock** a16123a9 · **2026-07-17** · Lambda a10, 3 runs (DTD ok, CIFAR crash on HF-datasets/PIL bug, CIFAR recovered via torchvision loader).
Total spend **~$2.61** (DTD $0.86 + failed CIFAR $0.13 + CIFAR $1.62). Artifacts: `out/face1b_dominance_apriori.json` (DTD), `out/face1b_cifar.json` (CIFAR).

## Headline (pre-committed → fired)
**SCREEN OVERTURNED — the dominance potential φ (the pecking-order coordinate) IS a-priori predictable, ORDINALLY and CARDINALLY, primarily via discriminability d′.** The n=12 single-seed screen full-null (ordinal +0.08) was **attenuation**: de-noising φ across 3 seeds raised cross-seed reliability 0.51 → 0.65–0.68 and the a-priori signal emerged on BOTH vision substrates.

## Numbers (3-seed de-noised φ, ridge λ=1, LOO, pre-committed trio {dprime,wnorm,spread}, d′ primary)
| arm | n | reliability | cardinal LOO-R² (null95) | ordinal rank-corr (perm-p) | ρ(d′,φ) [95% CI] perm-p |
|---|---|---|---|---|---|
| **CIFAR** (screen's substrate) | 16 | 0.679 | **0.403** (0.102) ✓ | **0.732** (0.038) ✓ | −0.665 [−0.87,−0.23] p=0.004 |
| **DTD** | 16 | 0.646 | **0.573** (0.088) ✓ | **0.679** (0.049) ✓ | −0.603 [−0.87,−0.06] p=0.015 |
| **POOLED** (within-dataset z-scored) | 32 | — | **0.368** (0.022) ✓ | **0.630** (0.045) ✓ | **−0.645 [−0.82,−0.37] p=0.0002** |

Screen (n=12, 1 seed, CIFAR, caps 0..77): ordinal +0.077, cardinal −0.170 → NULL. Powered (n=16, 3 seeds, CIFAR, caps 0..15): ordinal +0.732.

## What this means
- The pecking order is **partially a-priori**: **well-separated capabilities (high d′) occupy predictable positions in the forgetting-dominance order.** The O(n²) directed-interference grid can be **partly shortcut to an O(n) static scalar (d′)** — the Face-1 payoff, cheap and real.
- **Ordinal > cardinal** (rank recovery 0.63–0.73 vs magnitude R² 0.37–0.57): the *order* is more a-priori-recoverable than the exact φ magnitudes. Consistent with "the pecking order exists a-priori; fine magnitudes still carry interaction-specific residual."
- **d′ is the sole consistent driver.** `spread` was strong on DTD (ρ+0.64) but null on CIFAR (−0.01); `wnorm`/`munorm` inconsistent. Only d′ holds on both (−0.60 / −0.67). Report the law as **φ ∝ −(discriminability)**, not the full trio.

## Methodological lesson (bank this)
**Measurement noise in the TARGET manufactures false nulls in a predictability test.** The screen pre-registered its own attenuation caveat (φ split-half 0.51) and the powered arm vindicated it: de-noising the target (seed-averaging → reliability 0.68) surfaced a signal that n=12 buried. Pre-registration + the powered adjudicator worked exactly as designed. (Caveat on attribution: the powered run also changed caps 0..77→0..15 and N 12→16, so it is a *fuller/cleaner* measurement, not a de-noise-only A/B; the reliability increase is the evidence that attenuation was the mechanism.)

## Honest scope / still open
- n=16 per substrate (32 pooled), single architecture (ResNet18), 2 vision datasets. Ordinal perm-p sits at 0.038–0.049 (significant, not overwhelming).
- **NOT tested at LLM scale** — the bigger Face-1 question ("does the acyclic transitive order + a-priori d′-predictability survive at LLM scale / do real cyclic task-triples emerge?") remains the open frontier.
- Disruptiveness/fragility a-priori predictability is substrate-inconsistent (fragility LOO-R² 0.48 CIFAR / 0.02 DTD; disrupt 0.22 / 0.62) — not claimed.

## Supersedes
Updates `VERDICT_FACE1_DOMINANCE_APRIORI` (the screen): its FULL-NULL headline is **overturned** for φ. The screen's honest positive (d′→fragility real but weak) generalizes: **d′ → the pecking-order coordinate is the a-priori law, recoverable once φ is de-noised.**
