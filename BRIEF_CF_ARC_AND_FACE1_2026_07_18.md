# BRIEF — The Catastrophic-Forgetting Arc + the Face-1 Frontier Extension

**Nathan Humphrey · 2026-07-18 · audit-grounded synthesis of `D:/IC_experiments`**
Audit basis: 4 read-only agents over the tree (2026-07-17 clusters + 2026-07-18 cf_cascade & float_module). Rule throughout: **trust artifacts (`*.json`, `VERDICT_*.md`), not prose footers** (per `STATUS_RECONCILIATION_2026_07_14.md`).

---

# PART 1 — THE CF ARC (the whole program)

## 1.0 One-sentence thesis
Catastrophic forgetting, machine unlearning, and continual-learning protection are three faces of **one geometric quantity — the cosine between two capabilities' discriminative directions** in a representation. Established across text / vision / audio; this session we showed the *"globally orderable, no cycles"* version of that story is **regime-specific** (holds frozen-vision, breaks at LLM scale).

## 1.1 The map — five sub-arcs (they are NOT one codebase, and that matters)
| sub-arc | dir | what it is | status |
|---|---|---|---|
| **cf_cascade** | `cf_cascade/` | ORIGINAL LLM discovery arc (Jun 21–Jul 9): the *mechanism* of protection/forgetting in OPT/Pythia/BLOOM… | mature, self-corrected |
| **interference_law** | `interference_law/` | Vision/multimodal FORMALIZATION (Jul 11+): the cosine *law* itself + its regime structure | the scientific core |
| **application anchors** | `detector_cf, librarian_cf, audio_cf, alibi_multi_cf, tier_abstraction` | real-system + modality tests of the law | mixed (2 anchors, several nulls) |
| **float_module** | `float_module/` | SEPARATE operator/RMD-SRC arc (Jun 14–17); touches CF only narrowly | own thing; 2 real cores + 1 dead thread |
| **Face-1** (this session) | `interference_law/` | the pecking-order EXTENSION (Jul 17–18) | see Part 2 |

## 1.2 The scientific throughline (artifact-backed)
1. **Discovery (cf_cascade).** In LLMs, "childhood" pre-training protects a capability against later interference. The mechanism is **reorganization, not suppression** (Probe-7/7b: Δrobust +0.356, 13σ, N=100 paired; the suppression term ΔD2→0 at the clean dose) — childhood makes the *same* weight movement ~3× less damaging. Protection is **directional, not global flatness** (Probe-8v3: global curvature g≤0 at all magnitudes, 10–16σ — refutes the wide-basin story) and lives in a **broad subspace, not one axis** (Probe-18: nullifying the single interference axis leaves protection intact, +0.122≈baseline +0.149).
2. **The eroder (cf_cascade).** Only **BLOOM** erodes instead of protecting; Probe-14/15 break the confound to a **conjunction: ALiBi × multilingual** (Falcon=ALiBi+English protects +0.21; XGLM=sinusoidal+multilingual protects +0.149; only both-together erodes). The energetic mechanism (`frac_in_V`) is **refuted** — it's direction within the subspace, not energy.
3. **Formalization (interference_law).** The pairwise **frozen-readout cosine law**: `forget ≈ −0.445·cos(d_A,d_B) + 0.266`, N=2450 CIFAR-100 class-pairs, held-out R²≈0.45–0.52, ρ=−0.74. The *same* cosine governs unlearning collateral (∝+cos) and protection leakage (∝+cos). Modality-general (vision/text/audio).
4. **Regime structure (interference_law).** The law is a **razor-thin property of exact frozen-ness**: the instant the backbone moves (even rank-2 LoRA) direction stops governing and **magnitude (‖Δθ‖) takes over** — a sharp boundary, not a smooth crossover (ρ −0.74 → −0.07 at rank 2). The theoretically-correct t=0 NTK gradient cosine does **not** predict full-fine-tune forgetting (ρ=+0.016±0.020, 3 seeds — the one clean multi-seed null).
5. **Plastic structure (interference_law).** With a plastic backbone the directed forgetting matrix is **asymmetric** and reduces to **two per-capability scalars**: `F[i,j] ≈ μ + fragility_i + disruptiveness_j`. A Helmholtz-Hodge decomposition shows a **transitive dominance order** (split-half ρ 0.51/0.59) with **cyclic residual = noise** (ρ≈0) — "globally orderable, no rock-paper-scissors." A general plastic *directional* law was chased and **retired** (CIFAR-specific, died on DTD).
6. **Scope (interference_law).** The cosine is a **forgetting law, not a transfer law** (predicts backward retention ρ 0.73, predicts neither forward regime, ρ≈0). It has **no clean in-context twin** — gradient-free (in-context) interference is **dose-governed, not cosine** (a null; the apparent cosine structure was a label-token artifact). It **composes over streams** recency-weighted (γ≈0.4–0.5), path-dependent, intensifying with length.

## 1.3 SOLIDITY LEDGER (the honest audit)

### SOLID (survive their own falsifiers / N≥100 / multi-seed / control-passed)
- **Frozen cosine law** (interference_law §2): N=2450, held-out R²≈0.5, ρ−0.74, reproduced 2 recipes + phase-diagram.
- **Gradient-alignment null** (§8): ρ=+0.016±0.020, **3 seeds, confound-clean** — the single most rigorous result in the tree.
- **Transitive pecking order via Hodge** (§11, vision): split-half control, replicated CIFAR+DTD.
- **Sharp frozen→plastic boundary** (§17): robust across 3 LRs + independent Thread-6.
- **Two-scalar plastic decomposition** (§10): fragility∝−d′, disruptiveness∝‖Δθ‖, verified.
- **Reorganization-not-suppression** (cf_cascade P7/7b): N=100 paired, 13σ, dispositive, lit-novel.
- **Not-global-flatness** (P8v3): 10–16σ, dual-cell.
- **Ω-geometric-law REFUTED → P∝forget-magnitude** (P9b): β_forget 0.915 vs β_Ω 0.034, R²0.80 — a model self-correction.
- **Floor-battery discipline**: re-scoped the whole cf_cascade program from "+0.055" to a **~2σ / +0.03** dissociation. The reason the surviving claims are trustworthy.
- **RMD-SRC engine + "formation" signal** (float_module — *separate* arc): substrate-blind hash-locked falsifier engine reproduces SP500/HCC/Berkeley and honestly refuses Produc/WorldBank; formation signal reads learning MLP→ViT→5 LLM families→6.9b and LEADS grokking generalization (27/28).
- **Face-1b vision** (this session): φ a-priori predictable via d′, pooled ρ(d′,φ)=−0.645, CI[−0.82,−0.37].
- **Face-1c LLM cycles** (this session): survived a pre-registered projection-confound control (see Part 2).

### SHAKY (real but over-exposed / under-powered / confounded)
- **§2 "quantitative cross-substrate transfer, match <0.01"** — the *measured* 0.689 is EXTERNAL prose, N=1 system, and cf_cascade's own audit shows the deepfake swap↔diffusion **antagonism is DOMAIN-driven (cos_domain −0.91), not forgery-driven (forgery cos +0.67)**. The flashiest-sounding claim; the weakest-backed. Do not lean on it.
- **ALiBi×multilingual conjunction** — the most-promoted cf_cascade "novel" finding rests on **n=1 eroder (BLOOM)**, one corner cell; the causal mechanism (Probe-17) is **unresolved/anchor-confounded**.
- **Probe-18 broad-subspace** — N=15, single dose/seed-family.
- **Near-universal single-seed, no bootstrap CIs, hash (not group-disjoint) splits** across interference_law's headline correlations — deltas/structure hold, absolute levels inflated.
- **float_module "unification" thread** — v1 DEGENERATE (constant no-room prediction, `pred_acc=1.0` artifact), v2 free-test aborted + zero-width bootstrap CIs (`ci_lo==delta` ×8). The operator/free-probability unification **did not fire.**
- **float_module CF-immunity headline** (1.000 vs 0.514) — near true-by-construction, no JSON artifact.

### DEAD / refuted (honest negatives — these ARE the value)
OPT +0.055 "load-bearing" (→ really +0.006) · protection=pinning/suppression · global-flatness mechanism · the Ω geometric law · the 8v4 "OPT-specific/Pythia-generic" dissociation (retracted as threat-magnitude artifact) · σ_sem (semantic-similarity) predictor · frac_in_V energetic mechanism · Pythia depth-funnel universality · general plastic *directional* law (CIFAR-specific) · in-context cosine twin (dose not cosine) · float_module factorization-room predictor (falsified game-one, 7/7 fresh no-room) · float_module joint unification + capability-derisk (ceiling).

### GENUINELY OPEN (no artifact on disk)
- **ALiBi×multilingual causal mechanism** — characterized only by elimination; the positive directional test was never fired; rests on n=1.
- **Probe-3 (dose) + Probe-4 (dual)** — "highest-value open GPU work"; no artifacts; Probe-4's disjointness gate is broken.
- **What governs forward TRANSFER** (the axis the cosine does NOT) — field-open; the natural next probe (effective-rank / principal-angle horse-race).
- **Face-1 next rungs** (Part 2).

## 1.4 The meta-finding (both audits converge on this)
**The arc's most reliable output is its discipline of self-correction.** At least five headline claims were killed by the program's *own* later, better-powered probes: the +0.055 magnitude, the Ω-law, the 8v4 dissociation, the operator unification, and (adjacent) the variance-betting edge. **Operating rule: trust paired-differential + N≥100 + multi-seed + control-passed results; treat every absolute effect against the ~0.02–0.03 floor as provisional.**

---

# PART 2 — WHAT WE JUST DID (the Face-1 frontier extension, Jul 17–18)

## 2.1 How we got here
Audit → literature hunt (5 parallel scouts). The hunt found **two genuinely-new lenses** (not repackaged known law): (1) **Hodge geometry of task interference** — the scout found *zero* prior art bridging HodgeRank/Hodge decomposition with any NN interference matrix; (2) **plasticity as an order parameter** for a forgetting phase transition (contradicts Dayi & Chen COLT'25). We pursued lens #1.

## 2.2 The question
The dominance potential **φᵢ = (fragilityᵢ − disruptivenessᵢ)/2** is the coordinate that *defines* the transitive pecking order. **Is φ predictable a-priori from frozen base-task statistics?** If yes, the O(n²) interference grid collapses to an O(n) per-task scalar.

## 2.3 The three-step result
- **Face-1a — cheap $0-local screen** (n=12, 1 seed, CIFAR). **FULL NULL**: φ not a-priori predictable (ordinal +0.08, cardinal −0.17). Provenance-clean (recomputed predictors bit-matched `fragility.json`, ρ=1.0000). Pre-registered its own attenuation caveat (φ split-half reliability 0.51).
- **Face-1b — powered adjudicator, vision** (Lambda a10, 3 seeds × CIFAR+DTD, de-noised φ, ~$2.61). **SCREEN OVERTURNED.** De-noising raised φ reliability 0.51→0.68 and the signal emerged: **φ IS a-priori predictable via discriminability d′** — CIFAR ordinal 0.732 (p.038), DTD 0.679 (p.049), **pooled n=32 ρ(d′,φ)=−0.645, CI[−0.82,−0.37], perm-p 0.0002**. The order is **transitive/acyclic**. The screen's null was pure attenuation (measurement noise in the *target* manufactured a false null). Only d′ holds across both substrates.
- **Face-1c — LLM scale** (Qwen2.5-1.5B, banking77, LoRA causal-LM directed matrix, ~$4.47 incl control). **BOTH vision results FLIP.** (i) the d′→φ law does **NOT** transfer (projection-sensitive, +0.04…+0.65, sign-flipped vs vision); (ii) the forgetting order carries a **reproducible cyclic (rock-paper-scissors) component** — curl-ρ 0.37, where vision's was noise (≈0).
- **Face-1c projection-confound control** (12 projections: dims {8,12,24,48} × seeds {0,1,2}, from one cached set of 48 fine-tunes). **CYCLES PERSIST** — curl-ρ≥0.30 in 71% of in-band points (bar 66%), median 0.373. **Decisive tell: curl-ρ is *highest* at dim=48** (0.36/0.53/0.44), where headroom is smallest — a low-dim-projection artifact would vanish at high dim; the opposite happened. The falsifier was tested and **rejected**.

## 2.4 The frontier sentence
**"Interference is globally orderable, no cyclic tasks" is a property of the frozen-vision regime — NOT of interference in general.** On a plastic LLM, fine-tuning produces reproducible **non-transitive forgetting cycles** that vision lacks, and discriminability stops predicting the order. This is genuinely new (the Hodge-on-interference lens is, per the lit-hunt, unpublished), established through a pre-registered screen + powered arm + confound control.

## 2.5 Honest scope on Face-1c (not fatal, bounds the claim)
Weak forgetting (medFgt 0.02–0.06 → low SNR; effect real-but-small) · all readouts are random-projection (raw features saturate; gold-standard = de-saturate without a projection, e.g. harder negatives) · single model / dataset / layer, n=16 tasks · a reproducible dominance component coexists (not *purely* cyclic).

## 2.6 Method discipline this session (why to trust it)
Pre-reg → falsifier-first → fire on `go`; priors stated to lose (they lost twice, on the more interesting side each time) · **8 cheap smokes (~$1.8) caught 4 bugs** (env pins, a causal-LM design bug where a separate head absorbed the learning, the saturation trap) before any real spend · every surprising claim got its pre-registered control · all Lambda instances self-terminated, cost-logged. **Total Face-1 spend ≈ $7.**

## 2.7 Next rungs (specced, not fired)
Face-1c robustness: stronger-perturbation SNR + a non-projection de-saturation + a 2nd model/dataset → is the cyclic finding a law? · Lens #2 (plasticity phase transition): finite-width scaling of the frozen→plastic boundary, critical-exponent framing · The forward-transfer predictor hunt (§13's open axis).

---

*Provenance: all claims trace to pre-registered SHA256-locked probes with mandatory controls; production systems read-only; nothing deployed. Files: `COSINE_GEOMETRY_OF_INTERFERENCE_2026_07_11.md` (core synthesis), `STATUS_RECONCILIATION_2026_07_14.md` (staleness anchor), `cf_cascade/VERDICT_*`, `interference_law/{PREREG,VERDICT}_FACE1{,B,C}_*`, `interference_law/face1{b,c}_*_lambda.py`, `float_module/VERDICT_*`.*
