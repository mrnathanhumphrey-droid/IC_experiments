# VERDICT — Face-1c LLM (cheap first cut): BOTH vision conclusions flip — PROVISIONAL, one serious confound

**PREREG lock** (PREREG_FACE1C_LLM) · **2026-07-17** · Qwen2.5-1.5B, banking77, LoRA causal-LM directed-F, mid-layer 14, 1536→12 random projection (de-saturation), 16 caps × 3 seeds × 200 steps. `out/face1c_llm.json`. Cost: full run $1.28 (39 min A100) + 8 de-risking smokes $1.77 = **~$3.05**.

## Headline (pre-committed → fired), with a caveat that downgrades it to PROVISIONAL
**At LLM scale, the directed forgetting order has a REPRODUCIBLE CYCLIC component AND the vision d′→φ a-priori law does NOT transfer.** Both vision results (VERDICT_FACE1B) flip. **BUT** the cyclic finding is confounded with the de-saturation projection (see Threat) — treat as a provisional screen result pending a control, NOT a settled claim.

## Numbers (medFgt=0.053, in-band; non-saturating regime engineered — pre-reg saturation-guard invoked)
| question | vision (FACE1B) | LLM (this run) | verdict |
|---|---|---|---|
| **transitivity** cross-seed curl-ρ | ≈0 (noise → acyclic) | **0.394** (≥0.30 threshold → reproducible cycles) | **FLIPS → cyclic-real** |
| dominance cross-seed ρ | 0.51–0.59 | **0.788** (reproducible dominance too) | both components reproduce |
| antisym split (per-seed) | grad-dominated | **curl-dominated** (curl% 0.65–0.79 > grad% 0.24–0.40) | mostly cyclic |
| **d′→φ a-priori law** ρ(d′,φ) | −0.60 to −0.67 (CI excl 0) | **+0.038, CI [−0.47,0.60]** (incl 0); trio LOO-R² −0.42 | **FAILS → vision-specific** |

Note: ordinal −0.485 (p.024) is NOT a "reverse law" — it's LOO overfit in the negative-R² regime (trio doesn't generalize at n=16); the honest reading is d′ simply does not predict φ here.

## Interpretation (if it survives the control)
- The pecking order is NOT a universal geometric law. On frozen ResNet vision features it's transitive and a-priori (d′). On a plastic LLM (LoRA-SFT), the SAME construction yields **reproducible rock-paper-scissors task-triples** and d′ loses all predictive power. → Interference structure is substrate-dependent; the "globally orderable, no cycles" claim is a property of the frozen-vision regime, not of interference in general.
- This is the *more surprising* branch (my prior bet transitivity would hold — it lost).

## THREAT TO VALIDITY (load-bearing — why this is PROVISIONAL)
1. **Projection confound (#1).** Banking77 intents saturate Qwen probes at AUC0=1.0 (no headroom), so a fixed 1536→12 random projection was required for measurability. Vision needed NO projection. A low-dim projection can induce/inflate apparent cyclic geometry. Cross-seed reproducibility holds the projection FIXED (varies only LoRA seed) → projection-dependence is UNTESTED. **Required control: re-run at ≥3 projection seeds and ≥2 dims (e.g. 8, 24); the cyclic finding is believable only if curl-ρ ≥ 0.30 PERSISTS across projections.** If curl-ρ collapses when the projection changes, the cycles are an artifact.
2. **Weak forgetting** (medFgt 0.053, low end of band) → low SNR; φ small.
3. **Single condition:** one model (1.5B), one dataset (banking77), one layer (14), n=16. No replication.
4. curl-ρ 0.394 clears the pre-committed 0.30 but is not overwhelming; and a reproducible DOMINANCE component (0.788) coexists — so "not purely transitive," not "purely cyclic."

## Status
PROVISIONAL SCREEN. The pre-committed decision fired cyclic-real + law-fail, but the projection confound is serious enough that the cyclic claim must NOT be banked as settled. Next step (deliberately NOT auto-fired — new spend): the **projection control** above (~$1.3, ~40 min). Only if curl-ρ persists across projections does "cycles emerge at LLM scale" become a real finding worth pursuing (harder-negative de-saturation without projection would be the gold-standard follow-up).

## Priors: I bet transitivity HOLDS + d′-law weakens-but-holds. BOTH lost — cycles appeared, d′-law vanished. Honest loss; the surprising branch.

---

# CONTROL RESULT (2026-07-18): the cyclic finding SURVIVES the projection confound

`face1c_control_lambda.py` → `out/face1c_control.json`. Ran the 48 LoRA-SFTs ONCE, cached raw 1536-d activations, swept **12 projections** (dims {8,12,24,48} × seeds {0,1,2}). Control $1.19 + smoke $0.23. Face-1c total ~$4.47.

**HEADLINE: CYCLES PERSIST — not a projection artifact. curl-ρ ≥ 0.30 in 71% of in-band grid points (pre-committed bar 66%), median curl-ρ = 0.373.** Across all 12 variants curl-ρ ∈ [0.21, 0.53], dominance-ρ ∈ [0.52, 0.94] (both reproduce).

**Decisive de-risk:** curl-ρ does NOT decay as projection dim rises — it is *highest at dim=48* (0.36/0.53/0.44), where headroom is smallest. A low-dim-projection artifact would vanish at high dim; the opposite happened → the cyclic structure is intrinsic to the LLM forgetting geometry, not induced by the 12-d readout.

## Where Face-1c now stands (CONFIRMED, with honest scope)
1. **Non-transitive forgetting cycles at LLM scale are REAL** (survived a pre-registered confound control). The vision claim "interference is globally orderable, no cyclic tasks" is a property of the FROZEN-VISION regime, not of interference in general. On a plastic LLM (LoRA-SFT), the directed forgetting order carries a reproducible cyclic (rock-paper-scissors) component that coexists with a reproducible dominance tendency. **This is the genuinely new frontier result.**
2. **The a-priori d′→φ law does NOT transfer.** Across the grid ρ(d′,φ) is projection-sensitive (+0.04 … +0.65) and sign-FLIPPED vs vision's clean −0.6 — i.e. no stable law. Vision-specific, confirmed.

## Remaining honest caveats (not fatal, but bound the claim)
- **Weak forgetting** (medFgt 0.02–0.06 across grid) → low SNR; the cyclic effect is REAL (reproducible across seeds AND projections) but SMALL. A stronger-perturbation regime (more steps / higher rank) would raise SNR and is the natural robustness follow-up.
- **Projection family:** all readouts are random-Gaussian projections (raw features saturate). Persistence across dim 8→48 + 3 seeds substantially de-risks this, but the GOLD-STANDARD confirmation is de-saturating WITHOUT a projection (harder negatives / k-way accuracy metric).
- Single model (Qwen2.5-1.5B), single dataset (banking77), single layer (14), n=16 tasks, LoRA r=16. No cross-model/dataset replication yet.
- Not purely cyclic — a reproducible dominance component coexists (dom-ρ 0.5–0.9).

## Bottom line
A genuinely novel LLM-scale finding — **forgetting-interference cycles that vision lacks** — established through a pre-registered screen + a pre-registered confound control, with the falsifier (projection artifact) explicitly tested and rejected. Cheap ($4.47 all-in), honest, and frontier. Next robustness rung (if pursued): stronger-perturbation SNR + a non-projection de-saturation, then a 2nd model/dataset.

---

# P1 PLACEBO / NULL CONTROL (2026-07-18): cyclic finding SURVIVES the null — GENUINE, not a pipeline artifact

Prompted by external pushback (projection-manufactures-curl #1; low-SNR Hodge-hallucination #2). `face1c_placebo_lambda.py` → `out/face1c_placebo.json`. Three conditions on one base/projection/caps/seeds: **REAL** (task-specific LoRA), **RANDTEXT** (task-AGNOSTIC LoRA, matched forgetting), **NOISE** (Gaussian weight-noise, ‖Δθ‖ matched per-column). 16 caps × 3 seeds, projections {12,24}×{0,1}. $2.17 + smoke $0.24.

**RESULT — curl-ρ is REAL-ONLY:**
| condition | curl-ρ (4 projections) | dom-ρ | med medFgt |
|---|---|---|---|
| **REAL** | **0.331, 0.364, 0.296, 0.322** (median 0.331) | 0.44–0.79 | 0.038 |
| RANDTEXT (matched forgetting) | 0.001, 0.017, 0.007, −0.005 (**~0**) | 0.47–0.72 | 0.053 |
| NOISE (matched ‖Δθ‖) | −0.05, 0.11, −0.02, −0.05 (**~0**) | 0.22–0.66 | 0.016 |

**Pre-committed GENUINE verdict fired:** real curl-ρ 0.331 clears placebo-max 0.006 by ≥0.15.

**Kills #1 and #2:**
- #1 — placebos use the *identical* projection; if the projection made the curl, they'd show it. They don't. Projection is not the source.
- #2 — RANDTEXT is the airtight control: *real LoRA, matched forgetting magnitude* (medFgt 0.053 ≥ real 0.038), same low-SNR regime, same Hodge → curl 0.006. Low-SNR Hodge is NOT hallucinating the curl; it's signal.

**Clean dissociation (bonus):** dominance-ρ reproduces in ALL conditions (incl. both placebos) → the transitive/fragility component is partly INTRINSIC to tasks. Only the **cyclic residual is exclusively task-specific-interference-driven**. Hodge is correctly separating an intrinsic dominance axis from a genuine cyclic one — exactly the right behavior, and it argues the whole decomposition is sound.

## Where Face-1c stands now (well-controlled)
The LLM-scale cyclic (rock-paper-scissors) forgetting structure is established through: pre-reg screen → projection-variation control (curl persists across 12 projections, strongest at high dim) → **placebo/null control (curl real-only; two independent nulls ~0)**. Three of Gemini's/the-audit's four concerns (projection, SNR-hallucination, plus the earlier attenuation worry) are now addressed by controls.

## Remaining open (honest): critique #3 — single model/dataset
Still Qwen2.5-1.5B / banking77 only (n=1 model, n=1 domain-cramped dataset). The cf_cascade "n=1 eroder (BLOOM)" lesson applies. **Next rung = P4: a 2nd architecture (Llama-3.2-1B / Pythia-1.4b / Gemma-2-2b) + a 2nd, diverse task suite (clinc150 / cross-domain).** Also useful: P3 (raise medFgt off the 0.02–0.06 floor to lift SNR — the effect is real but small). Face-1c all-in now ~$6.88.

---

# P4 CROSS-SUBSTRATE REPLICATION (2026-07-18, $1.41): the STRONG cyclic claim does NOT generalize — pre-committed DOWNGRADE fired

**PREREG lock** (PREREG_FACE1C_P4) · Pythia-1.4b (GPT-NeoX, ≠Qwen) + clinc150 (150 intents / 10 domains, cross-domain ≠ all-banking). Arch-agnostic LoRA `target_modules='all-linear'`, `LAYER=n_layers//2` auto. 3 cells × 16 caps × 3 seeds × 200 steps × projection grid {8,12,24,48}×{0,1,2}. `face1c_p4_lambda.py` → `out/face1c_p4.json`. a10, 66 min, $1.41 (smoke $0.16 + real $1.41; ledger-logged, self-terminated). Face-1c all-in now ~$8.3, Face-1 arc total ~$10.9.

**RESULT — in-band median curl-ρ (pre-committed bar 0.30 in ≥66% of projections):**
| cell | axis tested | curl-ρ | dom-ρ | frac≥0.30 | persists |
|---|---|---|---|---|---|
| **pythia1.4b:clinc150** | BOTH new (money) | **0.194** | 0.388 | 0/12 | **NO** |
| pythia1.4b:banking77 | architecture | 0.204 | 0.576 | 0/12 | NO |
| qwen:clinc150 | dataset | 0.221 | 0.357 | 1/12 (0.083) | NO |

vs original Qwen×banking77 curl-ρ ≈ **0.37**. **Swapping EITHER axis alone drops curl-ρ to ~0.20, below the bar.** The pre-committed falsifier (both-new curl-ρ < 0.30) fired → **DOWNGRADE**.

## Verdict (three honest layers)
1. **STRONG claim FALSIFIED as pre-registered.** "Reproducible rock-paper-scissors cycles (curl-ρ≥0.30) at LLM scale" was **specific to the Qwen2.5-1.5B × banking77 corner.** Neither architecture nor dataset generalizes it — the amplified corner needed BOTH. Gemini critique #3 (cf_cascade "n=1 eroder = BLOOM") lands. My prior was REPLICATES (~70%); it LOST — the surprising branch again. **This honest negative is the win of P4.**
2. **Weak positive residual, not yet callable.** All 36 grid points are POSITIVE (medians 0.19–0.22), and the placebo put the null floor ~0 ON QWEN/BANKING77. IF that ~0 floor holds on the new substrates, a genuine-but-weak sub-threshold cycle exists everywhere; but WITHOUT a matched placebo here, ~0.20 could instead be an elevated pipeline floor (→ effectively absent). **Strong claim is down either way.** Disambiguation = **P4b: one RANDTEXT placebo on pythia:clinc150 (~$1.4)** — weak-real (placebo ~0) vs absent (placebo ~0.20).
3. **What GENERALIZES cleanly:** the **dominance / pecking-order (transitive) axis** reproduces across arch AND data (dom-ρ 0.36–0.58 every cell) — consistent with the placebo's "fragility is partly INTRINSIC." And the vision **d′→φ a-priori law stays dead/unstable** (ρ swings +0.68 … −0.59, no stable sign) — vision-specific, re-confirmed on new substrate. → The Hodge decomposition's TRANSITIVE half is the robust, general finding; the CYCLIC half is real-but-fragile, Qwen/banking77-amplified.

## Corrected bottom line for the whole Face-1c cyclic arc
The genuinely-general result is **substrate-dependent interference STRUCTURE with a reproducible transitive (dominance) core**; the cyclic (non-transitive) component is present-but-weak and only clears a "reproducible cycles" bar in the Qwen×banking77 regime. The earlier headline ("LLM forgetting cycles that vision lacks — a new frontier result") must be **downgraded to:** *dominance/pecking-order generalizes across arch+data; a weak cyclic residual exists but robust reproducible cycles are Qwen/banking77-specific.* Controls now run: projection-variation (passed), placebo/null (passed, Qwen/bank77), attenuation (addressed), **cross-substrate (FAILED for the strong claim, DOWNGRADE).**

**Next (optional):** P4b placebo to settle weak-real vs absent; P3 SNR push (though P4 shows higher medFgt did NOT buy more curl reproducibility — cells at medFgt 0.07–0.09 gave curl-ρ ~0.20, so SNR is likely not the lever). Honest stopping point: the strong cyclic claim is retracted to substrate-specific; the transitive-dominance finding stands and generalizes.
