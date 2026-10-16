# VERDICT — Why BLOOM erodes: the confound break. Eroder = a CONJUNCTION, not one knob; frac_in_V mechanism REFUTED (direction, not energy). 2026-07-09.

Consolidates Probe-14 (XGLM, LOCK b30d6579) + Probe-15 (Falcon-rw-1b, LOCK 623aa5bd) + the BLOOM-frac re-run. Follows the
Probe-11/12/13 reckoning, which left the open question "why is BLOOM the lone eroder — ALiBi or corpus?" All runs banked, $0
(all instances self-terminated; the patched launcher's teardown held through 2 infra flakes + 3 early-stops this session).

## Headline
The childhood→protection dose-response SIGN is architecture-dependent, and the eroder (BLOOM-560M) needs a **CONJUNCTION** of
properties — **ALiBi positional encoding AND multilingual pretraining**. Neither alone flips the sign. And the mechanism is
**directional, not energetic**: protectors and the eroder reorganize the SAME fraction of childhood movement into the task-A
Fisher curvature subspace V; what differs is the direction within V. The registered `frac_child_in_V` prediction (BLOOM low)
is REFUTED.

## The 2×2 (positional × corpus), 5 cells — dose-response of directional protection N_phase2 vs childhood dose
| cell | positional | corpus | size | N_phase2 (gentle→strong) | slope | class |
|---|---|---|---|---|---|---|
| OPT-350M | learned-abs | English | 350M | +0.071 → +0.157 | + | protect |
| Pythia-410M | rotary | English | 410M | +0.074 → +0.113 | + | protect |
| **Falcon-rw-1b** | **ALiBi** | English | 1.3B | +0.04 → +0.21 (N=9) | + | **protect** |
| **XGLM-564M** | sinusoidal | **multilingual** | 564M | +0.020 → +0.149 (N=27) | + | **protect** |
| **BLOOM-560M** | **ALiBi** | **multilingual** | 560M | +0.044 → −0.032 (crosses 0) | − | **ERODE** |

- **ALiBi alone does NOT erode**: Falcon-rw-1b (ALiBi + English, 1.3B) is a clear protector (N_phase2 +0.21 plateau, z≫5, N=9).
- **Multilingual alone does NOT erode**: XGLM (sinusoidal + multilingual, 564M) is a clear protector (N_phase2 +0.149, z≈9.5, N=27).
- BLOOM is the **unique cell with both**. → the eroder is an **ALiBi × multilingual interaction**, OR a BLOOM-specific third factor
  (leading candidate: BLOOM's **post-embed LayerNorm**, which OPT/Pythia/Falcon/XGLM lack). NOT explained by either knob alone.
- **Scale is exonerated too**: OPT-350M and OPT-1.3B both protect; Falcon protects at 1.3B. Size does not flip the sign.

## The mechanism finding: frac_in_V REFUTED — it's DIRECTION, not ENERGY (honest null)
New read-only metric `frac_child_in_V` = fraction of childhood weight displacement (ch − w_A) landing inside V (chance ≈ k/cin ≈
0.10; math unit-checked: random→0.125, in-V→1.0, V orthonormal).
| cell | frac_child_in_V (gentle→strong) |
|---|---|
| XGLM (protect) | 0.447 → 0.365 |
| Falcon (protect) | 0.461 → 0.203 |
| **BLOOM (erode)** | **0.435 → 0.276** |
All three are HIGH (~3–4× chance) and **mutually indistinguishable**. Registered prediction was BLOOM-low; it is BLOOM-HIGH,
≈ protectors. → "reorganizes inside V" is UNIVERSAL; energy-in-subspace does NOT discriminate protect from erode. **The lever is
the DIRECTION within V** (alignment of the in-V childhood displacement with the interference direction), which a magnitude ratio
cannot see. This was pre-flagged in both locks as the fallback. → next probe = cosine(in-V childhood displacement, phase-2
interference direction); this is Thread A.

## What survives / what's retired
- **SURVIVES + STRENGTHENS:** concentration on the interference direction (N_phase2 ≫ N_random) across 5 cells & 3 sizes; the
  dose-response-SIGN law (Probe-13); the corpus-vs-positional confound is now BROKEN (single-factor explanations empirically
  refuted). The "why BLOOM" question is answered to the level of "a conjunction, not one knob."
- **REFUTED:** the `frac_child_in_V` energetic-mechanism hypothesis (my registered prediction). Honest null — reported, not spun.
- **STILL OPEN:** (a) the DIRECTIONAL mechanism (cosine metric — Thread A, turns the null into a positive result); (b)
  conjunction-vs-post-embed-LN disambiguation (ablate BLOOM's post-embed LN, or find a 2nd ALiBi+multilingual eroder — n=1
  eroder caps generalization); (c) does it appear in a modern RoPE architecture (sets the reach).

## PROBE-18 RESULT (controllability / positive-direction, LOCK d8832e72, N=15) — NULL: single-axis nullify FAILS.
Steered the childhood gradient orthogonal to the interference direction dVp (vs a random-dir control). OPT-350m:
| arm | N_phase2 | |
|---|---|---|
| baseline | +0.149 ± 0.026 | native protection |
| **orth_dVp** | **+0.122 ± 0.038** | NOT nullified (≈ baseline, ~0.6σ) |
| orth_rand (control) | +0.158 ± 0.025 | ✓ ≈ baseline |
Forbidding the childhood from MOVING ALONG dVp did NOT switch off protection → protection is NOT the childhood's movement along
the single interference axis; it reshapes curvature along dVp via movement in MANY OTHER in-V directions → the operative
"direction" is a BROAD SUBSPACE, not one realized axis (consistent with / refines the Probe-7 reorganization picture). The
causal-directional proof did not land via this lever; nullify-controllability REFUTED. Control clean (orth_rand ≈ baseline).
BLOOM secondary NOT fired (primary high-power answered). Process: held to landed N=15 (n=5 mid-run was +0.021 → would have
mis-called nullify). File `gateA/probe18_steer_opt350m.json`. (Amplify/flip untested; the broad-subspace nature makes a clean
single-lever controllability demo unlikely — the WHY-thread's directional mechanism is characterized by elimination + this
bound, not a positive single-axis handle.)

## Application surface (repo scope, $0)
Two real CF surfaces in the user's tools: **the Librarian** (SportsThought — fine-tuned cross-encoder reranker / Pole-A adapter,
extended 3→6 sports; ALREADY instrumented with `b2_cf.py` measuring retention AUC with an inV-vs-random-swap control — the SAME
measurement as probe13 — and a live deployed rerank-antagonism symptom) and **the physics detector CNN** (continual-learned
across generator paradigms; `cl_gpm_diffusion.py`). DeepDive (frozen-API prompting) and the "NBA motion tracker" (tabular
tracking stats, classical modeling) are NOT CF-applicable. Best application probe = **Librarian sequential-domain surge**:
continue-train the reranker on a newly-added domain, measure prior-sport retention AUC/recall (not P@1 — saturates) with the
inV-vs-Vswap control + the joint-train baseline that already exists. Thread A (cosine metric) and this share the SAME Fisher-V
measurement → the mechanism probe plugs into the product.

## Scope / caveats (honest)
- Falcon N=9, XGLM N=27, BLOOM re-run N=25 (orig N=50 backed up `_N50_orig.json`) — sign results, not tight effect-size
  comparisons; all signs are z≫5. n=1 eroder architecture (BLOOM) — the conjunction rests on one corner cell.
- One task-A (SST-2), one phase-2 (subj), one childhood (mix5). frac_in_V is energy-weighted over carriers; XGLM/BLOOM shape-
  matched so their frac baselines coincide (clean contrast); Falcon is 1.3B (different shape) but its frac is still high.
- Lit-novelty NOT yet assessed (3-agent adversarial pass IN FLIGHT) — do not claim novelty until it lands.

## LIT-CHECK (2026-07-09, 5-agent adversarial pass — gates novelty)
- **C1 (ALiBi×multilingual conjunction flips the SIGN of a warmup's retention effect) = NOVEL, but FRAGILE.** No prior work links
  positional encoding to the SIGN of forgetting, nor gates it on the ALiBi+multilingual conjunction. Closest (all ADJACENT, none
  scoop): Mirzadeh "Architecture Matters in CL" (arXiv:2202.00275, width/depth not PE); Liu & Niehues "Conditions for CF in
  Multilingual Translation" (2510.19546, multilingual conditions but no PE, downplays architecture); Cossu 2205.09357 /
  Ramasesh ICLR'22 (arch-dependent forgetting MAGNITUDE); Ruiz-García Sci.Reports 2022 (arXiv:2108.03940 — arch flips forgetting
  SIGN, but toy algorithmic task, no PE, no auxiliary phase). 5-agent pass converged; agents self-caught 2 hallucinated scoops
  (Steele 2603.02224 energy-frac; He 2602.20796 Fisher thesis) → trust raised; C1 rests on absence-of-evidence, not proof. SKEPTIC WARNING: a 2×2 sign-flip on ONE arch × ONE corpus-property
  reads as a confound — must rule out that "ALiBi×multilingual" is a proxy for flatter cross-lingual minima, which would collapse
  C1 into C3. → highest-value next test = CURVATURE/FLATNESS disambiguation (below).
- **C2 (direction-not-magnitude in Fisher V) = PARTIALLY-KNOWN.** "Direction beats magnitude" is published (Fisher alignment-ratio
  magnitude-null result ~MixSD 2605.16865 [attribution UNCERTAIN — verify]; Steele "Subspace Geometry Governs CF in LoRA"
  2603.02224, gradient-subspace/principal-angle, rank-independent; AlignGuard-LoRA 2508.02079). **NOVEL/unclaimed = OUR specific
  energy-matched NULL: protectors & eroder deposit the SAME frac_in_V (~0.2–0.46) so a metric-matched energy measure FAILS to
  discriminate.** → the frac_in_V null is a KEEPER contribution; the (planned) cosine "direction discriminates" result would
  REPLICATE known work → cosine DEMOTED to internal validation.
- **C3 (warmup reorganizes → less damage/unit movement, slope not suppression) = KNOWN.** Mehta 2112.09153 (JMLR'23, flatter minima,
  "comparable distance moved," curvature-coeff reduction) scoops nearly verbatim; Mirzadeh 2006.06958 (wide minima); Springer
  2605.02105 (SAM mid-training, explicitly slope-not-step-size).
- FLAGS: several scoops are 2026-dated arXiv (2601–2606) → concurrent-vs-prior needs date-check before any writeup; one agent
  self-retracted a hallucinated energy-fraction "confirmation" of Steele 2603.02224 (it reports effective-rank, NOT energy).
- **NET: novelty = (1) C1 conjunction sign-flip + (2) the energy-matched frac_in_V null. Defense of C1 = curvature disambiguation
  + break n=1. Cosine metric demoted. Application (Librarian) is orthogonal to novelty and stands.**

## ★ PROBE-16 RESULT (curvature disambiguation, LOCK 2a97de80, N=8) — C1 SURVIVES the flatness kill-test.
Task-A Fisher spectrum (anisotropy A=λmax/trace [sharp→high]; eff_rank ER=trace²/Σλ² [flat→high]); scale-robust, cross-arch.
| cell | class | baseline A | baseline ER | childhood@7e-6 dA_frac | dER_frac |
|---|---|---|---|---|---|
| XGLM | protect | **0.710 (sharpest)** | 2.35 | −0.010 | +0.144 |
| Falcon | protect | 0.492 | 5.05 | −0.164 | +0.767 |
| **BLOOM** | **ERODE** | 0.494 | **6.72 (flattest)** | −0.065 | +0.419 |
- **Reduction (a) intrinsic-sharpness = DEAD (deterministic, no seed noise):** the eroder (BLOOM) has the FLATTEST minimum
  (ER 6.72), the SHARPEST cell (XGLM, A 0.710) PROTECTS. Baseline sharpness runs OPPOSITE to "sharp minima forget more."
- **Reduction (b) childhood-sharpens = DEAD:** ALL cells FLATTEN at the eroding dose (dA≤0, dER>0); nobody sharpens. Flattening
  degree Falcon(+0.77) > BLOOM(+0.42) > XGLM(+0.14) is NON-MONOTONE with outcome (XGLM flattens LEAST yet protects; BLOOM
  flattens MORE than XGLM yet erodes). Flattening cannot produce NEGATIVE protection.
- ⇒ scalar curvature does NOT explain the protect/erode sign ⇒ the effect is DIRECTIONAL (consistent w/ Probe-13 N_random≈0). **C1
  UPGRADED: novel-fragile → NOVEL, survived its hardest lit-flagged reduction.** Files `gateA/probe16_curvature_{xglm564m,falcon_rw1b,bloom560m}.json`.
- CAVEATS: n=8 (stable; baseline deterministic). ELIMINATION result (not-flatness) — the POSITIVE directional characterization
  (cosine of in-V childhood disp vs interference dir) is still to be shown. n=1 ERODER arch — conjunction generalization still
  needs the post-embed-LN ablation / 2nd ALiBi+multilingual model.

## HIGHEST-VALUE NEXT (reprioritized by lit-check; #1 now DONE)
1. ~~C1 curvature disambiguation~~ **DONE (Probe-16) — C1 survived; scalar-flatness reduction refuted.**
2. **Break n=1 eroder** — post-embed-LN ablation on BLOOM (retrain w_A with it off → does erosion stop?) or a 2nd ALiBi+multilingual model.
   - **Probe-17 (LN ablation, LOCK 60f7056d) — ATTEMPTED, INCONCLUSIVE (anchor-confounded).** bloom560m_noln
     (`transformer.word_embeddings_layernorm`→Identity), w_A retrained. Removing the LN cost ~6pt task-A (anchor 0.83→0.7725,
     below the pre-registered 0.78 flag). Dose-response N=25: BLOOM-noLN is a FLAT NULL — N_phase2 within ±0.02 of zero at every
     dose (7e-6: −0.009±0.024), no trend; NOT a protector, NOT a clear eroder (overlaps both 0 and BLOOM's −0.022). Weaker anchor
     shrank all effect sizes → cannot separate "LN irrelevant" / "LN attenuates" / "weaker-deposit artifact." **Confound-free
     takeaway: removing the LN did NOT restore protection → post-embed LN is NOT SOLELY responsible for the sign.** Fork
     conjunction-vs-LN NOT cleanly resolved. (Process note: I called it "LN-not-cause/superimposable" at n=5 — premature; landed
     N=25 is a muddier null. Anchor drop foreseeable + flagged; ablation was not the best disambiguator.) Files
     `gateA/probe13_doseresponse_bloom560m_noln.json`, `build_wA_bloom560m_noln.json`.
   - Cleaner disentanglers still owed: a 2nd ALiBi+multilingual model, or an anchor-preserving LN test. The conjunction-OR-LN
     hedge in the Headline STANDS until one lands.
3. **(demoted) cosine directional metric** — internal validation only; replicates Steele's direction>magnitude.
4. **Application (own track)** — Librarian sequential-domain surge.

## Status
Probe-14/15 CLOSED. Confound broken (conjunction, not single-knob); frac_in_V null (direction not energy). Files
`gateA/probe13_doseresponse_{xglm564m,falcon_rw1b,bloom560m}.json`, ckpts `gateA/{xglm564m,falcon_rw1b}_sst2_wA.pt`. Next:
lit-check (gates) → Thread A cosine metric ‖ Thread B Librarian sequential-surge (parallel, per user). Launcher scp-hang FIXED
+ fire-validated.
