# The Cosine Geometry of Interference: one variable governs learning, forgetting, unlearning, and protection

**Nathan Humphrey · 2026-07-11 · internal technical note**

## Abstract

Catastrophic forgetting, machine unlearning, and continual-learning protection are usually studied as separate problems with
separate machinery (EWC, GPM, OGD; gradient-ascent unlearning; rehearsal). We show they are three faces of one geometric quantity:
the **cosine between two capabilities' discriminative directions** in a frozen representation. Define, for a capability *p*, the
direction `d_p = normalize(mean(features | p) − mean(features | reference))` — the direction a linear readout moves to detect *p*.
Then across a linear readout on a frozen backbone:

- **Forgetting** of *A* when learning *B* rises as `cos(d_A, d_B)` falls — a calibrated, out-of-sample-predictive law
  (`forget ≈ −0.445·cos + 0.266`, held-out R² ≈ 0.5) that transfers *quantitatively* to independent production systems.
- **Unlearning collateral** on a protected capability rises as `+cos(target, protected)` — anti-parallel capabilities excise
  cleanly; aligned ones are inseparable.
- **Engineered protection** leaks to non-target capabilities as `+cos` — you can protect a cosine *cluster*, not an arbitrary subset.

The relationship is **modality-general** — the same structure appears in a text reranker, a deepfake-image detector, and a
text-to-speech (audio) detector — and **quantitative**: a law calibrated on CIFAR-100 class pairs predicts a production deepfake
detector's measured collapse to within 0.01 AUC. The organizing role of the cosine gives a single, cheap, predictive decision
variable for when to route, what will forget, what can be unlearned, and what can be protected.

---

> ## ⚠ Erratum / correction (2026-07-27) — the deepfake-detector cosine is domain-confounded
>
> The recurring **deepfake-detector** example below — swap↔diffusion cos ≈ **−0.93/−0.95**, used as the
> "anti-parallel forgery paradigms" case (§2 quantitative transfer, §3 erase-diffusion, §5 summary table) —
> rests on a cosine that is **dominated by DOMAIN distance, not forgery-mechanism geometry.** A later confound
> control (`v2/learned_detector/antagonism_confound_check.py` → `antagonism_confound_result.json`; detector-cf
> domain-control arc) decomposes the shared shift into a pure-forgery and a pure-domain component:
>
> | quantity | cos vs swap | ‖·‖ |
> |---|---|---|
> | `d_shared` = diff_fake − shared_real (the −0.95 as reported) | **−0.954** | — |
> | `d_forgery` = diff_fake − **own-domain** real (domain-controlled) | **+0.669** | 1.29 |
> | `d_domain` = real_A − real_B (pure real-domain offset) | **−0.909** | 7.52 |
>
> The pure forgery direction is **+0.67 (ALIGNED, not anti-parallel)**; the domain vector is **5.8× larger** in
> norm, so the observed −0.95 is domain-dominated (`domain_artifact = True`). **What this does and does not change:**
> - **The LAW is unaffected.** `forget ≈ −0.445·cos + 0.266` predicts forgetting from *whatever* cosine it is
>   fed; feeding it the deepfake detector's (domain-dominated) −0.93 still predicted its measured collapse. The
>   quantitative-transfer demonstration stands **as a prediction from the measured cosine**.
> - **The MECHANISTIC label is wrong.** "Diffusion is an anti-parallel *forgery* paradigm" → it is an
>   anti-parallel **domain**. Read every "anti-parallel" claim about the deepfake paradigms below as a statement
>   about domain distance between the paradigms' image sources, not forgery geometry.
> - **The OPERATIONAL results survive, re-attributed.** Erase-diffusion-cleanly and route-to-own-head still work
>   (R4 domain-controlled re-run: the downstream wins HOLD) — but they work because the paradigms are
>   **domain-separated**, not because their forgeries are anti-parallel. On own-real references the swap-family
>   antagonism collapses (−0.700→−0.212) and the max-ensemble route merges diffusion toward swap (0.964→0.683).
>
> The text-reranker (§2) and audio-TTS (§5) examples are **not** affected by this note. See
> `project_detector_cf_domain_control` + the R4 verdict for the full re-attribution.

## 1. The variable

For a frozen feature extractor `φ` and a set of "capabilities" (paradigms, classes, tasks) each with positive examples and a shared
reference/negative pool, the discriminative direction of capability *p* is

```
d_p = normalize( mean_x∈p φ(x) − mean_x∈ref φ(x) )
```

This is the direction a linear readout `w·φ(x)+b` aligns with to detect *p*. Two capabilities *A*, *B* interact through
`cos(d_A, d_B)`: **+1** = same direction (redundant), **0** = orthogonal (independent), **−1** = opposed (antagonistic). Everything
below is a statement about this one number.

## 2. Forgetting is a calibrated function of −cos (Thread #1)

**Substrate.** CIFAR-100, frozen ImageNet ResNet18 (512-d). 50 classes as capabilities, a shared negative pool from 10 reserved
classes, 2450 ordered class-pairs. For each pair: train a readout on *A*, continue-train on *B*, measure *A*'s AUC drop.

**Result.** Forgetting is monotone-decreasing in cosine, spanning a ~7× range (0.058 at cos≈+0.5 to 0.409 at cos≈−0.4).
Spearman ρ(cos, forget) = **−0.74**; splitting pairs and fitting `forget ≈ −0.445·cos + 0.266` on half predicts the other half at
**held-out R² ≈ 0.45–0.52**, MAE ≈ 0.08. The **decision boundary** c*(forget≥0.10) ≈ +0.2–0.3: below it, expect >0.10 AUC loss →
route/isolate; above it, sharing is cheap. This upgrades the sign heuristic to a calibrated predictor.

**Quantitative cross-substrate transfer.** Plugging *other systems'* measured cosines into the CIFAR-fit law:
- Deepfake detector, swap↔diffusion cos −0.93 → predicted forget **0.68**; *measured* swap collapse 0.913→0.224 = **0.689**. Match < 0.01.
- Text reranker, rules↔fantasy cos +0.67 → predicted **−0.03** (≈0, benign); *measured* erosion +0.008. Match.

(Absolute-AUC scales differ across substrates, so the 0.68-vs-0.689 coincidence is partly fortuitous — but sign and magnitude
landing this close across three independent systems is strong evidence the law is quantitatively transferable, not merely monotone.)

## 3. Unlearning collateral is a function of +cos (Thread #3)

Take a universal detector head over four generator paradigms and try to **erase one**: (a) DIRECTIONAL — remove the head's response
to the target's direction orthogonalized against the others; (b) NAIVE — relabel target→negative and fine-tune; (c) RANDOM control.

- **Erase diffusion** (anti-parallel to the others, cos ≈ −0.9): clean. Diffusion 0.904→**0.204**, others move ≤ 0.04. Removing an
  anti-parallel capability *spares* the rest.
- **Erase swap** (aligned with the "swap-family" celebdf/dfdc, cos +0.6/+0.8): impossible to isolate. Swap only 0.656→0.477 and it
  **drags celebdf/dfdc down 0.07–0.08**; no method cleanly separates them.

**Collateral ∝ +cos(target, protected).** A capability is surgically unlearnable to the degree it is anti-parallel to what you keep.
(Methodological note, reported honestly: the pre-registered "specificity ratio" rewards no-ops — RANDOM scored highest on the
entangled target by doing almost nothing — so we report raw erase-vs-collateral instead.)

## 4. Protection leaks along +cos (Thread #4)

The inverse problem: *design* protection for a chosen set *S* by projecting a strong interference (here, collapsing the head toward
diffusion-only) off `span{d_p : p∈S}`, and ask whether the realized protection profile equals the desired one.

- Protecting `{swap}` raises swap (+0.153) **but leaks +0.065 to aligned celebdf/dfdc** and −0.008 to anti-parallel diffusion — you
  cannot protect swap alone.
- Protecting `{diffusion}` is **independent** of the swap-family (no leak).
- Protecting the whole family `{swap,celebdf,dfdc}` is the clean realizable profile.

**The realizable protection profiles are exactly the cosine clusters.** Protection-leakage is positive to aligned capabilities,
~0 to anti-parallel — the same geometry as §2 and §3, a third time.

## 5. Modality generality (Thread #5) and the learn-together case

**Audio.** Frozen log-mel features + linear real-vs-fake heads over 7 TTS generators. The router recovers the same structure without
tuning: six neural-TTS generators form a compatible cluster (mutual cos +0.55…+0.96); **speecht5 is a lone anti-parallel antagonist**
(cos −0.11…−0.37). Forgetting-vs-cosine ρ = −0.52 (monotone by bin; the most-aligned pairs coexist at forget 0.11), and cluster-aware
max-ensemble routing recovers the worst paradigm from 0.00→0.93. Same law, third modality (after text and vision).

**Learn-together (route arms, prior work).** Clustering capabilities by cosine and giving each anti-parallel cluster its own head +
max-ensemble scoring recovers a production detector's diffusion detection 0.68→0.96 label-free; GPM (which assumes a free lane)
instead *starves* the newcomer. Within a moderately-aligned cluster there is a fixed retention budget — head-count is the dial.

## 5b. The frozen/plastic DICHOTOMY (Threads #6, #6b, #1)

The cosine law is a READOUT-level phenomenon. When the backbone is made plastic (fine-tuned), the governing variable CHANGES:

| regime | governing variable | law |
|---|---|---|
| **frozen backbone** (linear readout) | DIRECTION | forget ∝ −cos(d_A, d_B) — ρ −0.74, held-out R² ≈ 0.5, quantitative, modality-general |
| **plastic backbone** (fine-tune) | MAGNITUDE | forget ∝ ‖Δθ‖ (weight-space distance) — ρ +0.65; direction-agnostic (feature/Δθ/gradient cosines all ρ≈0, #6b) |

A frozen readout cares **which way** the new task points relative to the old; a plastic representation cares only **how far** it moves.
Forgetting under plasticity is governed by weight-space travel distance, not by pairwise alignment — established cleanly with the
theoretically-correct function-space (NTK gradient) cosine nulling at ρ=+0.02 despite full dynamic range (#6b), and ‖Δθ‖ winning the
predictor bake-off at ρ=+0.65 (#1). (Honest: the hypothesised feature-drift ‖Δφ‖ came 3rd, ρ=+0.15 — the damaging travel is not
globally visible on a neutral probe set.)

### Phase diagram of the readout law (Thread #5)
The readout law's SHAPE is scale-universal (every unsaturated backbone: forget ∝ −cos, ρ −0.64…−0.75) but its PARAMETERS sit on a
phase diagram: CNN scale SHRINKS the magnitude and the route-boundary (medFgt 0.23→0.11, c* +0.31→+0.10 across ResNet18→152);
pretraining OBJECTIVE sets the slope (self-supervised DINOv2 steepest at −0.82, ~2× supervised CNNs); and some architectures
(ViT/ConvNeXt) have frozen features so interference-robust the law saturates (unmeasurable without a harder task). c* is a coordinate,
not a constant.

## 6. The unified picture

The discriminative-direction cosine is the single organizing variable of the interference phenomenology:

| operation | governing relation | consequence |
|---|---|---|
| **learn together** | cluster by cosine | share a head within a cluster; isolate the anti-parallel one |
| **forget** (§2) | forget ∝ −cos (R²≈0.5, predictive) | route below c*≈+0.2–0.3 |
| **unlearn** (§3) | collateral ∝ +cos | anti-parallel excises clean; aligned is entangled |
| **protect** (§4) | leakage ∝ +cos | engineerable only over cosine clusters |

modality-general (text/vision/audio) · quantitative (predicts production systems) · cheap ($0 linear probes on frozen features).

## 7. Applications already realized

- **Text reranker (Librarian):** rules↔fantasy cos +0.67 → *do not* route; joint training is correct; the "keeper" collision is an
  upstream retrieval / hard-negative issue, not a reranker-direction antagonism. (The diagnostic overturned a proposed re-architecture.)
- **Deepfake detector:** swap↔diffusion cos −0.93 → route diffusion to its own head; label-free max-ensemble recovers it 0.68→0.96.
- **Portable rule:** compute one cosine to decide route/don't-route, predict forgetting magnitude, judge unlearnability, and bound
  what protection is achievable — before spending any training compute.

## 8. Limitations (honest)

- **Head-level — now TESTED, and the law is READOUT-SPECIFIC (Thread #6).** All the strong results are linear readouts over *frozen*
  backbones. When the backbone is made *plastic* (LoRA fine-tune on CIFAR-100/ResNet18), NONE of three cosines (feature, weight-update
  Δθ, Fisher-subspace) strongly predicts representational forgetting: the feature cosine — well-ranged — drops from head-level
  ρ=−0.74 to **ρ≈−0.12** (nearly flat), and Δθ/Fisher show no dynamic range (near-orthogonal in high-dim LoRA space; a confound).
  Representational forgetting is instead dominated by *which* B-task (how disruptive it is), not A–B alignment — consistent with
  Probe-18's broad-subspace reorganization. **The cosine law lives at the interface between a FIXED representation and a LINEAR
  decision, not in the representation's own plastic dynamics.** — **NOW SETTLED (Thread #6b):** the open question ("does a
  function-space/NTK cosine on a FULL fine-tune revive it?") is answered NO. With full fine-tuning (no LoRA confound) and the
  theoretically-correct t=0 loss-gradient alignment cos(g_A,g_B) — which has *genuine* dynamic range this time (~[−0.20,+0.28]) — the
  correlation with forgetting is **ρ = +0.016 ± 0.020 (3 seeds), i.e. ZERO** (raw and within-B). The first-order NTK loss-transfer
  picture does not describe re-probe representational forgetting; it is a higher-order global reorganization effect. The boundary is
  established cleanly, not conjectured. The feature cosine retains only a faint −0.12 echo (~1.5% variance). See
  `VERDICT_REPRESENTATIONAL_LAW_2026_07_11.md` and `VERDICT_GRADIENT_ALIGNMENT_2026_07_11.md`.
- **The saturation trap, three times.** The interference *sign/magnitude* is only measurable when forgetting is **partial**. It bit
  the audio run (log-mel saturates real-vs-synth AUC to 1.0), forced a hardened-regime check on CIFAR, and made two synthetic
  ALiBi×multilingual designs untestable (copy tasks forget to chance instantly). Any interference measurement must first engineer a
  non-saturating regime.
- **Estimated directions.** `d_p` are sample means; splits are hash-based (not always group-disjoint), inflating absolute AUCs — the
  *deltas* and *structure* are the claim, not absolute levels. Single seed on most sweeps.
- **Open mechanistic question.** *Why* the BLOOM (ALiBi × multilingual) conjunction flips the childhood-protection sign is not
  settled. It is directional, not a flatness artifact (Probe-16), but a controlled from-scratch test at tiny synthetic scale is
  null (the copy task saturates). A faithful test needs real small language models on real monolingual vs multilingual text — the
  next run (§9).

## 9. Open thread: the ALiBi×multilingual mechanism (Thread #2)

The novel observation from the base arc — that only the ALiBi×multilingual conjunction erodes (BLOOM the lone eroder), directional
rather than flatness-driven — remains unexplained at the causal level. Tiny synthetic 2×2 grids cannot test it (saturation). The
deliberate next experiment is a controlled from-scratch grid of **real** small LMs: {ALiBi, RoPE} × {monolingual, multilingual real
text}, trained to competence, then run through the childhood→interference protocol in a non-saturating regime. That run is scoped and
fired separately.

---

# Addendum (2026-07-13): the directed matrix, and the limits of the cosine

The sections above are readout-level and treat forgetting as a symmetric function of cos(d_A,d_B). A later block of work built the
*directed* forgetting matrix (plastic backbone: deployed frozen victim head, backbone full-fine-tuned on the perturber) and pushed on
what the cosine does and does not govern there. The results narrow the claim.

## 10. Interference is asymmetric, and reduces to two per-capability scalars

The directed matrix F[i←j] (forgetting of victim i after fine-tuning on perturber j) is strongly **asymmetric**: ρ(F_ij, F_ji) ≈ 0,
‖antisym‖/‖sym‖ ≈ 0.28. Since cos(d_i,d_j) is symmetric by construction, the cosine **cannot** be the whole story for directed
forgetting. Decomposing the matrix, the reproducible structure is **two per-capability scalars**:

```
F[i,j] ≈ μ + fragility_i + disruptiveness_j + noise
```

- **fragility_i** — how easily capability i is damaged. Best single base-measurable correlate is discriminability: ρ(d′_i, fragility) ≈ −0.5
  (well-separated capabilities resist forgetting). A margin-Jacobian ‖∂margin_i/∂θ‖ instrument was tried; the *raw* norm is scale-confounded
  (wrong sign) and the fractional version is untested.
- **disruptiveness_j** — how much j damages others; ≈ ‖Δθ_j‖ (weight-space travel), matching §5b, when magnitude has range (at fixed
  magnitude, e.g. equal-strength binary perturbers, it is direction-driven instead).

In a plastic directed matrix the *victim's identity* (fragility) explains more variance than the cosine (ΔR² ≈ 0.23 for the row effect vs
cosine R² ≈ 0.07). fragility ≠ disruptiveness (ρ ≈ −0.24): being fragile and being destructive are distinct axes.
(`VERDICT_ASYMMETRY_*`, `VERDICT_FRAGILITY_*`.)

## 11. The asymmetry is a transitive pecking order, not cycles

Helmholtz-Hodge decomposition of the antisymmetric part K = gradient (transitive dominance h_i − h_j) + curl (cyclic). A split-half test
(build F twice from disjoint image halves) separates signal from noise: the **dominance potential** h_i = (fragility_i − disruptiveness_i)/2
reproduces (ρ 0.51 CIFAR / 0.59 DTD), but the **cyclic residual does not** (ρ ≈ 0). So the ~50% apparent cyclicity is measurement noise;
interference is **globally orderable** — a transitive net-predator/prey ranking, no rock-paper-scissors. (`VERDICT_CYCLIC_*`.)

## 12. A plastic *directional* law — chased hard, then retired

We hunted a weight-space directional law for plastic forgetting: `forget(i←j) ∝ −cos(g_i, Δθ_j)`, with g_i = ∂margin_i/∂θ the victim's
margin-Jacobian and Δθ_j the perturber's displacement. The first-order term initially fit CIFAR at ρ ≈ +0.45, and the alignment (not the
magnitudes) carried it — encouraging, since it seemed to recover directional structure that gradient-cosine (#6b) and static feature
projections had missed, hiding in small deviations from high-dimensional orthogonality. **Hardening dismantled the generality of it:**

- **Architecture switch** (ResNet18 → MobileNetV3 → ViT): the pooled +0.45 deflated to +0.08–0.10; only the double-centered interaction
  survived (+0.15 / +0.30 / +0.36), and it did survive a pure transformer.
- **Dataset switch** (CIFAR → DTD textures / GTSRB signs, architecture held): the interaction **died** (DTD align_dc = −0.035). The law was
  CIFAR-specific.
- **Shape / descriptor / low-rank:** the directed interaction is real (asymmetric, mildly concentrated at small N) but **structureless** —
  no geometric descriptor fits it off CIFAR (feature, Jacobian, gradient, and centroid cosines all ρ ≈ 0 on DTD), and it is neither
  low-rank nor completable (best held-out completion R² ≈ 0.2 comes from the two main effects alone; adding interaction rank *hurts*).

Conclusion: the plastic per-pair directional interaction is real but has **no general law**; the durable plastic description is the
two-scalar magnitude structure of §10, not direction. The explicit lesson: **single-substrate robustness — even across three
architectures — is not generality; switching the training *data* was the falsifier.** (`VERDICT_DIRLAW_*`, `VERDICT_SHAPE_*`,
`VERDICT_CHARACTERIZE_*`, `VERDICT_LOWRANK_*`.)

## 13. The cosine is a forgetting law, not a transfer law

Whether the cosine also governs *transfer* (does aligned B help A, not just hurt it less?) was tested in three regimes, with headroom
engineered (base AUC ≈ 0.8) so improvement was possible:

| regime | what is measured | governed by cos(d_A,d_B)? |
|---|---|---|
| backward (keep old A-head after learning B) | retention | **yes** — severity, ρ 0.55–0.73; but one-sided, no transfer tail |
| forward, frozen (B-head init for few-shot A) | learnability | no — ρ ≈ 0, net ≈ 0 (warm-start, help ≈ hurt) |
| forward, plastic (few-shot A on B-fine-tuned rep) | learnability | no — ρ ≈ 0, **net negative** (−0.055; B-training narrows the rep) |

Same features, same cosine: it predicts backward forgetting (ρ 0.73) and neither forward regime (ρ ≈ 0) — and the random projection used
for headroom preserved the backward signal, so the forward nulls are real, not artifacts. The cosine governs **interference** (which way you
point when *overwriting* a capability), and says nothing about whether learning B helps you *learn* A. Forgetting and transfer are different
mechanisms. A side finding: plastic fine-tuning on B **net-hurts** other tasks' few-shot learnability (62–68% of pairs), alignment-agnostic —
a representational-specialization cost. (`VERDICT_TRANSFER_*`, `VERDICT_FORWARD_*`, `VERDICT_FWD_PLASTIC_*`.)

## 14. Where this leaves the claim

Two things stand, hardened:
1. **The frozen readout cosine law** (§2–§6): forget ∝ −cos, modality-general, quantitative — and now specifically a *forgetting/
   interference* law, with its signed severity axis mapped from severe (anti) to coexistence (aligned), and shown *not* to govern transfer.
2. **The plastic two-scalar structure** (§10–§11): F ≈ μ + fragility + disruptiveness + noise, with a transitive dominance order; robust
   across five architectures and three image datasets.

One thing was retired: a general plastic *directional* law (§12) — real per-pair structure, but no descriptor, no low rank, CIFAR-specific.
Open, and deliberately not claimed: the plastic two-scalar structure is vision-only so far (CIFAR/DTD/GTSRB) — its cross-modal generality is
untested, and by the §12 lesson should not be assumed. What governs transfer (the axis the cosine does *not*) is uncharacterized beyond "not
alignment."

---

# Addendum (2026-07-14): three extensions — a gradient-free substrate (null), stream composition (holds), and the frozen/plastic boundary (sharp)

Three questions the readout law left open: does it have a *gradient-free* analog (interference from in-context demonstrations, no weight
update); does the pairwise law *compose* when a capability is overwritten by a *stream* of tasks rather than one; and is the frozen→plastic
transition — whose endpoints §5b/§8 fixed but whose interior was never mapped — a smooth crossover or a sharp boundary. The first is a null;
the second holds and generalizes; the third is a discontinuity.

## 15. The cosine law has no clean in-context twin — gradient-free interference is dose-governed

An LLM "forgets" a task with zero weight updates: pack task-B demonstrations into the context and task-A few-shot recall degrades. We asked
whether *that* interference obeys the same cosine geometry, using discriminative directions read from *activations* under a prompt instead of
from frozen features (Qwen2.5-1.5B, ag_news one-vs-rest capabilities, retroactive setup: A-demos → B-demos → A-query). It does not.

- **Dose, not cosine.** Sweeping the intervening B-block length (dose 4→64), net interference grows monotonically (mean AUC-drop −0.024→+0.011)
  but the per-pair ordering does **not** track −cos: ρ(cos, drop) wobbles near zero past dose 16 (−0.04, −0.10, −0.03), with no march toward the
  −0.5 the weight-based law predicts. An apparent ρ=−0.50 at dose 64 was small-N noise — it fell to −0.03 when eval was tightened (N 80→160).
- **The low-dose reversal was a label artifact.** At short streams there was a reproducible *facilitation* (anti-parallel B *helps* A, ρ≈+0.4).
  A matched-label recast (uniform cue + yes/no labels, so label-space is identical and cosine varies by content) collapsed the effect — but it
  also collapsed the cosine range (`[−0.89,+0.76]→[−0.46,0]`, no aligned pairs) and tanked headroom to near-chance, so the isolation is not
  clean. The control ordering (replay-A helps > structured-B mild > neutral-filler hurts) is recency/coherence, not geometry.
- **The instrument is confounded.** That the cosine matrix collapses when label tokens are equalized suggests a large part of the in-context
  "cosine structure" was label/cue-token-driven, not semantic. The cosine law is a statement about a *representation*, not about tokens in a
  context window; the gradient-free substrate is closed as a null for this question. (`probe_ic_dose.py`, `probe_ic_facilitation.py`.)

## 16. The forgetting law composes over a stream — recency-weighted and path-dependent (holds on CIFAR/DTD/GTSRB)

The pairwise law overwrites A with a single B. Real continual learning overwrites A with a *stream* B₁→…→B_k. In the §2 regime (frozen
ResNet18 features, class-vs-neg linear readout, continue-train the head sequentially on each Bᵢ, measure A's surviving AUC), the pairwise
cosines compose — but not commutatively.

- **Composition rule = recency-weighted sum of pairwise −cos.** Predicting stream-forget from the perturbers' cosines, a recency-weighted sum
  (later perturbers heavier) beats both a flat additive sum and a last-only model, on all three datasets: recency-weighted ρ = +0.567 / +0.535 /
  +0.580 (CIFAR/DTD/GTSRB) vs last-only +0.505/+0.509/+0.532 vs flat-additive +0.461/+0.438/+0.500. A γ-sweep puts the decay at γ≈0.4–0.5
  (recency-weighted degrades to flat as γ→1).
- **Path-dependent.** Holding the perturber *set* fixed and varying the *order*, a stream ending in an anti-parallel task forgets most:
  within-set ρ(last-perturber cos, forget) = −0.53 / −0.39 / −0.39. The fraction of stream-forget variance attributable to order is 0.34 / 0.26 /
  0.19 — present on all three, largest on CIFAR, smallest on GTSRB (so mostly set-governed there, but order still bites; the "Markovian" auto-label
  on GTSRB is a threshold crossing on a continuum, not a qualitative flip).
- **It generalizes.** Unlike the plastic *directional* law of §12 (fit CIFAR, died on DTD), the recency-weighted composition survives the dataset
  switch — the same rule on textures and traffic signs as on natural images. (`probe_stream_compose.py`, `stream_xdata_lambda.py`.)

**Path-dependence grows with stream length.** Sweeping K=2→8 (CIFAR), the fraction of stream-forget variance attributable to order rises
monotonically (order_frac 0.16 / 0.22 / 0.34 / 0.41 / 0.45), and the recency-weighted model's edge over the order-blind flat-additive widens
(+0.05 → +0.19): flat-additive plateaus (~0.46, it cannot see order) while recency-weighted climbs to +0.645 at K=8. The within-set recency
signature locks in at ρ≈−0.5 for K≥3 (it is weak at K=2, where "first vs last" is barely a distinction). Longer streams are more recency-dominated
— later perturbers overwrite the geometric imprint of earlier ones, so which capabilities survive depends increasingly on *when* each perturber
hit, not just which were in the stream. Headroom held across the sweep (stream-forget range ~[0, 0.39], no saturation at K=8).

Honest scope: single seed per dataset, one head-training recipe; the composition ρ ≈ 0.5–0.6 is moderate (the cosines explain much, not all, of
stream forgetting); sequential SGD is inherently order-sensitive, so the substantive claim is that the path-dependence is *predictable*
(recency-weighted), consistently recency-*shaped*, and *intensifies with length* — not merely that it exists. The K-sweep is CIFAR-only so far.

## 17. The frozen↔plastic transition is a sharp boundary, not a smooth crossover

§5b/§8 established the two endpoints — frozen readout is direction-governed (forget ∝ −cos, ρ −0.74), a LoRA-rank-32 backbone is
disruptiveness-governed (feature-cosine ρ −0.12) — and left the rank sweep between them unrun. Mapping it (CIFAR/ResNet18, conv-LoRA on
layer3+4, order parameter = LoRA rank {1,2,4,8,16,32}; frozen-re-probe forgetting; PREREG cba92440):

The cosine direction law does **not** decay gradually with plasticity — it collapses at the frozen→plastic boundary itself. At the lowest
*reliable* plastic rank (rank 2, partial forgetting 0.066) the clean within-B alignment signal is ρ = **−0.068 — already gone** — versus the
frozen −0.74, and it stays near zero through rank 32 (within-B −0.13 … −0.21). There is no monotone decay across the rank decade; **any**
backbone motion, even a rank-2 LoRA, evaporates it. So the cosine law is a razor-thin property of *exact* frozen-ness: it lives at the interface
between a fixed representation and a linear decision, and the instant the representation is allowed to move, direction stops governing and
disruptiveness (which-B) takes over (per-B feature-drift → per-B forgetting ρ ≈ +0.94 at low rank). This sharpens §8's boundary claim from
"readout-specific" to "discontinuous at the frozen boundary."

Honest caveat: this run cleanly establishes the *direction collapse* (robust — Thread-6 at gentler LR independently found ρ −0.12 at rank 32), but
**not** how disruptiveness varies *within* the plastic regime. Disruptiveness IS substantial there — per-B feature-drift → per-B forgetting ρ ≈ +0.7–0.9,
and a gentle-LR (4e-3) re-run corrected rank-32's which-B main effect from a saturated η²_B = 0.03 to a true **η²_B ≈ 0.26** (confirming the near-zero
was purely the B_LR headroom bump saturating high ranks). But the η²_B-vs-*rank* trend is **not isolable**, and a matched-forgetting run (per-rank step calibration at LR 6e-3, `probe_phase_matched.py`) established
*why*: (i) very low rank is **capacity-capped** — a rank-2 LoRA forgets only ~0.02 even at 2600 steps, it physically cannot reach the 0.10 target, so it can't be
matched; (ii) rank and forget-magnitude stay **entangled** — in all four runs rank 32 forgot *more* than rank 8 at any step budget (even 493 steps gave 0.143), so it
could never be brought to an equal-or-lower forgetting level; (iii) η²_B is **condition-sensitive** — rank 32's value came out 0.03 / 0.26 / 0.055 across the three LR
settings. Across all runs η²_B is lower at rank 32 than rank 8, but since rank 32 always forgot more, "falls with rank" is inseparable from "forgot more." What is robust:
disruptiveness is real at low–mid rank (per-B drift → per-B forgetting ρ ≈ +0.8) and **decouples at rank 32** (ρ_drift ≈ −0.3, drift stops predicting forgetting). The clean
η²_B(rank) law does not exist to be measured; the boundary-collapse headline does not depend on it. (`probe_phase_rank.py`, `probe_phase_matched.py`.)

---

*Provenance: all claims trace to pre-registered, SHA256-locked probes with mandatory controls, run $0-local except one dataset
download. Sandboxes: `D:/IC_experiments/{interference_law, detector_cf, audio_cf, alibi_multi_cf}`. Verdicts and locks per thread on
disk. Production systems were read-only throughout; nothing was deployed.*
