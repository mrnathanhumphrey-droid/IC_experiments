# Decomposition & Unification — Seams and Bridges for Tier Abstraction

**Date:** 2026-06-12. Synthesis from two independent literature hunts (parallel agent + manual).
Purpose: decompose the five mechanisms to their active ingredient, find where prior literatures
miss / fit, and state the unifying bridge that turns the bushclear into a design principle.

## The unifying claim (the bridge)

> **Tier abstraction emerges iff (a) the upper tier's objective+channel make fine (within-coarse)
> distinctions MORE EXPENSIVE to preserve than coarse (across-coarse) distinctions — a "fine-price
> differential" — AND (b) a bottleneck exists to convert that price differential into actual
> information loss. The bottleneck is the CONVERTER; it is inert without a price differential to
> act on. Every known mechanism is just a different way to PRICE fine-detail.**

Pricing methods, decomposed:
| Arm | Mechanism | How it prices fine-detail | Price level |
|---|---|---|---|
| A (null) | bottleneck alone, fine on z_trans | none — fine is the only objective, uniformly compressed | **0 differential** |
| C (+0.13) | objective compartmentalization | fine OMITTED from z_trans → infinite price | omission |
| D | adversarial (GRL) | fine actively PENALIZED (×α) | penalty (tunable) |
| 7 | HSIC / covariance | fine statistically penalized (non-adversarial) | penalty (stable variant) |
| E | stacked level-supervision | upper label only rewards coarse | supervision-level |
| F | asymmetric channels | coarse (few bits) → cheap channel; fine (many bits) priced out | bit-cost |

**This predicts the whole experiment on one axis:** selectivity should rise monotonically with the
fine-price differential, and any mechanism that fails to create one (Arm F-uniform, Arm A) gives ~0.
That collapses five separate literatures onto a single dependent variable.

**It also explains v0.1's null mechanically:** Arm A had a strong bottleneck but ZERO price
differential (one fine objective, compressed uniformly) → the converter had nothing to differentially
squeeze → uniform loss → no abstraction. This matches the information-theoretic fact (Agglomerative-IB;
PMC9222931 2022) that **IB alone does not yield a hierarchy** — extra structural constraint is required.
Our "fine-price differential" IS that constraint, named and made tunable.

## Where the people miss their mark (each literature's blind spot)

- **IB / VIB (Tishby; Alemi 2017):** treats the bottleneck as the abstraction-MAKER. Misses that it's
  only a converter — inert without a price differential. Our Arm A is the counterexample they don't have.
- **Disentanglement (β-VAE Higgins 2017; Locatello 2019):** measures factor-INDEPENDENCE (MIG/DCI/FactorVAE),
  never hierarchy-structured error. Has no instrument for "do the mistakes respect the class tree?" →
  our **selectivity index** is the missing instrument.
- **Adversarial invariance (Ganin 2015; Edwards & Storkey 2016; Madras 2018):** points the GRL at
  nuisance/sensitive attributes to REMOVE them for fairness/DA. Misses that the SAME machinery aimed at
  fine-class BUILDS abstraction. Direction is even inverted in the one close vision paper (Coarse→Fine
  OSR, CVPR 2024, purges COARSE keeps fine; we purge fine keep coarse).
- **Hierarchical VAE / capsules (Ladder-VAE Sønderby 2016; NVAE; capsules; GLOM):** build depth-of-abstraction
  GENERATIVELY but never supervise NAMED levels nor measure hierarchy-locked selectivity; capsules/GLOM
  don't scale past toy. The supervised-level-graded + MEASURED version is open.
- **Multi-task routing (Standley 2020 "Which tasks together"):** knows competing objectives should be
  separated, measures TASK ACCURACY. Misses that the separation produces a measurable abstraction GRADIENT.
- **Fei et al 2022 COLING (Coarse-to-Fine HMTL):** their "coarse-to-fine" is TASK granularity (bottom shared
  / mid grouped / top per-task), DETERMINISTIC, no bottleneck, measures task accuracy + graphical distance.
  A different axis from class-abstraction-via-compression. Not a collision — a clean contrast.
- **Encoding Hierarchical Information (2112.10844):** loss design for hierarchical-classification accuracy
  (graphical distance, conditional label use). Does the per-level structure but (per both hunts) does NOT
  vary bottleneck pressure → β-as-the-abstraction-knob + the inverted-U being hierarchy-locked is our edge.

## Where they fit together (the seams)

1. **Standley (separate competing objectives) + IB (rate-distortion)** = our Arm C. Separating fine off
   z_trans removes the competition AND lets the rate constraint preferentially drop fine. Two literatures,
   one mechanism — the seam is "objective omission = infinite price."
2. **β-VAE (per-dim KL asymmetry emerges spontaneously) + Nested Dropout (Rippel 2014, ordered cost) +
   our Arm F (imposed per-channel cost)** = the SAME capacity-allocation axis approached spontaneously /
   by-ordering / by-fiat. Seam: capacity-allocation-as-abstraction. (Landmine: β-VAE shows the spontaneous
   version exists — so Arm F must beat its UNIFORM control to claim the imposition matters.)
3. **Adversarial invariance (D) + HSIC (7)** = identical goal (penalty-price fine), adversarial vs
   statistical-dependence. Two routes to the same price. Run both → isolates "does the route matter?"
4. **Locatello's impossibility ("no structure without inductive bias") + our positive** = the cleanest
   bridge: Locatello says you NEED a bias but is agnostic WHICH. **Tier-compartmentalization is the named,
   minimal inductive bias** that breaks the impossibility for hierarchical abstraction. Our null confirms
   Locatello; our positive identifies the bias.

## The bridges we build (contribution claims, defensible per both hunts)

1. **A unified "pricing" account** of tier abstraction that subsumes IB, disentanglement-by-capacity,
   adversarial invariance, and multi-task routing as special cases of one variable (fine-price differential),
   with the bottleneck as converter. Falsifiable: selectivity ~ monotone in price differential.
2. **The selectivity index** — a hierarchy-structured error-concentration metric the disentanglement field
   lacks. Puts incommensurable mechanisms on one scale.
3. **β tuned to a NAMED abstraction level** — showing the rate-distortion inverted-U peak is *hierarchy-locked*
   (Burgess gives the inverted-U; nobody shows the optimum aligns surviving info with a class tier).
4. **The four-mechanism head-to-head on one metric + one transfer judge** — "which inductive bias most
   cheaply/robustly buys hierarchy-respecting abstraction, and does it pay?" A genuine synthesis gap.

## Landmines adopted into method (the "robust & confirmed" tax)

- **L1 — Selectivity confound:** "errors concentrate in true superclass" can be inflated by simply being
  GOOD at superclass (which we train z_trans to be). → ADD a **matched-super-accuracy control**: selectivity
  must exceed what a representation with the same sup-acc but no fine-price differential achieves. (Pre-reg G6.)
- **L2 — Adversarial leakage (D):** beating the TRAINING adversary ≠ removing fine. → ADD a **post-hoc,
  higher-capacity (MLP) adversary probe** of z_trans for fine-recoverability; that, not the training loss,
  is the real removal test. (Pre-reg G7.) Expect α-instability (well-documented).
- **L3 — Posterior collapse (β-VAE; Ladder-VAE):** an "abstract" upper tier might just be COLLAPSED/inactive
  (trivially discards fine). Rule out via the super-floor gate + KL>0 check. May need KL warm-up for Arm E.
- **L4 — "Few-bit superclass" is an assumption (F):** measure conditional MI before claiming the cheap
  channel carries coarse; otherwise F's premise is unfalsified.
- **L5 — P5 must pre-register BOTH outcomes:** Locatello 2019 (ICML best paper) is the dominant NEGATIVE
  (disentanglement does NOT reduce sample complexity); van Steenkiste 2019 is positive for abstract reasoning
  / few-shot. Our transfer judge could disconfirm — that is a real, reportable result, not a failure.

## Sources (key anchors)
- Locatello et al. 2019 ICML, arXiv:1811.12359 (the impossibility / must-cite frame)
- van Steenkiste et al. 2019 NeurIPS, arXiv:1905.12506 (disentanglement helps abstract reasoning — P5-positive)
- Alemi et al. 2017 ICLR, arXiv:1612.00410 (Deep VIB — our converter)
- Burgess et al. 2018, arXiv:1804.03599 (rate-distortion inverted-U, posterior collapse)
- Ganin & Lempitsky 2015, arXiv:1409.7495 (GRL — Arm D)
- Rippel et al. 2014 ICML, arXiv:1402.0915 (nested dropout — Arm F ancestor)
- Standley et al. 2020 ICML, arXiv:1905.07553 (which tasks together — Arm C meta-principle)
- Behjati et al. 2023 EMNLP Findings (Nonparametric VIB → layer abstraction — Arm E nearest neighbor)
- Fischer 2020, arXiv:2002.05379 (CEB — cleaner v0.2 objective candidate)
- Fei et al. 2022 COLING, arXiv:2208.09129 (Coarse-to-Fine HMTL — task-granularity, the contrast)
- Encoding Hierarchical Information, arXiv:2112.10844 (hierarchical loss design — no bottleneck-knob)
