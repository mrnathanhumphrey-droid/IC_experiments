# VERDICT — NEW-1a''''-b: DATASET switch BREAKS the directional law (PREREG ea04454c, locked 2026-07-13). Landed, $0 local 609s, resnet18, DTD + GTSRB, 24×6 each.

## RESULT: NEGATIVE — a major scope correction. The directional INTERACTION (align_dc) that survived the ARCHITECTURE switch does NOT survive a DATASET switch. On DTD (textures, clean non-saturated) align_dc = −0.035 (gone); on GTSRB (signs) +0.079 (below the +0.15 bar). Same resnet18, same task, only data changed. → the plastic directional law forget∝−cos(g_i,Δθ_p) is CIFAR-SPECIFIC, not a general property of plastic fine-tuning. What DOES survive across architecture AND dataset: the MAGNITUDE pattern (gmag wrong-signed = margin-protects). Capstone DOWNGRADED.

## The table (resnet18, class-vs-neg held; only the DATASET varies)
  dataset (resnet18)   medFgt  align(pooled)  align_dc(interaction)  gmag     cos_range         DS-P1 DS-P2
  CIFAR-100 (baseline)  0.219     +0.090          +0.153            −0.304   [−0.004,+0.014]      —     —
  DTD (textures)        0.261     +0.103          −0.035            −0.282   [−0.009,+0.009]      ✗     ✓
  GTSRB (signs)         0.328     −0.090          +0.079            −0.182   [−0.022,+0.012]      ✗     ✓
  DS-P1 = align_dc ≥ +0.15 ; DS-P2 = align_dc > gmag AND gmag ≤ +0.10 (magnitude-not-carrier).

## Reading it — honest, no softening
- **DS-P1 FAILS on both datasets.** DTD is the decisive clean test (textures, no objects, forgetting 0.26 = non-saturated, no heavy
  columns): align_dc = −0.035 → the directional interaction is ABSENT on a non-CIFAR domain. GTSRB corroborates (+0.079, below bar; and it
  has near-saturated columns — perturb 29 medFgt 0.58 — that further compress any signal). The interaction that was +0.15 (resnet18/CIFAR),
  +0.30 (mobilenet/CIFAR), +0.36 (vit/CIFAR) does NOT reproduce when the DATA leaves CIFAR.
- **The architecture-invariance was real but CIFAR-bound.** align_dc held across resnet/mobilenet/vit — but ALL on CIFAR-100. Holding the
  architecture and switching the DATA (the cleaner isolation) kills it. So the directional interaction is tied to CIFAR-100's structure,
  not to plastic fine-tuning per se. (Whether mobilenet/vit would revive it on DTD is an open question — but the resnet18 apples-to-apples
  comparison, +0.153 CIFAR → −0.035 DTD, is a clean same-architecture falsification.)
- **DS-P2 holds on both: gmag wrong-signed (−0.28 DTD, −0.18 GTSRB).** The MAGNITUDE/margin structure — well-separated capabilities are
  robust, fragility ≈ margin — replicates across architecture AND dataset. THIS is the dataset-general finding, not the directional term.

## The corrected arc state (what is robust vs what is not)
  FINDING                                            robust across…                         status
  Frozen 4-op cosine law (feature-space −cos)        text/vision/audio (earlier arc)        SOLID, multi-modal
  Plastic MARGIN-PROTECTS / fragility≈d′             arch (5 backbones) + dataset (CIFAR/DTD/GTSRB)   ROBUST
  Plastic DISRUPTIVENESS ≈ ‖Δθ‖ (main effect)        arch + dataset                         ROBUST (main effect)
  Plastic DIRECTIONAL interaction −cos(g_i,Δθ_p)     arch (on CIFAR) only; FAILS DTD/GTSRB  CIFAR-SPECIFIC — NOT general
The three-term plastic law "forget ≈ fragility ⊕ disruptiveness ⊕ directional-interaction" holds its FIRST TWO terms universally; the
THIRD (directional) term is CIFAR-specific and must NOT be claimed as a general law. 1a''''s "found the plastic directional law" is
DOWNGRADED to "a CIFAR-specific directional interaction."

## Why this matters / lesson
The capstone survived TWO probes designed to test it from the same substrate (1a''' setup + architecture variation on CIFAR) and looked
general. It took switching the ACTUAL DATA — the axis never varied all session — to falsify the generality. Single-substrate robustness
(even across architectures) is NOT generality. The magnitude/margin structure is the durable plastic-regime finding; the directional
alignment is not established beyond CIFAR.

## Caveats (fair to the negative too)
Single seed; resnet18 only for the dataset switch (the arch that had the WEAKEST CIFAR interaction, +0.15 — so this is a conservative test;
a strong-interaction architecture like ViT on DTD would be the fairer follow-up, and could in principle revive it). GTSRB has some
near-saturated perturber columns (mild signal compression); DTD is the clean null and it is unambiguous (−0.035). 24×6=144 pairs (SE≈0.08),
so DTD −0.035 and GTSRB +0.079 are statistically ~0; the CIFAR mobilenet/vit values (+0.30/+0.36) are clearly larger. Follow-up to fully
settle CIFAR-specific-vs-weak: DTD/GTSRB with mobilenet+vit (the strong-interaction arches); a 3rd, more natural-image dataset (e.g. STL/
Flowers) between textures and CIFAR. But the headline stands: the directional law does NOT replicate off CIFAR with matched architecture.

## Bottom line
DATASET switch = the honest falsifier. The plastic directional law (−cos(g_i,Δθ_p)) is CIFAR-specific, not general — DOWNGRADE the
capstone. The durable, substrate-general plastic finding is MARGIN-PROTECTS (fragility ≈ separation) + disruptiveness ≈ ‖Δθ‖. Third
hardening axis (task family) is now lower priority — the directional claim is already bounded by this dataset result.
