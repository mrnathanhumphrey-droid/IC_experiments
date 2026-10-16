# PREREG — NEW-1a'''': does the plastic DIRECTIONAL LAW survive an ARCHITECTURE switch? (harden the capstone)

Author: arc (CF interference). Date: 2026-07-13. Status: PRE-REGISTERED, SHA256-lock before run. Substrate switch: CIFAR-100 held,
ARCHITECTURE varied. FULL fine-tune, deployed frozen-head metric. $0 local. Self-contained multi-arch pipeline.

## The gap this closes (user-flagged)
Every plastic-regime finding this session (#7/#7b dichotomy, #2/#2b dynamics, NEW-1 asymmetry, 1a/1a'/1a''' fragility + the DIRECTIONAL
LAW forget∝−cos(g_i,Δθ_p)) is ResNet18 / CIFAR-100 / class-vs-neg — we varied WHICH classes, never the architecture. The capstone
(1a''', lock 49633fa5, ρ+0.45 alignment / +0.30 interaction) rests on ONE architecture, and g_i=∂margin/∂θ is architecture-specific
(ResNet18 BN+conv). If the law is a conv/BN artifact, a transformer will break it. This probe switches architecture (the sharpest single
axis for a Jacobian law), CIFAR-100 held constant to isolate architecture.

## Design (same decomposition, 3 architectures)
Backbones (all ImageNet-pretrained, 224px, ImageNet-norm, feature-dim d varies): resnet18 (residual CNN+BN, d512 — internal baseline);
mobilenet_v3_small (depthwise-separable + SE + hardswish, d576 — a very different CNN); vit_b_16 (pure transformer, LayerNorm+attention,
NO conv/BN, d768 — the DECISIVE contrast). VICTIMS = classes 0..23, PERTURBERS = [50,55,60,65,70,75] vs neg, NEG 90-99 (24×6 = 144 pairs,
interaction dof (23)(5)=115). Per backbone: deployed head w_i FROZEN; full-FT on each perturber; g_i=∂margin_i/∂θ; the 1a''' decomposition
(fo_full / align=−cos(g_i,Δθ_p) / align_double-centered / gmag_broadcast / mag_only). Per-arch FT tuned to partial forgetting; band-checked.

## PRE-REGISTERED PREDICTIONS
1a''''-P1 — LAW REPLICATES. On EACH non-ResNet backbone (mobilenet, vit), pooled ρ(align, forget) ≥ +0.25. (resnet18@224 internal
    baseline expected ≥+0.35, replicating 1a'''@128 +0.45 up to resolution.) The alignment −cos(g_i,Δθ_p) predicts directed forgetting
    regardless of architecture. FALSIFIED on a backbone if < +0.10 there.
1a''''-P2 — INTERACTION REPLICATES. Double-centered ρ(align_dc, forget_dc) ≥ +0.15 on EACH non-ResNet backbone (removing victim+perturber
    main effects). The GENUINE per-pair directional interaction is architecture-general, not ResNet-specific. FALSIFIED < +0.05.
1a''''-P3 — MAGNITUDE PATTERN REPLICATES. On each backbone, ρ(align) > ρ(gmag_broadcast) AND ρ(gmag_broadcast) ≤ +0.10 — alignment (not
    magnitude) carries the prediction everywhere; the raw victim-Jacobian magnitude is not the carrier (as in 1a').
1a''''-P4 — THE TRANSFORMER TEST (decisive). vit_b_16 (no conv, no BN — maximally unlike ResNet) satisfies P1 AND P2. → the directional
    law is a property of plastic FULL-FINE-TUNING, not of convolutional/BN geometry. This is the load-bearing outcome.

## Outcome grid (locked)
- P1+P2 on both non-ResNet backbones (esp. ViT, P4): capstone HARDENED — forget∝−cos(g_i,Δθ_p) is ARCHITECTURE-GENERAL, not a ResNet18
  artifact. The plastic directional law stands across CNN families AND transformers.
- Holds on mobilenet but NOT vit: the law is CONV/CNN-specific (conv-Jacobian geometry) — important scoping, NOT a general plastic law.
- Fails on non-ResNet: the +0.45 was ResNet18-specific → RETRACT the general-directional-law claim; report as substrate-bound.

## Non-saturation (per architecture)
Require each backbone's median forget ∈ [0.03, 0.45]. Different arches need different FT strength: resnet18 300/5e-4, mobilenet 300/5e-4,
vit 300/1e-4 (transformers want lower LR). If a backbone lands out of band, its law-test is FLAGGED (can't read alignment off a saturated/
dead matrix) and reported as such, not silently included. d′-free (AUC/forget metric).

## Config (locked)
torchvision IMAGENET1K_V1 weights, 224px, ImageNet-norm, fc/heads/classifier→Identity. FULL FT wd 1e-4, per-arch LR above, 300 steps, 5-cls
head is BINARY (class_p vs neg, 2-way). 24 victims / 6 perturbers / NEG 90-99 / PER_CLASS 100. NJ=128 for Jacobians. seed 0. Metrics =
Spearman (pooled + double-centered) per backbone. Single seed (architecture generalization is the claim, not a 3rd-decimal). No production
system, no network beyond weight download, pure local CIFAR.
