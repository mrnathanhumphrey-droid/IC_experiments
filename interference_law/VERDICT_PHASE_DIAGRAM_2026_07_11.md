# VERDICT — Thread #5: PHASE DIAGRAM of interference (PREREG dcf864d2, locked 2026-07-11). Landed, $0 local (12 min, 9 backbones).

## RESULT: the readout law's SHAPE is scale-robust (always forget ∝ −cos) but its MAGNITUDE shrinks with scale; pretraining OBJECTIVE changes the slope sharply; and ViT/ConvNeXt frozen features are so interference-robust the law SATURATES (unmeasurable) there.

## The phase diagram
  backbone          dim    slope     rho     R2      c*     medFgt   note
  resnet18          512   −0.446   −0.754   0.552   +0.313   0.231
  resnet50          2048  −0.384   −0.711   0.430   +0.104   0.137
  resnet152         2048  −0.424   −0.713   0.360   +0.098   0.119
  wide_resnet50_2   2048  −0.328   −0.671   0.334   +0.109   0.107
  dinov2_vits14     384   −0.816   −0.637   0.267   +0.119   0.141    ← SSL: STEEPEST slope
  vit_b_16          768   −0.131   −0.500   0.089   −0.211   0.016    [SATURATED]
  vit_l_16          1024  −0.151   −0.526   0.110   −0.191   0.017    [SATURATED]
  convnext_small    768   −0.203   −0.530   0.187   −0.118   0.044    [SATURATED]
  clip_vitb32       —     SKIP (get_image_features return-type bug; bonus, re-runnable)

## Findings along the three axes
### 1. CNN SCALE (resnet18 → 50 → 152, wide50): the law HOLDS, and forgetting SHRINKS with scale
All CNNs unsaturated, ρ −0.67…−0.75 (strong law), slope in the −0.33…−0.45 band. As scale rises: **median forgetting drops
0.231 → 0.137 → 0.119 → 0.107** and **c* drops +0.31 → ~+0.10**. Bigger CNNs are more interference-ROBUST (more capacity/room), so
you need a MORE anti-parallel capability (lower c*) before routing pays off. The law's SHAPE (negative slope) is invariant; its
MAGNITUDE (how much you forget, where the route-boundary sits) is a decreasing function of scale.

### 2. ARCHITECTURE: ViT & ConvNeXt frozen features SATURATE — a finding, not a gap
ViT-B/L and ConvNeXt show near-ZERO forgetting (medFgt 0.016–0.044): their frozen features are so robust that a continue-trained
linear probe barely drifts. The slopes (−0.13…−0.20) and negative c* are UNRELIABLE (flagged saturated — the recurring saturation
trap). Interpretation: transformer / modern-CNN frozen representations are dramatically more interference-robust than classic CNNs at
the linear-readout level. (To measure the law there needs a harder task — within-superclass negatives — a follow-up.) NOT evidence the
law fails for ViTs; evidence there's nothing to forget at this task difficulty.

### 3. PRETRAINING OBJECTIVE: self-supervised (DINOv2) has the STEEPEST slope
DINOv2 (SSL ViT-S) is unsaturated (medFgt 0.141) and has slope **−0.816** — roughly 2× steeper than any supervised CNN (~−0.4). So
in the self-supervised feature space, directional interference is SHARPER: aligned capabilities coexist even better, anti-parallel
ones collide even harder. The pretraining objective materially reshapes the interference geometry (SSL > supervised in slope). CLIP
(contrastive) would have completed the objective axis but hit a code bug (skipped; re-runnable).

## Bottom line
The interference law is UNIVERSAL IN SHAPE (every unsaturated backbone: forget ∝ −cos, ρ −0.64…−0.75) but its PARAMETERS live on a
phase diagram: scale shrinks the magnitude and the route-boundary; the pretraining objective sets the slope (SSL steepest); and some
architectures (ViT/ConvNeXt) are so robust the law is unmeasurable without a harder task. "Should I route?" has a scale- and
objective-dependent answer — c* is not a universal constant, it's a coordinate on this diagram.

## Caveats
Saturation (ViT/ConvNeXt/borderline) flagged and EXCLUDED from the slope trend — the clean comparison is within the unsaturated set
(CNN family + DINOv2). Feature-dim varies 384–2048 (AUC/cosine are dim-robust; noted). Each backbone uses its own preprocessing.
Single seed per backbone (the cross-backbone TREND is the claim, not a 3rd-decimal slope). CLIP skipped (bonus). CIFAR-100 base-AUC is
high for all (strong backbones) — the within-superclass-harder-task variant would de-saturate ViT/ConvNeXt and is the natural follow-up.
