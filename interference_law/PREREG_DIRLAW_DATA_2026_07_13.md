# PREREG — NEW-1a''''-b: does the directional law survive a DATASET switch? (2nd hardening axis)

Author: arc (CF interference). Date: 2026-07-13. Status: PRE-REGISTERED, SHA256-lock before run. Architecture (ResNet18) + task (binary
class-vs-neg) HELD; DATASET varied off CIFAR-100. FULL fine-tune, deployed frozen-head metric. $0 local. Self-contained.

## The gap (user-flagged, axis 2 of 3)
Arch-switch (1a'''', lock 8b4cf4c6) hardened the ARCHITECTURE axis: the directional law's invariant = the double-centered interaction
align_dc, replicates resnet18/mobilenet/vit (+0.15/+0.30/+0.36). But everything was still CIFAR-100. This switches the DATA DOMAIN,
holding ResNet18 + class-vs-neg + 224px + 24×6 fixed (matched to the arch-switch resnet18 row, align_dc +0.153) so ONLY the dataset changes.

## Design (2 non-CIFAR, non-object domains)
- DTD (Describable Textures, 47 classes, ~120/class) — TEXTURES, maximally unlike CIFAR generic objects. The decisive domain.
- GTSRB (German Traffic Sign Recognition, 43 classes) — traffic SIGNS, a structured non-object domain, different again from DTD.
Each: VICTIMS = class idx 0..23, PERTURBERS = idx 24..29, NEG = idx 30..39 (24×6 = 144 pairs). ResNet18 IMAGENET1K_V1, 224px, ImageNet-norm,
deployed binary head w_i FROZEN; full-FT on each perturber (class_p vs neg); g_i=∂margin_i/∂θ; the 1a''' decomposition (align / align_dc /
gmag). Same everything as the arch-switch resnet18 arm EXCEPT the dataset.

## PRE-REGISTERED PREDICTIONS
DS-P1 — INTERACTION REPLICATES ON A NEW DATASET. On EACH of DTD and GTSRB, double-centered ρ(align_dc, forget_dc) ≥ +0.15 (the arch-switch
    resnet18 bar). The directional interaction −cos(g_i,Δθ_p) predicts directed forgetting on data domains OTHER than CIFAR. FALSIFIED on a
    dataset if < +0.05 there.
DS-P2 — MAGNITUDE STILL NOT THE CARRIER. On each dataset, ρ(gmag_broadcast, forget) ≤ +0.10 and align_dc > gmag — replicating the
    "alignment carries it, magnitude wrong/weak" pattern (P3 of arch-switch, held on all 3 architectures).
DS-P3 — DECISIVE (non-object domain). DTD (textures — no objects, no scenes) satisfies DS-P1 → the directional law is NOT tied to
    object-recognition / CIFAR structure; it is a property of plastic fine-tuning on ANY discrimination task. Load-bearing outcome.

## Outcome grid (locked)
- DS-P1 on both (esp. DTD): directional law is DATASET-GENERAL — architecture-general (1a'''') AND domain-general. Two of three hardening
  axes closed; the law is robust to what you train on, not just how the net is built.
- Holds on GTSRB but NOT DTD: possibly tied to natural-image statistics — scope note.
- Fails on both non-CIFAR: the law was CIFAR-specific → major scope correction, retract "general directional law".

## Non-saturation (per dataset)
ResNet18 is ImageNet-pretrained; DTD/GTSRB are OOD for it, so full-FT dynamics differ. Require each dataset's median forget ∈ [0.03,0.45];
start FT 300/5e-4; if a dataset saturates (>0.45) or is dead (<0.03), FLAG and note (can't read align off a saturated/dead matrix). d′-free
(AUC/forget metric). Image counts: DTD combine train+val+test (partition 1) → ~120/class; GTSRB train split (plenty). PER_CLASS=100.

## Config (locked)
torchvision ResNet18 IMAGENET1K_V1, fc→Identity+Flatten, 224px, ImageNet-norm. torchvision.datasets.DTD (splits combined) + GTSRB (train),
auto-download to scratch. 24 victims / 6 perturbers / 10 neg (class idx 0..39 per dataset), PER_CLASS 100, FULL FT wd 1e-4 300/5e-4, NJ 128,
seed 0. Metrics = Spearman (pooled + double-centered) per dataset. Single seed (dataset generalization is the claim). Local only; one-time
dataset download from torchvision mirrors.
