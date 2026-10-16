# PRE-REG — Thread #5(fun): PHASE DIAGRAM of interference — does the readout law's SLOPE / decision-boundary c* shift with backbone SCALE, ARCHITECTURE, and PRETRAINING OBJECTIVE? 2026-07-11.

SHA256-lock before running. #1 calibrated the readout interference law on ResNet18 (forget ≈ −0.445·cos+0.266, held-out R²≈0.5,
c*(0.10)≈+0.2–0.3). This sweeps the FROZEN backbone across a grid to map how the law's parameters move — the "phase diagram" of
interference. Same forget-vs-cosine machinery as #1; only the frozen feature extractor changes.

## Grid (three axes)
- **CNN SCALE:** resnet18 (512-d, reference) → resnet50 (2048) → resnet152 (2048).
- **WIDTH:** wide_resnet50_2.
- **ARCHITECTURE:** vit_b_16 (768), vit_l_16 (1024), convnext_small.
- **PRETRAINING OBJECTIVE (bonus, try/except):** CLIP ViT-B/32 (contrastive) and DINOv2 ViT-S/14 (self-supervised) — different
  geometry from supervised-ImageNet. Included if their deps/weights load; skipped gracefully otherwise (reported).

## Method (identical to #1 per backbone)
CIFAR-100, C=40 held-in classes as capabilities vs a shared negative pool (reserved classes). Each frozen backbone → penultimate
features (each model's own weights.transforms() for resize/normalize). Per class: linear probe base AUC + discriminative direction
d_c. Pairwise sweep (all C×(C−1)=1560 ordered pairs): train head on A, continue-train on B, measure A's AUC drop. Fit forget ~ cos:
Spearman ρ, held-out R² (fit half, predict half), slope, and c*(forget≥0.10). Report per backbone.

## Pre-registered read (locked before running)
- **THE PHASE DIAGRAM:** tabulate {slope, ρ, held-out R², c*, feature-dim, base-AUC median, median-forget} per backbone. The result
  IS the table + how slope/c* move along each axis.
- **SCALE:** does the slope FLATTEN with scale (bigger models = more capacity/room → forgetting less cosine-dependent) or STEEPEN
  (sharper directional structure)? Report the resnet18→50→152 trend.
- **ARCHITECTURE:** do ViTs vs CNNs give different slopes/c* (attention vs conv geometry)?
- **OBJECTIVE:** do CLIP/DINO (non-supervised) differ systematically from supervised ImageNet? (contrastive/SSL directions may be
  more/less antagonistic.)
- **INVARIANCE (the boring-but-important outcome):** if slope & c* are ~constant across the whole grid, the law is
  backbone-UNIVERSAL (a strong, clean result — the cosine law is a property of the task geometry, not the backbone).
- **NULL/CONFOUND:** if a backbone SATURATES forgetting (base-AUC≈1.0 → forget pinned high, the recurring saturation trap), its
  slope is unreliable — FLAG it (report base-AUC + median-forget; a saturated backbone is excluded from the slope trend, not spun).

## Confounds (locked)
1. Saturation (Nth time): stronger backbones separate CIFAR even more → higher base AUC → risk of pinned forgetting. Report base-AUC
   median + median-forget per backbone; flag any with median-forget outside [0.05,0.45] as slope-unreliable.
2. Feature-dim differs across backbones (512–2048) — cosine/AUC are dim-robust (rank-based AUC, normalized dirs), but note it.
3. Each backbone uses ITS OWN preprocessing transform (fair; not a shared resize).
4. Single seed per backbone for the sweep (the SLOPE across a 1560-pair fit is stable; the cross-backbone TREND is the claim, not a
   3rd-decimal slope). Multi-seed the reference (resnet18, already 2 regimes in #1).

## Compute / ops
Multi-backbone → the BIG run: downloads (some GB: ViT-L, CLIP, DINO) + feature extraction (~8500 imgs/backbone at 224px) + 1560-pair
sweep each. LOCAL BACKGROUND ($0, my GPU) — est ~40–70 min incl downloads. `probe_phase_diagram.py` (generic backbone loader +
#1's sweep). py_compile + a 2-backbone smoke, then fire full in background. Output `out/phase_diagram.json`.

## Status
PREREG written. Next: SHA256-lock → build → 2-backbone smoke → FIRE full in background → (meanwhile pursue #1 disruptiveness) → read
the phase diagram → verdict + bank.
