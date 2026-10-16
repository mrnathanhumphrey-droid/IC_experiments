# PREREG — NEW-3: is the forgetting law also the TRANSFER law? One signed cosine axis from transfer to forgetting.

Author: arc (CF interference). Date: 2026-07-13. Status: PRE-REGISTERED, SHA256-lock before run. FROZEN resnet18 features, sequential
head continual-learning, CIFAR + DTD. $0 local.

## The question
Every probe so far measured only the NEGATIVE half: forget(i←j) ≥ 0 (victim degrades). Because base AUC saturated at ~1.0, victims could
not IMPROVE — backward TRANSFER (forget < 0) was structurally invisible. NEW-3 gives HEADROOM (hard victim task, base AUC ~0.8) and asks:
does backward transfer exist, and does the SAME signed feature-cosine axis govern the full continuum — aligned pairs TRANSFER (Δ>0),
anti-aligned pairs FORGET (Δ<0)? If yes, the forgetting law IS the transfer law (one signed law, not two phenomena).

## Design (frozen regime, with headroom — where the cosine law is cleanest, #1 R²≈0.5)
Frozen resnet18 features. VICTIM task i = class_i vs the POOLED OTHER capabilities (1-vs-rest among the cap set → HARD, base AUC ~0.75-0.90,
headroom to improve OR degrade). N = 20 caps per dataset (CIFAR + DTD). d_i = ddir(feat_i, feat_others). SEQUENTIAL head continual learning:
train linear head on task A(i), record AUC_i0; CLONE it, continue-train on task B(j) (moderate steps), record AUC_i1 = the B-trained head
re-evaluated on task A. RETENTION CHANGE Δ(i←j) = AUC_i1 − AUC_i0 (Δ>0 = backward TRANSFER, Δ<0 = FORGETTING). 3 head seeds (avg). Descriptor
= cos(d_i, d_j).

## PRE-REGISTERED PREDICTIONS
NEW3-P1 — TRANSFER EXISTS. With headroom (median base AUC ∈ [0.70, 0.92]), a meaningful fraction of ordered pairs show backward transfer:
    ≥ 15% of pairs have Δ ≥ +0.01 (victim's deployed head IMPROVES after training on the perturber). FALSIFIED if < 5% (transfer absent —
    forgetting is one-sided even with headroom).
NEW3-P2 — ONE SIGNED LAW (forgetting law = transfer law). cos(d_i,d_j) predicts the SIGNED retention change across the full range:
    ρ(cos(d_i,d_j), Δ) ≥ +0.40 — aligned pairs (high +cos) TRANSFER (Δ>0), anti-aligned (low/−cos) FORGET (Δ<0). The same cosine axis
    governs both directions → transfer and forgetting are ONE signed continuum, not two mechanisms. FALSIFIED if ρ < +0.20.
NEW3-P3 — THE ZERO-CROSSING. There is a cosine threshold c₀ where mean Δ crosses zero (below c₀ forgetting, above transfer). Report c₀
    (binned Δ-vs-cos curve). This is the transfer/forget boundary — the analog of the routing threshold c* but for the signed axis.
NEW3-P4 — DATASET-GENERAL. NEW3-P1 ∧ NEW3-P2 hold on BOTH CIFAR and DTD.

## Outcome grid (locked)
- P1+P2+P4: the cosine law is SIGNED and GENERAL — one axis runs transfer(+cos) → coexistence(0) → forgetting(−cos), on both datasets.
  Backward transfer is real and is the positive tail of the SAME law. Completes the phenomenon (both halves, one law). Frozen-regime result
  (recall the frozen 4-op cosine law is the SOLID, multimodal part of the arc — this extends it to the transfer side).
- P1 holds, P2 weak: transfer exists but is NOT cosine-governed → transfer is a separate axis from forgetting (two phenomena).
- P1 fails (no transfer even with headroom): forgetting is one-sided; continual learning here only degrades, never helps → the law has no
  positive tail in this setting.

## Non-saturation / headroom
This probe INVERTS the usual gate: it REQUIRES headroom — median base AUC ∈ [0.70, 0.92] (must be BELOW ceiling so transfer is possible).
If base AUC > 0.95 (saturated, no headroom) the transfer question is untestable → tune the victim task HARDER (more/harder negatives, fewer
head-train samples) before reading. If base AUC < 0.65 (too hard, noisy) tune easier. Continue-train steps moderate (not full overwrite).

## Config (locked)
frozen resnet18 IMAGENET1K_V1 224px features (extracted once). VICTIM = class_i vs pooled other-caps; head = logistic, AdamW; initial train
+ continue-train (moderate steps). 20 caps per dataset (CIFAR + DTD), PER_CLASS 100, 3 head seeds. Metrics = signed-Δ distribution (transfer
fraction) + Spearman(cos, Δ) + binned Δ-vs-cos curve (zero-crossing c₀). Local; DTD downloaded. NO backbone fine-tune (frozen — this is the
readout-level regime).
