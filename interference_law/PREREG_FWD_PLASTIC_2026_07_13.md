# PREREG — NEW-3c: PLASTIC forward transfer — does B-training the backbone improve A's LEARNABILITY, and is it cosine-governed?

Author: arc (CF interference). Date: 2026-07-13. Status: PRE-REGISTERED, SHA256-lock before run. PLASTIC resnet18 (full-FT on B), few-shot A
on the resulting representation, CIFAR + DTD. $0 local.

## The question NEW-3b left open
NEW-3b (frozen) found forward transfer via head warm-start is COSINE-AGNOSTIC (features fixed → B can only give a head prior). NEW-3c tests
the DEEPER regime: full-FT the backbone on B (representation MOVES), then LEARN a few-shot A on the B-trained representation. Does the B-
representation help learn A better than the base representation, and is THAT cosine-governed? This dissociates from NEW-3b: if aligned B
SHARPENS A's features, plastic forward transfer is cosine-governed at the REPRESENTATION level even though head warm-start was agnostic.
Distinct from backward FORGETTING (old A-head degrades): here A is LEARNED FRESH few-shot on the new representation (learnability, not retention).

## Design
Frozen-map headroom (random-proj-24d applied consistently to all features). Per cap: task = class vs pooled-others; d = ddir @ base;
few-shot train (8 pos/24 neg), held-out test. θ0 = base backbone.
  cold(A)   = few-shot A-head AUC on θ0 (projected) features
  θ_B       = full-FT(base, task B, 300 steps)  [backbone moved toward separating B]
  onB(A,B)  = few-shot A-head AUC on θ_B (projected) features
  PLASTIC FORWARD TRANSFER  FT(A<-B) = onB(A,B) - cold(A)     (FT>0 = B-representation helps LEARN A; FT<0 = hurts)
3 few-shot seeds avg. Descriptor = cos(d_A, d_B) @ base. 20 caps per dataset (CIFAR + DTD).

## PRE-REGISTERED PREDICTIONS
3c-P1 — NET SIGN. Report mean FT: does moving the backbone toward B, on average, HELP (net>0) or HURT (net<0) learning a few-shot A? And
    the positive fraction. [Prior: net slightly NEGATIVE — B-full-FT specializes the representation for B, generally reducing other tasks'
    few-shot learnability — but modulated by alignment.] No pass/fail; characterize the net effect.
3c-P2 — IS PLASTIC FORWARD TRANSFER COSINE-GOVERNED? (the key, dissociation test). ρ(cos(d_A,d_B), FT) >= +0.30 → aligned B improves A's
    few-shot LEARNABILITY (representation-level transfer follows alignment), UNLIKE the frozen head-warmstart (NEW-3b ρ~0). FALSIFIED if
    ρ < +0.15 → plastic forward transfer is ALSO cosine-agnostic (or forgetting-dominated) → cosine governs ONLY forgetting, at every level.
3c-P3 — DATASET-GENERAL. 3c-P2 holds (or fails) consistently on BOTH CIFAR and DTD.

## Outcome grid (locked)
- P2 holds (ρ>=+0.30): cosine governs PLASTIC forward transfer → at the REPRESENTATION level the cosine is a GENERAL alignment law
  (governs both learnability-transfer AND retention-forgetting); the frozen NEW-3b agnostic result was because head-warmstart != feature
  transfer. Nuanced completion: cosine = interference axis at the head level, alignment axis at the representation level.
- P2 fails (ρ<+0.15): plastic forward transfer is cosine-agnostic/forgetting-dominated too → cosine is purely a FORGETTING law at ALL
  levels; transfer is never alignment-governed. Clean, definitive close: forgetting and transfer are different mechanisms everywhere.

## Non-saturation / headroom
Require cold few-shot AUC median in [0.60, 0.90] (random-proj-24d gave 0.80 in NEW-3b). B full-FT at 300/5e-4 (moves backbone substantially,
~‖Δθ‖ 12 per prior). If cold saturates (>0.92) lower D_PROJ; if <0.55 raise it. Read FT only in the headroom band.

## Config (locked)
resnet18 IMAGENET1K_V1 224px; perturber task B = class_B vs pooled-others (full-FT backbone+head, 300/5e-4); few-shot A = 8 pos/24 neg,
K=30 head steps on projected features; random-proj-24d fixed map; 3 few-shot seeds; 20 caps per dataset (CIFAR + DTD), PER_CLASS 100.
Metrics = FT net/fraction + Spearman(cos, FT) + FT-vs-cos curve. Local; DTD downloaded. PLASTIC regime (backbone full-FT — the deeper test).
