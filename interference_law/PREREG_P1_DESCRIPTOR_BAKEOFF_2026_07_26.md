# PREREG — P1: descriptor BAKE-OFF for the reliable frozen forward-transfer signal (FT)

**Locked 2026-07-26.** Transfer-geometry thread, phase P1. $0 local, frozen resnet18, D_PROJ=24, CIFAR+DTD. Reuses P0 machinery. Target = the reliable FT established by P0 (r_SB 0.95-0.98, cosine-blind rho~0 CI excl +/-0.30).

## Question
P0 proved FT (frozen head-init warm-start forward transfer, FT(i<-j)=AUC_A_i(from B_j full-head init, few-shot) - cold) is a HIGHLY RELIABLE signal that the discriminative cosine cannot predict. **What geometric/mechanistic descriptor X(i,j) DOES predict it?**

## Scoping notes (honest, pre-committed)
- **LogME is N/A in the frozen regime** and is DEFERRED to the plastic phase. Reason: LogME scores a SOURCE MODEL's features for a target; here all source tasks share the SAME frozen backbone -> identical features -> LogME(i<-j) is constant over source j and cannot predict FT's j-variation. LogME becomes the prior-to-lose only when B reshapes features (plastic regime).
- **Direction-family descriptors are PRE-REGISTERED AS EXPECTED NULL.** In a frozen linear-head regime, subspace overlap / CKA / init-descent alignment / zero-shot head-AUC are all ~monotone in the discriminative cosine, which P0 showed is null (rho~0). We compute them to CONFIRM the collapse (a null here is a prediction met, not a surprise). Published negatives corroborate: CKA-fails-as-transfer-predictor (2601.03168).
- **Live hypothesis = scale/conditioning family.** If FT is reliable yet direction-blind, the mechanism is likely an OPTIMIZATION/CONDITIONING property of B's head init (norm, how B's head sits in A's feature covariance, initial gradient magnitude), not a direction alignment.

## Descriptor battery (all on cached 24-d projected feats + deterministically-trained full heads w_j; correlated with reliable FT over off-diag n(n-1) cells)
DIRECTION FAMILY (expected null):
- COS = cos(d_i, d_j)                         [the forgetting geometry; known null]
- X1 subspace overlap = mean(svdvals(U_i^T U_j)^2), U=top-5 PCs of pos feats   [symmetric multi-D reuse]
- X4 linear CKA(pos_i, pos_j)                 [symmetric representation similarity]
- Xzs zero-shot AUC_i(w_j)                    [B's head evaluated on A test; direction-driven, scale-invariant]
SCALE / CONDITIONING FAMILY (live hypothesis):
- X0 = ||w_j||                                [B head norm; per-j scale]
- X3 = ||U_i^T w_j||^2 / ||w_j||^2            [fraction of B's head living in A's top-5 feature subspace; direction-normalized conditioning]
- X6 = ||grad_w L_i(w_j)||                    [initial few-shot gradient norm at the B-init; step dynamics]
- X7 = L_i(w_j) BCE                           [initial few-shot loss at B-init; joint scale+dir+bias]

## Stats
Per descriptor: Spearman rho(X, FT) + bootstrap 95% CI (resample off-diag cells, B=2000) + permutation null p (2000). Per dataset (CIFAR, DTD). FT target recomputed NSEED=10 (stable; P0 showed r_SB 0.95+).

## Pre-registered decision (commit BEFORE numbers)
- **HIT (thread advances to powered P2):** at least one descriptor X with |rho(X,FT)| >= 0.30 AND boot 95% CI excluding 0, REPLICATING with the same sign on BOTH datasets, and beating COS (which is ~0). Report which FAMILY wins.
  - If a SCALE/CONDITIONING descriptor wins -> "frozen forward transfer is an optimization/conditioning effect, orthogonal to the discriminative cosine" = mechanism found.
  - If a DIRECTION descriptor unexpectedly wins -> re-examine (would contradict P0's cosine null).
- **NULL (frozen transfer geometrically structureless):** no descriptor clears 0.30 with cross-dataset replication. This is STILL a publishable contrarian result given P0 (reliable transfer, no low-order geometric predictor) AND it is the pre-registered MOTIVATION to move to the PLASTIC regime (B reshapes features; feature-reuse geometry + LogME genuinely defined) for the real second-law hunt.
- Cross-dataset NON-replication (pops on one, null on other) = treat as NULL (the DTD data-switch is the honesty test, per NEW dirlaw-data lesson).

## Falsifier-first
Cheap ($0). Every outcome is decision-relevant: HIT -> P2 powered law; NULL -> plastic regime is next (not a dead end). No descriptor is allowed to be declared a law without cross-dataset replication + CI + perm-null.
