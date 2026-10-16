# PREREG — P1c (round-2): OPTIMIZATION-LANDSCAPE descriptors for the proven frozen-FT INTERACTION

**Locked 2026-07-26.** Transfer-geometry thread, phase P1c. $0 local, frozen resnet18 D_PROJ=24, CIFAR+DTD. Reuses P1/P1b. ROUND 2 of the descriptor hunt -> strict anti-fishing bar.

## Target (proven by P1b)
Reliable frozen FT(i<-j) is dominated (~76% of reliable variance, comp-rel 1.00, both datasets) by the pairwise INTERACTION resid_ij = FT - mu - row_i(target) - col_j(source). It is scale-INVARIANT (survives norm-control) and direction-BLIND (P0/P1: cosine/subspace/CKA all null). So the descriptor must be a genuine (i,j) INTERACTION that is neither a direction-alignment nor a magnitude. Mechanism hypothesis: how B's head w_j pre-positions task-i's few-shot GD in its loss LANDSCAPE (reachability / curvature / feature-usage), beyond direction & scale.

## Primary target = INTERACTION RESIDUAL (not raw FT)
Correlate each descriptor with resid_ij (doubly-centered). RATIONALE: a descriptor that is secretly a source(j)-only or target(i)-only main effect correlates ~0 with the residual by construction -> the residual is the honest filter for genuine interaction structure. Also report corr with raw FT for continuity. All descriptors use UNIT-normalized [w_j,b_j] (scale controlled, since interaction is scale-invariant and we do NOT want to re-discover the ||w|| source effect).

## Descriptor battery (pairwise i<-j; cached feats + trained heads; unit-normalized w_j)
- Y1 REACHABILITY = ||proj_{S_i}(w_j)||^2, S_i = top-5 PCs of task-i cold few-shot GRADIENT span (directions GD actually explores).
- Y2 NEWTON-DECREMENT = g_i(w_j)^T H_i(w_j)^-1 g_i(w_j) (few-shot logistic grad/Hessian at w_j; descent speed / basin quality).
- Y3 SCORE-SPREAD = var(X_few_i @ w_j + b_j) (how the init distributes task-i few-shot points).
- Y4 WHITENED-ALIGN = w_j^T C_i d_i, C_i = task-i pos feature covariance (metric-aware alignment; differs from raw cosine).
- Y5 FEATURE-USAGE-OVERLAP = spearman(|w_j|, |d_i|) over 24 coords (UNSIGNED coordinate-usage overlap; sign-blind so decoupled from cosine). [star candidate]
- Y6 CURVATURE-TRACE = trace(H_i(w_j)) (active signal magnitude at the init).

## Stats
Per descriptor x {resid, rawFT}: Spearman rho + bootstrap 95% CI (B=2000) + permutation null p (2000). Per dataset CIFAR+DTD. FT/resid from NSEED=10 (P1b-grade reliability).

## Pre-registered decision (STRICT, round-2 anti-fishing)
- **HIT:** some descriptor with |rho(X, RESID)| >= 0.30, boot CI excluding 0, SAME-SIGN replication on BOTH datasets, AND perm_p < 0.01. (Higher bar than P1's implicit 0.05 because this is round 2 / multiple batteries.) -> advance that descriptor to a powered P2 law.
- **NULL:** no descriptor clears the bar with cross-dataset replication. -> the interaction resists optimization-landscape descriptors too. HONEST outcome: bank the strengthened negative (reliable transfer interaction defeats direction, scale, subspace, CKA, reachability, curvature, feature-usage) AND pivot to the PLASTIC regime with a sharpened question. The P0+P1+P1b+P1c package stands regardless.
- Report the family/mechanism of any HIT. Non-replication across datasets = NULL.

## Falsifier-first
$0, cached. Every outcome decision-relevant. No descriptor becomes a "law" without cross-dataset replication + CI + perm_p<0.01. Guards against round-2 fishing by (i) strict bar, (ii) residual (not raw FT) as primary target, (iii) cross-dataset replication requirement.
