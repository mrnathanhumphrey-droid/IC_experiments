# VERDICT — P1b: the reliable frozen-FT variance is DOMINATED by a pairwise INTERACTION that is orthogonal to all tested geometry AND survives norm-control (2026-07-26). $0 local, frozen resnet18 D_PROJ=24, CIFAR+DTD, NSEED=10. Corrects P1's "structureless" over-claim.

## RESULT: Frozen forward transfer is NOT purely scale and NOT structureless. Decomposing FT(i<-j) = mu + row_i(target) + col_j(source) + resid_ij(interaction): the RELIABLE variance is dominated by the pairwise INTERACTION on BOTH datasets (reliable-share ~0.76), with interaction component-reliability = 1.00. The ||w_j|| scale signal (P1) IS the SOURCE main-effect (col) -- rho(||w_j||,col) -0.60 cifar / -0.24 dtd, and norm-control removes 57%/23% of source variance -- but source is only 11-20% of reliable variance. The dominant interaction SURVIVES norm-control fully intact (share rises 0.76->0.84 cifar / 0.76->0.78 dtd; FT reliability unchanged 0.98/0.94). => there IS a rich, reliable, scale-invariant pairwise transfer signal; the P1 descriptor battery (cosine, subspace overlap, CKA, zero-shot-AUC, norm, grad-norm, loss) simply does not capture it.

## The numbers (reliable-variance share = var_frac x component_reliability, renormalized; off-diag cells, 10 seeds)
  dataset  FT r_SB   target(row)  source(col)  INTERACTION(resid)   rho(||w||,col)  col-var norm/raw
  CIFAR    0.976     0.05         0.20         0.76                 -0.60           0.43
  DTD      0.945     0.13         0.11         0.76                 -0.24           0.77
  norm-control: interaction reliable-share -> 0.84 (cifar) / 0.78 (dtd); FT reliability 0.973 / 0.940 (essentially unchanged).
  component reliabilities: interaction=1.00 both; source(col)=0.99-1.00; target(row)=0.64-0.72 (noisier).

## Correction to P1
P1 VERDICT said "geometrically STRUCTURELESS." That over-claimed. The literal P1 result stands (no descriptor reaches |rho|>=0.30). But P1b shows the reliable FT is NOT structureless: 76% of its reliable variance is a perfectly-reliable pairwise interaction orthogonal to the whole battery. Correct statement: "frozen forward transfer is dominated by a reliable pairwise interaction that the standard geometry descriptors (cosine/subspace/CKA/scale) fail to capture." The ||w_j|| finding is real but is only the (minor) SOURCE main-effect, not the main story.

## Reading it (mechanism)
- Frozen FT(i<-j) = quality of task-j's trained head w_j as an INIT for task-i few-shot GD (K=30 steps), minus cold. The reliable signal is a genuine pairwise INIT x LANDSCAPE interaction: how well w_j pre-positions few-shot GD in task i's specific loss landscape, beyond w_j's norm (source effect) and task-i's init-sensitivity (target effect).
- It is scale-invariant (survives norm-control) and direction-blind (P0/P1) -> it is NOT a cosine/subspace alignment and NOT a magnitude effect. Candidate: optimization-REACHABILITY (is w_j inside the subspace few-shot GD explores for task i) or curvature/basin geometry -- untested.

## Why this matters
This REVIVES the frozen regime as a legitimate, cheap ($0, cached) place to hunt a transfer geometry. The target is now PROVEN to exist (reliable, dominant, un-captured) rather than assumed. The standard transferability descriptors are demonstrably insufficient -- which is itself a novel, publishable point given the crowded transferability-estimation literature (LogME/CKA/subspace all target the kinds of quantities that here fail).

## Bottom line
Frozen forward transfer: reliable (r_SB 0.94-0.98), cosine-blind (P0), dominated by a pairwise interaction (P1b, ~76% reliable share, comp-rel 1.00) that is orthogonal to cosine/subspace/CKA/scale and survives norm-control. The ||w_j|| effect is only the minor source main-effect. >>> NEXT (fork): (A) CHEAP $0 frozen round-2 -- pre-registered bake-off of OPTIMIZATION-LANDSCAPE / REACHABILITY descriptors targeting the proven interaction (gradient-span projection, curvature/basin alignment, per-example pre-separation); vs (B) PLASTIC regime (feature-reuse geometry + LogME). Recommend A first (signal proven there, $0), strict bar (|rho|>=0.30 + cross-dataset replication + perm-null) to avoid round-2 fishing.
