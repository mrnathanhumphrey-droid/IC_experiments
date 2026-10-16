# VERDICT — NEW-2: interference matrix is NOT low-rank / NOT completable (PREREG 7c1f8288, locked 2026-07-13). Landed, $0 local 1340s, resnet18, CIFAR + DTD, 24×24.

## RESULT: NEGATIVE, and it CORRECTS the shape-probe's "low-rank" read. The directed forgetting matrix does NOT complete from a sample: best held-out R² is 0.19–0.27 (CIFAR/DTD) and that is the BIAS-ONLY (fragility+disruptiveness) model; adding ANY low-rank interaction makes held-out prediction strictly WORSE (k=1 → negative R²). The interaction is NOT low-rank (eff-rank ~8–9 of 24, diffuse spectrum) and NOT predictively completable. Only the two per-capability main effects carry (weak) signal; the per-pair interaction is matrix-level noise.

## The completion curve (held-out R² vs model rank k, 50% observed, 5 mask seeds)
  dataset   k0(bias)  k1     k2     k3     k4     k6     k8   | best  interaction: effrank  top-6 var%
  CIFAR      +0.27   −0.19  −0.53  −0.77  −0.82  −0.66  −0.17 | k0    9.1   [21,14,13,10,9,6]
  DTD        +0.19   −0.61  −0.86  −1.19  −1.11  −0.94  −0.34 | k0    8.4   [22,16,13,10,8,6]
  (NEW2-P1 completable R²≥0.5: FAIL both. NEW2-P2 interaction adds ≥+0.05: FAIL both, ΔR²=0.000. NEW2-P3 effrank ≤ N/3=8: FAIL, ~8–9 & diffuse.)

## Reading it
- **NOT completable (P1 fail).** Best R² 0.27/0.19 << 0.50 bar. And that best is k=0 = pure main effects. So you can predict a held-out
  forgetting entry only ~20–27% as well as knowing its victim-fragility and perturber-disruptiveness — and NOT AT ALL better by modeling
  the interaction.
- **Adding interaction rank HURTS.** k=1 already drops held-out R² below zero (worse than the mean); it worsens monotonically to k~3–4.
  Fitting a low-rank interaction to observed entries OVERFITS and generalizes terribly → the per-pair interaction has NO generalizable
  low-rank structure. It is, from a completion standpoint, noise.
- **The interaction is NOT low-rank (P3 fail).** Eff-rank 9.1/8.4 of 24; top mode only ~21%, gradual decay across many modes. Diffuse.
- **This CORRECTS 1a''''-c.** The shape probe at N=16 reported DTD interaction "low-rank, eff-rank 4.4, top-2=48%" and I called it a real
  low-rank shape. At N=24 the eff-rank is 8.4 and the spectrum is diffuse, and the COMPLETION test (the honest generalization test) fails.
  The N=16 "low-rank" was a small-N artifact + the shuffle-null tested concentration, not completability. DOWNGRADE: the interaction is
  NOT low-rank; it is diffuse and non-completable. (The asymmetry from 1a''''-c stands — that's a main-effect/matrix property, not a rank claim.)

## Where this leaves the interference-interaction story (now firmly bounded)
The per-pair directional INTERACTION is REAL (matrices are asymmetric; it beats a shuffle-null in mild concentration at small N) but it is
STRUCTURELESS at every handle we tried: NO geometric descriptor fits it (1a''''-d: feature/Jacobian/gradient/centroid cosines all ~0 on DTD),
and it is NOT low-rank / NOT completable (this). It is high-dimensional, idiosyncratic per-pair variation with no exploitable structure.
The ONLY predictable structure in the interference matrix is the two per-capability MAIN EFFECTS — fragility (≈ margin/d′) and disruptiveness
(≈ ‖Δθ‖) — and even those complete only weakly (R² ~0.2).

## The durable arc state (unchanged, reinforced)
  SOLID:   Frozen 4-op cosine law (feature-space −cos, readout-level, multimodal).
  ROBUST:  Plastic MAGNITUDE structure — MARGIN-PROTECTS/fragility≈d′ + disruptiveness≈‖Δθ‖ + ASYMMETRY-general (arch + dataset).
  BOUNDED/NULL: Plastic per-pair directional INTERACTION — real but structureless: no geometric law (CIFAR-specific weak −cos only), NOT
               low-rank, NOT completable. The session capstone's "general plastic directional law" is fully retired; the durable plastic
               finding is MAGNITUDE/margin, not direction, and the interaction has no exploitable low-dim structure.

## Caveats
Single seed for the matrices (5 mask seeds for completion variance); resnet18; N=24; soft-impute completion at 50% observed. R² could rise
at higher observed fractions but the k>0 HURTING (negative ΔR²) is the decisive signal — it is not a data-scarcity artifact, it is
overfitting a non-low-rank interaction. GTSRB excluded (saturated). Completion tested the FULL matrix (bias+interaction); main-effects R²~0.2
is itself modest (fragility/disrupt are weak absolute predictors, consistent with 1a's LOO).

## Bottom line
The interference matrix is NOT a low-dimensional, completable object. Its only predictable content is two weak per-capability scalars
(fragility, disruptiveness); the per-pair interaction is real but has no geometric law and no low-rank structure — it does not complete.
This CLOSES the interference-structure line (descriptor + low-rank both null). The durable science: frozen cosine law + plastic margin/
magnitude/asymmetry. Natural, well-bounded endpoint for the whole capability-interference-matrix thread.
