# VERDICT — NEW-1a''''-c: the SHAPE of the interference (PREREG 44ba13f8, locked 2026-07-13). Landed, $0 local 1346s, resnet18, square 16×16 directed matrices, CIFAR/DTD/GTSRB.

## RESULT: The directional STRUCTURE is dataset-GENERAL; the −cos LAW is CIFAR-specific. All three matrices are ASYMMETRIC (P1 ✓ all), and DTD/GTSRB carry a REAL LOW-RANK interaction (P2 ✓, ~4 eff-dims, beats shuffle null) — so there IS genuine per-pair directional shape off CIFAR. But −cos(g_i,Δθ_j) captures it ONLY on CIFAR (+0.293); it misses DTD (+0.060) and anti-aligns on GTSRB (−0.225). → interference is directionally-shaped everywhere, its geometry is dataset-dependent, and our −cos law describes only CIFAR's. Downgrade REDIRECTED: structure general, law CIFAR-bound, general descriptor OPEN.

## The table (resnet18, square directed matrix, 16 caps as victim=perturber)
  dataset  medFgt  ‖K‖/‖S‖  ρ(Fij,Fji)  interFrac  interReal(low-rank)  effRank  domRank1%  −cos-captures(align_dc,DC)
  CIFAR     0.247    0.273     +0.098      0.645      False (diffuse)       6.3      40%        +0.293
  DTD       0.259    0.334     −0.274      0.583      True  (s0 .88>null .73) 4.4      46%        +0.060
  GTSRB     0.385*   0.231     +0.664      0.305      True  (s0 2.0>null 1.5)  4.0      25%        −0.225
  P1 = asymmetric (‖K‖/‖S‖≥.15 & ρ<.70) ; P2 = interaction top-SV > shuffle-null 95th pct. (*GTSRB partly SATURATED — see caveat.)

## Reading it
- **ASYMMETRY GENERAL (P1 ✓ all three).** ‖K‖/‖S‖ 0.27/0.33/0.23; DTD the MOST directional (ρ(Fij,Fji)=−0.274). The interference is
  non-symmetric on every dataset → the asymmetry/fragility structure is dataset-general, strengthening the durable plastic finding.
- **REAL LOW-RANK INTERACTION off CIFAR (P2 ✓ DTD+GTSRB).** DTD/GTSRB double-centered interactions beat the entry-shuffle null and sit in
  ~4 effective dimensions → there IS concentrated per-pair directional shape off CIFAR, not merely fragility×disruptiveness row/col imbalance.
- **−cos IS A CIFAR-SPECIFIC DESCRIPTOR.** align_dc(−cos, real interaction): CIFAR +0.293 (−cos was built here, captures it), DTD +0.060
  (≈0, misses it), GTSRB −0.225 (wrong direction). So the SPECIFIC hypothesis −cos(g_i,Δθ_j) describes CIFAR's interaction geometry and
  NOT the (real, low-rank) interaction geometry on DTD/GTSRB. Confirms + explains 1a''''-b: −cos is CIFAR-bound; the interaction it was
  fitting is not the same geometry that lives on other datasets.
- **INVERSION worth noting:** where −cos works (CIFAR) the interaction is DIFFUSE (effrank 6.3, top-SV below null); where −cos fails
  (DTD/GTSRB) the interaction is LOW-RANK (effrank ~4, top-SV above null). CIFAR's interaction is a spread-out −cos-aligned signal; DTD's
  is a concentrated signal in a few directions that −cos is orthogonal to. Different geometries, both real.

## The redirect (constructive, not a dead end)
The dataset switch (1a''''-b) said "the directional LAW is CIFAR-specific." This probe refines that: the directional STRUCTURE (asymmetry +
real low-rank interaction) is GENERAL — present on textures and signs — but its GEOMETRY is dataset-dependent, and −cos only fits CIFAR's.
So the open target is no longer "does a directional law exist off CIFAR" (it does — a low-rank one) but "WHAT are the ~4 interaction
directions on DTD/GTSRB, and is there a general descriptor?" The shape is real and compact; we just don't have its law.

## Caveats (honest)
- P2's shuffle-null tests CONCENTRATION (low-rankness), NOT existence. CIFAR's P2=False means its interaction is DIFFUSE, not absent —
  −cos still correlates +0.293, so CIFAR's interaction exists but is spread across dimensions. Don't read CIFAR-P2=False as "no interaction."
- **GTSRB is partly SATURATED**: median off-diag 0.385 but many perturber columns hit 0.64–0.96 (near-total forgetting). Saturation inflates
  symmetry (ρ(Fij,Fji)=+0.664, the highest) and distorts the interaction. GTSRB's numbers are the LEAST trustworthy; DTD is the CLEAN
  off-CIFAR case (inband 0.259, no saturated columns) and it is unambiguous: strongly asymmetric, real low-rank interaction, −cos ≈ 0 (+0.06).
- Single seed; resnet18 only; 16×16 (240 pairs); dominance rank-1 % (40/46/25) echoes NEW-1's ~46% CIFAR — mostly-non-transitive on all.
- Diagonal excluded; Jacobian over all params; −cos self-terms excluded.

## Bottom line
Directional/asymmetric interference is GENERAL (all datasets); a REAL low-rank per-pair interaction exists off CIFAR (DTD clean); but the
−cos(g_i,Δθ_j) LAW is a CIFAR-specific descriptor that does not fit the (differently-shaped) interaction on other datasets. The capstone's
STRUCTURE claim is rehabilitated (interference is directionally shaped everywhere); its LAW claim stays CIFAR-bound. Durable across
everything: asymmetry + margin-protects. Open: characterize the DTD/GTSRB low-rank interaction subspace (its ~4 directions) — the shape has
a compact form we have not yet named.
