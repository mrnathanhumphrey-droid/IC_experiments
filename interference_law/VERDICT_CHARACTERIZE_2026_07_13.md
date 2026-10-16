# VERDICT — NEW-1a''''-d: what describes the interaction? NO general geometric law (PREREG 40287dfa, locked 2026-07-13). Landed, $0 local 924s, resnet18, CIFAR + DTD, 16×16.

## RESULT: NULL on the descriptor hunt. My lead (feature cosine = general descriptor) is FALSIFIED. On DTD, ALL four candidate geometries — feature −cos(d_i,d_j), Jacobian −cos(g_i,Δθ_j), gradient −cos(g_i,g_j), centroid −cos(μ_i,μ_j) — land at ~+0.07–0.08 (noise, SE≈0.065): NONE describe DTD's real low-rank interaction. On CIFAR the best (Jacobian +0.248) is weak and the interaction is DIFFUSE. → the interference is directionally shaped (asymmetric + low-rank on DTD) but has NO general geometric law among the natural candidates; −cos was a weak, CIFAR-specific fit.

## The table (ρ of double-centered descriptor vs double-centered directed matrix; n=240 offdiag, SE≈0.065)
  dataset  feature-cos  Jacobian-cos  gradient-cos  centroid-cos   best        interaction SVD (top-4 var%)
  CIFAR      +0.191       +0.248         +0.016        +0.182     Jacobian    22/19/16/13  (DIFFUSE, no dominant mode)
  DTD        +0.075       +0.072        −0.022        +0.080     centroid    25/23/14/10  (LOW-RANK, top-2 = 48%)

## Reading it
- **DTD: no descriptor fits.** All candidates ~+0.07–0.08 = statistically zero. The DTD interaction is REAL and LOW-RANK (top-2 SVD modes =
  48% variance; shape-probe confirmed it beats the shuffle null) but it is NOT −cos of feature directions, NOT −cos of the margin-Jacobian
  with displacement, NOT gradient alignment, NOT centroid similarity. The compact shape exists; its geometry is outside our tested hypotheses.
- **CIFAR: weak and diffuse even at its best.** Jacobian +0.248 wins but feature (+0.191) and centroid (+0.182) are close — three different
  cosines correlate similarly weakly, and the interaction spreads across modes (22/19/16/13%, no dominant one). So the CIFAR "directional
  law" (−cos) explains only a modest, diffuse slice — consistent with, and reinforcing, the deflation of the original +0.45 headline.
- **Gradient cosine (−cos(g_i,g_j)) is dead on both** (+0.016 / −0.022) — replicates #6b's null; margin-Jacobian ALIGNMENT between two
  victims carries nothing; it was the Jacobian-vs-DISPLACEMENT term (JC) that had the (weak, CIFAR-only) signal.
- **The dominant Jacobian on CIFAR is not general:** JC CIFAR +0.248 → DTD +0.072. The −cos(g_i,Δθ_j) law does not transfer, as 1a''''-b/-c
  already showed; here it is put head-to-head with alternatives and still only wins (weakly) on CIFAR.

## The honest end-state of the directional thread
The interference DIRECTIONAL STRUCTURE is established and general: asymmetric on every dataset (1a''''-c), with a real low-rank per-pair
interaction off CIFAR. But its LAW is NOT: no general geometric descriptor was found. The best we have is −cos(g_i,Δθ_j), which fits CIFAR
weakly (+0.25) and diffusely and does not transfer. The DTD low-rank subspace (mode-1 victims {7,15,8,6}, perturbers {9,6,11,2}, u1·v1
corr 0.15 → giver≠taker structure) is real but uninterpreted — its descriptor is outside {feature, Jacobian, gradient, centroid} cosines
(candidates NOT exhausted: class-confusion, data-level similarity, higher-order/nonlinear geometry remain untested).

## What survived the whole hardening + characterization arc (the durable findings)
  ROBUST (arch + dataset):  asymmetry-general (interference is directional everywhere) ; MARGIN-PROTECTS / fragility ≈ d′ (well-separated
                            capabilities resist forgetting) ; disruptiveness ≈ ‖Δθ‖ (perturber travel).
  BOUNDED / CIFAR-weak:     the directional INTERACTION is real+general as STRUCTURE but has NO general geometric LAW; −cos(g_i,Δθ_j) is a
                            weak CIFAR-specific fit, not the universal directional law the session capstone (1a''') claimed.
  SOLID (earlier arc):      frozen 4-op cosine law (feature-space, readout-level, multimodal).

## Caveats
Single seed; resnet18; 16×16 (240 pairs, SE≈0.065 → DTD descriptors indistinguishable from 0, CIFAR +0.19–0.25 are ~3–4 SE weak-real).
Four candidate descriptors tested, NOT exhaustive (confusion matrix / data-similarity / nonlinear untested — the DTD shape could have a
non-geometric or higher-order descriptor). GTSRB excluded (saturated). The SVD subspace is real but its physical meaning is uncracked.

## Bottom line
The descriptor hunt is a clean NULL: the plastic directional interaction is real and low-rank off CIFAR but describable by NONE of the
natural cosine geometries; −cos is CIFAR-specific and weak. The directional thread is now thoroughly bounded — real structure, no general
law. The durable, substrate-general findings are asymmetry + margin-protects + disruptiveness. Natural resting point for the directional line.
