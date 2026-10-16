# VERDICT — DOMAIN-CONTROLLED re-test: the diffusion "antagonist" of the 4-op cosine law is a DOMAIN ARTIFACT; swap-family geometry is REAL (PREREG 2026-07-26). $0 local CPU, frozen prod_swap_seed0.pt 128-d, SUB=260, BOOT=500, corpus face_swap_corpus.

## RESULT: The load-bearing dramatic fact of the 4-op cosine "law" -- diffusion is anti-parallel to the swap family (cos -0.94), hence routes to its own head / is cleanly excisable -- is DOMAIN-DRIVEN, not forgery-driven. Under domain control (each family's direction referenced to its OWN dataset's real pool instead of the shared ffpp/real pool), the swap-diffusion cosine goes -0.937 -> +0.575 (CI [-0.34,+0.85] EXCLUDES the shared -0.94), the operative C_shared perfectly rank-tracks C_domain (rho +1.00) over C_forgery (+0.54), and the ROUTING DECISION REVERSES: diffusion isolates into its own head under C_shared but JOINS the swap family under C_forgery. NUANCE (real signal): the swap-family INTERNAL geometry (celebdf-dfdc 0.973->0.915, swap-celebdf 0.489->0.400, swap-dfdc 0.341->0.687) is ROBUST to domain control = genuine forgery similarity. So: swap-family clustering is real; the "diffusion = anti-parallel antagonist" story is a domain (dataset) artifact.

## The numbers (cos, [95% bootstrap CI]; SUB=260, B=500)
  pair                shared (ffpp-ref)      forgery (own-real ref)     domain      read
  swap-diffusion      -0.937 [-0.95,-0.90]   +0.575 [-0.34,+0.85]       ~0(swap=ffpp)  ANTAGONISM DESTROYED (CIs non-overlap)
  diffusion-celebdf   -0.349 [-0.43,-0.26]   -0.011 [-0.61,+0.47]       -0.325      collapses to ~0
  diffusion-dfdc      -0.191 [-0.32,-0.07]   +0.233 [-0.54,+0.70]       -0.083      flips positive
  swap-celebdf        +0.489 [+0.32,+0.63]   +0.400 [-0.15,+0.69]       n/a         ROBUST (real forgery)
  swap-dfdc           +0.341 [+0.15,+0.52]   +0.687 [-0.06,+0.95]       n/a         robust/stronger
  celebdf-dfdc        +0.973 [+0.95,+0.98]   +0.915 [-0.07,+0.97]       +0.947      ROBUST (real forgery)
  corr(C_shared,C_forgery)=+0.538   corr(C_shared,C_domain)=+1.000  -> operative cosine tracks DOMAIN
  routing@tau0: SHARED=[[swap,celebdf,dfdc],[diffusion]]  FORGERY=[[swap,diffusion,celebdf,dfdc]]  -> diffusion isolation REVERSES

## Pre-registered conditions (reported explicitly, no goalpost-moving)
- prereg strict "forgery-cos CI > 0": NOT MET (CI [-0.34,+0.85]; wide, data-limited by only 247 diffusion reals). Reported, not hidden.
- antagonism_destroyed (forgery CI excludes shared -0.94): MET.
- forgery point positive (+0.575): MET.
- shared tracks domain (rho +1.0 > +0.54): MET.
- routing reverses (isolate->join): MET.
- swap-family robust to domain control: MET.
=> Weight of evidence: DOMAIN ARTIFACT for the diffusion antagonist; the single strict CI criterion falls short only due to the 247-real data cap.

## What this overturns / qualifies (honest, high-integrity self-audit)
- The 4-op "one FORGERY cosine governs LEARN/FORGET (route) / UNLEARN (collateral inverted cos) / PROTECT" framing is CONFOUNDED WHERE DIFFUSION IS THE ANTAGONIST. "Diffusion is anti-parallel, cleanly excisable" (VERDICT_DIRECTIONAL_UNLEARNING) is a DOMAIN effect: diffusion is cleanly removable because it is OUT-OF-DATASET, not because its forgery mechanism is anti-parallel. A diffusion generator IN the ffpp domain would (by the +0.57 forgery cosine) NOT be cleanly excisable -- opposite the arc's implied general claim.
- The audio_cf / text "cross-substrate triangulation" that leans on "vision detector cos -0.93 catastrophic" (VERDICT_AUDIO_TRIANGULATION) is undermined at its vision anchor -- the -0.93 is domain, not forgery.
- The OPERATIONAL wins remain REAL: the detector genuinely isolates and cleanly excises the diffusion generator. But the CAUSE is domain distance (a different source dataset), not anti-parallel forgery geometry. Real capability, wrong mechanism story.
- PRESERVED: the swap-family (swap/celebdf/dfdc) cosine structure is domain-robust and IS real forgery-similarity geometry; routing/protection WITHIN the swap family is legitimately forgery-driven.

## Caveats
Bootstrap CIs wide for diffusion pairs (247 diffusion reals = deepfakeface/real cap). No diffusion-on-ffpp faces exist on disk, so a same-source swap-vs-diffusion contrast is impossible; the own-reference control is the cleanest achievable and is decisive on CI-nonoverlap + routing reversal. Mean-difference directions, 128-d frozen features, filename-hash (not identity-disjoint) as in the original probes; deltas/structure are the claim.

## Bottom line
The 4-op cosine "law" splits under domain control into (a) a REAL forgery-similarity geometry within the swap family, and (b) a DOMAIN-DISTANCE artifact masquerading as forgery antagonism for the out-of-dataset diffusion generator. The dramatic "antagonist" narrative and its cross-modal triangulation do not survive; the swap-family operational geometry does. High-integrity correction: real operational capability, confounded causal framing. USER writes; Claude banked numbers in out/domain_control.json + this verdict.
