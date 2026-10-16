# VERDICT — Thread #4: curvature engineering (PREREG 128d757a, locked 2026-07-11). Landed, $0 local CPU. ARC-CLOSING.

## RESULT: protection is ENGINEERABLE but CONSTRAINED by the cosine geometry — you can only protect cosine CLUSTERS, not arbitrary subsets.
Interference = "collapse toward diffusion-only" (erodes swap/celebdf/dfdc). Engineer protection of a chosen set S by GPM-projecting
the interference off span{d_p : p in S}. Realized profile vs desired S:

  protect S            swap  diffusion  celebdf   dfdc
  {} baseline          0.442   0.991     0.613    0.750     (swap-family eroded)
  {swap}               0.595   0.984     0.711    0.782     self +0.153, LEAKS +0.065 to celebdf/dfdc, −0.008 to diffusion
  {celebdf}            0.512   0.991     0.777    0.846     self +0.164, LEAKS +0.083 to swap/dfdc,    +0.000 to diffusion
  {dfdc}               0.488   0.992     0.756    0.854     self +0.104, LEAKS +0.095 to swap/celebdf, +0.000 to diffusion
  {diffusion}          0.381   0.981     0.427    0.495     self ~0 (at ceiling), leak to family −0.167 (independent/anti)
  {swap,celebdf,dfdc}  0.584   0.984     0.759    0.872     family retained as a UNIT (clean cluster protection)
  {swap,diffusion}     0.633   0.947     0.659    0.591

## The three arc-closing claims — all met
1. **Engineering works:** protecting S raises S's retention vs baseline (swap +0.153, celebdf +0.164, dfdc +0.104). Aimable.
2. **Constrained by entanglement:** protecting ANY swap-family singleton LEAKS to its aligned neighbours (+0.065…+0.095) — you CANNOT
   protect swap while letting celebdf/dfdc forget. The family moves as a unit. Protecting the whole family {swap,celebdf,dfdc} is the
   clean realizable profile.
3. **Anti-parallel independence:** protecting {diffusion} does NOT raise the swap-family (leak to anti ≈ 0/negative); protecting the
   family does NOT raise diffusion. Anti-parallel capabilities protect independently.

## Leakage follows +cos — the SAME geometry, a THIRD time
protection-leak(S→p) is POSITIVE for aligned p (cos>0.3: +0.065…+0.095) and ~0 for anti-parallel p (cos<0: −0.008…0.000). Protection
spills along +cos, exactly as FORGETTING spills along −cos (#1) and UNLEARNING-COLLATERAL spills along +cos (#3).

## ★★★★★ THE ARC, CLOSED: one cosine geometry governs FOUR operations
The discriminative-direction cosine between capabilities is the single organizing variable for the entire interference phenomenology:
- **LEARN-together (route arms):** cluster by cosine — share a head within a cluster, isolate the anti-parallel one.
- **FORGET (#1 interference law):** forget ∝ −cos, calibrated & out-of-sample predictive (R²≈0.5), predicts detector+Librarian.
- **UNLEARN (#3):** collateral ∝ +cos — anti-parallel capabilities excise cleanly, aligned ones are entangled/inseparable.
- **PROTECT (#4):** protection leaks ∝ +cos — you can engineer protection of a cosine CLUSTER, not an arbitrary singleton.
And it is MODALITY-GENERAL (text/vision/audio, #5) and quantitative (#1 predicts the detector's 0.69 collapse from cos −0.93).

## Caveats (honest)
Head-level (frozen backbone) — engineering the linear readout's curvature, not representational. GPM-projection of the interference
gradient stands in for a designed "childhood curriculum" (the curriculum-DATA construction is a heavier follow-up). Filename-hash
split (abs AUC inflated; deltas/leakage-structure = the claim). Single seed. Directions are estimated means. Mechanism demonstration
that PROTECTION obeys the cosine geometry — not a deployable continual-learning controller.
