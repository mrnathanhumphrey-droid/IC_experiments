# PRE-REG — Thread #1: calibrate the cosine→forgetting INTERFERENCE LAW into a predictive curve. 2026-07-11.

SHA256-lock before the run. The arc has 3 anecdotes (detector cos−0.93 catastrophic, Librarian cos+0.67 benign, audio ρ=−0.52
in-rank) but no CALIBRATED curve — and #5 showed a saturated task hides the graded magnitude. This thread builds the graded
forgetting-vs-cosine curve on a RICH, NON-SATURATING real substrate, turns the sign-heuristic into an out-of-sample PREDICTOR, and
extracts the route/don't-route DECISION BOUNDARY (the cosine below which forgetting is large enough to justify a separate head).

## SUBSTRATE (user chose "new real domain"): CIFAR-100, frozen ImageNet ResNet18
- 100 fine classes (20 superclasses × 5) → each class = a "paradigm" (the detector/audio analog). Hundreds of natural class-pair
  directions spanning the whole cosine range: within-superclass pairs ALIGNED + hard (aquatic-mammals, trees, flowers), cross-
  superclass pairs more orthogonal/opposed + easy. This built-in difficulty gradient is what fixes #5's saturation.
- Frozen ResNet18 (ImageNet1K weights, cached) → 512-d penultimate (avgpool) features. NO backbone training. Extract once → cache npy.
- **SHARED NEGATIVE POOL** (label 0) = a fixed random sample of images drawn across a RESERVED set of classes, identical for every
  class-task → the only thing differing between paradigms is the POSITIVE class (the shared-reals design, third time).
- Frozen features → linear head (class-vs-shared-neg). Metric = rank-AUC. Deterministic index-hash split.

## NON-SATURATION GUARD (locked — the explicit fix for #5)
Target median base one-vs-shared AUC in **[0.75, 0.97]**. After extraction, print the base-AUC distribution. If median > 0.98
(saturated), harden BEFORE the pair sweep by either (a) restricting shared-negatives to WITHIN-superclass images, or (b) cutting head
epochs, until median ∈ [0.75,0.97]. The chosen setting is logged. (A saturated run would repeat #5's flaw and is disqualified.)

## Design
1. **Cosine matrix.** d_c = normalize(mean(feat class_c) − mean(feat shared_neg)) for each of C classes. Full C×C cosine.
2. **Pairwise forgetting (the curve).** For a large set of ordered pairs (all C×(C−1) if C≤50 ≈ ≤2450, else a cosine-STRATIFIED
   sample covering [−1,+1]): train head on A → AUC_A; continue-train on B (plain SGD, no GPM) → AUC_A'. forget = AUC_A − AUC_A'.
   Record (cos(d_A,d_B), forget, base_AUC_A) per pair.
3. **Calibrated predictor.** Fit forget ~ f(cos) (monotone / linear + binned deciles). SPLIT pairs 50/50; fit on half, predict the
   other half → held-out R² and MAE. Report the binned curve with error bars.
4. **Decision boundary.** The cosine c* where mean forget crosses a route-worthwhile threshold (report the curve; mark c* at
   forget=0.05 and 0.10). Below c* → route; above → share.
5. **Cross-modal overlay.** Drop the audio pairs (from #5 `audio_seq.json`) onto the same (cos, forget) axes → is the slope/sign
   consistent across vision+audio? (text: the Librarian single point +0.67/benign as a landmark.)

## Pre-registered read (locked before the run)
- **LAW HOLDS (calibrated):** forget is monotone-decreasing in cos — Spearman ρ < −0.4 (significant) AND held-out R² > 0.20 (cosine
  predicts forgetting on unseen pairs). Report c* (the decision boundary). This upgrades the law from "sign heuristic" to "predictor."
- **GRADED (fixes #5):** with the non-saturation guard, forgetting spans a real range (not pinned at ~1.0) and the aligned end shows
  low forgetting (< 0.05) while the anti-parallel end shows high (> 0.15) — a genuine curve, not a step.
- **CROSS-MODAL:** the CIFAR slope sign matches audio (#5) — negative. If the two substrates give consistent c* (within a modality-
  scale factor) that is a strong modality-general claim; if slopes differ in sign, the law is substrate-conditional → report.
- **NULL:** ρ ≈ 0 / non-monotone / held-out R² < 0.05 → cosine does NOT predict forgetting on a rich substrate → the 3 anecdotes were
  special cases, the law is weaker than claimed. Report honestly (this would be the most important negative of the program).

## Caveats
Head-level only (frozen backbone) — representational forgetting deferred (that's a later thread). One-vs-shared-neg framing (not full
multiclass CL). Index-hash split (not group-disjoint). Single seed for the sweep (report; multi-seed the headline c* if cheap).
CIFAR/ResNet18 is ONE substrate — cross-modal overlay guards against substrate-specific artifacts but a second rich substrate (text
20NG) is the natural replication if the law holds. Continue-train uses plain SGD (no GPM) — measures RAW directional interference.

## Ops
`probe_interference_law.py`: torchvision ResNet18 frozen feats on CIFAR-100 (cache), reuse the detector-probe head/auc/
discriminative_dir/continue_train. py_compile + --smoke (5 classes) before full. $0 local GPU, minutes. Output
`out/interference_law.json` (cosine matrix + per-pair (cos,forget) + fitted curve + held-out R² + c* + cross-modal overlay).

## Status
PREREG written; ResNet18 weights cached; CIFAR-100 fetching (HF). Next: confirm data → SHA256-lock → build → --smoke →
non-saturation guard check → full sweep → fit curve + c* + cross-modal overlay → bank.
