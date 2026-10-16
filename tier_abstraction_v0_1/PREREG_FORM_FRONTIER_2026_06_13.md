# PRE-REGISTRATION — The Form Frontier ("what KIND of abstraction pays?")

**Date:** 2026-06-13. Locked BEFORE running. Motivation: C5 tested ONE form of abstraction —
continuous, 8-dim, **Gaussian-prior, flat-Euclidean** VIB — and got a near-tie (borderline). The
"does it pay" question therefore has two orthogonal axes: the REGIME of utility (data scarcity /
shift / task — see PREREG_PAY_FRONTIER) and the FORM of the abstraction (this doc).

## The five forms (what each varies, holding selectivity-production constant)
- **Form 1 — Geometry of the latent.** Euclidean (current) vs **hyperbolic** (negative curvature,
  low-distortion tree embedding — Nickel & Kiela; Ganea) vs simplex/categorical vs product spaces.
  *Prediction: if abstraction IS a tree (CIFAR-100 fine⊂super is a tree), hyperbolic pays where
  Euclidean was borderline.* Small architectural change, sharp prediction, lit anchor. **RUN FIRST.**
- **Form 2 — Discrete vs continuous.** VQ-VIB / Gumbel-softmax codewords. Mixed priors → a real test.
- **Form 3 — Compositional vs holistic.** Multiple parallel z_trans modules, one per attribute axis;
  hold out novel COMBINATIONS (Lake & Baroni / SCAN). Canonical "abstraction pays" case. Biggest lift.
- **Form 4 — What gets supervised.** Self-supervised / contrastive / causal abstraction (vary what
  makes the latent abstract, holding the latent constant). Bigger change, less clean prior.
- **Form 5 — Granularity / depth (bit budget).** Shallower (8 dims for 20 supers ≈ 4.3 bits, easy)
  vs deeper (1 dim for ~6 macro). Risk: redundant with Axis 1 (both probe bit-budget). Low priority.

## EXECUTION ORDER (locked)
**Axis 1 (data scarcity) -> Form 1 (hyperbolic) -> Form 3 (compositional).** Forms 2/4/5 deferred.
Form 3 runs LAST and only with a hardened held-out-combination protocol (Locatello's "designed the
task to favor my method" pathology is acute for compositional task-construction).

## Form 1 — locked design (hyperbolic z_trans)
- **Change ONLY z_trans's geometry**; everything else identical to Arm C (fine<-z_low Euclidean,
  super<-z_trans, beta=0.1). Encoder emits tangent vector v in R^8; sample in tangent space (VIB:
  v ~ N(mu,sigma), KL to N(0,I) — the wrapped-normal approximation, Nagano et al.); map to the
  Poincare ball z = exp_0(v) = tanh(||v||) * v/||v||. Super head = hyperbolic distance to learned
  per-super prototypes p_s in the ball: logit_s = -d_hyp(z, p_s); CE over logits.
- **PAIRED comparison: hyperbolic z_trans vs Euclidean z_trans (Arm C), same wiring.** Evaluated on
  (a) selectivity, (b) C5 super-transfer (8000-pool), (c) Axis-1 low-data super-transfer.
- **CONFIRM Form-1-pays** iff hyperbolic beats Euclidean z_trans on coarse transfer by >= +2% top-1
  (paired, CI excludes 0) on EITHER the C5 task OR any Axis-1 fit-pool <= 1000. **Secondary:** does
  hyperbolic raise selectivity above Euclidean's +0.13?
- **REFUTE** if hyperbolic <= Euclidean within 2% everywhere -> geometry is not the missing piece.

### Numerical landmines (hyperbolic ops are finicky — pre-empt)
- Project into the ball every step: z <- z * min(1, (1-eps)/||z||), eps~1e-5 (else d_hyp -> inf).
- Clamp arcosh argument >= 1+eps; clamp 1-||x||^2 >= eps in the distance denominator.
- exp_0 at ||v||->0: tanh(||v||)/||v|| -> 1; guard with eps.
- float32 with clamping first; escalate to float64 only if unstable. Lower lr if boundary blowups.

## Verdict logic
Form-1 CONFIRMS -> "the FORM (geometry) was the missing piece" — a substantial finding even if the
Euclidean version is functionally null. Form-1 REFUTES -> geometry isn't it; move to Form 3 (or, if
Axis-1 also nulled, the program needs a structural reset, not more form-tuning).
