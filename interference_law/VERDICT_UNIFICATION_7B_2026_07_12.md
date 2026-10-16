# VERDICT — Thread #7b: UNIFICATION, w_A-projected re-test (PREREG 8cffe1c8, locked 2026-07-12). Landed, $0 local, n=120, non-saturated (median forget_old 0.099).

## RESULT: NOT rescued. Projecting onto the DEPLOYED head axis w_A (instead of #7's centroid-diff d_A) gave the SAME wrong sign — raw on-axis margin collapse ρ −0.25. All pre-registered predictions P1b–P4b FALSE. The one near-perfect predictor (d′ along w_A, ρ +0.82) is DEFINITIONAL (AUC≈Φ(d′/√2)), not a directional law. #7 + #7b together CLOSE the unification: the frozen/plastic dichotomy is IRREDUCIBLE by this family of tests.

## Scoreboard (predictor vs forget_old = deployed frozen-A-head AUC drop)
  predictor           pooled ρ   pooled r    within-B ρ    pre-registered expectation      outcome
  onaxis_w (P1b)       −0.246     −0.117      −0.297       ≥ +0.50                          FAIL — wrong sign (as #7)
  on_absA              −0.246     −0.144      −0.266       (paired w/ off for P2b)          wrong sign
  offaxis_w (P2b)      −0.074     +0.071      −0.151       weak (<0.30)                     ~weak, but P1b fail makes moot
  dtheta ‖Δθ‖          +0.261     +0.554      (per-B n/a)  positive                         still the real magnitude predictor
  cosAB (frozen)       −0.149     −0.227      —            (original coordinate)            faint negative whisper, persists
  cosAB → onaxis_w (P4b) +0.161     —          —            ≤ −0.30 (loop closure)           FAIL — wrong sign
  ddprime (d′ drop)    +0.816     +0.885      —            SECONDARY (flagged definitional) DEFINITIONAL, not evidence

## Reading it honestly
- **Two independent projection axes, same falsification.** #7 projected onto the centroid-difference d_A (ρ −0.26); #7b onto the
  deployed head's logistic weight w_A (ρ −0.25). Both give margin-collapse the WRONG sign vs forgetting. The "I projected onto the
  wrong axis" escape hatch from #7 is now CLOSED — the deployed axis fails identically. The unification is not a projection artifact.
- **The +0.82 is a mirror, not a window.** d′ = margin / noise along w_A, and for near-Gaussian scores AUC = Φ(d′/√2), so Δd′ tracks
  ΔAUC ALMOST BY CONSTRUCTION. I pre-registered it as "secondary, flagged definitional" precisely so I couldn't cash it as a result
  now. It tells us the deployed-head metric is Gaussian-clean; it does NOT tell us interference is directional. Reporting it as a
  discovery would be exactly the overclaim the prereg exists to prevent.
- **‖Δθ‖ remains the real predictor** (pearson +0.55 between-B): plastic forgetting is MAGNITUDE-governed, as #1 crowned. Nothing here
  dethrones it; the directional reconstructions all failed.
- **The frozen −cos law leaves a persistent FAINT residual** (cosAB ρ −0.15 to −0.20 across #7/#7b, correct sign): with a fully plastic
  backbone, anti-parallel B still forgets marginally more. Real but weak — a whisper, not the governing variable.

## The wrong-sign puzzle (POST-HOC observation, NOT a conclusion — flagged for a possible #7c)
Raw margin collapse anti-correlates (−0.25) while normalized d′-drop correlates (+0.82) → the NOISE (within-class spread along w_A),
not the centroid margin, drives the deployed-head forgetting: classes forget when their score distribution along w_A gets FUZZIER
(variance up), often WHILE the centroid margin holds or grows. i.e. plastic forgetting ≈ variance inflation along the decision axis,
not centroid drag. This is a statement about WHICH MOMENT of the score distribution moves — non-definitional and mechanistically
interesting — but it was NOT pre-registered. If pursued it is a NEW prereg (#7c: decompose Δd′ into margin vs noise moments, test
whether variance-inflation is B-direction-dependent). NOT claimed here.

## Bottom line for the arc
UNIFICATION THREAD CLOSED (two honest attempts, both falsified). The frozen/plastic dichotomy STANDS as irreducible under directional
margin-projection: frozen readout forgetting is direction-governed (−cos, 4-op law); plastic backbone forgetting is magnitude-governed
(‖Δθ‖), with only a faint directional residual (cosAB ~−0.18). The two laws are the two faces of interference across the frozen/plastic
boundary — genuinely two mechanisms, not one law in two projections.

## Caveats
Single seed, ResNet18, one FT strength (partial regime confirmed, median 0.099). d′ Gaussian-approx is the reason for +0.82 — verified
by construction, not measured. The variance-inflation observation is post-hoc (→ #7c if wanted). cosAB residual (−0.15/−0.20) is stable
in sign across #7+#7b but weak in magnitude. This CLOSES the "collapse the dichotomy" line; DYNAMICS (Direction 2) is the untouched axis.
