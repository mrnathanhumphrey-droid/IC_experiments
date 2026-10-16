# WILSON PROMPT — ratify the GPT-2 generalization verdict ("same flow, different type") + adjudicate the attention seed "miss"

## Ask
Adversarial pass on the GPT-2-medium dye-program generalization. The verdict is "same flow, different
type" — confirm or contest. Lead the pass on the ONE flagged anomaly (attention seed-localization went
opposite-direction to prediction), which I believe RESOLVES to a bundled-scalar confound but needs Wilson's
ruling because it touches the core "spectrum sets the profile" mechanism.

## The result (four-descriptor table, GPT-2/AG vs Pythia/AG anchor)
| descriptor | Pythia (two-scale) | GPT-2 (continuum) | locked prediction | verdict |
|---|---|---|---|---|
| HOLD laminar + curvature-ordering | spatial z 59-105, arrival +0.9 | spatial z 35-87, arrival +0.90->1.00 (z 9-34), laminar=true | universal | HELD |
| MOVE seed-loc up-proj (c_fc) | PR 5.0 | PR 18.5 | higher (less localized) | hit |
| MOVE seed-loc down-proj (c_proj) | PR 1.7 | PR 17.3 | higher (less localized) | hit (sharpest call) |
| MOVE seed-loc attention | PR 10.3 | PR 4.1 | higher | MISS (went MORE localized) |
| channel carrier | Wdown slowest | c_proj wets 37->1024 only by step200 (others step1) | down-proj channels | hit (in dynamics) |
| GRADING (Dye2) | transient 1.4x peak -> 1.08x | mean 1.06x, monotone, NO overshoot | <1.4x | HELD (richer: shapeless not just weaker) |

## What's solidly confirmed (my read, contest if wrong)
1. UNIVERSAL (substrate-independent): flow exists, spatially coherent, arrives in curvature order. Both nets,
   both Dye3 gates, clean. => laminar curvature-ordered flow is a CF PROPERTY, not Pythia-specific. Headline
   survives the net axis.
2. SPECTRUM-SET (substrate-specific) confirmed for grading+saturation: GPT-2 continuum spectrum (gpt2_stage0.json:
   all 3 carriers "continuum-ish", early_gap_ratios ~1.02-1.24) predicted NO sharp curvature cliff -> grading came
   in not just weaker but SHAPELESS (monotone creep to 1.06x, no transient overshoot; per-carrier c_attn 1.06/
   c_fc 1.10/c_proj 1.03). The absence-of-overshoot is a STRONGER confirmation than "smaller overshoot" — overshoot
   was the two-scale signature; no two-scale => no overshoot. This is the mechanism working at the level of SHAPE.

## The attention miss — my proposed resolution (needs Wilson ruling)
Predicted attention seed-PR >10.3 (less localized, like the other carriers on a continuum); got 4.1 (MORE localized).
Opposite-direction, so it can't be "effect smaller than thought." Two things ruled out / found:

- RULED OUT (rescue that FAILED): "attention seeded sharp because its LOCAL spectrum is two-scale" — FALSE.
  gpt2_stage0.json shows attn.c_attn is "continuum-ish" like the others (early_gap_ratios ~1.08-1.24, the gentlest).
  So a local-two-scale explanation does NOT hold. Honest: the global-spectrum-shape story does NOT cleanly predict
  the attention seed.

- WHAT I THINK IT ACTUALLY IS (bundled-scalar confound — the session's recurring species): seed-PR bundles TWO
  distinct quantities — (a) INITIAL-seed concentration at step 0, and (b) SPREAD dynamics after. Resolved per-step
  (gpt2_dye1.json wetted_count): attention wets 8->1024 AT STEP 1 (instant flood) — IDENTICAL to c_fc (31->1024 step1).
  Only c_proj channels (37->1024 by step200). So attention's SPREAD is diffuse/flooding exactly as the continuum
  prediction wants; it's only the INITIAL seed (step0 PR 4.1) that's tight. The prediction was about spread-localization
  (diffuse vs channeled); attention's spread IS diffuse. The "miss" is that seed-PR scalar conflated tight-initial-seed
  with spread, and they point opposite ways for attention.
  => Proposed verdict: the spectrum-shape mechanism HOLDS for spread (attention floods like c_fc, confirming continuum);
  the tight INITIAL seed is a SEPARATE attention-architecture property (fused Conv1D QKV concentrates the initial pull)
  that the spectrum-shape predictor was never actually about. Mechanism is correct-for-spread, the scalar miscounted.

## What I want Wilson to rule
1. Is the bundled-scalar resolution legit, or am I rescuing the miss post-hoc by switching observables (the same move
   the discipline is suspicious of)? The defense: the per-step wetted_count is the SAME corroborating-trajectory
   evidence that overrode the Yelp "free" and the Dye2 "same rate" — reading the resolved trajectory over the bundled
   scalar is the established pattern, not a new rescue. But Wilson should check it's not motivated.
2. Does the attention initial-seed-concentration being architecture-set (not spectrum-set) DENT "spectrum sets the
   profile", or just SCOPE it (spectrum sets SPREAD profile; architecture sets INITIAL concentration)? My read: scopes,
   doesn't dent — but it means the clean "two-scale->sharp seed, continuum->diffuse seed" rule should be restated as
   about SPREAD not initial-seed.
3. Overall verdict: is "same flow, different type" confirmed at the level that lets it be the headline (flow universal,
   transport-profile spectrum-set, with the initial-seed-concentration carved out as architecture-set)?

## Scope still open (not for this ruling, for the record)
- Two nets (Pythia + GPT-2), one task-B anchor (AG via SST-2). Task-axis dye (Yelp/Subj) not run — but static shape +
  generator already known overlap-dependent from earlier, so task-axis is more confirmatory than the net axis was.
- If Wilson confirms: the generalization is done at the net axis; next is either task-axis dye (confirmatory) or pivot
  to a BUILD that uses the characterization (Nate's stated priority: dope shit > novelty).

## User additions
