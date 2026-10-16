# VERDICT — scale de-confound SETTLED: collision, not exodus. 2026-06-22.

## Question
The ML-agent's scale reframe: as models scale, the childhood deposit's raw out-of-V% climbs to 82-86% — "the map's
substrate drains out of the curvature basis V; the signal left the basis." Their own flagged confound: V is a fixed
K=128-dim slice and the ambient carrier dim grows (768→2048), so V covers 17%→6% by construction — a fixed-rank basis in
a growing space captures less mechanically. Does the deposit REALLY drain, or is the rise the fixed-K/growing-d artifact?

## Instrument (clean settle)
`cloud/geom_basis_lambda.py` (6 cells, A100) measures the deposit's in-basis energy fraction per carrier under:
- fixed top-128 (the original V), null = 128/in_dim
- ENERGY-90% basis (K_e90 dirs for 90% Fisher mass — SCALE-INVARIANT, removes the fixed-K confound), null = K_e90/in_dim
- exact Rrand random null (empirical) — validates the analytic K/in_dim null
`settle_basis.py` → `gateA/settle_basis.json`. Decisive = EXCESS_e90 = median_carrier(fe90 − K_e90/in_dim) vs d_model.

## Result
rand−null ≈ 0 at EVERY cell (−0.003..−0.000) → analytic K/in_dim null is exact; methodology sound.

| cell | d_model | Ke90 | EXCESS_e90 | excess(top128) |
|---|---|---|---|---|
| opt125m | 768 | 22 | +0.057 | +0.125 |
| opt350m | 1024 | 14 | +0.070 | +0.160 |
| opt1_3b | 2048 | 26 | +0.060 | +0.119 |
| pythia160m | 768 | 3 | +0.095 | +0.336 |
| pythia410m | 1024 | 1* | +0.040* | +0.204 |
| pythia1b | 2048 | 26 | +0.097 | +0.149 |

- OPT EXCESS_e90 vs scale: +0.057 → +0.070 → +0.060 (net +0.002, FLAT).
- Pythia EXCESS_e90 vs scale: +0.095 → +0.040* → +0.097 (net +0.002, FLAT at endpoints).

## VERDICT — COLLISION, NOT EXODUS
Under the scale-invariant basis + exact null, the deposit's curvature-enrichment is STABLE across scale for BOTH families
— it does NOT drain. The raw out-of-V% rise to 82-86% (and the apparent decline in fixed-top-128 excess, Pythia
0.336→0.149) is the fixed-K/growing-d mechanical artifact; it vanishes under the energy-90% basis. The deposit stays
curvature-enriched at every scale. Therefore the map's scale-inertness (B-D→0 at 1B) is NOT the deposit leaving the
measurement basis (exodus) — it is COLLISION: the deposit stays put while deposit↔surge overlap rises (G5 0.344→0.410→
0.439 across OPT scale) and B-D falls. This is what DSI (the per-step collision instrument, in flight) measures directly.

## Caveats
- pythia410m Ke90=1 (degenerate single-direction energy basis — that cell's Fisher spectrum is extremely steep);
  its +0.040 is the 1-dim quirk. The 768 (+0.095) and 2048 (+0.097) endpoints bracket it and both hold → Pythia trend
  robust at endpoints.
- This settles "the DEPOSIT drains out of V." It does NOT touch the ML-agent's separate, valid point that D−A
  (non-destructive channel) and A_forget are genuinely out-of-V by construction — those remain worth the task-A-agnostic
  basis build (the fly-around / out-of-V lane).
