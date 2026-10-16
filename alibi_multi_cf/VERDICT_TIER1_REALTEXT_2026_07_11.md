# VERDICT — Thread #2 Tier-1 (REAL TEXT, Lambda) (PREREG 21a63965, locked 2026-07-11). Landed, a100 us-east-1, ~$2.70 (+$1.60 crashed 1st run). NULL — no conjunction; the effect is SCALE-DEPENDENT.

## RESULT: the ALiBi×multilingual conjunction does NOT reproduce in controlled from-scratch small LMs on real text. The childhood-protection effect is ~0 in ALL cells. This is a VALID null (non-saturating, competent) → the BLOOM effect lives at scale, not in a ~25M grid.

## The full 2×2 (from-scratch d384/L6/16k BPE, 2 seeds, SST→AGNews protection protocol)
  cell          accA0   retA_noc   PROTECTION (childhood − nochild; NEG = BLOOM signature)
  alibi mono    0.707    0.707      +0.007 ± 0.002
  alibi multi   0.711    0.697      −0.005 ± 0.015
  rope  mono    0.702    0.700      −0.006 ± 0.021
  rope  multi   0.724    0.705      +0.002 ± 0.028
CONJUNCTION test: ALiBi×multi = −0.005; others_min = −0.006 (rope_mono, LOWER); margin −0.001 vs 2·SE 0.064 → FALSE.
All four protections are within noise of ZERO; ALiBi×multi is not a distinct eroder.

## Why this null is CREDIBLE (unlike Tier-0's saturation nulls)
- **The measurement was valid this time.** accA0 ≈ 0.70–0.72 (the from-scratch LMs reached real SST competence) and retA_noc ≈
  0.69–0.71 = genuine PARTIAL forgetting (the non-saturating regime the medium local check promised, [0.55,0.80] band). So the
  protection SIGN had full dynamic range to appear — and it didn't, in any cell.
- **Proper error bars.** With 2 varied seeds the crashed 1st-run's single-seed −0.020 collapsed to −0.005 ± 0.015 (≈ 0). The apparent
  signal was seed noise.
- **The whole phenomenon, not just the conjunction, is ~0.** Childhood warmup does not measurably protect task-A against interference
  in ANY cell at this scale. There is no effect to attribute to a knob.

## Interpretation (pre-registered NULL clause, met)
The BLOOM childhood-protection sign — and the ALiBi×multilingual conjunction that flips it — does NOT reduce to a controlled
from-scratch small-LM grid, even with real multilingual text (en+de+es+ru+hi+ar, 4 scripts) and a valid non-saturating measurement.
The effect appears to REQUIRE the scale / rich pretraining of the ≥560M multilingual checkpoints where it was originally observed.
The original finding stands as a real observation on real checkpoints (banked, novel; Probe-16 showed it is directional, not a
flatness artifact) — but it has NO small-scale causal reduction. **Thread #2 is CLOSED: the conjunction is scale-dependent; not
reproducible in miniature.** This BOUNDS the phenomenon (a real scientific outcome), rather than explaining its micro-mechanism.

## What would be needed to go further (not recommended now)
A faithful mechanistic test would require training grids at the scale where the effect exists (≥300M–560M, more tokens) with
controlled PE×multilinguality — a large, expensive campaign (many GPU-days), not a single instance. Given (a) the effect is already
banked as a real observation and (b) the Tier-1 null shows small controlled models don't carry it, the cost/value of a scale campaign
is poor unless the mechanism becomes independently important. Recommend PARK.

## Process notes (honest)
- 1st run crashed on the RoPE cells: `apply_rope` used a position cache built for full SEQ while LM-pretraining feeds SEQ−1 tokens
  (shape mismatch). Cause: local validation only ran an ALiBi cell (ONECELL=alibi_multi), never exercising the RoPE+pretrain path.
  Fixed (slice cos/sin to actual L), re-validated a RoPE cell locally, re-fired. Both runs self-terminated cleanly (no billing
  runaway — the finally-block + watchdog worked).
- Bug-fix was crash-only; the pre-registered design + read (2×2, protection sign, conjunction criterion) were unchanged, so the lock
  (21a63965) holds.
- Also fixed: the 2 "runs" per cell now use different seeds (1st run had identical seeds → fake ±0.000 SEs).

## Caveats
Small from-scratch LMs (~25M) — a null is bounded-by-scale evidence, not proof the mechanism is absent at all scales (it is present at
≥560M). 6 languages, one architecture family, 2 seeds. SST→AGNews English downstream (multilingual knob acts through the base only,
as in BLOOM). The valid non-saturating measurement is the key strength that makes this null informative rather than ambiguous.
