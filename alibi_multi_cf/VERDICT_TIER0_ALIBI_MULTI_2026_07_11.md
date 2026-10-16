# VERDICT — Thread #2 Tier-0 gate (PREREG 140d742c, locked 2026-07-11). Landed, $0 local GPU. NULL / INCONCLUSIVE.

## OUTCOME: the childhood-protection phenomenon does NOT reproduce in a tiny from-scratch synthetic 2x2. No conjunction signal.

## What happened (with the confound found + fixed honestly)
1. **Task-competence confound (caught + fixed).** First config (d64/H4) let RoPE learn copy@2 (base 1.00) but NOT ALiBi (base 0.26)
   — copy@D favors RoPE, so protection was being measured on an UN-learned ALiBi model = noise. Capacity test → at **d128/H8/L3**
   both reach competence (ALiBi 0.995/0.969 on copy@2/@5, RoPE 1.000). Fair 2x2 restored before reading anything.
2. **At competence: PROTECTION ≈ 0 in ALL cells.** alibi_mono +0.029±0.029 (≈1σ, noise), alibi_multi −0.001, rope_mono +0.000,
   rope_multi +0.000. ALiBi×multi is NOT an erosion outlier. margin +0.001 vs 2·pooledSE 0.057 → NO conjunction.
3. **Non-saturation sweep (the #1/#5 lesson applied).** Retention after phase-B collapses to near-chance (retA_noc 0.03–0.24) at
   EVERY phase-B strength (30→250 steps) — copy@D interferes CATASTROPHICALLY, there is no partial-forgetting regime where a subtle
   protective SIGN could live. PROTECTION stays ~0 across the whole sweep; the only faint positives are in MONO cells (noise), the
   OPPOSITE of the BLOOM-erodes-in-multi story. No ALiBi×multi signature at any interference level.

## Why this is a NULL, not a finding against the conjunction (honest read)
Two non-exclusive reasons the tiny synthetic gate can't test the conjunction:
- **(a) No dynamic range.** copy@D forgets to chance almost immediately (1.0→~0.05), squeezing PROTECTION (childhood−nochildhood) to
  ~0 — the same saturation that muddied #5, now total. A subtle directional-protection sign has no room to appear.
- **(b) Wrong substrate for the mechanism.** The arc's childhood-protection effect was measured on RICHLY-PRETRAINED LLMs with real
  task-A knowledge and (for BLOOM) REAL multilinguality. A tiny model doing a synthetic copy rule with disjoint-vocab "languages"
  likely lacks the representational structure the effect lives in. Absence of evidence, not evidence of absence.

## GO / NO-GO on Lambda (the gate's purpose)
Tier-0 CANNOT cheaply confirm the ALiBi×multilingual conjunction → per the pre-reg, **Lambda is now a JUDGMENT CALL, not a
confirmation run.** Critically, a faithful Tier-1 is NOT "same synthetic, bigger" (this null suggests the synthetic task is the
problem, and scale won't fix a saturating copy task) — it would be **real small LMs trained from scratch on real MONO vs real
MULTILINGUAL text, ALiBi vs RoPE**, then the childhood→interference protocol. That is a heavier, more faithful ~$40–80 Lambda run
with an uncertain outcome. RECOMMENDATION: do NOT auto-fire; either park #2's causal-mechanism (the BLOOM conjunction FINDING +
Probe-16's "directional not flatness" already stand, banked, as the novel result) or commit to the real-text Tier-1 deliberately.

## What Tier-0 DID buy ($0)
- A clean demonstration that the childhood-protection sign is NOT a generic property of PE×breadth in small models — it needs either
  real dynamic range or real-LM structure. That BOUNDS the phenomenon (it's not trivially reproducible), which is itself informative.
- The saturation lesson recurs a THIRD time (after #5, #1): the protective/interference SIGN is only measurable when forgetting is
  PARTIAL. Any faithful mechanism test — synthetic or real — must engineer a non-saturating interference regime first.

## Caveats
Stylized (copy@D, disjoint-vocab multilingual), tiny, one architecture family, 4 seeds. A NULL here is genuinely ambiguous
(setup-limited vs scale-limited). The confound-fix (d128/H8) and non-saturation sweep make the "no cheap signal" conclusion robust,
but do not adjudicate whether real-text scale would show the conjunction.

## ── v2 REDESIGN (cross-lingual interference + multilingual-BASE knob; PREREG 690a5b42) — ALSO NULL ──
Fixed v1's two weaknesses: shared copy@2 rule across LANGUAGES (interference = learning a NEW language, closer to real CF) and the
"multilingual" knob = whether the BASE model is multilingual (BLOOM's actual condition). Full run (multi-base 1600 steps):
- **Non-saturation guard FAILED AGAIN:** the phase-B tuner could NOT land retA_noc in [0.30,0.75] at ANY interference strength
  (20→170 steps all gave retA_noc 0.03–0.07). Cross-lingual copy forgetting is ALSO all-or-nothing — lang-0 collapses to near-chance
  as soon as the copy circuit retrains for lang-5. No partial-forgetting regime exists for this task family either.
- **2x2 PROTECTION (all positive, huge noise):** alibi_mono +0.088±0.030, alibi_multi +0.151±0.160, rope_mono +0.070±0.179,
  rope_multi +0.345±0.155. margin −0.082, 2·pooledSE 0.479 → NO conjunction. If anything rope×multi protects MOST (opposite of the
  BLOOM-erodes-in-multi hypothesis). Measured on floored retention → dominated by noise.
- **DEFINITIVE Tier-0 conclusion:** TWO independent synthetic designs (distance-switch, cross-lingual) BOTH lack a non-saturating
  regime → neither can cleanly measure the protective SIGN. The tiny synthetic route is EXHAUSTED. The childhood-protection
  phenomenon does not reproduce testably in tiny from-scratch models — strong indication it requires real-LM structure (rich
  pretraining / real language). A faithful test needs the real-text Lambda Tier-1; the cheap gate cannot decide it.
- **FINAL GO/NO-GO:** cheap route done, both nulls. Genuine park-or-spend fork: accept the banked BLOOM finding (conjunction eroder +
  Probe-16 directional-not-flatness) as the standing result, OR commit deliberately to the ~$40–80 real-text Lambda run. Do NOT
  auto-fire. Saturation lesson holds a 3rd+4th time: the interference SIGN is only measurable when forgetting is PARTIAL.
