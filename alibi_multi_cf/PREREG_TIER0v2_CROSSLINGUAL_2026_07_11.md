# PRE-REG — Thread #2 Tier-0 v2 (redesign): cross-lingual interference + multilingual-BASE knob. 2026-07-11.

SHA256-lock before the run. Tier-0 v1 (copy@D distance-switch, LOCK 140d742c) was a NULL because copy@D forgets to chance instantly
(no dynamic range) and the multilingual proxy was weak. v2 fixes BOTH: (1) shared copy rule across LANGUAGES so interference is
cross-lingual and PARTIAL (a non-saturating regime — the #1/#5/#2 lesson), and (2) the "multilingual" knob = whether the BASE model
is multilingual, which is exactly BLOOM's condition. Still $0 local, still a GO/NO-GO gate for Lambda.

## HARD ISOLATION / cost
$0 LOCAL GPU. Pure synthetic (in-process). Tiny models (d128/H8/L3, confirmed competent for BOTH ALiBi & RoPE on copy@2 — 0.995).
Writes only ./out/. No downloads, no Lambda.

## Design (only PE and base-multilinguality vary)
- Task = **copy@2** (target[i]=input[i−2]); both PEs learn it to competence. Tokens carry a LANGUAGE = one of K=6 disjoint vocab
  blocks (BLOCK=32, VOCAB=192). A sequence is monolingual (all tokens from one block); the LANGUAGE is the variable.
- **task-A (protected)** = copy@2 in language-0.  **task-B (interference)** = copy@2 in language-5 (UNSEEN by either base).
- **BASE (the 2x2 second knob):** MONO-base = trained on language-0 only. MULTI-base = trained on languages {0,1,2,3} mixed. Both
  must reach comparable lang-0 base competence (checked + reported; if multi-base lang-0 acc lags mono by >0.05, raise multi steps).
- **PE (first knob):** ALiBi vs RoPE. Model width/depth/heads/lr/steps/seeds IDENTICAL across all 4 cells.
- **Protocol:** base → two branches: no-childhood (→ phase-B lang-5 → retention lang-0) and childhood (→ extra lang-0 warmup → phase-B
  lang-5 → retention lang-0). **PROTECTION = retA_ch − retA_noc.** Positive = childhood protects lang-0 vs the new language; NEGATIVE
  = childhood ERODES (BLOOM signature). N seeds, mean±SE.
- **Non-saturation guard (locked):** phase-B (lang-5) steps tuned so retA_noc (lang-0 retention, no childhood) lands in [0.30,0.75]
  — a PARTIAL-forgetting regime with dynamic range. Quick phase-B-step sweep on the multi cell picks the operating point BEFORE the
  2x2; the chosen step count is logged. If NO phase-B step count yields retA_noc ∈ [0.30,0.75] (cross-lingual interference is all-or-
  nothing), that itself is reported as a design-limit null.

## Pre-registered read (locked before the run)
- **CONJUNCTION REPRODUCES (fire Lambda):** PROTECTION NEGATIVE for ALiBi×multi-base AND ≥ it by > 2 pooled-SE for the other three
  (ALiBi×mono, RoPE×mono, RoPE×multi). Neither ALiBi alone nor multilingual-base alone erodes; the CONJUNCTION does. → strong signal.
- **SINGLE-KNOB:** both ALiBi cells erode (PE alone) OR both multi-base cells erode (breadth alone) → report as single-knob, not
  conjunction.
- **NULL:** protection ≈ 0 / same sign across all cells / interaction within noise → tiny synthetic still misses it even with a
  non-saturating cross-lingual task → the effect needs real-LM scale; Lambda becomes a deliberate scale-test, not a confirmation.

## Caveats (honest)
Disjoint-vocab "languages" still lack real cross-lingual subword sharing/script structure (a real-text Tier-1 is the faithful
upgrade). copy@2 is a stylized task (chosen because both PEs learn it, isolating PE from difficulty). Tiny from-scratch models. Few
seeds. A clean conjunction-flip = strong positive; a null remains ambiguous (setup vs scale). This is a GO/NO-GO gate, not a
standalone mechanistic claim.

## Ops
`probe_tier0v2_crosslingual.py` (reuse the v1 TinyLM/ALiBi/RoPE + train/eval; add language-blocked data + multilingual-base +
phase-B-step non-saturation tuner). py_compile + --smoke before full. $0 local GPU, minutes. Output `out/tier0v2_crosslingual.json`.

## Status
PREREG written; d128/H8 competence confirmed. Next: SHA256-lock → build → tune phase-B to retA_noc∈[0.30,0.75] → full 2x2×seeds →
read conjunction interaction → GO/NO-GO on Lambda.
