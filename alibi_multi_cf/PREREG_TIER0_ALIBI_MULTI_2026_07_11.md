# PRE-REG — Thread #2 Tier-0 feasibility gate: does the ALiBi×multilingual CONJUNCTION flip the CF retention-sign in a controlled from-scratch 2×2? 2026-07-11.

SHA256-lock before the run. The arc's novel finding: BLOOM (ALiBi × multilingual) is the LONE ERODER — extra childhood warmup on
task-A makes task-A LESS robust to later interference (sign flips negative), while mono/RoPE models PROTECT (sign positive). We never
proved WHICH knob causes it. This $0 local gate trains a controlled 2×2 — {ALiBi, RoPE} × {monolingual, multilingual} — tiny
transformers FROM SCRATCH (only those 2 knobs vary) and measures the retention-sign per cell. Decides whether a Lambda scale-up
(Tier-1/2, real multilingual text, larger models) is worth firing.

## HARD ISOLATION / cost
$0 LOCAL GPU. Pure synthetic data (generated in-process), tiny models (~1-2M params). Writes only ./out/. No downloads, no Lambda.

## The retention-sign metric (faithful to probe13's PROTECTION quantity, minimized)
For each cell: (0) train a base model on task-A to competence. Then two branches from the SAME base checkpoint:
  - **no-childhood:** immediately train task-B; measure task-A retention retA_noc.
  - **childhood:** extra warmup steps on task-A, THEN train task-B; measure task-A retention retA_ch.
**PROTECTION = retA_ch − retA_noc.** POSITIVE = childhood protects A against B-interference (OPT/RoPE/mono signature);
NEGATIVE = childhood ERODES (the BLOOM signature). This is probe13's sign (childhood-effect on retention) without the full
Fisher-subspace apparatus — the SIGN is measured directly. Multiple seeds; report mean±SE.

## The 2×2 (only PE and vocab-breadth vary; everything else identical)
- **Position encoding:** ALiBi (per-head linear-distance bias added to attention scores) vs RoPE (rotary q/k rotation). Standard impls.
- **"Multilingual" (stylized proxy):** MONO = tokens from ONE vocab block; MULTI = tokens from K=4 DISJOINT vocab blocks ("languages"),
  each sequence sampled from all blocks. A minimal proxy for "input spans multiple sub-languages" (flagged; real multilingual text is
  the Tier-1 upgrade). Model width/depth/heads/steps/lr/seeds IDENTICAL across all 4 cells.
- Task = position-sensitive COPY/induction (target = token D positions back), so PE genuinely matters. task-A = copy@D_A, task-B =
  copy@D_B (D_A≠D_B) → they interfere (different relative-position rule). Retention = task-A next-token accuracy.

## Pre-registered read (locked before the run)
- **CONJUNCTION REPRODUCES (fire Lambda):** PROTECTION is NEGATIVE for the ALiBi×MULTI cell AND positive (or ≥ ALiBi×multi by a clear
  margin > 2 pooled SE) for the other three (ALiBi×mono, RoPE×mono, RoPE×multi). I.e. neither ALiBi alone nor multi alone erodes; the
  CONJUNCTION does — the BLOOM signature in miniature. → strong signal, scale up to real multilingual text on Lambda.
- **SINGLE-KNOB (not a conjunction):** if ALiBi cells both erode (regardless of vocab) → it's PE alone; if multi cells both erode →
  it's breadth alone. Reported as such — changes the mechanistic claim (still informative, still may warrant Lambda but with a
  different hypothesis).
- **NULL / doesn't reproduce at tiny scale:** all 4 cells same sign (all protect or all erode), or the 2×2 interaction is within noise.
  → the effect needs scale/real-multilinguality OR this stylized setup misses the mechanism. Honest ambiguous null: report, and Lambda
  becomes a judgment call (does the phenomenon NEED scale) rather than a confirmation run. A null here SAVES the spend.

## Caveats (honest — this is a stylized analog, not a replica)
Disjoint-vocab-blocks is a crude proxy for multilinguality (no shared subwords / script structure). Copy@D is a stylized PE-sensitive
task, not language modeling. Tiny from-scratch models lack the rich pretraining where the original effect lives → a NULL is genuinely
ambiguous (absence of evidence), whereas a clean conjunction-flip is strong POSITIVE evidence the mechanism is scale-invariant.
Single architecture family, synthetic data, few seeds. This gate is a GO/NO-GO for Lambda, not a standalone mechanistic claim.

## Ops
`probe_tier0_alibi_multi.py`: tiny decoder transformer with switchable ALiBi/RoPE, synthetic copy@D data with mono/multi vocab, the
base→{childhood,no-childhood}→phase-B→retention protocol, 2×2 × N seeds. py_compile + --smoke (1 seed, tiny) before full. $0 local
GPU, minutes. Output `out/tier0_alibi_multi.json` (PROTECTION mean±SE per cell + the 2×2 interaction).

## Status
PREREG written; metric confirmed faithful to probe13's protection sign. Next: SHA256-lock → build → --smoke → full 2×2×seeds →
read the conjunction interaction → GO/NO-GO on Lambda Tier-1.
