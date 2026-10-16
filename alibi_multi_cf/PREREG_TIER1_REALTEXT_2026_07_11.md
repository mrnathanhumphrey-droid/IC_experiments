# PRE-REG — Thread #2 Tier-1 (REAL TEXT, Lambda): does the ALiBi×multilingual CONJUNCTION flip the CF childhood-protection sign in from-scratch small LMs on REAL text? 2026-07-11.

SHA256-lock before firing. Tier-0 (synthetic, LOCKS 140d742c + 690a5b42) was NULL ×2 — copy tasks saturate (no partial-forgetting
regime). This Tier-1 uses REAL text + a REAL downstream interference, where the medium local check CONFIRMED a non-saturating regime
exists (base SST accA0=0.654, retA_noc=0.618 in-band, protection +0.056 measurable). It is the faithful test of the BLOOM conjecture.

## Compute / cost (GPU spend — user-authorized "fire the big real-text Lambda")
Lambda, ML-CF lane, a100-first (a10 EXCLUDED). One instance runs the FULL 2×2 (8 from-scratch pretrains). Estimated ~1.5–2.5h →
**~$3–8 at a100 $1.99/hr** (medium check calibrated the per-pretrain cost). 6h watchdog + always-terminate finally-block cap runaway.

## Design (only PE and pretrain-linguality vary; shared tokenizer)
- **2×2 = {ALiBi, RoPE} × {mono (en), multi (en+de+es+ru+hi+ar)}.** Tiny GPT (d=384, L=6, H=6, vocab=16k BPE), trained from scratch.
- **Shared BPE tokenizer** trained on the multilingual mix → identical tokenization across all cells; ONLY pretrain data + PE differ.
- **Same token budget** for mono and multi pretrain (PT_CHARS per bucket); multi splits it across 6 languages/scripts (Latin/Cyrillic/
  Devanagari/Arabic — BLOOM-like diversity).
- **Protocol (faithful to probe13's protection sign):** pretrain base → fine-tune task-A (SST-2 sentiment) → measure accA0 →
  two branches: NO-CHILDHOOD (→ task-B AG-News interference → re-probe SST retention retA_noc) and CHILDHOOD (extra SST warmup →
  task-B → retA_ch). **PROTECTION = retA_ch − retA_noc.** NEG = childhood ERODES = BLOOM signature. 2 seeds/cell, mean±SE.
- **Non-saturation:** phaseB fixed at 200 (medium check: retA_noc=0.618 at 150, in the [0.55,0.80] band). retA_noc reported per cell →
  post-hoc verify it stayed in-band; if a cell saturates (retA_noc≈0.5) that cell is flagged.

## Pre-registered read (locked before firing)
- **CONJUNCTION REPRODUCES:** PROTECTION NEGATIVE for ALiBi×multi AND ≥ it by > 2 pooled-SE for the other three (ALiBi×mono,
  RoPE×mono, RoPE×multi). Neither knob alone erodes; the conjunction does — the BLOOM signature in a controlled from-scratch grid.
  → confirms the ALiBi×multilingual mechanism is real and scale-reachable.
- **SINGLE-KNOB:** both ALiBi cells erode (PE alone) OR both multi cells erode (breadth alone) → report as single-knob, revises the
  mechanistic claim.
- **NULL:** protection ≈ same sign across all cells / interaction within noise → even faithful real-text from-scratch small LMs do not
  reproduce it → the effect needs the SCALE / rich pretraining of the original ≥560M multilingual checkpoints; the base BLOOM finding
  stands as an observation without a small-scale causal reduction. Honest — a null here bounds where the effect lives.

## Caveats (honest)
Small from-scratch LMs (~25M) reach only modest SST competence (accA0~0.65 medium; full a bit higher) — the effect may need more
capacity/pretraining than one instance affords; a null is therefore ambiguous (scale vs mechanism), a clean conjunction-flip is strong
positive. Downstream = SST→AGNews (English), so the MULTILINGUAL knob acts through the BASE model only (as in BLOOM). 6 languages,
one architecture family, 2 seeds, one phaseB. Retention = linear re-probe of the interfered body. GPU non-determinism across the 2
seeds. This is the faithful controlled test the synthetic Tier-0 could not be.

## Ops
`alibi_tier1_lambda.py` (self-contained: corpus stream → shared BPE → 2×2 pretrain → SST/AGNews protocol → 2×2 protection JSON);
launcher `cf_cascade/cloud/lambda_launch_alibi_tier1.py` (ML-CF, a100-first no-a10, scp-hang fix, 6h watchdog, always-terminate).
Local: py_compile clean, MODE=smoke plumbing OK, MEDIUM check confirmed competence + non-saturation. Output `out/alibi_tier1.json`.

## Status
PREREG written; pipeline locally validated (competence + non-sat regime confirmed). Next: SHA256-lock → verify Lambda auth/ssh →
FIRE launcher (background) → monitor → pull 2×2 → read conjunction → verdict + bank.
