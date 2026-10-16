# PRE-REG — Probe-17 (conjunction-vs-LN fork): is BLOOM's erosion caused by its post-embed LayerNorm, or by the ALiBi×multilingual conjunction? 2026-07-09.

SHA256-lock before GPU. Probe-16 showed BLOOM's erosion is NOT the known flatness story (C1 survives). But our own verdict hedges
the CAUSE: "ALiBi×multilingual INTERACTION **OR** BLOOM's post-embed LayerNorm" (BLOOM uniquely applies a LayerNorm right after
the token embeddings; OPT/Pythia/Falcon/XGLM do not). This probe resolves that fork by ablating the post-embed LN and re-running
the dose-response. This is the single biggest remaining caveat (n=1 eroder → is it even the conjunction?).

## Design
- Build **bloom560m_noln**: BLOOM-560M with `transformer.word_embeddings_layernorm` → `nn.Identity()`, fine-tuned on SST-2 to a
  w_A anchor (buildwa). BLOOM uses pre-LN blocks (each block has its own `input_layernorm`) so block-0 re-normalizes → the removal
  is cushioned; fine-tune should still reach a solid anchor. (Smoke-verified: ablate → forward OK → strict reload OK; state_dict
  correctly drops the LN params.)
- **probe13 dose-response** on bloom560m_noln, DOSES = **3e-7, 1e-6, 2e-6, 3e-6, 7e-6** (BLOOM's grid, for direct overlay on the
  BLOOM eroder curve). N=25. ANCHOR gate = buildwa anchor ± 0.02 (via ANCHOR env). Same BLOOM Megatron carriers (72).

## Pre-registered read (locked before result) — overlay bloom560m_noln on BLOOM (+0.044→−0.032, eroder)
- **LN-IS-CAUSE:** bloom_noln STOPS eroding — N_phase2 slope ≥ 0, does NOT cross negative (becomes a protector like the others).
  ⇒ the post-embed LN was necessary for erosion ⇒ C1's cause is **BLOOM's post-embed LayerNorm**, not the ALiBi×multilingual
  conjunction. (Still novel — nobody links post-embed LN to a warmup-erosion sign — but the claim is RE-FRAMED; retract the
  "conjunction" wording in favor of "post-embed-LN".)
- **LN-NOT-CAUSE:** bloom_noln STILL erodes — N_phase2 crosses negative like BLOOM. ⇒ the LN is not necessary ⇒ erosion is the
  ALiBi×multilingual property (or ALiBi itself) ⇒ STRENGTHENS C1 as stated and removes the LN alternative.
- **PARTIAL:** attenuated erosion (crosses zero later / shallower negative slope) ⇒ the LN contributes but is not the sole cause ⇒
  report both factors.

## VOID / confound guards
- **anchor** for bloom_noln is unknown a priori → gate = buildwa's measured anchor ± 0.02 (ANCHOR env). If buildwa anchor lands
  FAR below BLOOM's 0.83 (e.g. < 0.78), FLAG: the ablation degraded task-A learnability → the erosion comparison is confounded
  (a crippled model, not a clean LN removal); report as a caveat, do not over-interpret a null.
- childhood-destructive at gentlest dose → VOID-soft, refire gentler (as with BLOOM). dataset fail → VOID.
- CAVEAT (inherent): ablating a component a pretrained model was trained with + re-fine-tuning is a perturbation, not a
  from-scratch control. Block-0 input_layernorm cushions it (smoke-verified) but the result is "LN removed + adapted," not "LN
  never existed." Interpret as necessity, not clean sufficiency.

## Ops
Harnesses `build_wA_lambda.py` (+bloom560m_noln, ablation hook after load) and `probe13_doseresponse_lambda.py` (+bloom560m_noln,
ablation BEFORE load_state_dict so structure matches the noln ckpt) — py_compile-clean; ablation smoke-verified. Launcher:
bloom560m_noln in CKPT_MAP; scp-hang fix live; a100-first (us-east-1 $1.99), a10 excluded. Reuses BLOOM carriers/machinery.

## Scope (fire plan)
buildwa bloom560m_noln (~$2) + probe13 bloom560m_noln N=25 (~$4). 2 fires.

## Status
NOT fired (harness edits done + py_compile-clean + ablation smoke-verified: LN→Identity, forward OK, strict reload OK). Next:
LOCK → fire buildwa → set ANCHOR from output (flag if <0.78) → fire probe13 → overlay N_phase2-vs-dose on BLOOM (LN-IS/NOT/PARTIAL cause).
