# PRE-REG — Probe-15 (Hunt-BLOOM confirm): does ALiBi cause erosion in an ENGLISH model too? 2026-07-09.

SHA256-lock before GPU. Probe-14 (XGLM, LOCK b30d6579) exonerated the multilingual corpus: XGLM (multilingual, **sinusoidal**)
PROTECTS (N_phase2 +0.020→+0.149, growing) — opposite of BLOOM (multilingual, **ALiBi**, eroder). So BLOOM's erosion points to
its **ALiBi positional scheme**. This probe tests that directly with an ALiBi model trained on an **English** corpus: if ALiBi
erodes even in English, ALiBi is the mechanism and corpus is fully cleared. (Supersedes the MPT-1b plan — mosaicml repos are 404
and the community mirror's 2023 MosaicGPT custom code is incompatible with current transformers: `all_tied_weights_keys` crash.
`tiiuae/falcon-rw-1b` gives the same ALiBi+English test, HF-native, zero custom code.)

## The discriminator
**falcon-rw-1b** (`FalconForCausalLM`, HF-native): **alibi=True**, trained on **RefinedWeb (English)**, 24 layers / 2048 hidden
(1.3B), standard MHA with **fused** `query_key_value` → the **same Megatron carrier scheme as BLOOM** (`self_attention.query_key_
value`, `mlp.dense_h_to_4h`, `mlp.dense_4h_to_h`; 72 carriers). Falcon shares **ALiBi** with BLOOM but differs on corpus (English
vs multilingual) and size (1.3B vs 560M). Since scale does NOT flip the protection sign (OPT-350M +0.075 and OPT-1.3B +0.090 both
protect), an eroding Falcon at 1.3B implicates ALiBi, not size.

## DV + x-axis (identical to Probe-13/14 for direct overlay on BLOOM & XGLM)
- **N_phase2** (protection of realized phase-2 interference dir @ magp), **N_random** (matched-mag control), **N_conc**.
- x-axis = childhood **dose (lr)**, DOSES = **3e-7, 1e-6, 2e-6, 3e-6, 7e-6** (BLOOM's grid). Record retA_drop per dose.
- **`frac_child_in_V`** (childhood displacement energy landing in V; chance ≈ k/cin). Compare to XGLM's 0.37–0.45.

## Design
1. **buildwa falcon_rw1b** on SST-2 → w_A anchor + reload-confirm.
2. **probe13 dose-response falcon_rw1b**, DOSES above, **N=25** paired seeds (early lesson: N=25 gives tight CIs; skip the
   early-stop dance), K_RAND=3, NoCH shared across doses. ANCHOR gate = buildwa anchor ± 0.02 (via ANCHOR env; hard-VOID if off).
   Incremental JSON dump per seed. `frac_child_in_V` dumped per dose. HF-native → no trust_remote_code, no attn_impl handling.

## Pre-registered read (locked before result)
Overlay Falcon N_phase2-vs-dose on BLOOM (ALiBi) and XGLM (sinusoidal):
- **H-ALiBi-CAUSES — Falcon ERODES** (N_phase2 slope < 0, crosses negative like BLOOM): ALiBi causes erosion **even in English**
  → **ALiBi is the mechanism**, corpus fully exonerated. The strongest possible confirmation. Predict frac_child_in_V low
  (fails to reorganize in V, like BLOOM).
- **H-ALiBi-INSUFFICIENT — Falcon PROTECTS** (slope > 0, like OPT/Pythia/XGLM): ALiBi **alone** does not cause erosion →
  BLOOM's erosion is an ALiBi×multilingual interaction, or its post-embed LayerNorm, or something else BLOOM-specific. Re-opens
  the question toward post-embed LN / BigScience specifics.

## VOID / guard conditions
- anchor off > 0.02 → hard VOID. dataset load fail → hard VOID (incremental dump guards partial).
- childhood destructive at gentlest dose (retA collapses at 3e-7) → VOID-soft, refire gentler grid (1e-7, 3e-7). Falcon-rw-1b
  is 1.3B; if 7e-6 top dose is over-hot, the curve's low end still carries the sign.

## Ops
Harnesses `build_wA_lambda.py` (+falcon_rw1b) and `probe13_doseresponse_lambda.py` (+falcon_rw1b, FALCON_C=BLOOM scheme,
ANCHOR-env gate, frac_child_in_V dump) — py_compile-clean; Falcon config/carriers/verbalizer unit-checked locally (alibi=True,
72 carriers, pos/neg distinct, HF-native meta-instantiate OK). Launcher: falcon_rw1b in CKPT_MAP; scp-hang fix live; a100-first
(us-east-1 $1.99), a10 excluded. 1.3B → a100 40GB mem-safe (fp32 + AdamW ~16GB). requests Bearer, ML-CF, self-terminating.

## Scope (fire plan)
buildwa falcon_rw1b (~$2–4) + probe13 falcon_rw1b N=25 (~$4). Parallel-OK with the BLOOM-frac re-run already in flight.

## Status
NOT built-new/fired (harness edits done + parsed + config-unit-checked; MPT smoke-test ruled out custom-code path). Next: LOCK →
fire buildwa → set ANCHOR from output → fire probe13 → read N_phase2-vs-dose (H-ALiBi-CAUSES vs -INSUFFICIENT) + frac_child_in_V.
