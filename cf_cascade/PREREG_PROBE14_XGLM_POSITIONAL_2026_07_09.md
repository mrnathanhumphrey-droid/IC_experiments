# PRE-REG — Probe-14 (Hunt-BLOOM): is BLOOM's inverted dose-response a CORPUS property or a POSITIONAL/arch property? 2026-07-09.

SHA256-lock before GPU. Follow-up to the Probe-11/12/13 reckoning: BLOOM-560M is the lone **eroder** — its childhood→protection
dose-response has NEGATIVE slope (N_phase2 +0.044→−0.022, crosses 0 at ~1.5e-6), inverse to the three protectors
(OPT-350M +0.075, Pythia-410M +0.044, OPT-1.3B +0.090). The banked-rows readout (READOUT_bloom_fingerprint_2026_07_09.md)
sharpened this: protectors run N_random slightly NEGATIVE (pay a tiny generic cost to buy CONCENTRATED protection on the
interference dir); BLOOM runs N_random slightly POSITIVE while its interference-dir protection goes negative — an inverted,
anti-concentration signature. BLOOM confounds three axes vs OPT/Pythia at once: **ALiBi** positional, **multilingual** corpus,
and post-embed LayerNorm. One negative exemplar cannot separate them. This probe breaks the confound with one well-chosen cell.

## The discriminator
**XGLM-564M** (`XGLMForCausalLM`, HF-native): 24 layers / 1024 hidden / 16 heads — **shape-matched to BLOOM-560M**
(24L/1024h/16h), **also multilingual** (CC100-XL, 30 langs), but **sinusoidal** positional (not ALiBi), OPT-named submodules
(`q/k/v/out_proj`, `fc1/fc2`) so the OPT carrier list drops in. Because XGLM matches BLOOM on size AND multilinguality and
differs mainly on the **positional scheme**, XGLM-vs-BLOOM isolates **ALiBi vs sinusoidal at matched size + matched corpus-type.**

## DV + x-axis (identical to Probe-13 so XGLM overlays directly on BLOOM)
- **N_phase2** = childhood's reduction of task-A damage along the realized phase-2 (subj) interference direction @ magp.
  **N_random** = same for a top-16 random dir (matched-mag control). Paired NoCH vs CH. **N_conc = N_phase2 − N_random.**
- **x-axis = childhood dose (lr)**, same DOSES as BLOOM (below). Also record retA_drop per dose (note: on the DOSE axis BLOOM
  was clean-monotone; retA_drop was the noisy axis — so we read on dose).
- **MECHANISM (new, read-only): `frac_child_in_V`** = energy-weighted fraction of the childhood weight displacement (ch − w_A)
  that lands inside the task-A curvature subspace V. Per-carrier chance = k/cin (0.125 for 1024-dim carriers, 0.031 for fc2);
  since XGLM and BLOOM are shape-matched, their chance baselines coincide → the XGLM-vs-BLOOM frac contrast is clean.

## Design
1. **buildwa XGLM-564M** on SST-2 (same recipe as BLOOM/OPT-1.3B) → w_A anchor + reload-confirm. Records anchor_acc.
2. **probe13 dose-response XGLM**, DOSES = **3e-7, 1e-6, 2e-6, 3e-6, 7e-6** (BLOOM's grid, for direct overlay), N=50 paired
   seeds/dose, K_RAND=3, NoCH shared across doses. ANCHOR gate = buildwa anchor_acc ± 0.02 (passed via ANCHOR env; hard-VOID if off).
   Incremental JSON dump per seed. `frac_child_in_V` dumped per dose.

## Pre-registered read (locked before the result)
Overlay XGLM's N_phase2-vs-dose on BLOOM's:
- **H1 — XGLM PROTECTS** (N_phase2 slope > 0, N_phase2 > 0 at strong dose, N_conc grows, like OPT/Pythia): the multilingual
  corpus is NOT sufficient to cause erosion → BLOOM's erosion points to its **ALiBi positional scheme (or post-embed LN)**.
  Next fire = MPT-1b (ALiBi + English) as the confirmatory ALiBi test.
- **H2 — XGLM ERODES** (N_phase2 slope < 0, crosses negative like BLOOM): the inverted response is a **multilingual/corpus-class**
  property, NOT a one-off — it generalizes across two architectures that share only the multilingual regime. Major finding.
- **Mechanism prediction (registered):** whichever way N_phase2 goes, `frac_child_in_V` should track it — a protector's childhood
  concentrates in V (frac above chance and above BLOOM's); an eroder's does not. If XGLM protects but its frac ≈ BLOOM's, the
  mechanism is directional (in-V but anti-aligned), not energetic — flagged for a follow-up cosine-with-dVp dump.

## VOID / guard conditions
- **anchor off > 0.02** (buildwa didn't reach a solid SST-2 deposit) → hard VOID.
- **dataset load fail** → hard VOID (incremental dump guards partial).
- **childhood destructive at the gentlest dose** (retA collapses far below anchor at 3e-7, as happened to BLOOM/OPT-1.3B in v1):
  VOID-soft → refire a gentler grid (1e-7, 3e-7 anchored lower). Do NOT interpret a destructive-baseline curve as erosion.

## Scope (fire plan)
- **This fire (confound break):** XGLM buildwa (~$2) + XGLM probe13 (~$2–4). Single a100 each (a10 excluded). Answers H1 vs H2.
- **Optional mechanism contrast (defer, gated on above):** re-run BLOOM probe13 with the new `frac_child_in_V` dump (BLOOM ckpt
  exists; ~$2) to put XGLM's frac beside BLOOM's. OPT-350M/Pythia frac later if the mechanism looks live.

## Ops
Harnesses `cloud/build_wA_lambda.py` (+xglm564m cell) and `cloud/probe13_doseresponse_lambda.py` (+xglm564m cell, ANCHOR-env
gate, read-only frac_child_in_V dump) — both py_compile-clean; frac_in_V projection math unit-checked (chance k/cin, in-V=1.0,
V orthonormal). Launcher `lambda_launch_generic.py`: xglm564m in CKPT_MAP, ANCHOR added to probe13 pass-through, **scp-hang fix
now in place** (keepalive + 3× retry + 6h run watchdog — closes the recurring bill-forever hang). requests Bearer, ML-CF key,
self-terminating in finally, GPU_PREF excludes a10.

## Status
NOT built-new/fired (harness edits done + parsed + unit-checked). Next: LOCK → CPU-dry-run buildwa+probe13 → fire buildwa →
set ANCHOR from its output → fire probe13 → read N_phase2 vs dose (H1/H2) + frac_child_in_V.
