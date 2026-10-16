# SCOPE — Models × Sizes matrix for the CF dissociations. 2026-06-21. Parallel agent. PRE-FIRE (no cells fired yet).

## Objective (user-locked)
**De-confound family-vs-size** for the four banked cross-arch dissociations. Every prior result sits in ONE size band
(~350–410M): Pythia-410m, OPT-350m, GPT2-medium. Family and size are unidentified. This matrix adds the size axis to
identify each dissociation as family-driven, size-driven, or a single-point artifact. Scaling trajectories are the bonus.
Span: small-to-mid (all fp32 on one A100). De-confound-first sequencing.

## The four dissociations under test (anchors at ~350M)
| D | Pythia-410m | OPT-350m | probe script (parameterize MODEL) |
|---|---|---|---|
| **D1 antagonism** | −0.23 head-on | +0.09 orthogonal | rung2_fragile step1 `antag` (cheap, in the fragile run) |
| **D2 protection decomp** | non-destruct≈all, map inert (B≈D) | map 74% load-bearing (B−D +0.118) | dispositive (`opt_dispositive`-style: A/B/D, 30-seed) |
| **D3 SPIRAL depth-geometry** | eff-rank collapse, inert | MLP contraction, load-bearing, opposite sign | breadth_pass + crosspair (funnel) + taskavar_seed (twist/seed) |
| **D4 POKE occupy** | inert | fragile-specific sink −0.125 | poke_test (occupy + matched-regime) |
Backbone (seed/M1/aimed/fragile=task-A) rides along in rung-1/2 (cheap; expected to replicate).

## MESH with the other agent (do NOT cross)
- **Other agent = Layer 1 (G1–G6 geometry vocabulary + baseline script `b333znaa4`) + Layer 3 (RMD SRC: family vs size
  vs irreducible, hash-locked random-grouping null).** They produce, per cell, dispositive B−D + the G1–G6 vector from
  THEIR single script, and adjudicate the filled matrix.
- **THIS agent = Layer 2: feed the model matrix.** Deliverable = pinned **w_A checkpoints per (family,size)** that their
  script stamps onto. I do NOT build geometry metrics or RMD-SRC (their lane). RMD-SRC unit = the MODEL (or normalized
  carrier-role), since OPT(6 carriers) vs Pythia(3) aren't model×carrier-poolable across families.

## Size ladder (locked: small-to-mid; aligned to the other agent's matrix)
- **Pythia:** 160m · **410m (DONE)** · 1b
- **OPT:** 125m · **350m (DONE)** · 1.3b   ·   **GPT2 = 3rd family for RMD-SRC power (deferred cascade)**
- Matched-size bands for de-confound: **A ~125–160M** (Pythia-160m, OPT-125m) · **B ~350–410M (DONE anchors)** ·
  **C ~1–1.3B** (Pythia-1b, OPT-1.3b).

## w_A FEED (fired 2026-06-21, parallel) — Layer-2 deliverable
4 new cells training to **matched-acc ~0.80** (de-confound vs w_A-accuracy), saved as `gateA/<tag>_sst2_wA.pt` +
`<tag>_wa.json` (acc/stop_step/matched flag): `pythia160m`, `pythia1b`, `opt125m`, `opt1_3b`. Band-B anchors already on
disk: `pythia410m_sst2_subsetmin.pt` (0.77), `opt350m_sst2_wA.pt` (0.82). Scripts: `wa_trainer_lambda.py` (family-
agnostic, MODEL+TAG args) + `lambda_launch_wa.py` (a100 default, h100 fallback for big). The other agent's script stamps
these to emit B−D + G1–G6 per cell.

## Cells to fire (de-confound-first; 410m/350m reused as band B anchors)
NEW model-points (each = train w_A → regime-calibrate → D1–D4 + backbone):
1. Pythia-70m  2. Pythia-160m  3. Pythia-1.4b  4. OPT-125m  5. OPT-1.3b
→ 5 new cells. ~4–6 GPU-hr each (rung-1/2 ~0.5h + dispositive ~2h + SPIRAL gen ~0.8h + POKE ~1h) → ~20–30 GPU-hr total.
GPT2 cascade fill-in (T5) and large models (≥2.7B) DEFERRED per de-confound-first + small-mid.

## De-confound logic (what the design identifies, per dissociation)
- **Within-family scaling:** D(Pythia, {70,160,410,1400}m) and D(OPT, {125,350,1300}m) — is D size-dependent inside a family?
- **Cross-family at matched size (bands A/B/C):** is the Pythia−OPT gap CONSTANT across sizes (→ FAMILY) or does it
  close/open with size (→ SIZE-interaction / the 350M gap was a regime artifact)?
- **Sign-flip within a family across size → SIZE-driven** (family label was misattributed at the single 350M point).

## Pre-registered hypotheses (lock before firing — anti-fishing)
- **D1 antagonism:** family sets the sign (NeoX tied-embed vs OPT) → gap ~constant across bands. ALT: strengthens with size.
- **D2 protection decomp:** SIZE-plausible (more capacity → map load-bearing) → OPT-style map-load-bearing emerges with
  size in BOTH families. ALT: family-fixed.
- **D3 SPIRAL sign:** family (arch depth-organization) → opposite sign persists across sizes.
- **D4 POKE:** tracks D1 (antagonism) → wherever antagonism is orthogonal/positive, sink appears.
Either outcome real for each; the design is built to separate them, not confirm a side.

## Methodology guards (carry; T6 is GATING not optional)
- **Per-size regime calibration BEFORE each cell's probes** (the disambiguator lesson): match w_A task-A accuracy band
  + forget-severity precheck across sizes. Bigger models hit higher w_A acc / different forget regime → uncontrolled,
  size-differences are regime-confounds. Each cell logs w_A acc + A_forget; HARD-STOP if outside the matched band.
- Per-(layer,carrier) never pooled. Effect size vs the anchor (not bare t). Reproduce-known-quantity per fire
  (e.g., 410m/350m re-anchor). Swap control on any positive. Matched-N. STOP-on-missing → generate, don't proxy.
- **Matched-measurement-template** (the 4th catch): each family's depth-signature measured in its OWN signature
  (Pythia eff-rank, OPT MLP-contraction) — don't impose one family's template on another (the SPIRAL-OPT-1 v1 error).

## Fire order (on go)
Phase 1 — Pythia ladder D1+D2 (antagonism + protection) at 70m/160m/1.4b → the cleanest within-family size axis.
Phase 2 — OPT 125m/1.3b D1+D2 → completes bands A/C for the cross-family de-confound on antagonism+protection.
Phase 3 — D3 (SPIRAL) + D4 (POKE) across the same 5 cells.
Phase 4 — synthesis: 4 dissociation surfaces over (family × size); identify family vs size per dissociation.
Each phase: pre-reg lock → regime-calibrate → fire → reproduce-guard → bank on rows.

## PHASE 3 — controlled-SHAPE axis (aspect-ratio first; user-locked 2026-06-21)
Rationale: "family" is a confounded bundle (data+tokenizer+pretraining+arch). Size de-confounds size; the controlled-
shape axis isolates ONE arch knob so RMD-SRC can group by `deep-narrow` / `parallel-block` instead of uninterpretable
`family`. Chosen knob: **aspect ratio (deep-narrow vs shallow-wide at FIXED param count)** — targets the SPIRAL
depth-geometry dissociation (which is literally about depth).

### GATING BRIDGE cell (fires first; whole phase depends on it)
Every banked dissociation was measured on PRETRAINED models. From-scratch ≠ pretrained regime. So before any ablation:
build a from-scratch (random-init `from_config`) model matching a fully-characterized arch (OPT-350m config), train
task-A=SST to whatever acc it reaches (extended steps, matched-acc early-stop), and read the BACKBONE (com/M1/antag/
aimed) vs the pretrained OPT anchor (M1 0.86, antag +0.09, partial +0.29).
- **Reproduces backbone AND reaches the acc band → from-scratch regime is valid → fire the aspect-ratio sweep.**
- **Can't reach the acc band, or backbone absent → pretraining is load-bearing → aspect-ratio shapes need PRETRAINED
  variants (expensive) or a from-scratch-learnable synthetic task-A.** Honest fork; do NOT run the sweep blind.
Script `bridge_fromscratch_lambda.py` (staged; reuses rung2_fragile_opt helpers + `from_config`). Fires when GPT2
size cells land (sizing complete).

### Aspect-ratio sweep (gated on bridge pass)
Fixed param budget (~matched to a band, e.g. ~160M), vary depth/width: deep-narrow (many layers, small d_model) vs
shallow-wide (few layers, large d_model), ≥2 intermediate points. Each: train task-A matched-acc → SPIRAL (funnel/twist/
seed-load-bearing) + antagonism. RMD-SRC then groups by aspect-ratio. Same tripwires (matched-acc, per-carrier never
pooled, swap control, reproduce-anchor). New Layer-2 cells for the other agent's stamping.

## Deferred (explicit, not silently dropped)
GPT2 cascade (tracer-only today; needs rung-1/2/3+SPIRAL at ≥2 sizes); large models ≥2.7B (extend only if a
dissociation shows a live size-trend in small-mid); full Cartesian product (wasteful — much is confirmatory backbone).
