# HANDOFF — size×family matrix + depth-surgery axis (pre-compact). 2026-06-22. Parallel agent.

## STANDING RULES (locked this session — carry verbatim)
- **I RUN WHAT WE HAVE. I never tell the user something is "the other agent's", never defer/pass work, never
  self-coordinate lanes. The USER coordinates who does what; if I coordinate I miss things.** (Hard correction 2026-06-22.)
- **I fire ALL GPU work I need to Lambda myself** (launchers in `cloud/`, key=ML-CF, `D:/physics_detector/.env` +
  ssh `ml_cf_ed25519`). Hard finally-terminate. H100 available for big models ("big boys"); a100 fits ≤~1.5B fp32.
- **NEVER pipe a background launcher through `| head`/`| tail`.** When head exits, the launcher gets SIGPIPE and dies
  BEFORE its finally-terminate → orphaned billing instances. Fire with run_in_background and NO pipe. (Self-caught
  2026-06-22: 3 taskavar launchers SIGPIPE-orphaned f00662/94f255/5e9ec; terminated by exact printed instance_id.)
- **Orphan cleanup = match by the instance_id each launcher PRINTS, never by "newest/idle".** Other lanes run their own
  fleet on the same key (their logs = `pl.log`; mine = `tav_*/cpr_*/wa_*.log`). Probe `ps`+`ls ~/*.log` to classify;
  terminate ONLY ids my own launcher output recorded. (d79fdc looked like mine by uptime but was another lane's.)
- **Six tripwires bind every substantive action:** T1 constraint-diff (no silent narrowing) · T2 STOP-on-missing →
  GENERATE it myself, never proxy · T3 verdict←two rows + reproduce-anchor · T4 reconcile-before-code + flag my own
  errors · T5 grep before asking / inspect schema before coding · T6 lede-first. Name the gating tripwire before acting.
- Matched-acc feed (early-stop ~0.80) so size/depth ≠ w_A-accuracy confound. Per-(layer,carrier) never pooled.
  Curvature-controlled = clean read. Swap control on any positive. Reproduce a known quantity before banking.

## WHAT'S DONE + BANKED
- **SPIRAL-OPT v1+v2** (`VERDICT_SPIRAL_OPT_2026_06_21.md`, `VERDICT_SPIRAL_OPT_V2_2026_06_21.md`): OPT funnel does NOT
  replicate Pythia's (eff-rank collapse absent); OPT has its OWN signature (MLP fc1/fc2 resolution+universality
  contraction, ρ−0.84). OPT-3-v2 = depth-structure LOAD-BEARING (seed concentrates deep, swap survived). OPT-4 twist
  mixed (fc1 spirals, fc2 contract-only). OPT-CARRIER null. Pre-regs locked (eadc8832, 59ebf808).
- **9-CELL SIZE MATRIX (Layer-2 feed) COMPLETE** — pinned w_A, matched band [0.77–0.88], `gateA/<tag>_sst2_wA.pt`:
  Pythia {160m 0.79, 410m 0.77(=`pythia410m_sst2_subsetmin.pt`), 1b 0.82} · OPT {125m 0.83, 350m 0.82(=`opt350m_sst2_wA.pt`),
  1.3b 0.80} · GPT2 {small 0.84, med 0.87, xl 0.88}. **GPT2 family runs HOT (0.84–0.88) = acc covariate to carry.**
  Scripts: `cloud/wa_trainer_lambda.py` + `cloud/lambda_launch_wa.py` (MODEL TAG [GPU]).
- **From-scratch BRIDGE: FAILED** — random-init OPT can't learn SST (acc 0.49=chance), backbone absent → **pretraining
  is load-bearing** → from-scratch arch-ablation dead. (`cloud/bridge_fromscratch_lambda.py`, `surgery_bridge_opt.json`.)
- **SURGERY-BRIDGE: PASSED** — half-depth (24→12) PRETRAINED OPT re-stabilizes (SST 0.777) + keeps backbone (M1 0.71,
  aimed partial 0.49). Layer-surgery is the valid depth-in-regime path. **Preview: depth shifts backbone** (aimed
  partial 0.29→0.49, M1 0.86→0.71, antag +0.09→0). (`cloud/surgery_bridge_opt_lambda.py`.)
- **DEPTH AXIS w_A COMPLETE** — OPT-350m surgered (evenly-spaced layers + reindex layer_idx) + matched-acc retrain:
  `gateA/opt350m_d{6,12,18}_sst2_wA.pt` (acc 0.773/0.820/0.800) + d24 = full anchor. Scripts:
  `cloud/surgery_wa_lambda.py` (MODEL KEEP TAG) + `cloud/lambda_launch_surgery_wa.py` (MODEL KEEP TAG [GPU]).

## IN FLIGHT (fired pre-compact — will land during/after)
- **Depth-variant CROSSPAIR**: `b6qpn1jvr` (d6), `b0dgh9430` (d12), `b9ifj221v` (d18). Each surgers→loads ckpt→
  extracts crosspair → `gateA/crosspair_residual_optd{6,12,18}.npz/.json`. Scripts: `cloud/crosspair_depth_lambda.py`
  (MODEL KEEP CKPT TAG; has `_surger` + tagged output) + `cloud/lambda_launch_depth_cpr.py` (MODEL KEEP CKPTNAME TAG [GPU]).

## NEXT STEPS (on landing / resume)
1. **Funnel-vs-depth read (CPU, local):** for d6/d12/d18 + full-24 (`crosspair_residual_opt.npz`), compute the opt1
   crosspair3 `a_curvres`/`c` contraction per carrier (fc1/fc2 esp.) vs depth. Does the MLP funnel (ρ−0.84 @ full)
   strengthen/weaken/hold with depth. opt1 logic in `opt1.py` (crosspair3 substrate); write a crosspair-only depth
   reader (no breadth needed for the funnel signal). Reproduce-anchor: full-24 must match the banked spiral_opt1 fc1 −0.84.
2. **Extend to full SPIRAL per depth:** `cloud/breadth_depth_lambda.py` ALREADY adapted (MODEL KEEP CKPT TAG + `_surger`)
   — needs a launcher (mirror `lambda_launch_depth_cpr.py`, JSONS = breadth's task set, pull `breadth_pass_<TAG>`).
   Then `taskavar_seed` per depth (adapt `cloud/taskavar_seed_opt_lambda.py` with `_surger`+CKPT) for opt3/opt4 (twist +
   seed-load-bearing). Run opt1/opt3/opt4 across depths → depth×SPIRAL-dissociation surface.
3. **PHASE-3 aspect-ratio goal** (user-locked): isolate depth from the confounded "family" axis; the surgery depth axis
   IS that (depth-in-regime). Aspect-ratio-at-fixed-params would need pretrained width-variants (deferred).

## SCOPE DOC
`SCOPE_SIZE_MATRIX_2026_06_21.md` — full matrix design, mesh notes, Phase-3 plan (now realized via layer-surgery since
from-scratch failed). Cross-program matrix context: size de-confounds size; surgery depth-axis isolates depth.
