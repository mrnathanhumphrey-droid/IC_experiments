# STATUS RECONCILIATION — D:/IC_experiments, 2026-07-14

**READ THIS BEFORE TRUSTING ANY "open / NOT fired / deferred / NEXT" LINE IN THIS TREE.**

Produced by a staleness audit (3 parallel read-only agents over 8 arcs) after a near-miss: a thread was proposed for re-run because
`cf_cascade/TODO_diffusion_program.md` said "(open)", but `gateA/pythia_dispositive.json` had already answered it three weeks earlier
(~$5 + 90 min of GPU saved by checking the disk instead of the prose).

## The pathology (systematic, not incidental)

**Status prose in this tree is written BEFORE the run and never revisited.** Artifacts routinely land minutes-to-hours after the line
declaring the thread open.

- **PREREG `Status: ... NOT fired` footers are a staleness GENERATOR — 10/10 July preregs in cf_cascade are wrong.** The footer is
  boilerplate from design time; nothing flips it on fire.
- **VERDICT_*.md files are RELIABLE.** PREREG footers, "NEXT" sections, HANDOFF status lines, and the big consolidation docs are NOT.
- `cf_cascade/CONSOLIDATION_forgetting_flow.md` (80KB master doc) is ~1 day behind the confound break it summarizes.
- Tally: **cf_cascade ~14 stale-closed · float_module 6 · small arcs 3 fired-and-forgotten + 1 cross-arc orphan link.**

**RULE: trust artifacts (`gateA/*.json`, `out/*.json`, `lambda_out/*.json`, `results/*.json`, `cloud/*.log`), not prose. Compare mtimes.**

## KNOWN FALSE CLAIM ON DISK — do not cite

`float_module/JOINT_EXPERIMENT_RESULT.json` (v1) reports **"UNIFICATION LANDS"**. It is **degenerate/vacuous**: every setting resolves
`no-room` and `struct_degree: 1`, so `predictor_accuracy: 1.0` is a constant-prediction artifact, and `free_holds_where_room: false` —
yet the verdict string asserts the free-convolution test matched where room resolved. **No room was resolved.** Logic bug at
`float_module/joint_experiment_lambda.py:176` (`elif pred_acc >= 0.75 and (free_holds or not free_where_room)`).
Independently flagged by two audit agents AND already warned in `cf_cascade/HANDOFF_FLOOR_AND_PROBE5_2026_06_25.md:50` ("do not bank").
Use `JOINT_EXPERIMENT_V2_RESULT.json` instead — but note V2's bootstrap CIs also look degenerate (`ci_lo == delta` exactly in all 8 settings).

## STALE-CLOSED (docs say open; artifact answers it) — do NOT re-run these

### cf_cascade — the "NOT fired" footers are wrong across the board
| thread | stale claim at | actual answer (artifact) |
|---|---|---|
| Pythia B-D at N=30 | `TODO_diffusion_program.md` "(open)" | `gateA/pythia_dispositive.json`: N=30, **B-D = -0.006+-0.015**, CI [-0.036,+0.024] EXCLUDES OPT's +0.055 -> **cross-arch dissociation REAL at matched power** (line now corrected in-place) |
| Probe-7b non-destructive dose | `HANDOFF_FLOOR:42` "NOT yet built/fired" | `gateA/probe7b_displacement_opt350m.json` (N=100) + `VERDICT_PROBE7_REORGANIZATION` -> reorganization confirmed |
| opt1_3b / pythia1b B-D | `VERDICT_THREAD2_ASPECT:15,16,92` "pending" | `bd_opt1_3b.json` **+0.000**, `bd_pythia1b.json` **+0.006** (both N=30) -> Pythia flat across size; OPT decays +0.110->0 |
| G5 -> B-D loop-closer | `VERDICT_THREAD1_G5:46,49-50` "NOT run" | `gateA/mediation_familysize.txt`: slope **-0.635**, R2 0.848, right-signed; grouping **UNDERPOWERED** (p=0.20 vs floor 0.10) |
| opt1_3b "STRUCTURAL HOLE, ladder tops at 350m" | `TODO:92` | `geom_opt1_3b.json`, `bd_opt1_3b.json` — re-fired w/ grad-ckpt; hole CLOSED |
| Probe-10 sim-rehearsal + format-vs-direction post-hoc | `PREREG_PROBE10*:71`, `VERDICT_PROBE10:49` | `probe10_simrehearsal_{opt350m,pythia410m}.json`, `probe10post_dirsim_opt350m.json` |
| Probe-11 / 11v2 / 11v3 / 12 | four PREREG footers "NOT fired" | `probe11_realizedalt_{opt350m,pythia410m,bloom560m,opt13b}.json`, `probe8v4_specificity_bloom560m.json` |
| Probe-13 dose-response | `PREREG_PROBE13:49` "NOT built/fired" | 7 cells: `probe13_doseresponse_{opt350m,pythia410m,opt13b,xglm564m,bloom560m,falcon_rw1b,bloom560m_noln}.json` |
| Probe-14 XGLM / Probe-15 Falcon | `PREREG_PROBE14:62`, `PREREG_PROBE15:52` | `build_wA_{xglm564m,falcon_rw1b}.json` + `VERDICT_PROBE14_15`: **CLOSED**, confound broken -> CONJUNCTION |
| Probe-16 curvature | `PREREG_PROBE16:52` "NOT fired" | `probe16_curvature_{bloom560m,falcon_rw1b,xglm564m}.json` -> scalar flatness REFUTED; C1 upgraded to NOVEL |
| Probe-17 LN ablation | `PREREG_PROBE17:44` "NOT fired" | RAN -> **INCONCLUSIVE** (anchor-confounded 0.83->0.7725 < 0.78 flag). The conjunction-vs-LN hedge genuinely stands |
| Probe-18 steer | `PREREG_PROBE18:46` "NOT fired" | `probe18_steer_opt350m.json`: orth_dVp +0.122+-0.038 ~= baseline +0.149 -> single-axis nullify **REFUTED**; protection is a BROAD subspace |
| "why BLOOM erodes — OPEN" | `CONSOLIDATION:856` | Answered by `VERDICT_PROBE14_15_CONFOUND_BREAK`: **CONJUNCTION** (ALiBi x multilingual), not one knob |

### float_module
| thread | stale claim at | actual answer |
|---|---|---|
| Phase 2b LLM size ladder | `VERDICT_LLM_PHASE2:32` "still owed" | RAN same day. Local ladder's "only 410m holds" was a **HARNESS FAILURE, not a scale boundary** — Lambda re-run: **2.8b FORM_HOLDS** (0.902), **6.9b FORM_HOLDS** (0.890). Scale axis closed to 6.9b. `lambda_out/pythia{28b,69b}_result.json` |
| §6.B RELM (z_low/z_trans) | `III_FINDING_CAPABILITY_REGIME:54` "separate go" | RAN 10h later: `III_RESULT_6B_TRANSFER` — **PASS (marginal)** armC .3364 vs armB .3016, **+0.0349**, CI [.0216,.0484]. Secondary: observational +0.457 was **entirely readout capacity** (matched-readout delta -0.0004) |
| Phase-1 efficient hand-in budget | `VERDICT_PHASE1_FRAME_DISCOVERY_FINAL:40` "NOT started" | Started 20 min later; became a ~25-artifact Phase-2 program through 06-17 |
| Free-convolution joint | `PRE_REG_FREE_CONVOLUTION_TEST:1` / `PRESTAGE:1` "HELD, NOT FIRED" | FIRED twice. V2: **"PREDICTOR WORKS, FREE TEST INCONCLUSIVE/CLASSICAL"** (deg x locality 0.875 vs degree-only 0.50). v1 verdict INVALID (see above) |
| Tooth-2 gate (b) | `TOOTH2_RESOLUTION:44,64` "UNTESTED / OPEN" | RAN 4 min later -> gate (b) **FAILS as predicted**; superseded by `TOOTH2_RESOLUTION_v2` |

### small arcs
| thread | stale claim at | actual answer |
|---|---|---|
| **librarian Phase-1 sequential** | `PREREG_PHASE1_LIBRARIAN_SEQ:53` "NOT run full" | **RAN 2026-07-11** (`out/seq_result.json`) and sat unbanked 3 days. **Verdict now written**: `VERDICT_PHASE1_LIBRARIAN_SEQ_2026_07_11.md` — NON-DESTRUCTIVE OVERLAP (retention flat; surge inV 0.35-0.40 vs inSwap 0.0003) |
| **tier_abstraction thread #4** ("does abstraction PAY downstream?") | `VERDICT_THREAD2_ARM_C:80-81`, `VERDICT_TIER_ABSTRACTION:88` "open for v0.2" | **ANSWERED YES — cross-arc.** `tier_abstraction/results/run_6b_transfer_result.json` IS float_module's `III_RESULT_6B_TRANSFER` (identical: armC .336/.3364, armB .302/.3016, +0.0349, CI [.0216,.0484]) -> **COMPARTMENTALIZATION-TRANSFERS**. Banked in float_module; neither arc links the other |
| detector_cf seq full run | `PREREG_DETECTOR_SEQ:48` "NOT run full" | `out/seq_detector.json` (`smoke:false`); rows reported verbatim inside `VERDICT_DETECTOR_ROUTE` |
| detector_cf ROUTE+GPM | `VERDICT_DETECTOR_ROUTE:37-40` "NOT run" | Fired + verdicted **6 minutes later** (`VERDICT_DETECTOR_ROUTE_GPM`) |
| alibi TIER0v2 crosslingual | looks orphaned (no `VERDICT_TIER0v2` file) | **MISFILED, not missing**: v2 verdict was APPENDED into `VERDICT_TIER0_ALIBI_MULTI` (lock 690a5b42) -> ALSO NULL (margin -0.082 vs 2SE 0.479). ⚠️ a filename-based orphan check FALSE-POSITIVES here |

## GENUINELY OPEN (artifact-verified: no result exists)

1. **Probe-3 (dose) + Probe-4 (dual)** — `CONSOLIDATION:879-880` calls these *"highest-value open GPU work."* No `gateA/probe3_*` / `probe4_*`.
   Needs re-spec: N>=100 paired; **Probe-4's disjointness gate is broken** (0.35 threshold < null ~0.38).
   ⚠️ **NAME TRAP:** `probe13_doseresponse` is a *different, later* probe — NOT Probe-3.
2. **`interference_law/PREREG_DYNAMICS_2C_2026_07_12.md`** — designed + locked, **never fired**. Asks: *is ‖Δθ‖ the SOLE governing variable for
   plastic forgetting, or does the PATH matter beyond it?* (LR sweep to convergence; also fixes two known #2b metric bugs). Note: the
   frozen-readout twin of this question was answered 2026-07-14 (§16: stream forgetting IS path-dependent/recency-weighted).
3. **Threads 1 / 2 / 4** (float_module<->CF bridge; adapter-sized-by-antagonism; operator x ML joint) — `TODO:73,75,77` "UNWORKED / SPECCED, NEVER FIRED".
4. **MPT-1b ALiBi+English confirmatory** — never ran (HF repo gated, needs HF_TOKEN); Falcon was substituted.
5. **2nd ALiBi+multilingual eroder / anchor-preserving LN test** — `VERDICT_PROBE14_15:139`. Probe-17 did NOT resolve it; the conjunction-vs-LN hedge stands.
6. **Pythia arms of Probe-5 / Probe-6** — no `probe6_displacement_pythia*`.
7. **Carrier-ablated-D run** — `TODO:227` "not wired; deferred".
8. **float_module:** 100M RELM net (spec+lock only, no artifact) · LLM-arm frontier / common-judge · 2nd real substrate for B-parsimony ·
   free-convolution **Axis B** (the joint tested Axis D<->Pillar E, not Axis B) · PRESTAGE derivation skeleton (trigger correctly never fired).
9. **detector_cf:** more-heads ceiling / identity-disjoint / multi-seed · curriculum-DATA construction (both explicitly "not owed").
10. **audio_cf:** operator-bank features (vs log-mel) · speaker-disjoint split (currently filename-hash).
11. **librarian_cf:** Phase-2 (localize / F2 antagonism — never specced) · order-permutation arm (deferred; now well-motivated by §16's path-dependence).
12. **tier_abstraction:** discrete/VQ bottleneck · Axis 2 (distribution shift) · Axis 3 (task-aligned pay) · (β,K) co-sweep.
13. **alibi_multi_cf:** Tier-1 was NULL -> a 2nd ALiBi+multi model / >=300M scale test is deliberately PARKED (needs GPU-days).

## AMBIGUOUS (left alone)
- `VERDICT_FLOOR_BATTERY:12` "fp32 hardware floor (not run)" — `floor_battery_opt350m_fp32.json` exists but PREDATES the doc; the "(not run)"
  likely refers to a different (Pythia) column. Not treated as stale.
- `PREREG_PROBE11v3_GENTLE:33` "NOT fired" — `probe11_realizedalt_{bloom560m,opt13b}.json` post-date it and are plausibly the v3 fire, but no
  cell tag distinguishes v2 from v3.
- `float_module` Phase-2 form program: last artifact `FORM_PANEL_S2` (06-17); **no VERDICT was ever written** — terminal state unrecorded.
