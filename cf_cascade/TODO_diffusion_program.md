# CF Diffusion Program — TODO

## ⚡ CURRENT (2026-06-21) — READ FIRST — consolidated (merged the desktop-sync `.new` conflict back in)

### One-line state
The mechanism layer of CF (Pythia + OPT) has closed honestly. Surviving headline: non-destructiveness protects, M2 pre-moved-capacity is the geometric mechanism, and **the map is real-but-inert at every structural level measured (6 tests)**. Multiple major retractions and a banked methodology win. Live remaining thread: the basis-reframe (Probe 2), to be run with the float-module B-falsifier methodology. Cross-program threads identified (Threads 1-5 below).

### What landed today (2026-06-21, full-day session)

**Cross-architecture climb completed on OPT-350m:**
- **Rung-1 (Gate-0a + freeze): PASSED.** OPT clears Gate-0a richer than Pythia (ESD multi-decade + autocorr void); SST-2 forms a flat-PSD basin. (−0.70/−113 λ_min were power-iter artifacts.)
- **Rung-2 (fragile-subspace characterization): PASSED — three-family mechanism with one bounded sub-claim.** OPT replicates seed-in-shared-steep-core (com 0.072), contested-core M1=0.859 (>Pythia 0.70), beyond-curvature aimed +0.288 (99% layers, ~Pythia +0.45), Cut-1-analogue (fragile-G≈topcurv-G≈0.087 vs chance 0.014 → task-A-owned). **The one divergence: antagonism does NOT replicate** — OPT contests the same curved dirs at cosine≈0 (+0.094, two measures), not Pythia's −0.23. Refined headline: **forgetting is driven by SHARING the steep directions, not ANTAGONISM there.** Pythia+GPT-2+OPT carry the core; "head-on" flavor is family-specific.
- **Rung-3 (non-destructiveness battery): MAGNITUDE retracted, DIRECTION reinstated at N=30 (see dispositive below).** Original: B−A=+0.160 / B−D=+0.118 / D−A=+0.041 (7/7), decomposition inverts vs Pythia (26% non-destr + 74% map). The reproduce-anchor fired at 7 seeds (att-3 B−D=+0.021; re-run B−D=+0.038 n.s.) — but the **OPT dispositive at N=30 resolves it**: B−D=+0.055±0.021 SIG. So the **map IS load-bearing on OPT**, just at HALF rung-3's magnitude (+0.118→+0.055) and ~61% of protection (not 74%). The 7-seed instability was per-seed SD ~0.10 swamping a real ~+0.055 effect, NOT absence. Substrates do NOT fully unify on #2b: OPT map load-bearing, Pythia B−D never run at N=30 (open). Original preserved: `dev_net2_opt_rung3orig.json`.
  - **[STATUS CORRECTION 2026-07-14 — this "(open)" is STALE; the Pythia dispositive DID run at N=30.]** `gateA/pythia_dispositive.json`: Pythia-410m, N=30, matched
    pinned regime (same lr_depth 5e-5/250, child 400/3e-5, pinned w_A, held-out subj), A_forget 0.215 (clears gate). **B−D = −0.006 ± 0.015 (11/30+), CI
    [−0.036,+0.024] — EXCLUDES OPT's +0.055.** Guards pass (mapfrac_B 0.340 vs mapfrac_D 0.018). Verdict: **CROSS-ARCH DISSOCIATION CONFIRMED at matched
    power** — map-dominant OPT vs non-destructiveness-dominant Pythia is REAL, not a power artifact. Do NOT re-run this; it is answered.

**Rung-4 mechanism (#2c on OPT) — attempted, voided, dispositive in flight:**
- **Toggle test (antagonism-controls-survival law): VOIDED, not killed.** Gradient-surgery knob confounded by perturbation magnitude (∝|c−c_natural|); D−A collapsed in lockstep with B−D (a map-independent control crashing identically can't be antagonism-specific). Reproduce-known anchor failed (c=0.0 surgery gave B−D=−0.015 not rung-3's +0.118). Hypothesis stays leading post-hoc; clean toggle needs a synthetic task-B family (data carrying prescribed cosine, no surgery) — heavy build, not fired.
- **Attempt-1: M3 probe confounded** (deposit-PCs curvature-derived; D rode B's deposit-residual nearly as much as B). Caught via M-stability + arm-D contrast. **Attempt-2: VOIDED** (auto-precheck drifted held-out→emotion; no dynamic range). **Attempt-3 (forced-regime + resid_B−resid_D + large-ε sharpness + reproduce-anchor): the anchor FIRED** (B−D=+0.021 ns below the +0.08 gate) → drove the rung-3 retraction.
- **Survives across all rung-4 attempts: M2-capacity directional both substrates** (fracV B,D<A: Pythia 0.348/0.410<0.461; OPT 0.220/0.226<0.244). The one mechanism that reproduces.
- **M1 honest catch:** symmetric fixed-norm perturbation cancels the linear term → ½ε²·vᵀHv → back to curvature at small ε. Only large-ε (basin width) is off-min-robust; "flattening" vs "walked off the bottom" may be inseparable with weight-space probes when phase-1 is an off-min walk. M1 flagged weak-secondary; M2+M3 carry the verdict.
- **OPT dispositive (30-seed forced-regime A/B/D): LANDED 2026-06-21 — SPLIT branch, map is REAL on OPT.** Pinned w_A (acc 0.823), forced subj, lr 5e-5/250, arms A/B/D. Not VOID (A_forget 0.237 ≥0.15; ret A/B/D 0.587/0.675/0.621). **B−A=+0.089±0.019 SIG** (23/30+, replay protects, REPRODUCES at N=30 — 7-seed nulls were underpowering, per-seed SD ~0.10). **B−D=+0.055±0.021 SIG** (22/30+, ~2.6 SE) — B,D differ ONLY by V-projection (mapfrac 0.219/0.014) → clean paired causal: **the MAP IS LOAD-BEARING on OPT**, ~61% of protection. **D−A=+0.034±0.023 n.s.** (map-independent residual below resolution). Mech tell: childhood loss-drop B/D 6.67/13.54 (map-stripped D learns FASTER, protects LESS). **Resolves the retraction as a POWER problem, not a fluctuation** — the three 7-seed B−D draws (+0.118/−0.003/+0.021, SE~0.04) all consistent with +0.055. Reproduce-anchor was right that rung-3's +0.16/74% was inflated ~2×; retraction OVERSHOT calling the map inert. `gateA/opt_dispositive.{json,npz}`; `cloud/opt_dispositive_lambda.py`.

**Basis-reframe pre-check (Probe-1 cross-task basis) — BUILT, RAN, PASSED both substrates.**
- Candidate C = top-K task-B Fisher eigvecs. After a **gate-2 logic-bug fix** (v1 penalized concentration — flat-noise=high effrank, concentrated=low effrank=GOOD; the v1 "ALL_PASS False" was the bug), and a K-sweep that resolved the distinctness (v1's top-128 0.648 was noise-padded suspect): **distinct ortho-frac HOLDS at genuine low rank** — OPT 0.50→0.71→0.65, Pythia 0.41→0.50→0.65; stable Grassmann@K16 OPT 0.36/Pythia 0.76 (≫chance 0.01); conditioned (effrank 14 OPT / 2 Pythia, top128 energy 0.97/0.999). Basis = the LOW-RANK task-B gradient subspace, ~50-71% OUTSIDE curvature, working K=16. Antagonism mean-grad cos ≈0 both (diagnostic). `gateA/basis_precheck.json`; `cloud/{basis_precheck_lambda,lambda_launch_basis}.py`.
- ★ This is **Probe-1's** basis (cross-task task-B geometry). **Probe-2** (Thread 3) needs a *task-A-agnostic* basis (different construction) — that's the higher-value live build.

**Spiral sub-program (parallel agent, depth-geometry hunt) — COMPLETE, real-but-inert close:**
- **SPIRAL-1: POSITIVE, INVERTED.** Pre-reg early-broad→late-specific; measured the opposite (early HIGH-dim/separated, deep COLLAPSE onto shared/universal contested core; ρ=−0.72, z_shuf=−3.5, curvature-controlled, 2 substrates, attention-carried). Keeper reframe: the deep collapse IS the universal contested core located in depth.
- **SPIRAL-2: NEGATIVE** (smooth funnel not fractal). **SPIRAL-3: NEGATIVE** (seed-concentrates-deep refuted ρ≈0; carrier-split n=3 can't adjudicate).
- **SPIRAL-4: POSITIVE — the TWIST.** Consecutive overlap 0.534, ρ_gap=−1.00, fully reorients top→bottom, attention-carried (qkv/h4h coherent; down-proj structureless = dye carrier-split). True contract+twist spiral. Nuance: twist is curvature-general (topcurv spirals identically; fragile rides the backbone).
- **SPIRAL-5: NEGATIVE — caught a pooling artifact mid-run.** Pooled n=72 +0.29 (z+2.5, would've banked) → within-carrier decomp killed it (qkv −0.19, h4h −0.01, 4hh −0.29). Ecological-fallacy manufacturing a positive. **CASCADED-BROAD: NEGATIVE** (C≈B; breadth is breadth regardless of depth-ordering).
- **Close:** map is a true contract+twist spiral, real geometry at every level (flat/funnel/spiral), load-bearing at NONE. SPIRAL-1 (deep-collapse-IS-the-core) is the one keeper.

**Poke axis (cross-arch fragile-subspace pre-loading) — COMPLETE.**
- Initial spec voided by CPU pre-check (matched-magnitude poke confounded by ~10⁶× stiffness ratio — fragile≡stiffest-16). Pythia: INERT (D1) + weak-generic sink (D2). OPT run1 VOIDED; run2 regime-confounded sink-hint. **Disambiguator (matched-regime Pythia at OPT's 0.115 forget): ARCHITECTURAL SINK CONFIRMED** (Pythia dead-flat inert, sink only on OPT; regime ruled out).
- **Two banked headlines:** (1) fragile subspace is NOT a protective handle on either arch; occupy-don't-project DEAD; (2) interventional bound on the antagonism law: antagonism controls effect-EXISTENCE (Pythia inert / OPT non-inert) but a surviving deposit is a SINK not a RESOURCE. "Survival enables protective occupation" REFUTED. Map real-but-inert under pre-occupation (sixth time). ~$9, 5 runs.

### The robust stack as of EOD 2026-06-21

**Characterization layer (UNTOUCHED, seed-stable geometry):**
- Rungs 1-2 across three families (Pythia, GPT-2, OPT): fragile-subspace mechanism replicates. Seed in shared steep core; contested core ≈ top-curvature; beyond-curvature aimed +0.45 (Pythia)/+0.288 (OPT); fragile subspace is task-A's, not architectural (within curvature basis).
- Bounded sub-claim: antagonism (Pythia −0.23 / GPT-2 / OPT ≈0) is family-specific; SHARING the steep directions, not antagonism, is the cross-family driver.

**Mechanism layer — CROSS-ARCH DISSOCIATION CONFIRMED at N=30 (both dispositives in): same magnitude, different mechanism.**
- **OPT (N=30): map-CARRIED.** B−D=+0.055 sig (~61% of protection), D−A=+0.034 n.s. (map-independent residual below resolution).
- **Pythia (N=30): NON-DESTRUCTIVENESS.** D−A=+0.079 sig (strip the map, protects fully), B−D=−0.006 n.s. CI [−0.036,+0.024] **excludes OPT's +0.055.** The #2b pattern CONFIRMED at matched power (not the 7-seed artifact the morning feared).
- **Both protect at EQUAL magnitude** (B−A ≈ +0.073 Pythia / +0.089 OPT) — they ROUTE protection differently. Invisible at 7 seeds (both B−D ≈ +0.02 ± 0.04); separates decisively at N=30. **Coherence tell holds both** (childhood loss-drop D faster than B: Pythia 8.77/15.28, OPT 6.67/13.54) → same surgery, same learning-asymmetry, opposite carrier = dissociation is clean, not confounded.
- **REOPENS the antagonism line with a real target:** rung-2's divergence (Pythia antagonism −0.23 head-on / OPT ≈0) plausibly explains it — Pythia head-on antagonism erodes the map deposit → inert; OPT no-antagonism → deposit survives → load-bearing. rung-4 M3 + toggle redesign are BACK IN PLAY (no longer moot) with a concrete cross-arch prediction.
- M2 pre-moved-capacity: fracV B,D<A both substrates, every run. **The one geometric mechanism that reproduces across both.**
- M3 (map-deposition): on OPT, the dispositive's B−D sig IS the causal map-deposition signal; on Pythia it's genuinely inert at N=30. M1 (basin-flattening): still conceptually confounded by off-min displacement; not adjudicated.

**Six real-but-inert map findings:** 1. #2b map inert for protection (Pythia B≈D). 2. SPIRAL-3 depth-funnel inert for seed/carrier-split. 3. SPIRAL-5 depth-twist inert. 4. Cascaded-broad depth-ordering inert (C≈B). 5. Pythia poke fragile-occupation inert. 6. OPT poke = SINK not resource.
- ⚠ **SCOPE after both dispositives:** #1 is Pythia-specific and now **CONFIRMED at N=30** (B−D=−0.006 n.s., map genuinely inert on Pythia). But on OPT at N=30 the replay-deposited map is NOT inert (B−D=+0.055 sig). So "map inert for protection" is **architecture-dependent, not universal** — true on Pythia, false on OPT. #5/#6 are a DIFFERENT intervention (pre-OCCUPYING the fragile subspace before training) than natural-replay deposit. Clean statement: pre-occupation is a sink (both archs); natural-replay map-deposit carries the majority of OPT's protection but nothing on Pythia.

**Methodology wins (banked as their own findings):**
- The reproduce-known anchor caught rung-3 as a fluctuation (confirmed by same-script re-run) — saved the program from spiraling on +0.118.
- ~12-14 measurement-validity catches across the program; new species this session: a knob/probe whose side-effect mimics the signal, caught by a control that should be invariant (toggle's D−A-in-lockstep, M3's deposit-PC confound, poke's 10⁶× stiffness pre-check, basis-precheck's gate-2 inversion).
- Float-module's enforced-not-asserted discipline confirmed transferable to CF measurement design.

### Turbulence parallel work (banked separately, related epistemically)
- Probe 12A H1-H5 ran. H1 (4/5) PASS; H2 (full j0+j4 tensor) PASS strongly with off-symmetry dirs; H5 (cubic amp) PASS; H3 (3rd-order C1) INCONCLUSIVE / basis-limited not data-limited; H4 (Hill-29 iso) FAILS at mid r.
- **Reframe:** CLM-244/245 said "data-limited"; pristine data → same failure → **basis-limited**. Residual is real inertial anisotropy (resid/noise 4.3, incompressibility holds 0.99, scale-flat), localized to diagonal d_111/d_000 (75%), in directions the j0+j4 basis structurally can't represent. **CLM-TURB-249** (verified finding, walks back CLM-245) + **CLM-TURB-250** (full j0+j4 2nd-rank tensor clean, relres 0.009-0.075) banked (commit 27ba2bc).
- H3↔H4 unification probe (`probe12A_unify_H3H4.py`, iso-div VALIDATED vs H4 to 0.000): simple unification refuted (aniso divergence 6× overshoot, wrong-signed); auto "different physics" REFUSED (overshoot from the failing H3 fit; sig-corr −0.75 contaminated by div_iso zero-crossing). Held at "two faces of the same un-representable 3rd-order structure"; "doesn't close" is a mid-inertial statement (gap 0.41 @ r/L=0.20 but 0.06-0.11 at large r).
- **Epistemic bridge to CF:** turbulence shows the residual lives in directions the dominant basis is geometrically blind to, found by asking what the dominant basis structurally can't represent. CF's basis-reframe (Probe 2) translates it: the beyond-curvature ~30% may live in a basis the curvature view is blind to (task-pair geometry, not task-A projected onto task-B).

### LIVE REMAINING THREADS (ranked by leverage, cross-program)

**Thread 3 (highest immediate leverage) — float-module B-falsifier for CF Probe 2.** Probe 2 (basis-causal: Cut-1 in a task-A-agnostic non-curvature basis) was to be built from scratch with its own pre-check. Float-module's `rmdsrc_parsimony.py` (hash-locked SHA `18c6e9d7`) is the validated methodology for "does a candidate structural grouping pay vs random-grouping null" — banked 7 substrates, cross-executor-blind P5 validated, M3 meta-finding flags it as the most discriminating + externally-validated falsifier. Mapping: per-unit OLS ↔ per-(task-A,layer,carrier) fragile orientation; group-shared ↔ shared fragile orientation under candidate basis; held-out LPPD ↔ held-out task-A's fragile overlap; parsimony_strict null ↔ random task-A-agnostic basis null; PAYS_STRICT ↔ real architectural fragility; PROVISIONALLY_IRREDUCIBLE ↔ no basis beats per-task-A (architectural fragility doesn't exist beyond curvature). Candidate task-A-agnostic bases (answer-blind): base-model (pre-FT) principal directions; cross-anchor-invariant (intersection across SST/AG/UCDP/code/Dolly fragile subspaces); multi-pair Reynolds basis (Reynolds-projector × harmonic-degree filtration from `CROSS_SYNTHESIS_operator_x_ML_2026_06_16.md`). CPU-only on existing artifacts. **Status: prompt specced, handed to parallel agent; does NOT block the dispositive.**

**Thread 1 (highest strategic leverage) — `float_module/catastrophic_forgetting_test.py` ↔ cf_cascade.** Float-module banked: append-only precondition-keyed records retain 1.000 where SGD weight-replacement forgets 0.514 (9 falsifiers clear) — CF immunity as a property of substrate commit-discipline, not optimizer geometry. Cross-lane question: is non-destructiveness what CF discovered as the only-thing-that-protects *because append-only is the only thing that scales to weight space*? Float-module's M2 capacity might be the neural-substrate version of append-only commit-discipline. Either resolution reshapes the writeup. Status: UNWORKED, write-up bridge candidate.

**Thread 2 (most generative) — tier-abstraction "fine-price differential" × CF antagonism law.** `tier_abstraction_v0_1/SEAMS_AND_BRIDGES.md`: abstraction emerges iff a price differential between fine/coarse distinctions; Arm A (price=0)→0, Arms C/D/E/F (price>0)→abstraction, Arm B (+0.02 supervision pulse). CF antagonism IS a fine-price differential on task-pair geometry (Pythia prices shared-dir deposits out→eroded; OPT orthogonality removes price→survive-but-interfere=SINK). Difference = whether the net has a place to absorb priced-out info (tier-abstraction has a compartmentalization arm; CF has only fixed weights). Prediction: **CF protection becomes possible iff you give an architectural sink — adapters/dedicated dims — sized by the antagonism geometry** (the second-spool idea, now with a cross-substrate prediction; Arm B predicts a small-but-real effect). Status: UNWORKED, adapter-design experiment.

**Thread 4 (highest readiness, never fired) — operator × ML joint experiment.** `CROSS_SYNTHESIS_operator_x_ML_2026_06_16.md`: spectral structural-degree predictor with the group, tested as free independence; Steps 1-4 specced. Both lanes converged on "structural degree = first non-vanishing harmonic in the substrate's natural basis." qx+1 closed-form structural degree (the (1,4)-eigenline of T_diag); vision rep degree = lowest Hermite degree with non-zero target projection; free-convolution (R-transform) spectral test for free-vs-classical. Status: SPECCED, NEVER FIRED. Adjacent to Thread 3 (both about identifying the right basis).

**Thread 5 (writeup-shape) — the discipline as a finding across substrates.** Pre-registered nulls + reproduce-known anchors + honest-negative-as-finding produces real bounds in three research types: CF (~12-14 catches; rung-3 retraction strongest), float-module (P1 specificity fail-then-pass; 7-substrate ledger M1-M4; PROVISIONALLY_IRREDUCIBLE as first-class verdict), tier-abstraction (Form-1 + Form-3 both REFUTED). Cross-lane consistency is the methodology contribution. Status: WRITEUP, not a fire.

### Next moves (compute-ordered)
1. **BOTH dispositives DONE — cross-arch dissociation CONFIRMED at N=30.** OPT map-carried (B−D=+0.055 sig); Pythia non-destructiveness (B−D=−0.006 n.s., CI excludes OPT). Same magnitude, different mechanism. Banked to memory + this TODO. `gateA/{opt,pythia}_dispositive.{json,npz}`.
2. **ANTAGONISM LINE CLOSED — DEAD as the explanation; TOGGLE TERMINALLY DEAD (2026-06-21).** Matched-w_A parity (antag_space + opt_antag_space): at the dispositive w_A's the depth-task (subj) antagonism is near-zero on BOTH (Pythia +0.047 / OPT −0.007, neither head-on; the −0.23 was a rung-2-w_A artifact), and the residual runs OPPOSITE the hypothesis (OPT load-bearing yet MORE negative). Space doesn't matter (curv16≈curv128≈raw). The toggle: no head-on dial exists at either operating point + the synthetic-task construction failed its own gate (label-mix doesn't steer weight-grad cosine, Jacobian common-mode dominates). The dissociation is now **REAL-BUT-UNEXPLAINED.** `gateA/{antag_space,opt_antag_space}.{json,npz}`; `cloud/{antag_space,opt_antag_space,toggle_v2_construct}_lambda.py`.
3. **★ Dissociation EXPLANATION — BOTH candidate mechanisms now DOWN at matched conditions → writeup with the honest close.**
   - **rung-4 M3 on OPT (matched N=30, pinned w_A): deposit-routing NOT the mechanism.** Anchor reproduced (B−D +0.060, mapfrac 0.219/0.014, A_forget 0.241). M3 resid_B−D stat-positive (+0.017, 30/30, 6/6 carriers, M-stab 0.614) but **QUARANTINED** by the verdict→rows tripwire: the inline **swap control** killed deposit-specificity (vs deposit-M +0.017 vs top-curv +0.009, margin +0.008 < 0.01 → ~half is generic curvature-family routing, the Thread-3 species caught verbatim); generic-childhood baseline resid_D−A=+0.063 dwarfs the B-increment (3.7×); +0.017 ≪ the +0.060 protection. M2-capacity reproduces geometrically (fracV B,D<A) but doesn't distinguish B/D. `gateA/opt_m3.{json,npz}`; `cloud/{opt_m3_lambda,lambda_launch_optm3}.py`.
   - **CONCLUSION: OPT's load-bearing map is real-but-unexplained at the mechanism layer.** Antagonism (matched-w_A) AND deposit-routing (matched-M3) both tested clean at matched conditions and neither explains it. The architecture-level *cause* of the dissociation is out of scope.
   - **NEXT = WRITEUP** OR the geometry hunt below (in progress).

4. **★ COMMENSURABLE GEOMETRY CLASSIFICATION — the explanation hunt (2026-06-21, in progress).** Built `geom_classify_lambda.py`: 6 dimensionless cross-arch metrics on V (task-A curvature subspace) at pinned w_A, per-carrier never pooled. **2-model baseline DONE (OPT vs Pythia, `gateA/geom_classify.{json,npz}`): vocabulary VALIDATED + coherent "crowding vs room" candidate.** G2 spectral decay OPT 2.48 / Pythia 5.92 (huge), G3 surge-fracV 0.31/0.51 (~13σ), G4 surge↔topcurv 0.29/0.41, ★G5 deposit↔surge 0.420/0.463 (~2.7σ, **pre-reg hypothesis SUPPORTED** — OPT deposits OUT of surge's way), G6 mapfrac 0.28/0.35 (reproduces #2b). STORY: Pythia=crowded (steep spectrum→surge+deposit pile into stiff V→deposit overwritten→inert); OPT=roomy (flat spectrum→deposit out of surge's way→survives→protective). G2/G3 bigger levers than G5. CAVEATS: G1 non-differentiator; G1/G2 null-test DEGENERATE (deterministic→noise=0→trivially "classifies") — **FIX before matrix: cross-carrier bootstrap null for deterministic metrics**; n=2 = co-occurrence not causation.
   - **PIPELINE:** Layer1 geometry vocab → Layer2 `size×family` MODEL MATRIX → **Layer3 RMD SRC** (`rmdsrc_parsimony.py`, hash `18c6e9d7`) tests which grouping {family/size/B−D-sign} PAYS vs random-grouping null. Cross-family unit = MODEL (6 vs 3 carriers not poolable).
   - **★ MATRIX RAN (2026-06-22, 5/6 cells; G1/G2 bootstrap-null fix applied): `gateA/geom_matrix.{json,npz}`.** OPT{125m,350m} + Pythia{160m,410m,1b}; **opt1_3b OOM'd in AdamW on 40GB (re-running 8-bit/80GB) = STRUCTURAL HOLE, OPT ladder tops at 350m.** RESULT: **G4 (surge↔stiff) + G5★ (deposit↔surge) SEPARATE families across the available cells** — OPT G5 0.34/0.41 vs Pythia 0.53/0.50/0.49 (pythia1b stays Pythia-like), ~12×bootSE. **G2/G3/G6 are NOT robust family axes** (the n=2 baseline's huge G2 gap was an artifact — pythia1b G2=1.81 < OPT; ALSO G2 precision-fragile: same pythia410m ckpt 5.92 float64 vs 3.43 float32 → DEMOTE G2; G3/G6 are SIZE-driven, Pythia→OPT-like as it scales). STRUCTURE: "how much into V"(G3,G6)=size-dependent; "where vs surge"(G4,G5)=family-fixed-across-available-cells. ⚠⚠ **HEDGES (Wilson 2026-06-22, [[feedback_separation_SE_is_not_prediction_2026_06_22]]): (1) the 12×SE is FAMILY-SEPARATION, NOT a tested G5→B−D PREDICTION — G5↔B−D is CO-OCCURRENCE across 5 well-separated points until the RMD-SRC loop-closer runs vs random null; do not transfer the SE. (2) "family-fixed at ALL OPT sizes" is EXTRAPOLATION — only 2 OPT cells, no high-size mirror of pythia1b; honest claim = "across the Pythia ladder + the 2 available OPT cells."**
   - **BRIDGE FAILED (parallel agent, 2026-06-22): from-scratch OPT-350m-config can't learn task-A (SST 0.490=chance; aimed 0.049 vs anchor 0.29; M1/com degenerate). → PRETRAINING IS LOAD-BEARING; dissociation lives in pretrained→finetuned regime; from-scratch aspect-ratio ablation DEAD (no features to forget). Gate earned its keep (~20-30 GPU-hr saved).** ★ ASPECT-RATIO likely MOOT for free: matrix has 2 iso-aspect cross-family pairs (OPT350m/Pythia410m both 1024×24 ~42.7; OPT125m/Pythia160m both 768×12 ~64) and BOTH dissociate on G5 → aspect ratio doesn't explain the family split (pending config rubber-stamp). REDIRECT architectural-cause hunt to positional-encoding (learned vs rotary) / qkv-layout (split vs fused) / activation, via feature-swap on PRETRAINED models (keeps pretraining).
   - **★ DEPTH AXIS DONE (2026-06-22, surgered OPT-350m d6/12/18/24, `gateA/geom_depth.{json,npz}`):** proper bootstrapped-monotone-slope test (NOT the script's loose range>2SE auto-verdict, which false-positived all 6 — superseded). **G3 surge-fracV (0.381→0.319) + G6 deposit-fracV (0.327→0.281) TREND monotone-DOWN with depth** = SCALE-driven (same direction as the matrix size-trend → G3/G6 decrease with both size AND depth). **G4 + G5★ FLAT across depth** (G5 0.436/0.393/0.396/0.426 non-monotone, slope CI incl 0). → **G5 is invariant to BOTH size and depth = family-fixed, de-confounded on both axes.** The family-G5 separation is not a size or depth artifact. ⚠ still CO-OCCURRENCE (G5 tracks family not scale; G5→B−D prediction is the loop-closer's job). Methodology: range>2×single-cell-SE is a too-loose MOVES null (flags wobble); use bootstrapped monotone-slope.
   - **B−D FEED FIRING (2026-06-22, 7 cells, `bd_feed_lambda.py` parameterized by CELL):** opt125m/pythia160m/opt_d6/d12/d18/opt1_3b/pythia1b → bd_<cell>.{json,npz} = model B−D + per-carrier resid + A_forget VOID guard. Big cells grad-checkpointed. opt1_3b geometry re-firing (grad-ckpt) closes the matrix hole. Launcher hardened: 429-retry+20s backoff (10 tries), 0-20s politeness jitter, 900s boot timeout.
   - **NEXT (loop-closer):** when B−D-per-model + opt1_3b land → RMD-SRC on [G4,G5,size,depth] vs B−D, grouping {family/size/depth/B−D-sign} vs RANDOM NULL. Pre-reg: depth-grouping of G5 should NOT pay (G5 depth-flat); family-grouping is the live candidate. **Run BOTH with and without the top-OPT cell, and with opt1_3b imputed at favorable AND unfavorable G5 — if family-grouping only pays under a favorable imputation, DISQUALIFIED.** Random-null verdict is the ONLY thing that promotes co-occurrence→tested prediction.
4. **Thread 3 (B-falsifier for Probe 2)** — handed to parallel agent (CPU-only). **SPIRAL-OPT** — OPT depth-substrate (`crosspair_residual_opt.npz`, `breadth_pass_opt.npz`) being generated this session for the parallel analysis agent.
5. Writeup headline: **protection mechanism is architecture-dependent (OPT map-deposition / Pythia non-destructiveness), at equal magnitude — and the cause is NOT antagonism (matched-w_A refuted it); it's an open architecture question.** Pair with the basis-reframe verdict. ★ DISCIPLINE addition: matched-conditions species (matched-N / matched-architecture / matched-w_A) — cross-condition mechanism inferences require matched conditions; the cheap measurement gate caught the toggle's dead premise for ~$1 before $6-9 of arms.
4. **Thread 4 (joint operator × ML)** if Thread 3 finds the basis.
5. **Thread 2 (adapter-sized-by-antagonism second spool)** if the writeup needs a "what next" — live novel intervention with a cross-substrate prediction.

### Parked / lower priority
- Toggle redesign (synthetic task-B family). **PROMOTED to next-moves #2** — no longer moot: the cross-arch dissociation gives it a real antagonism→map-survival target to test. (Heavy build; redesign data-carries the cosine, no surgery.)
- Cross-window/continuum-overlap sweep (CF-physics bridge). Untouched.
- DND + CSCL full-text reads (parked from breadth-pass frame).
- Probe 1 (cross-task basis M3 re-measurement) — subsumed (basis pre-check showed the basis exists/distinct; M3 inert).

---

## PRIOR SECTIONS (pre-2026-06-21 detail, preserved for reference — current state is the 2026-06-21 section above)

## (2026-06-20, superseded) — prior CURRENT
- Characterization program CAPPED. Cut 1 done: task-A varied (SST-2→AG→UCDP) → **the fragile subspace is TASK-A's, not
  the architecture's**. Bounded headline: universality is ACROSS-task-B / SET-BY-task-A / attention-carried / graded.
- 7-arm CONTROL TEST = INCONCLUSIVE (timed+targeted protection): fragile=stiffest so no stiff-matched targeting control
  exists; timed-early insufficient on real-forgetting AG. Banked NEGATIVE.
- **RESIDUAL FRAGILE TEST = DONE → CLEAN NEGATIVE (lever capped at curvature).** Control finally constructible
  (Grassmann(RES,CURV)=0, λ_RES≈λ_randorthog≈6e-8 ≪ λ_CURV=0.1). AG (valid): curvres 0.680 < curv 0.722 (>2SE, residual HURTS),
  curvres≈currand, res-alone≈naive → triple-confirmed negative. MECHANISM (near-forced): λ_RES≈0 → residual is FLAT → protecting it
  inert by construction. seed↔core +0.45 beyond-curvature is real in the AIM but aims into FLAT space, INERT for damage. Subj=VOID.

## ✅ DONE (2026-06-20) — BREADTH-PASS v1/v2/v3: task-A's neighborhood is a learnable map, GRADED BY DISTANCE FROM TASK-A
- v1: held-out map predicts a never-seen task-B's pull, raw 0.83 vs shuffle 0.00, partial **0.66** 72/72 layers. FLAGGED not
  banked — only null was dirty (permuted-SST = sentiment-shaped, polluted LOO map).
- v2 (tighten the floor): map fit = REAL only; TRUE nulls (randtok=pure geometry, nltext=TinyStories LM no task-format).
  Decomp: real 0.652 = ~0.40 generic (randtok 0.401) + ~0.16 language (nltext 0.561) + task-specific lift. Attn 0.700>mlp 0.628.
- **v3 LEAVE-ONE-TYPE-OUT = HONEST FINAL SHAPE, graded.** cross-type **0.627** (vs language floor 0.561, +0.065) does NOT collapse.
  Within-type wins (same 0.743 vs cross 0.650) BUT cross-type gen is REAL & GRADED by distance from task-A (sentiment): topic
  0.69-0.72 / subj 0.66 / emotion 0.63 / sentiment 0.61 → **code 0.58 ≈floor, ucdp 0.51 BELOW floor** (alien kinds get zero benefit).
  Neighborhood CENTERED on task-A, decays with distance = developmental shape. Bounds #3 to the neighborhood. attn 0.683>mlp 0.598.
- LIT closed: 2603.07415 (Cheng) + CSCL Yoo2024 — neither occupies the claim. Scripts gateA/breadth_{pass,typeout}.{json,npz} + cloud/breadth_pass_lambda.py.

## ✅ DONE (2026-06-20) — EXPERIMENT #1 BREADTH-EXTEND: extensibility confounded; map STABILITY + SELF-PLACEMENT clean (firm v3)
- A/B: predict same held-out code & ucdp under ORIG-9 vs ENRICHED-14 stream (added math/sci/nli→code, newscat/yahoo→conflict).
- Auto-verdict "HARD LIMIT" REFUSED = confounded: NO extension (code Δ+0.028, ucdp Δ−0.029, noise) BUT (1) bridges landed DOWN
  with code (math 0.555/sci 0.511 not spanning), (2) floor moved 0.561→0.639 → "wall" not proven (bridge-quality check caught it).
- CLEAN POSITIVES firm up v3: STABILITY (5 new kinds barely moved ORIG, |Δ|≤0.03) + SELF-PLACEMENT (newscat 0.73/yahoo 0.62 near,
  math 0.55/sci 0.51 far w/ code) + CLUSTER STRUCTURE (affect/topic 0.62-0.72 | STEM/code 0.51-0.61 | conflict 0.49; STEM bridges
  JOINED code's cluster → alien kinds in DISCONNECTED regions). Artifacts gateA/breadth_extend.{json,npz} + cloud/breadth_extend_lambda.py.
- #1 cleanly IDs in-neighborhood tasks for #2 (topic/subjectivity/emotion IN; STEM/conflict OUT).

## ✅ DONE (2026-06-20) — #2 THE DEVELOPMENTAL EXPERIMENT: CLAIM LANDS (in-neighborhood) [⚠ B>C reading RETRACTED by #2b]
- 3 arms × 7 paired seeds. Held-out=subj (forgot 0.263). Childhood=NON-sentiment stream. SST retention (pre 0.753): A 0.527 /
  B 0.653 / C 0.538. B−A=+0.126±0.024, B−C=+0.115±0.025. Childhood made depth 3× less destructive. Scramble verified.
- CAVEATS: master-none partially violated; 1 dissent seed; single net/task-A. Artifacts gateA/dev_experiment.{json,npz} + cloud/dev_experiment_lambda.py.

## ✅ DONE (2026-06-20) — #2b NON-DESTRUCTIVE NO-MAP CONTROL: DEFLATION, the map does NOT route the surge
- Arm D = same stream as B but childhood grads projected off task-A top-128 curvature (non-destr + no-map). 4 arms × 7 seeds.
  Retention A 0.527 / B 0.653 / C 0.538 / D 0.670. **B−D=−0.018±0.018 (B≈D, map adds NOTHING); D−A=+0.144±0.018 (~8SE).**
  Guards passed (D non-destr 0.741≥B 0.722, no-map mapfrac 0.018 vs B 0.341, trained drop 15.3).
- RETRACTS #2's "it's the map" → "it's NON-DESTRUCTIVENESS not the map". v3 map REAL but CAUSALLY INERT for protection.
  Artifacts gateA/dev_experiment2b.{json,npz} + cloud/dev_experiment2b_lambda.py.

## ✅ DONE (2026-06-20) — #2c MECHANISM (flattening M1 vs pre-moved-capacity M2): LEANS PRE-MOVED CAPACITY
- M2 (redirection) CLEAN: D's phase-2 Δw lands LESS in task-A V16 (0.214 vs A 0.306, +0.092 ~9SE, 5/5) = capacity signature.
- M1 (flattening) CONFOUNDED (empirical Fisher off-min → ratio 441 = grad-inflation; auto "unchanged" = code bug). Pure-flattening
  DISFAVORED, capacity SUPPORTED. [Superseded by rung-4 M1 redesign → off-min intrinsic, M1 flagged-weak.] Artifacts gateA/dev_experiment2c.{json,npz}.

## 🗂 FRAME (reference) — BREADTH-PASS / "missing childhood" (Nateism, 2026-06-20)
**★ CANONICAL FRAME DOC: `BREADTH_PASS_developmental_frame.md`** (Wilson writeup — developmental reframe "CF = missing
childhood", metaphor→test chain, lit-map, white-space, full queue). Frame: forgetting = two learning steps smashed into one.
Nets do only DEPTH (yank); they skip BREADTH (placing task-B relative to known). The breadth pass = the net learning task-A's
task-neighborhood map as an explicit step-1 so step-2 allocation becomes readable. Discipline: HELD-OUT task-B is THE control;
per-layer never pooled; SEED not contest; chance baseline; don't bank off in-stream tasks.

## ⏸ SUPERSEDED (historical — GPT-2 generalization, first-half adjudication, dye program, Stage B; all resolved, see memory APEX)
### GPT-2 GENERALIZATION (2026-06-19) — "same flow, different type"
- GPT-2 CONTINUUM (gap 3.4-3.9×) vs Pythia TWO-SCALE (10-25×). HOLD laminar+ordering universal HELD (Dye3 z35-87, arrival +0.9-1.0).
  Seed MIXED (down-proj 17.3≫1.7 ✅ sharpest; up 18.5 ✅; attn 4.1 ❌). Carrier-split: GPT-2 NO channel (spectrum-tied) vs Pythia Wdown.
  Tooling gpt2_{stage0,dye1,dye2,dye3}*.py, basis gpt2_dye_basis.npz. Conv1D: FA=g@gᵀ, proj=Vkᵀ@dw, inj=outer(v,u).
### LAMBDA OPS (2026-06-19)
- **FUTURE GPU RUNS → LAMBDA, not local** (user directive). Tooling cloud/lambda_launch_cf.py etc. Gotcha: stale Pillow → `--upgrade Pillow`.
  A100 SXM4 $1.99/hr, 410M run 158s/~$0.27. Resilient _req retry-on-5xx; hard finally-terminate.
### FIRST-HALF / TRAFFIC ADJUDICATION (2026-06-19) — REOPENED then characterized
- TEST #1 free-lane EXISTS (task-neutral Fisher overlap 0.12-0.18, ~82-88% task-A-private; overturned Stage-B "no free lane").
  TEST #2a along-path antagonism cos=−0.23 (shared core antagonistic). k-sweep + substrate firmups PASSED.
- ★★★ PROJECTION LEVER WORKS (gateA/projection_lever.json): projected r64 A=0.74 (~94% forgetting prevented) AND B=0.895 not starved.
  Method=GPM (known); contribution = the characterization. FIXED CONTESTED CORE real (z+197) but DEFLATES to task-A dominant-curvature
  (M1 Grassmann 0.70, M2 +0.71) = mundane (Gur-Ari/Wu/Papyan). LIT-check: GPM/EWC/PCGrad/Mirzadeh reproductions; diffusion = surviving white space.
- ★★★★ DYE PROGRAM (Dye1 passive / Dye2 active / Dye3 flow): forgetting SEEDS localized high-curvature droplet → LAMINAR traveling
  front (z24-105) → early rate CURVATURE-GRADED (stiff 1.4× faster) → SATURATES to direction-indep extent. Carrier-split qkv+up broad / Wdown channel.
### STAGE A/B (encapsulation) — RESOLVED to reverse-after
- No irreversible core; full weight-reversibility; redundant double-encoding (attn +0.90 / mlp +0.87). Encapsulation = reverse-after.
- Stage B adjudication NEGATIVE: task-A & task-B Fisher monotone-aligned (no high-A-low-B subspace) → protect-one-pathway not buildable
  within curvature basis (basis-induced-alignment caveat → motivated the breadth/basis-reframe work). Recipe = redundancy-based pathway reverse-after.

## PHYSICS LANE (parked, separate Wilson)
- σ_cpl existence LOCKED+committed (PSD gate 0/1500 neg). Peak-LOCATION needs continuum task-B. Catch #8 (plan_G): bridge-prediction VOID;
  measurements (β=0.409/CLM-114/K62-refutation) UNTOUCHED. The reversible-cell of Stage B IS the σ_cpl reversible-vs-dissipated split.

## STANDING DISCIPLINE (the harness — measurement-validity catches)
Same species each time: a chosen observable/knob whose confound mimicked the signal, fixed by switching to the distribution-robust
observable or a should-be-invariant control. 1. within-shell null → full-shuffle. 2. K-FAC PSD-blind → Lanczos λ_min. 3. var-of-squared
→ MSD. 4. recovery-alone → 2D recovery×importance. 5. "B retained" trivial → magnitude control. 6. (2026-06) toggle magnitude side-effect
→ D−A-in-lockstep control. 7. M3 deposit-PC confound → arm-D contrast. 8. basis-precheck gate-2 inversion → K-sweep + concentration=good.
9. rung-3 fluctuation → reproduce-known hard-stop anchor (same-script re-run confirmed).
- NEVER bank a script's printed verdict without the corroboration check. Per-direction objects in basis-independent decompositions.
  Separate exciting-prospect from proven-result. Pre-registration caught the theory's intuition wrong many times while keeping the right half.

## G5→B−D MEDIATION PRE-REG (LOCKED 2026-06-22, before opt1_3b + pythia1b B−D land — Wilson)
SCOPE OF CURRENT CLAIM (do not overstate): G5 is FAMILY-ORGANIZED — locked, hole-robust (parallel agent RMD-SRC: G5 family-grouping
PAYS_STRICT, present p=.080, imputed p=.015–.025, survives worst-case PY-high imputation of the 1.3b slot; only feature of 6; G1/G3/G6
size-driven/COST, G4 collapses under hostile imputation, G2 non-specific). Real opt1_3b G5=0.430 (OPT-leaning, sharpens not overturns).
B−D is NOT YET de-confounded from size — it exists at one band (350/410) until opt1_3b+pythia1b land; family & size CO-MOVE across the
6 size cells. The within-OPT G5↑(0.344→0.410)/B−D↓(+0.110→+0.055) trend is 2 points (125,350)+pending = the exact configuration where
size & family are still confounded ON THE OUTCOME even though G5 is de-confounded from size on the geometry. STRONGEST HONEST STATEMENT:
"G5 family-organized; B−D co-occurs with G5 at the one band where both exist, + suggestive within-OPT size trend." Partial-mediation
(G5 carries family/size, depth=residual) is the FRAME THE MEDIATION TESTS, not a current result.
THE BREAKER CELL = pythia1b B−D (report this de-confound EXPLICITLY in the verdict):
  • stays flat-negative (~−0.005, like 160m/410m) → protection is FAMILY (Pythia unprotected regardless of size; OPT within-size trend
    = "family-protected-anyway", size a within-family modulation).
  • moves toward zero/positive → protection partly SIZE-driven → family/size attribution on the outcome does NOT cleanly separate.
  opt1_3b B−D (G5=0.430, highest OPT) → predicted LOWEST OPT B−D if within-OPT G5→B−D real; if it rises instead, the trend was noise.
POWER FLOOR (pre-registered): n=6 size cells + handful valid depth cells is SMALL for a faceted mediation. Before claiming "G5 partially
mediates," the mediation MUST show it can SEPARATE the family/size component from the depth component. If underpowered to distinguish
(a) "G5 partially mediates; depth is a SEPARATE mechanism" from (b) "G5 fully mediates the non-depth part; depth is NOISE" → verdict is
"UNDERPOWERED — cannot distinguish (a) vs (b)", NOT "partial mediation." Pre-register the discriminating statistic + its min-detectable
effect at n-available BEFORE running.
DEPTH MAGNITUDES NOT BANKABLE: d12/d18/d24 = n=3, surgery-artifact-exposed. "B−D depth-built" is DIRECTIONALLY airtight (monotone
−0.039→+0.014→+0.055) but FORMALLY underpowered — use depth for the SIGN of the residual, never a coefficient/magnitude. d6 VOID (A_forget
0.114<0.15) — excluded from B−D, geometry still valid.
DON'T POOL: depth cells vs size cells measure different things → FACETED (family/size/depth), never pooled into one unit column.
Carrier-type split fc1/fc2/attn per parallel agent's dissociation (fc1 rotation depth-robust = G5-like; fc1/fc2 contraction depth-built).
DIVISION OF LABOR (agreed 2026-06-22): geometry-side grouping = parallel agent (DONE, G5 locked).
  FAMILY/SIZE side of G5→B−D mediation = ME (6 size cells, model-level B−D ~ G5 median, pythia1b=breaker, family-vs-size attribution
  reported explicitly) once opt1_3b+pythia1b B−D land. DEPTH side = parallel agent (fc1 funnel/seed-deep clock → depth residual of B−D on
  d6/12/18/24 surgery cells). NO SHARED UNITS — size ladder mine, depth cells theirs. Two-mechanism = confirmed iff BOTH sides land.
  Outcome is per-MODEL (retention contrast); carrier-type split on the OUTCOME would need a carrier-ablated-D run (not wired; deferred).
SIGN-ARCHITECTURE CRITERION (LOCKED 2026-06-22, mirrors parallel agent's depth disqualifier): G5 mediates an axis ONLY if right-signed
+ dynamic-range-sufficient. Family axis (OPT low-G5/+B−D vs Pythia high-G5/null) = G5↓→B−D↑; within-OPT size (G5 0.344→0.410, B−D
+0.110→+0.055) = ALSO G5↓→B−D↑ → family & size SIGN-CONSISTENT = precondition for G5 to jointly mediate them. Depth axis = G5↑→B−D↑
(SIGN-REVERSED) + G5 depth-swing 0.043 vs family-swing 0.18 (~4× too small) → G5 DISQUALIFIED as depth carrier (parallel agent verdict,
gateA/mediation_depth.json; NOT the n=3 Spearman +1.0 trap — variance-partition + sign argument). MY mediation MUST: (i) confirm G5's
family/size law right-signed + range-sufficient; (ii) treat ANY sign-reversal on my axis as a DISQUALIFIER not a weaker fit; (iii) pythia1b
tests whether the sign-law holds at high size or size breaks off. Depth-side per-carrier gradient is BROAD (fc1 +1.0/fc2 +0.70/attn +0.83-0.97)
NOT fc1-exclusive → carrier-split stays DESCRIPTIVE. Two-mechanism CONFIRMED iff my α (G5 family/size, right-signed) + their β (depth) hold.
RECIPE-COMMON-CAUSE REFINEMENT (LOCKED 2026-06-22, from parallel agent Thread-2 matched-shape control, VERDICT_THREAD2_ASPECT):
  Matched-shape OPT/Pythia pairs (768/12/3072=opt125m/pythia160m gap G5 -0.180/B-D +0.115; 1024/24/4096=opt350m/pythia410m -0.095/+0.060)
  -> BOTH G5 & B-D separate by family at BYTE-IDENTICAL architecture. Dissociation = PRETRAINING RECIPE/DATA, NOT geometry/shape/width
  (also resolves saddle-vs-basin = pretraining). CONSEQUENCE FOR MY MEDIATION: recipe is a COMMON CAUSE of G5 AND B-D, so the FAMILY offset
  CANNOT distinguish "G5 mediates family->B-D" from "recipe->G5 & recipe->B-D in parallel (G5 = marker)" -> family offset is CO-OCCURRENCE/
  common-cause, NEVER causal however clean. The CAUSAL-LEANING test = WITHIN-FAMILY size slope (recipe held constant): within OPT, G5-up as
  B-D-down is recipe-controlled evidence for G5->B-D. So alpha SPLITS: report (family offset = common-cause) vs (within-family slope =
  causal-leaning) SEPARATELY; never let the family offset's size inflate a causal claim. Secondary recipe x width G5 modulation
  (OPT-rises/Pythia-falls d_model, ~half family offset, n=3/fam) is NOT the headline; width-surgery DEFERRED (contingent on within-family
  slope becoming the crux). DECISION: parallel agent option (a) STOP width-surgery -- agreed; matched-shape settled the headline.

## RECIPE-WITHIN-PYTHIA: POWERED NULL + DESIGNED OPEN TEST (2026-06-22)
Q: does varying Pythia pretraining recipe (dedup data / training-maturity) at FIXED arch + matched w_A move the non-destructive floor (B-D~=0)?
NULL (powered, banked): recipe moves SEVERITY (A_forget) + GEOMETRY (K_E 74->3 across maturity) HARD but does NOT flip the MECHANISM. Matched-Af
  cells at floor: pythia160m std B-D=-0.005, step8k +0.006 (K_E=74 broad core yet floor = GEOMETRY DISSOCIATES FROM MECHANISM, the anchor),
  pythia410m std -0.006. step16k/50k EXCLUDED VOID/low-forget (Af .147/.157). Vacuums corroborate: Pythia410m D_K-A FLAT 0.082->0.085 (100% out-of-V,
  room-based), and Pythia-flat de-confounds OPT's decline (0.096->0.044) since both share lossdrop rise ~9->15.5 but only OPT declines -> OPT in-V real.
LIVE THREAD (DESIGNED, not loose end): dedup consistent-direction lean B-D +0.027(160m)/+0.021(410m) SAME SIGN both sizes BUT both forget ~3pp less
  -> severity-confounded, sub-2sigma each. KILL: not separable from A_forget difference yet.
  >>> OPEN TEST (specific design): re-run pythia{160m,410m} dedup vs std at MATCHED A_forget (tune childhood steps/lr per cell to equalize Af) + more
      seeds (N>=50) -> splits "engages the map" from "severity confound twice same direction." Fire only this; do NOT absorb the lean into the null.
BRACKETING: dissociation NOT reproduced by recipe-within-Pythia (coarser than dedup/maturity) + Thread-2 (survives byte-identical shape) => driver
  bracketed both sides: NOT shape, NOT within-family recipe knobs -> remaining = FULL cross-family pretraining pipeline (irreducible bundle). Levels
  ruled out, driver not isolated.

## DEDUP OPEN TEST = RESOLVED: KILLED (within reproducibility floor) 2026-06-22 overnight
Ran the designed severity-matched test (recipe_matched_lambda, ladder STEPS_DEPTH{180,250,350} N=40, std vs dedup at 160m+410m).
160m: at matched A_forget dedup>std (+0.055 gap, looked 2.7sigma). 410m: did NOT replicate -- dedup<std (opposite sign).
DECISIVE: std-410m recipe_matched B-D=+0.060+-0.017 vs the banked dispositive -0.006+-0.015 at IDENTICAL w_A (0.753) + identical
regime (steps/lr/stream/heldout/bf16 all equal) -> 0.066 gap between IDENTICAL runs = the B-D RUN-TO-RUN REPRODUCIBILITY FLOOR
(seed-set 30v40 + bf16 nondeterminism + lib drift; all noise, no regime effect). Floor (~0.05-0.066) >= dedup effect (0.027-0.055)
-> dedup lean is WITHIN NOISE. KILLED, not promoted. Null stands strengthened: recipe-within-Pythia (dedup AND maturity) does not
move the mechanism above the floor.
GENERAL CAUTION (banked): B-D effects <~0.05 at N<=40 are within the run-to-run floor; within-run SE UNDERSTATES it. OPT-350m +0.055
sits AT this floor too -- OPT survives only via converging-ladder (125m +0.110 >> floor, monotone) + orthogonal vacuums instrument
(OPT declines / Pythia flat). RULE: never trust a single-cell B-D near +-0.05; require multi-cell pattern + independent instrument.
TO ACTUALLY detect a true ~0.03 dedup effect: N>=150-200 with FIXED seeds across arms + deterministic kernels (fp32 or seeded bf16) to
push run-to-run var below the effect. Not worth it unless dedup becomes load-bearing for another reason.

## FLOOR-INDUCED RECLASSIFICATION of all program B-D claims (2026-06-22, Wilson — DO NOT SOFTEN)
The ~0.066 reproducibility floor (composition: bf16 non-determinism + seed-set 30v40 + library drift; measured from two nominally-
identical pythia410m runs, -0.006 vs +0.060, same w_A 0.753 + same regime) is now the program's QUANTITATIVE floor for single-cell B-D
trust at N<=40. It CALIBRATES the instrument and TIERS every B-D claim:
  TIER-A (survives, above floor OR multi-support):
    - OPT-125m B-D=+0.110 -- >> floor, standalone-trustworthy.
    - The OPT DISSOCIATION as a PATTERN: monotone ladder +0.110->+0.055->0.000 + vacuums (OPT D_K-A declines vs Pythia flat) +
      carrier-structure convergence. Survives as a PATTERN across cells+instruments, NOT as any single number.
  TIER-B (PROVISIONAL, single-cell <=~0.05 at N<=40 -- TEETH, do not soften):
    - ★ OPT-350m B-D=+0.055 standalone is PROVISIONAL -- it sits AT the floor. It survives ONLY via the ladder (125m +0.110 above floor,
      monotone), the vacuums instrument, and carrier-structure convergence -- NOT as a standalone number. In the writeup it must be
      presented as ladder/instrument-supported, never as an independent +0.055 measurement.
    - DEPTH-LADDER magnitudes (d12 -0.039 / d18 +0.014 / d24 +0.055) are ALL within the floor -> DIRECTION/monotone only, magnitudes
      NOT bankable (already flagged; now floor-quantified).
  TIER-C (UNAFFECTED -- nulls):
    - pythia160m/410m/1b ~=0, opt1.3b 0.000, step8k +0.006 -- these are floor-CONSISTENT-WITH-ZERO. The floor IS their uncertainty and
      0 is inside it, so the NULL holds robustly. The floor threatens POSITIVE near-floor claims, NOT null claims. Nulls stand.
  CONSEQUENCE: any future single-cell B-D positive near +-0.05 is provisional until N>=150-200 + fixed seeds + deterministic kernels,
  OR corroborated by an orthogonal instrument / multi-cell pattern. The dedup test that produced this floor converted a dead result
  into a calibration of the whole B-D instrument -- the most valuable output of the recipe sub-program.
