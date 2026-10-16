# CF Forgetting-Flow — Consolidation (2026-06-19)

> ### ⚠️ STALENESS BANNER (added 2026-07-14) — READ `D:/IC_experiments/STATUS_RECONCILIATION_2026_07_14.md` FIRST
> A staleness audit found this tree's status prose is systematically written BEFORE the run and never revisited.
> **This file is the highest-impact stale doc in the arc** and is ~1 day behind the confound break it summarizes:
> - **`:856` "why BLOOM erodes (ALiBi? corpus?) — OPEN" is STALE** → answered by `VERDICT_PROBE14_15_CONFOUND_BREAK` (07-10):
>   it is a **CONJUNCTION** (ALiBi × multilingual), not one knob.
> - **PREREG `Status: ... NOT fired` footers across this arc are wrong 10/10** (July preregs). Probes 10/11/12/13/14/15/16/17/18
>   have all FIRED and have artifacts in `gateA/`.
> - **Pythia B−D at N=30 is CLOSED** (`gateA/pythia_dispositive.json`: B−D = −0.006±0.015, CI excludes OPT's +0.055 → the
>   cross-arch dissociation is REAL at matched power). Do not re-run it.
>
> **TRUST ARTIFACTS (`gateA/*.json`, `cloud/*.log`), NOT THIS PROSE.** VERDICT_*.md files are reliable; this doc and the PREREG
> footers are not. Genuinely-open list lives in the reconciliation doc (headline: Probe-3 dose + Probe-4 dual, still unfired).

Working consolidation of the forgetting-as-flow lane. Pythia-410m, task-A = SST-2 (sentiment),
task-Bs = AG-News / Yelp / Subj. All measurements per-layer, never pooled.

> **★ LATEST (2026-07-09): Probe-14/15 confound break** — `VERDICT_PROBE14_15_CONFOUND_BREAK_2026_07_09.md`. BLOOM's dose-response
> erosion (Probe-13) is a **CONJUNCTION** (ALiBi × multilingual), not one knob: Falcon-rw-1b (ALiBi+English) and XGLM (sinusoidal+
> multilingual) both PROTECT; BLOOM (both) is the unique eroder → interaction or its post-embed LayerNorm. `frac_child_in_V`
> mechanism REFUTED (all cells ~0.2–0.46 energy in V; direction not energy is the lever). Threads: (A) cosine directional metric,
> (B) application → the Librarian reranker (already has `b2_cf` inV-vs-swap instrumentation). Lit-check in flight.

## The headline mechanism (3 nested layers)

Forgetting does not just load the loudest directions, and it does not aim per-pair. It has a
**universal, beyond-curvature, aimed component** — a fixed fragile subspace of task-A that any of
these task-Bs disturbs the same way.

| layer | what | measure | status |
|---|---|---|---|
| 1. Curvature backbone (~70%) | forgetting loads task-A's loudest Fisher directions | deflation M1 = 0.70 (core ≈ top-curvature) | **known object** (Gur-Ari / Wu / Papyan) |
| 2. Beyond-curvature aimed (~30%) | the flow's seed tracks *which* directions are contested, controlling for how loud | per-layer partial spearman(seed, contest \| λ) = **+0.45, 72/72 layers** | **novel**, survives curvature control |
| 3. Universal across task-Bs | layer-2 component is a *fixed* subspace every task-B shares (not per-pair) | cross-pair partial: contest **+0.64**, seed **+0.63**, 216/216 estimates, both objects agree | **mechanism, not description** |

Layer 1 is what the deflation found and banked as "not ours." Layers 2–3 reopen and sharpen it:
the core ≈ curvature claim still holds *for the backbone*, but on top of it sits a universal aimed
subspace the landscape literature does not predict.

### Where in DEPTH the universal core lives (SPIRAL sub-program, parallel agent, 2026-06-20)

The layer-3 universal core (cross-pair +0.64 — the fixed subspace every task-B fights task-A over) had no
depth address; it was measured per-layer-then-summarized. The SPIRAL sub-program located it. Re-analyzing the
existing per-layer artifacts (breadth `E` 9 task-Bs × 128 dirs; crosspair seed 4 task-Bs × 64 dirs; never pooled),
the neighborhood map is **LAYERED BY DEPTH as a smooth fine→coarse funnel**: early layers hold a HIGH-dimensional,
task-SEPARATED map (task-Bs occupy distinct directions); deep layers COLLAPSE every task-B onto a SHARED, low-dim
subspace. Curvature-controlled effective-rank of the cross-task stack falls ρ=−0.72, z_shuffle=−3.5 (attention
hardest; 2 substrates agree), monotone (SPIRAL-2: a smooth funnel, NOT recursive/fractal — folded-residual descent
and re-expansion both null vs phase-randomized surrogate). **The deep collapse IS the universal contested core,
located in depth.** Early layers keep tasks separate (low interference); deep layers smash them onto the shared
subspace = where the +0.64 universality lives. The map didn't *route* forgetting or *predict where it seeds* — it
told you *where the merge happens*. (SPIRAL-1's pre-registered direction was early-broad→late-specific; measured the
opposite — the wrong-and-kept-the-finding pre-reg killed the prior, not the result.)

**But the depth-structure is REAL-BUT-INERT for the mechanism** (SPIRAL-3, both hard predictions refuted clean):
seed-concentrates-deep is FALSE (ρ(seed,depth)≈0, |z|<0.8 across 3 basis-consistent seed measures — forgetting does
NOT seed preferentially in the collapsed layers); carrier-split-as-differential-collapse is unresolved (n=3 carriers;
map-prediction measure orders attn>mlp, contest measure anti-orders — genuinely ambiguous, not forced). A flagged-
not-banked bonus: seed co-locates with the *separated/high-dim* layers (ρ≈+0.50, opposite sign to prediction) —
possibly activity-scale-driven, held as needs-more.

**This is the SAME dissociation species as #2b.** #2b: the flat neighborhood map is real but causally inert *for
protection*. SPIRAL: the depth-funnel map is real but causally inert *for seed-location / carrier-split*. Two
independent confirmations that **the map — at every structural level measured (flat neighborhood, depth-funnel) — is
representation geometry that does NOT control the forgetting mechanism.** The mechanism is driven by
non-destructiveness (#2b) and curvature (the backbone), not by the map's structure. Artifacts
`gateA/spiral{1,2,3}.{py,json}`.

**SPIRAL-4 (the twist — it really is a spiral, not just a funnel/cone):** the funnel only measures the contested subspace's
RADIUS per depth. Measuring its ORIENTATION (per-layer weight-space eigenvectors, `taskA_variation.npz` fragile__sst2, 16-dim
subspaces in a common per-carrier space; guard reproduces Cut-1 G(sst,ag)=0.13), the contested subspace ROTATES coherently with
depth: consecutive-layer overlap 0.53 (qkv) decays perfectly monotone (ρ_gap=−1.00, shuffle-killed z−2.15) to ORTHOGONAL at the
stack ends, one-directional (drift away from L0 ρ−0.63, toward L23 ρ+0.95), nearest-orientation-neighbor = depth-neighbor 79% vs
9% chance (z+10.7). So the contested core both CONTRACTS and fully REORIENTS top→bottom = a genuine spiral. Carrier-split crisp:
qkv+up-proj spiral, down-proj incoherent (matches the dye carrier-split). NUANCE: the twist is curvature-GENERAL not
contested-specific — topcurv spirals identically and fragile/topcurv rotation profiles correlate ρ=0.90, so the contested
subspace rides the curvature backbone's depth-rotation (fragile≈topcurv, G=0.72). Load-bearingness of the twist is OPEN.
Artifacts `gateA/spiral4.{py,json}`.

**SPIRAL-5 (is the TWIST the load-bearing half? — NO, clean close).** SPIRAL-3 showed the CONTRACTION (funnel) doesn't
explain seed-location/carrier-split; the twist had a better prior (it reproduced the carrier-split crisply). Tested whether
rotation predicts seed-location/universality BEYOND curvature (the decisive control — the twist is curvature-general). Guard:
reproduced SPIRAL-4 consec 0.534 + Cut-1 G 0.126. (B) seed-location: for the carriers that actually twist (qkv/h4h) rotation-rate
adds NOTHING beyond curvature (partial +0.02 / −0.05); only the non-twisting 4hh shows a curvature-adjacent effect (+0.58) that
cannot be the twist. NEGATIVE. (A) carrier-split/universality: a pooled n=72 partial looked positive (+0.29, z+2.5) BUT
decomposing WITHIN carrier killed it (qkv −0.19, h4h −0.01, 4hh −0.29 — all ≤0) → it was a between-carrier POOLING ARTIFACT
(same ecological-fallacy species the program's per-layer law guards against, here manufacturing a positive). Between-carrier the
rotation split (qkv/h4h coherent vs 4hh incoherent) does NOT map onto universality under any measure (cross-task: 4hh≈h4h;
crosspair-contest: 4hh HIGHEST = anti-correlated with coherence) — the same n=3 measure-dependent wall SPIRAL-3 hit.
**VERDICT: the twist is REAL-BUT-INERT too.** Both halves of the depth-geometry — contraction (funnel) AND twist (spiral) — are
real representation geometry, both curvature-ridden, NEITHER load-bearing for the mechanism. **Depth-geometry question CLOSED:
the mechanism lives in non-destructiveness (#2b) + curvature (the backbone) + task-A sharing, NOT in the depth-structure (neither
its radius nor its orientation).** Artifacts `gateA/spiral5.{py,json}`.

**#3 CASCADED-BROAD CONSTRUCTION (2026-06-20) — closes the last door: the depth-structure is inert even when CONSTRUCTED.**
SPIRAL-3/5 showed the OBSERVED depth-geometry is inert; #2b's lesson was that CONSTRUCTION can be causal where observation
isn't. So we built the funnel deliberately, band-by-band in depth-order (broad@early→mid→deep), vs all-at-once whole-net broad
(#2's arm B), vs reverse-order (anti-cascade). Lambda A100, 4 arms × 7 paired seeds. Retention A/B/C/D = 0.527/0.653/0.610/0.630.
**C−B = −0.043 ± 0.034 (null, C even nominally lower); C−D = −0.020 (no direction effect).** Cascaded childhood still protects
over none (C−A +0.083, D−A +0.103, both clear) but NOT better than all-at-once — and B−A reproduced #2's +0.126 exactly (harness
valid). **So the #2b construction-asymmetry does NOT extend to depth-ORDERING: what protects is the breadth/non-destructiveness
itself, not how it's distributed across depth or in what order — breadth is breadth.** This CLOSES "is ANY form of the
depth-structure causal?" → NO (inert observed AND inert constructed). The whole spiral sub-program lands: a true contract+twist
spiral that is real representation geometry at every level and load-bearing at none; the mechanism is non-destructiveness +
curvature + sharing. Artifacts `gateA/dev_experiment3.{json,npz}`.

### POKE TEST (2026-06-20) — pre-loading the fragile subspace: SINK or RESOURCE? → NEITHER (no fragile-specific effect)
The program characterized the fragile/contested core but never POKED it. This pre-loads structure into it before task-B and
watches the surge. **Construction crux:** the naive matched-magnitude poke is CONFOUNDED — fragile is ~10⁶× stiffer (λ), so a
magnitude-matched weight poke is non-destructive on controls but catastrophic on fragile → B's "SINK" would be a trivial damage
artifact (the program's fragile=stiffest obstruction). So ran BOTH constructible designs on Pythia (the antagonistic case),
arms A/B/C/D (no-poke / fragile / random / complement) × 7 seeds, held-out subj. Guards all pass: arm A forgets 0.214; **poke
non-destructive (postpoke-B 0.744/0.745 ≈ pre 0.750 — so any effect is NOT a destruction confound)**; task-B not starved.
- **Design 1 — functional occupy (non-destructive, extra task-A training kept in the target subspace): INERT.** B−A = −0.001 ±
  0.034 (ns); not fragile-specific (B−C = −0.068 ns, if anything random-occupy nominally higher). Occupying the fragile core
  with task-A-consistent structure changes forgetting NOT AT ALL.
- **Design 2 — damage-matched perturb (δW into subspace, α tuned to equal Δtask-A-loss=0.05): weak SINK but GENERIC, not
  fragile-specific.** B−A = −0.039 ± 0.017 (sig, retains less) BUT B−C = −0.019 (ns) and complement sinks equally (D−A = −0.041
  sig) → any matched-damage perturbation costs ~0.02–0.04 retention regardless of WHERE; not a fragile-channel-priming effect.
  α B/C/D = 0.036/0.079/0.259 → fragile most damage-efficient (stiffest), but the realized equal-damage α ratio is only ~7×
  (a random perturbation samples the whole spectrum, not just λ_subspace — so the "10⁶×" is the raw-λ gap, not the poke gap).
- **VERDICT: the fragile subspace is NEITHER a SINK you can prime NOR a RESOURCE you can fill — its leak-behavior is independent
  of pre-load state** (occupy inert; perturb generic-not-specific). Consistent with displacement-not-accumulation and the
  whole real-but-inert family: the contested core is real and characterized but forgetting is NOT manipulable by pre-loading it.
  Matches the pre-registered Pythia (antagonistic) expectation = SINK/null, not deflect.
  Artifacts `gateA/poke_test.{json,npz}` + `cloud/poke_test_lambda.py`.

### POKE TEST on OPT (2026-06-21) — the cross-arch split: NO deflect (pre-reg refuted), regime-confounded sink-hint
Pre-registered: RESOURCE/deflect (B>A) more likely on OPT (orthogonal core, deposits survive) than Pythia. **Run 1 VOID**
(STEPS_DEPTH/LR=200/5e-5 from the Pythia recipe collapsed OPT-350m: arm A SST + subj BOTH to chance, degenerate — no dynamic
range; caught by reading per-seed, not the auto-verdict). **Run 2** added a depth-severity precheck (sweep LR×steps, pick a
readable arm-A window). Precheck found NO clean window in [0.58,0.72]×subj≥0.70 (settings clustered >0.72 or <0.58) → fell back
to lr=2e-5/steps=200 (arm-A SST 0.674 forgot 0.121, subj 0.88), flagged `no_clean_window`, high seed variance (one floored seed).
- **D1 occupy (headline) B−A = −0.078 ± 0.063 → ns (no deflect; nominally SINK-direction, NOT the predicted B>A).** RESOURCE
  refuted on OPT. BUT **fragile-occupy is fragile-specifically the worst: B−C = −0.125 SIG, B−D = −0.099 SIG** (B 0.596 < random
  0.721, complement 0.695, no-poke 0.674) — a fragile-specific SINK signal ABSENT on Pythia (Pythia D1 B−C −0.068 ns), OPPOSITE
  to the deflect prediction.
- **REGIME CONFOUND (flagged, per the discipline):** OPT measured at a NON-matched, gentler regime (forgot 0.121 / subj 0.92)
  vs Pythia's (forgot 0.214 / subj 0.88, 250/5e-5). So the OPT-vs-Pythia comparison is QUALITATIVE DIRECTION only, not
  matched-magnitude: "Pythia poke inert in its regime; OPT poke shows no deflect (ns B−A, fragile-specific sink-hint vs controls)
  in a gentler unmatched regime." The sink-hint is regime-confounded + noisy (no clean window, floored seed, control C nominally
  *helps* which is unexplained) → HINT not banked.
- **VERDICT: the pre-registered RESOURCE/deflect split did NOT appear — no architecture shows the fragile subspace as a
  protective handle.** Real-but-inert-as-a-handle holds on both (Pythia clean-inert; OPT no-deflect). The lone non-null is a
  noisy, regime-confounded fragile-specific *sink* on OPT (wrong direction) → needs the matched-regime Pythia disambiguator
  (re-measure Pythia D1 at OPT's 2e-5/200 regime: stays inert → OPT sink is architectural; also sinks → it's regime) before any
  claim. Artifacts `gateA/poke_test_opt.{json,npz}` + `cloud/poke_test_opt_lambda.py`.

### POKE DISAMBIGUATOR (2026-06-21) — regime ruled out: the OPT sink is ARCHITECTURAL (sink, not deflect). POKE AXIS CLOSED.
Re-measured Pythia D1-occupy at OPT's MATCHED forget severity (forget-match precheck → lr=2e-5/200 → Pythia arm-A forget 0.115,
within ±0.03 of OPT's 0.121 → MATCHED, confound removed). **Pythia stays DEAD-FLAT INERT at the matched gentle regime:**
retention A/B/C/D = 0.636/0.630/0.618/0.625, **B−C = +0.012 ns** (B−A −0.006, B−D +0.005) — vs OPT's B−C = −0.125 SIG at the
same severity. Same pipeline/method/regime → clean null on Pythia, sink on OPT ⇒ **the OPT fragile-specific sink is
ARCHITECTURAL, not a gentle-forgetting regime artifact** (pre-registered outcome 1). **MECHANISM — the dissociation:** the
antagonism/toggle law (antagonistic core → deposits eroded; orthogonal → survive) controls *whether the poke has any effect*
(Pythia eroded → inert; OPT survives → effect present), BUT a surviving deposit in the shared core is a **SINK (compounds
interference), NOT a RESOURCE (deflection)** — the OPPOSITE of the pre-registered "deposits survive → occupy to deflect." So
"antagonism controls deposit-survival" SURVIVES, but "survival enables protective occupation" is FALSE. **RESOURCE / occupy-
don't-project is DEAD on both architectures.** Caveat: the OPT run itself was noisy (no clean window, one floored seed); the
clean Pythia null under the identical pipeline strengthens the architectural read, but a clean-window OPT replication would fully
nail it. **POKE-AXIS HEADLINE (closed): the fragile subspace is NOT a protective handle — pre-occupation cannot deflect the
surge on either architecture; on OPT it mildly back-fires (architectural sink). Pre-loading the contested core is at best inert,
at worst a sink.** Artifacts `gateA/poke_pythia_matched.{json,npz}` + `cloud/poke_pythia_matched_lambda.py`.

## The measurement arc (and the catch that made it real)

1. **seed↔core overlap (aggregated)** → looked "curvature-mediated": index overlap 0.75 but the
   curvature-controlled partial collapsed to +0.18 (z+1.4, n.s.). Reported as PARTIAL.
2. **Per-layer re-run** → **flipped to BEYOND-CURVATURE**: partial +0.45, every one of 72 layers
   positive (t+29). The +0.18 was a **pooling / ecological-fallacy artifact** — layer-averaging
   contest and seed (where rank *j* is a different physical direction in each layer) blurred the
   per-layer pairing toward the shared curvature trend. Pair each layer's seed with *its own*
   contest and the structure is overwhelming. The aggregate also faked "qkv leaks 43% outside band /
   50% on-core" → per-layer truth is **10% out / 87% on-core**.
3. **Cross-pair (this run)** → **UNIVERSAL**: stripping curvature, every task-B pair still aligns at
   +0.64 (contest) / +0.63 (seed), 216/216. Both "where they compete" and "where forgetting aims"
   give the same answer.

**Law banked: per-layer, never pooled.** Pooling across layers manufactured a false null once and
would have here too.

## Why each verdict is trustworthy (guards)

Before reading any verdict, reproduce a known quantity:
- `align = 0.0` exact — recomputed Fisher eigenvalues == deflation's saved `lam__` → identical basis.
- `seedcheck = +1.000` — layer-mean of the per-layer seed == dye1's stored aggregated seed.
- `mean-contest reproduce = +1.000` — mean over the 3 task-Bs' contests == deflation's saved mean.

Same guard logic that made the per-layer flip credible. Bit-exact reproduction → the new resolution
is the same underlying object, just un-pooled.

## Corroboration discipline (banked)

- **Grassmann is vacuous** when both objects share one eigenbasis (axis-aligned subsets → principal
  angles 0/90 → Grassmann ≡ index overlap). Don't fake an "independent Grassmann that agrees."
- Substitute three non-identical statistics: **(a)** index overlap, **(b)** curvature-controlled
  partial correlation (the decisive one — only it separates "beyond curvature" from "both just
  loud"), **(c)** dissociation-set overlap (do pairs contest the same *off-curvature* directions?).
- **Effect size, not t.** At n=216 layer-estimates the t-stat is n-inflated (a trivial +0.05 effect
  gives t+6); read the partial *magnitude* against the within-pair benchmark (+0.45).

## Honest scope / caveats

- **3 task-Bs, all classification** (topic / sentiment / subjectivity, similar prompt framing).
  "Universal" = across these three. A structurally different task-B (generation, other domain) is the
  sharper test → **Option 3**.
- **Single net** (Pythia). Net-universality → **Option 2** (partly rode along in the GPT-2
  generalization, which held "same flow, different type").
- Profile is **strongly** shared (+0.64); the *exact* off-curvature direction sets overlap only
  **moderately** (dissociation Jaccard ~0.3) though significantly (z +2.6 … +5.2). So: universal in
  profile, partly-shared in precise low-curvature direction identity.

## Where this sits relative to the rest of the lane

- **Flow characterization (Pythia + GPT-2):** laminar, curvature-ordered, spectrum-set transport;
  "same flow, different type" survived the net axis (Wilson-ratified).
- **This consolidation** adds *where the flow seeds*: a universal beyond-curvature aimed subspace.
- The intervention shape Wilson sketched (condition/dedicate the contested core, timed to the seeding
  transient) is now **green-lit on aim** — the leak genuinely seeds at the contested directions, and
  those are shared across task-Bs. (Build is the user's call, not pursued here.)

## Artifacts

| file | what |
|---|---|
| `gateA/seed_core_overlap.{py,npz,json}` | aggregated seed↔core overlap (the curvature-mediated first read) |
| `gateA/perlayer_seed.npz` + `cloud/perlayer_seed_lambda.py` | per-layer seed; the BEYOND-CURVATURE flip (+0.45, 72/72) |
| `gateA/crosspair_residual.{npz,json}` + `cloud/crosspair_residual_lambda.py` | cross-pair residual; UNIVERSAL (+0.64/+0.63) |
| `gateA/deflation_artifacts.npz` | task-B-mean contest + curvature eigenvalues (layer 1) |
| launchers `cloud/lambda_launch_{perlayer,crosspair}.py` | Lambda A100; step-0 seed cheap, contest needs train-to-converge ×3; auto-terminate in finally |

## Option 3 — universality is GRADED, not binary (2026-06-20)

Adversarial test: does the universal subspace survive a *structurally different* task-B (vs the 3
classification anchors)? Anchors loaded from `crosspair_residual.npz` (only the new task-B trained →
the +0.64 anchor reference reproduces automatically; a new-task-B null can't be a pipeline artifact).
The **seed** is the clean read; the **contest** is convergence-confounded (see lesson below).

Seed code/gen-vs-anchor shared fraction **degrades monotonically with structural distance:**

| task-B contrast | seed vs anchor | shared frac | verdict |
|---|---|---|---|
| classification ↔ classification (opt 1) | +0.63 | 100% (baseline) | universal |
| classification ↔ NL-generation (Dolly) | +0.52 | 84% | universal-strong |
| classification ↔ **code-generation** | **+0.46** | **73%** (212/216 pos) | universal-strong (lower edge) |

All clear the bar (≥0.45) → the fragile subspace **is** a property of task-A and survives even a code
task-B — but **attenuates** with distance. **Carrier-resolved:** attention (qkv) stays *fully*
universal even for code (+0.67, anchor-level; dissociation Jaccard 0.50, z+5.4); the MLP carriers
erode (code up +0.37 / down +0.33). **The universal subspace lives mostly in attention.**

**Method lesson (banked):** for cross-objective-type comparison the **seed** (step-0 at w_A, identical
point, convergence-independent) is the clean object; the **contest** is **convergence-confounded** —
generation task-Bs don't memorize to loss 0.02 (Dolly 0.049 / code 0.104 vs anchors <0.02), so the
converged-Fisher contest isn't apples-to-apples. Contest came PARTIAL (55–61%) both times — a
convergence artifact, **not** family-specificity. Same confound species as the rest of the session
(an artifact mimicking the signal, fixed by switching to the confound-free object). Caught via the
seed/contest disagreement, not averaged away.

*Ops note:* humaneval/MBPP are unusable on the Lambda-Stack `datasets` version (systematic
`Invalid HF URI 'hf://datasets/<x>@<commit>/.huggingface…'` on script-style datasets). Use
plain-parquet community datasets (`iamtarun/python_code_instructions_18k_alpaca`, Dolly load fine).

## Cut 1 — vary task-A: the fragile subspace is task-A's, not the architecture's (2026-06-20)

The one variable never moved in the whole program was task-A (SST-2 throughout). Cut 1 moves it:
**SST-2 (sentiment) → AG (topic) → UCDP-GED type_of_violence (conflict-event typology).** All three
fine-tuned from Pythia *base* to clean basins (AG loss 0.017, UCDP 0.020 — UCDP cleared the freeze
gate, so no convergence confound). This is the **first cross-basis comparison**, so Grassmann is
finally non-vacuous (ambient weight-space vectors), and index overlap is now the ill-defined one.

Decisive question: is there a *fragility-specific architectural core* — attention directions invariant
across task-A beyond generic curvature overlap?

| comparison | attention fragile G | topcurv G | verdict |
|---|---|---|---|
| SST-2 vs AG | 0.13 (z+83) | 0.13 | — |
| SST-2 vs UCDP | 0.15 (z+98) | 0.15 | — |
| grand | **0.14** | **0.14** | **MOVES (~86%)** |

**Verdict: the attention fragile subspace MOVES — it's task-A's, not Pythia's.** Grassmann 0.14 is
above chance (z+90) but far below "stays put" (≥0.6) → ~86% of fragile directions relocate when task-A
changes. **Decisive control: fragile G ≈ topcurv G (0.14 ≈ 0.14)** → the ~14% shared is generic
top-curvature overlap, *not* fragility-specific. The architectural-fragility hypothesis is dead by the
exact measurement that would have revealed it. The control is a **ratio that cancels basis-mismatch**
(a basis artifact hits both legs equally), so the equality isn't an artifact. **No attention/MLP split**
(0.14 ≈ 0.10) → attention's specialness was task-B-invariance, *not* task-A-invariance — independent
corroboration. *(Wilson-ratified, survived 4 presses: moves-not-stays, control validity, convergence
closed, attention/MLP collapse corroborates.)*

**Bonus standalone (plasticity, not fragility):** a carrier's top-curvature subspace is ~86%
task-A-specific — fine-tuning remakes the curvature geometry per task; only ~2/16 directions are
architecturally fixed. Bank separately; it's about plasticity, may matter elsewhere.

**Implication:** the fragile subspace is *learned* structure (movable by how you train task-A), not
fixed architecture → the manipulation lever operates on influenceable structure = **more tractable.**

### Probe 2 (2026-06-21) — does architectural fragility hide in a NON-CURVATURE basis? NO (PROVISIONALLY_IRREDUCIBLE)
Cut-1 found fragile-G ≈ topcurv-G in the curvature basis (no architectural core). Probe 2 asks the basis-reframe: could
an architectural fragility core live in a basis the curvature view is blind to? Recast as a **B-falsifier** (float_module
methodology, family-lock SHA `18c6e9d7` verified intact). **The regression engine REFUSED_NO_FORM** (correct, per its M4
discipline): CF fragile subspaces are Grassmannian (no X→y panel; 3 units; candidate = basis not partition) → re-implemented
the *verdict-shape* natively (Grassmann alignment, leave-one-task-A-out, matched-rank-16 random null ≥60, search-corrected
across 2 bases, per-(layer,carrier) never pooled). Guard: reproduced Cut-1 fragile-G 0.114 ≈ topcurv-G 0.114 bit-stable.
- **Base-model basis** (genuinely non-curvature architectural prior = top-16 right-singular dirs of pre-FT W_base): **COSTS**
  — align 0.054 ≪ per-task-A curvature transfer 0.117 across all carriers. The architectural prior is NOT where fragility lives.
- **Cross-anchor-invariant basis** (curvature-family) looked **PAYS_STRICT** (0.129 > transfer 0.117, +0.013, beats random
  100% cells) — but the **decisive Cut-1-style control caught it as a FALSE POSITIVE:** cross-FRAGILE→fragile 0.1293 vs
  cross-TOPCURV→fragile 0.1284, fragile-specific lift **+0.0011 ± 0.0006 = NS** → the lift is **generic curvature**, not
  beyond-curvature fragility (the +0.013 over single-anchor transfer is a 2-anchor pooling effect, within the curvature family).
- **VERDICT: PROVISIONALLY_IRREDUCIBLE** (PROVISIONAL, n=3 anchors) — no tested basis carries architectural fragility beyond
  curvature; the genuinely non-curvature prior COSTS and the only "paying" basis pays via generic curvature. **Cut-1's "fragile
  is task-A's, not architectural" is BASIS-ROBUST for the tested bases → the basis-reframe is killed (provisionally).** Scope:
  n=3 task-A anchors on disk; more (code/Dolly/emotion as task-A) need GPU → escalation flag, out of CPU lane. Cross-lane note:
  float_module engine loaded clean + refused cleanly on form (no forced fit) — the B verdict-shape ports to NN-internals.
  Artifacts `gateA/probe2_grassmann.{json,npz}` + `probe2_grassmann_bfalsifier.py`.

### Probe 2 — THIRD challenger (Reynolds × harmonic, Thread 4, 2026-06-21): also generic curvature → Probe 2 HARDENS
The synthesis-doc's Reynolds-projector × harmonic-degree-filtration is the 3rd pre-registered challenger. Group declared
answer-blind: the ONLY architecturally-given discrete symmetry on a fragile-subspace *input* space is **S_H (MLP hidden-unit
permutation)**, acting on 4hh's input (4096); head-perm acts on the qkv *output* not input → no group on qkv/h4h (imposing one =
confound, refused). S_H's Reynolds projector → the 1-dim **all-ones invariant** (MLP mean-activation direction); filtration is
degenerate (invariant 1-dim, complement isotropic → no rich rank-16 basis). **Result:** the all-ones invariant STRONGLY loads
the fragile subspace (align 0.67, beats random 100% of 4hh cells) — but the **mandatory swap control kills the fragility-specific
reading: fragile−topcurv lift −0.0005 ± 0.0051 = NS → generic curvature** (the mean direction is high-curvature, equally in
topcurv). **Verdict: PAYS_NONSPECIFIC (generic curvature), NOT beyond-curvature architectural fragility.** (Same false-positive
species the swap control caught in challengers 1–2.) **PROBE 2 NOW HARDENS:** three pre-registered challengers — base-model
(COSTS), cross-anchor (PAYS_NONSPECIFIC), Reynolds×harmonic/S_H (PAYS_NONSPECIFIC) — ALL fail to find architectural fragility
beyond curvature. Cut-1's "fragile is task-A's, not architectural" is **basis-robust across all three named candidate bases →
the basis-reframe closes** (IRREDUCIBLE-proper for the tested bases; still n=3-PROVISIONAL on anchor count — more anchors=GPU).
Caveat: challenger 3 reaches only 4hh (rank-1 degenerate invariant); qkv/h4h afford no architectural group.
Artifacts `gateA/probe2_reynolds_challenger.json`. (Synthesis-doc joint experiment — qx+1 calibration + Hermite + room
predictor + free-convolution — fired separately to Lambda: `float_module/joint_experiment_lambda.py`.)

## The bounded headline (task-A axis closed)

> **A beyond-curvature aimed-forgetting subspace that is universal ACROSS task-B (even code, +0.46–0.63),
> SET BY task-A (moves ~86% when task-A changes), attention-carried, and graded with task-B distance.**
> It's about what a network *learns*, not what a network *is*.

## Control test — timed+targeted protection: INCONCLUSIVE, not confirmed (2026-06-20)

7-arm timing×targeting matrix (naive / fragile×{full,early,late} / randhi×early / randany×{early,full})
× {AG, code} × 3 seeds; GPM projection off the fragile attention subspace during a timing window;
retention = SST-2 held-out accuracy (pre 0.753). Thoroughness caught three confounds that block a clean
positive:

| arm | AG (forgets 0.17) | code (forgets 0.03) |
|---|---|---|
| naive | 0.582 | 0.712 |
| fragile×full (guard) | 0.688 | 0.769 |
| fragile×early | 0.640 | 0.751 |
| fragile×late | 0.554 | 0.717 |
| randhi×early | 0.619 | 0.690 |

1. **Targeting control structurally impossible** — the curvature-match *failed*: fragile mean-λ 9.44e-2 vs
   best nearest-λ non-fragile draw 1.28e-2 (7× less stiff). The fragile subspace **is** the stiffest ~16
   directions, so no equally-stiff non-fragile control exists → "fragile" can't be isolated from "stiff"
   when fragile = top-seed (consistent with deflation's fragile≈top-curvature). Press 1 unresolvable as designed.
2. **Timed-early insufficient where forgetting is real** — on AG, fragile×early (0.640) < fragile×full
   (0.688); need full-time. And early vs randhi-early (0.640 vs 0.619) overlap in error bars → not fragile-specific.
3. **code is a dead substrate** — forgets only 0.03, gradient mass not front-loaded (early 274 ≈ late 271,
   so the timing mechanism doesn't apply), random arms fell *below* naive. Its "HELD" is a small-signal artifact.

**Verdict: not confirmed.** The known lever reproduces (AG 0.582→0.688 full-time), but the **timed +
fragile-specific refinement does not survive a rigorous control.** Negative banked — it bounds what the
lever can claim. Redesign paths (only if pursued): (a) define fragile = curvature-*residual* so a
stiff-matched control is constructible; (b) high-forgetting task-Bs + ~8 seeds.

### Residual redesign — DONE (2026-06-20): CLEAN NEGATIVE, lever capped at curvature

Redesign path (a) run. The stiff-matched control is finally constructible (Grassmann(RES,CURV)=0.000;
λ_CURV=1.05e-1 vs λ_RES=5.96e-8 ≈ λ_randorthog=1.93e-8). AG (valid substrate, lever grips 0.595→0.722):
curvres 0.680 < curv 0.722 (−0.042, clears 2·SE — adding residual *hurts*), curvres ≈ currand 0.689 (not
specific), res-alone 0.635 ≈ naive (no protection). Triple-confirmed negative. **Mechanism (near-forced):**
λ_RES ≈ 6e-8 → the residual lives in *flat* task-A directions → you can't forget along a flat direction
(Δloss ≈ ½λΔ²), so protecting it is inert by construction. **Reconciles the arc:** the seed↔core +0.45
"beyond-curvature aim" is real in the seed's *aim* but aims into flat space — real for aim, inert for damage.
Forgetting-that-matters is entirely the curved directions, which the lever already covers. (Subj = VOID, lever
didn't grip.) **The manipulation lever is provably capped at curvature; the spool needs only the curvature core.**

## Breadth-pass v1+v2 (2026-06-20) — the neighborhood is a learnable map, with a real task-specific component

**Question (the path to #3 / seamstress):** can a map fit from a STREAM of task-Bs' step-0 seed geometry
(where each near-task pulls, per curvature direction, at SST-2 w_A — no training) predict a HELD-OUT task-B's
pull above chance and BEYOND the trivial curvature prior? Held-out is the whole control.

**Design:** 9 real task-Bs (agnews/yelp/subj/ucdp/code/emotion/dbpedia/imdb/tweet), leave-one-out map fit;
per-layer never pooled; curvature-controlled partial-spearman(map, seed | λ); seed-not-contest (convergence-clean).

- **v1** raw 0.83 vs shuffle 0.00, partial **0.66**, 72/72 layers. FLAGGED not banked: the only null was dirty
  (permuted-SST = sentiment-shaped, and it polluted the LOO map). The clean control wasn't clean.
- **v2** (tighten the floor, then bank): map fit = REAL only; nulls held-out only; added TRUE nulls sharing
  neither inputs nor format — **randtok** (synthetic random tokens = pure model gradient geometry), **nltext**
  (TinyStories LM, no classification cue).

**Result — the decomposition (the honest, bounded story):** real held-out partial **0.652** =
- **~0.40 generic model geometry** (randtok 0.401 — where this model's gradients land regardless of task),
- **+~0.16 language/format geometry** (nltext 0.561 over randtok),
- **+0.171 task-specific neighborhood LIFT** (real 0.652 − true-null floor 0.481, ≫ SE).

Task-specificity confirmed **twice, independently:** (1) real − floor = +0.171; (2) graded same-type 0.702 vs
diff-type 0.544 = +0.157. Consistency check: diff-type (0.544) ≈ nltext (0.561) → cross-*different*-type
prediction collapses to the language floor; *same*-type carries the real neighborhood signal. Attention carries
it harder (0.700 vs mlp 0.628 — every prior finding). nullperm 0.538 sanity-confirms it's sentiment-shaped.

**Verdict (v2):** the v1 0.66 was NOT generic — there's a real ~0.17 task-specific increment on a generic-geometry base.

**v3 — leave-one-TYPE-out (the decisive press: is that increment cross-type or just within-type near-duplicates?):**
Hold out an ENTIRE type, predict from OTHER-type members only. Result = **GRADED BY DISTANCE FROM TASK-A** (the honest
final shape, supersedes the "+0.17 flat" gloss):
- cross-type held-out **0.627** (vs LOO 0.652, language floor 0.561 → +0.065 aggregate) — does NOT collapse to floor.
- within-type DOES win (same-type has-sibling **0.743** vs cross-type has-sibling **0.650**, +0.093) — Wilson half-right.
- but cross-type generalization is REAL and **graded by structural distance from task-A (sentiment)**, per-task cross:
  agnews(topic) 0.721, dbpedia(topic) 0.689, subj 0.664, emotion 0.633, sentiment ~0.61 (ALL above language floor) →
  **code 0.582 ≈ floor, ucdp(violence-events) 0.509 BELOW floor.** The two alien task-KINDS get zero map benefit.
- attn 0.683 > mlp 0.598 cross (every prior finding).

**Synthesis:** the learnable neighborhood is **CENTERED ON TASK-A and DECAYS with structural distance** — strong within-type,
real-modest across the affect/text-classification family, collapsing to/below the language floor for alien task-kinds. This is
exactly the developmental shape: a childhood map helps with tasks NEAR what it saw; genuinely novel kinds still yank. It BOUNDS
#3 cleanly — allocation is decidable for the neighborhood, and a BROADER childhood (wider breadth stream) is what would extend
the neighborhood (the next experiment the bound points to). Artifacts `gateA/breadth_pass.{json,npz}` (v2) +
`gateA/breadth_typeout.{json,npz}` (v3) + `cloud/breadth_pass_lambda.py` (one script; type-out is current state; ~10 min/$0.30).

**Lit closed (full-text):** arXiv 2603.07415 (Cheng — info-theoretic + architectural CF, "context channel
capacity"; bundles a *named* null taxonomy: Hebbian-null, θ₀-memorizer, Sₙ barrier, impossibility triangle —
cite carefully if touching frozen-vs-learned/impossibility, but none is our subspace claim) and CSCL = Yoo 2024
(arXiv 2411.02460, Findings ACL 2025 — multilingual code-switch curriculum, no geometry/nulls; only adjacency is
it's a staged-curriculum remedy → distinguish any breadth-then-depth remedy from it). White space holds.

## Experiment #1 — breadth-extend (2026-06-20): does a broader childhood reach the alien kinds?

A/B in one run: predict the same held-out **code** & **ucdp** under the ORIG-9 stream vs an ENRICHED-14 stream (added
5 bridge kinds: math/science/nli toward code; news-category + yahoo-topics toward conflict-events).

**Auto-verdict said "HARD LIMIT" — REFUSED as confounded** (the built-in bridge-quality check caught it; same-species
catch as the pooling/aggregate ones):
- **No extension observed:** code 0.579→0.607 (Δ+0.028, ~1 SE), ucdp 0.515→0.487 (Δ−0.029). The Δs are noise-level.
- **Confound 1 (bridge quality):** the code-ward bridges land *down with code* (math 0.555, science 0.511, nli 0.590 —
  all ≤ code 0.607), not spanning toward it; the ucdp-ward "bridges" weren't genuine (newscat 0.734 is a *topic* task
  sitting with agnews, not a conflict-event intermediate). A bridge off the map can't extend it.
- **Confound 2 (moving floor):** language floor rose 0.561→0.639 (a broader, more diverse childhood makes the average map
  more language-generic → higher bar). "Below floor" is partly an artifact of comparing across stream compositions; Δ is robust.

**Clean positives — they FIRM UP v3:**
1. **Stability:** 5 new task-kinds barely moved the ORIG tasks (|Δ| ≤ 0.03). The centered-decaying map is a **stable property
   of task-A**, not an artifact of which task-Bs you streamed.
2. **Self-placement:** the 5 fresh, never-seen tasks **sorted themselves by distance from task-A** exactly as predicted —
   newscat 0.734 / yahoo 0.622 near (topic), snli 0.590 mid, math 0.555 / science 0.511 far (STEM, clustering *with* code).
3. **Cluster structure:** three regions — task-A's affect/topic neighborhood (0.62–0.72) | a separate STEM/symbolic cluster
   (math/science/**code** 0.51–0.61) | conflict-events lowest (ucdp 0.49). The STEM bridges **joined code's cluster** instead
   of pulling code into task-A's → suggestive that alien kinds live in **disconnected regions** (a cluster reading, NOT a
   proven wall).

**Status:** extensibility *not cleanly resolved* (couldn't supply a genuine spanning bridge; floor moved) — but #1 cleanly
**identifies the in-neighborhood tasks** (topic/subjectivity/emotion IN; STEM/conflict OUT), which de-risks #2. To nail
extensibility would need a fixed per-stream floor + a genuine spanning bridge (which may not exist among loadable text tasks —
itself informative). Artifacts `gateA/breadth_extend.{json,npz}` + `cloud/breadth_extend_lambda.py`.

## #2 — THE DEVELOPMENTAL EXPERIMENT (2026-06-20): CLAIM LANDS (in-neighborhood)

**A network given a childhood SLIVERS where one without it YANKS — and it's the map, not the mileage.** The terminal
prediction of the whole metaphor chain, confirmed in-neighborhood. 3 arms × 7 paired seeds, real training, no geometry.

- **Held-out depth task-B = subj** (forgot 0.263 in arm A; emotion excluded — only forgot 0.080 < 0.10).
- **Childhood stream = non-sentiment neighborhood** (agnews/dbpedia/yahoo/newscat/emotion) — *deliberately no sentiment
  datasets*, which would be task-A replay (CLS) rather than map-building. If childhood helps without rehearsing task-A, it's the map.
- **Result** (task-A = SST retention after depth, pre 0.753): A (no childhood) **0.527** · B (real childhood) **0.653** ·
  C (scrambled childhood) **0.538**.
  - **B − A = +0.126 ± 0.024 (~5 SE)** — childhood retains task-A better.
  - **B − C = +0.115 ± 0.025 (~4.6 SE)** — it's the **structure**, not the compute (C = matched compute/depth, labels shuffled).
  - task-B **not starved** (subj acc A 0.881 / B 0.861 / C 0.867) — genuine sliver (retain A *and* learn B), not rigidity.
- **Mechanism = the sliver itself:** the childhood made the depth phase **3× less destructive** to task-A (depth-damage
  A −0.226 vs B −0.069; C −0.128 in between). Traffic-guard: the street-map routes the new-task surge *around* task-A's main road.
- **Scramble verified clean by the loss:** real childhood learned structure (loss 9.50→0.79); scrambled **floored at
  9.69→2.33 ≈ log(#labels)** — could only predict the label marginal = structure destroyed exactly as intended.

**Honest caveats (banked, don't over-read):**
1. **Master-none partially violated** — real childhood final loss 0.79 = it *substantially learned* the stream (not a
   pristine "low-res survey"). Reframe "shallow childhood" → "**structured neighborhood pre-exposure**." BUT arm C matches
   compute/depth, so the decisive "structure-not-compute" claim holds regardless of how deep the childhood was.
2. **The scrambled control is noise-fitting** (itself somewhat destructive: C post-child SST 0.666 < B 0.722) — so part of
   B's edge is "real-structure training is less destructive than noise-fitting." That *is* the map helping, but a future
   non-destructive no-map control would tighten "map-specific."
3. One dissenting seed (seed 4: B<C); the paired stats clear it. Single net (Pythia-410m) / single task-A (SST) / single
   held-out (subj) / **in-neighborhood only** — silent on alien kinds. Option 2 (confirmatory net) still open.

Artifacts `gateA/dev_experiment.{json,npz}` + `cloud/dev_experiment_lambda.py`. (~24 min / ~$1.)

## #2b — non-destructive no-map control (2026-06-20): DEFLATION — the map does NOT route the surge

Built to isolate Mode 2 ("the map routes the phase-2 surge around task-A") from Mode 1 ("the map just leaves a better
non-destructive starting point"). #2's only no-map arm (C, scrambled) is *noise-fitting* = itself destructive, so B>C
conflated the two. Arm **D = same stream as B, but childhood gradients GPM-projected off task-A's top-128 curvature subspace
each step** → non-destructive AND no-map *by construction*, matched compute, same data. Re-ran all 4 arms × 7 paired seeds.

| arm | start (post-child) | phase-2 damage | final retention | mapfrac (writing into task-A subspace) |
|---|---|---|---|---|
| A no childhood | 0.753 | −0.227 | 0.527 | — |
| B real childhood (map) | 0.722 | **−0.070** | 0.653 | 0.341 |
| C scrambled (destructive) | 0.666 | −0.128 | 0.538 | 0.453 |
| **D non-destructive no-map** | 0.741 | **−0.070** | **0.670** | **0.018** |

- **B − D = −0.018 ± 0.018 → B ≈ D** (D nominally higher). The map adds **nothing**.
- **D − A = +0.144 ± 0.018 (~8 SE)** — a non-destructive *no-map* childhood helps hugely over none.
- **Phase-2 damage B = D = −0.070, identical** — D slivers exactly as well as B, with **no map**.
- **Guards all passed (valid, not void):** D non-destructive (0.741 ≥ B's 0.722), no-map (mapfrac 0.018 vs B's 0.341, ~19×
  less), trained (childhood loss dropped 15.3 in the complement — non-degenerate).

**Mechanism (from mapfrac):** protection tracks **not clobbering task-A's directions**, not map-building. C's noise-fitting
wrote the *most* into task-A's subspace (0.453 = its destructiveness); D wrote ~none (0.018, non-destructive); B built a map
(0.341). B and D protect **equally** → **the v3 map is real but causally INERT for forgetting-protection** (clean dissociation).

**Retraction & what stands:**
- **RETRACTED:** #2's "it's the map, not the compute" (B>C). B>C was **C's destructiveness** (noise-fitting clobbers task-A),
  not map evidence. Control for destructiveness (D) → the map advantage vanishes.
- **SURVIVES:** a non-destructive prior phase (map *or* map-free) leaves task-A ~3× more robust to subsequent depth.
- **DIES:** "the map routes the surge" (Mode 2). **The traffic-guard-with-a-street-map metaphor is deflated** — it's not
  route-knowledge that protects, it's that a non-destructive warm-up leaves task-A in a robust state.
- **NEW open question:** *why* does generic non-destructive training protect (basin flattening? pre-moved complement
  capacity?) — #2b kills the map reading, doesn't answer the sub-mechanism.
- Caveat: D could in principle route via a map *outside* top-128 (low-curvature tail), but that contradicts all prior
  structure-location work — parsimony says no.

Artifacts `gateA/dev_experiment2b.{json,npz}` + `cloud/dev_experiment2b_lambda.py` (~35 min / ~$1.4).

## #2c — mechanism: basin-flattening (M1) vs pre-moved-capacity (M2), 2026-06-20: LEANS PRE-MOVED CAPACITY

The *why* under the #2b survivor (non-destructive warm-up protects). Two measurements in a common frame (V = task-A top-16
curvature subspace at w_A). **Auto-verdict "PRE-MOVED CAPACITY (clean)" REFUSED** — one axis is broken; read both jointly.

- **M2 (redirection) — CLEAN, strong:** D's phase-2 depth Δw lands **less** in task-A's V16 than A's: `fracV16 A=0.306, D=0.214,
  paired A−D=+0.092 ± 0.010 (~9 SE)`, 5/5 seeds. The surge is **redirected away from task-A into the complement** = the
  pre-moved-capacity signature. Redirection is **inconsistent with *pure* basin-flattening** (which predicts *same* destination,
  less impact). Protection reproduced (D−A = +0.142, matches #2b); mapfrac D = 0.018 (guard holds).
- **M1 (flattening) — CONFOUNDED, uninterpretable as run:** I used the **empirical Fisher (Σ ggᵀ)** as the curvature proxy, but at
  w_D the model is *off* the SST minimum (acc 0.753→0.741), so SST gradients inflate → Fisher inflates. The ratio
  curvD16/A = **441** is **off-min gradient inflation, not basin-sharpening** (tell: B is *more* off-min than D yet ratio 199 < 441 —
  not even tracking degradation cleanly). The auto-label "unchanged ~1" was a binary-logic bug (only checked `<0.85`). Empirical
  Fisher **cannot** measure basin curvature off-min — need a **true Hessian** (HVP λ in V16 at w_A vs w_D).

**Verdict:** pure-flattening **disfavored**, pre-moved-capacity **supported**; whether flattening *co-contributes* (the "mixed" cell)
is **unresolved** pending the Hessian-based M1 redo (#2c-v2; HVP code now exists from rung-1). **Buildable direction supported:**
"pre-exercise the drain" (a capacity sink, *not* GPM) — pre-train the orthogonal complement so the depth surge has somewhere
non-destructive to flow. Artifacts `gateA/dev_experiment2c.{json,npz}` + `cloud/dev_experiment2c_lambda.py` (~22 min / ~$0.9).

## #3 reframed (Wilson's metaphor, 2026-06-20): the TRAFFIC GUARD, not the seamstress — [#2b deflated the MAP version; #2c REVIVES routing via CAPACITY]

**Update (#2c):** the *routing* idea revives — the surge genuinely **is** routed around task-A (M2). But the routing is **not** from a
street-map (neighborhood knowledge, dead since #2b); it's from a **pre-exercised drain** (the complement the childhood moved). So the
traffic guard doesn't route via *knowing the streets* — she routes because a **side-street was pre-cleared**. Map-knowledge stays dead;
capacity-routing is the live mechanism.

The arbiter (#3) is a **traffic guard after a Knicks win**, not a seamstress. Knicks win = the *surge* (the new task floods
in all at once — the yank's suddenness). The guard doesn't *build* anything (the seamstress implied growing a new
spool/adapter = allocation); she **routes existing flow in real time** so the streams don't pile up at the intersection =
**arbitration, not construction.** #3 reframes from *allocate new parameters* → *route the gradient flow already there,
around what's occupied.* The map = the street grid (the childhood); a guard can only route surges into neighborhoods she has
a map for (the v3 in-neighborhood bound). Roads = the curved/load-bearing directions; flat directions aren't streets, so
routing there does nothing (the residual negative). The contested core (antagonism −0.23, head-on) = the intersection she's
keeping from gridlocking. #2 is itself a traffic-guard test: zero new parameters — pure "does the street-map let the flow
route around task-A (sliver) instead of through it (yank)." **[UPDATE 2026-06-20: #2b DEFLATED the map-routing version of this
metaphor — D (non-destructive, no map) slivers as well as B (the map). The guard's STREET-MAP does NOT route the surge; what
protects is a non-destructive warm-up leaving task-A robust. The "Knicks-win surge" + "arbitration not construction" framing of #3
still stands, but the "routes via the neighborhood map" mechanism is dead. The metaphor needs another revision — what survives is
closer to "a non-disruptive prior flow settles the grid so the next surge disrupts task-A less," not "the guard reads a map to route."]**

## OPT-350m rung-3 — cross-architecture non-destructiveness (2026-06-20): MAP-INERT FAILS, the decomposition INVERTS

The #2/#2b A/B/C/D battery re-run on a 3rd family (OPT-350m; rungs 1-2 already passed there). ×7 paired seeds, held-out=subj
(arm-A forget 0.304 ≫ 0.10 dynamic range), pre_sst 0.817. **Guards all valid** (D non-destructive: post-child 0.783 ≥ B's 0.745,
mapfrac 0.015 ≪ 0.110; D genuinely trained: childhood loss 13.75→0.47 in the orthogonal complement; arm C scrambled floored at 2.26).

**I pre-registered the Pythia outcome (D−A large + B≈D, map inert) and it was WRONG.** The middle pre-committed branch fired.

| arm | retention (SST, pre 0.817) | vs A |
|---|---|---|
| A no childhood | 0.513 | — |
| C scrambled (destructive) | 0.554 | +0.041 |
| D non-destructive no-map | 0.555 | +0.041 |
| B real childhood (map) | **0.673** | **+0.160** |

- **B−A = +0.160 ± 0.028** (7/7 seeds) — slivers REPLICATE on a 3rd family. ✓
- **D−A = +0.041 ± 0.012** (7/7 seeds) — non-destructiveness protects, but **weakly**. The pre-registered falsifier (D−A≈0 → ties
  protection to antagonism) did **not** fire: protection survives even though OPT has no antagonism (rung-2: g_A·g_B0 ≈0). ✓
- **B−D = +0.118 ± 0.023** (7/7 seeds) — **the map is LOAD-BEARING on OPT** (✗ Pythia's B≈D map-inert).

**The decomposition inverted.** Real-childhood protection (+0.160) = non-destructiveness (+0.041, **26%**) + map/structure
(+0.118, **74%**). On Pythia it was non-destructiveness ≈ all, map ≈ 0.

**Why this is clean, not a confound.** B beats *both* controls by the same ~0.118: B−C = +0.119 (vs scrambled, matched compute) **and**
B−D = +0.118 (vs projected-out-of-task-A, matched compute). C has write-freedom but no structure; D has structure but no write-freedom;
**neither alone reaches B.** You need *both* structured content *and* freedom to deposit it into task-A's curvature subspace (B's
mapfrac 0.22) — that deposit *is* the map, and on OPT it's constructive rather than inert.

**Post-hoc hypothesis (flagged, not confirmed — needs its own test).** The rung-2 antagonism divergence *predicts* this one. Pythia has
head-on antagonism (−0.23) → a map deposited in the shared steep directions gets fought/eroded during depth → inert → non-destructiveness
(staying out) is the only lever. OPT has no antagonism (≈0, orthogonal) → the deposit *survives* → the map becomes load-bearing.
No-antagonism is exactly the condition under which writing a map into the shared subspace is safe. **Test: does toggling antagonism toggle
the map's load-bearingness?** Artifacts `gateA/dev_net2_opt.{json,npz}` + `cloud/{dev_net2_opt_lambda,lambda_launch_net2}.py` (~25 min / ~$0.8).

## RUNG-3.5 toggle test (2026-06-20): VOID / inconclusive — the knob confounds antagonism with perturbation magnitude

Tried to toggle the dial: depth-phase gradient surgery rotating each carrier's in-core (V = task-A top-128) gradient component to a
prescribed cosine c ∈ {+0.20 aligned, 0.0 orthogonal, −0.23 head-on} with the held-out task-A gradient, norm-preserving so *sharing*
stays fixed. Arms A/B/D × 7 seeds; childhood built once per seed, reused across levels; only the depth dynamics differ by antagonism.
Pre-registered: B−D decreases with antagonism (large at orthogonal, →0 at head-on); D−A antagonism-independent.

**The knob bit mechanically** — `post_cos` hit +0.200 / −0.000 / −0.230 exactly, sharing stable (0.52±0.03), `pre_cos ≈ −0.01`
re-confirmed OPT's natural antagonism ≈ 0. **Result:** B−D = +0.024±0.007 (c=+0.20) / −0.015±**0.047** (c=0.0) / −0.013±0.015 (c=−0.23) —
flat ~0. The script auto-called this a "clean kill." **I refused it — it's void, for two reasons:**

1. **Anchor failed.** At c=0.0, rung-3 with *natural* depth gave B−D = +0.118 (7/7 seeds). Here c=0.0 *surgery* gives −0.015±0.047 —
   doesn't reproduce, huge SE. Forcing every carrier *exactly* orthogonal is not OPT's natural near-orthogonal depth; the surgery wrecks
   the map signal even at the level meant to anchor the test.
2. **Smoking gun: D−A collapses in lockstep.** Non-destructiveness (D−A) is *map-independent* — yet it shows the same shape: alive at
   orthogonal (+0.152, even larger than rung-3's +0.041), dead at both extremes (+0.004, +0.034). If antagonism *specifically* eroded the
   map, B−D would shrink while D−A held flat. Instead **both** protections die at the extremes, where all arms collapse to the floor
   (retention ~0.49-0.53). The driver is **perturbation magnitude** — the rotation angle grows with |c − c_natural≈0|, and that magnitude
   crashes protection across the board — *not* antagonism.

**The script's own guards (cos_ok, share_ok) passed but don't catch a magnitude side-effect.** The reproduce-known anchor (not encoded as
a hard guard) plus the map-independent control (D−A) crashing in lockstep are what expose it. **Lesson: a knob that passes its own
mechanical guards can still be void via a side-effect the guards don't measure.**

**Verdict:** the unifying hypothesis (antagonism controls map-survival, rungs 2+3 = one law) is **neither confirmed nor killed** — it
stays a post-hoc hypothesis. **Redesign:** the knob must hold *perturbation magnitude constant* across levels (the current one doesn't).
The clean route is the spec's **synthetic task-B family** — construct task-B *data* whose natural gradient carries the prescribed cosine
(no surgery), so the manipulation magnitude is natural/constant and the c=0.0 level reproduces rung-3. Artifacts
`gateA/toggle_test_opt.{json,npz}` + `cloud/{toggle_test_opt_lambda,lambda_launch_toggle}.py` (~58 min / ~$1.9).

## Next (user's call)

- **Antagonism↔map toggle, REDESIGNED** (synthetic task-B family — cosine in the *data*, not gradient surgery) — the only clean way left
  to confirm-or-kill the rungs-2+3 unification; the surgery knob is void (magnitude confound, above).
- **RUNG-4** (#2c mechanism on OPT: basin-flattening M1 vs pre-moved-capacity M2) — the next ladder rung; gated, now unblocked.
- **Mechanism of Mode-1 protection** (the new open question) — WHY does a non-destructive prior phase (even map-free, arm D) leave
  task-A ~3× more robust to depth? Basin flattening (childhood lands task-A in a wider minimum)? Pre-moved orthogonal-complement
  capacity (depth finds room without overwriting task-A)? This is what #2b opened.
- **Master-none-enforced re-run** (cap childhood loss genuinely low-res) — still tightens "shallow childhood" framing (orthogonal to #2b).
- **Option 2** — different net (net-universality beyond GPT-2's qualitative hold). Confirmatory.
- **Broader-childhood spanning-bridge** — extensibility (still open from #1).
- The characterization program is **capped**: forgetting is task-A-set, attention-carried, universal
  across task-B, protectable full-time by the known lever — but *not* refinable by timed/fragile-specific
  protection without a redesign. Extend only on a redesign or a new axis.

---

# CONSOLIDATION UPDATE 2026-06-23 — localizing map load-bearingness (the DSI → core/shell arc)

The lane above characterized *that* a non-destructive prior phase makes task-A load-bearing (the ~30% aimed component,
Mode-1 protection) and that it dissociates across architecture (OPT-350m load-bearing, Pythia-410m inert; dead at 1B).
This block localizes the mechanism: **per-step** how the childhood deposit interacts with the phase-2 surge, and **where
in the deposit's subspace** the load-bearingness lives. Band = OPT-350m / Pythia-410m only. All per-(layer,carrier),
curvature-controlled, magnitude-matched, swap-gated, reproduce-anchored. Run-to-run floor ~0.066 (bank on within-run
paired differentials, demote absolute levels).

## Thread 1 — DSI swap-gate: load-bearingness is deposit-specific ABSORPTION, inertness is deposit-specific DEFLECTION  **[BANKED]**
Per-step deposit↔surge geometry (`cloud/dsi_lambda.py`, `VERDICT_DSI_2026_06_22.md`). On the faithful pre-reg gate
(real-B inV vs Bswap inV, cross-arm paired — floor cannot fake it):
- **OPT (load-bearing):** absorbing carriers v/k/fc1 pull the surge INTO V deposit-specifically (+0.07…+0.17 over a
  curvature-matched random deposit), deposit holds (surv ~0.98) = non-destructive sinks. Deflectors q/out/fc2 ≈ generic.
- **Pythia (inert):** every carrier pushes the surge OUT of V deposit-specifically and hard (−0.13…−0.30), yet protects
  nothing (B−D ≤ 0).
Both respond specifically to the real deposit, opposite sign and opposite consequence: **load-bearing where it absorbs,
inert where it only deflects.** (Two logged deviations: probe1.py's cosine `swap_gap` false-nulls — read inV; the Pythia
half of the pre-commit was wrong-signed, specific-deflect not generic.)

## Thread 2 — basisinv: the task-A curvature core is tiny, the deposit concentrates there  **[BANKED]**
`basisinv_*.json`: K_E90 ≈ 14 dims (OPT-350m) / ≈ 5–9 (Pythia-410m) hold 90% of task-A Fisher mass; the deposit is ~10×
enriched in the low-K core (enrich@K32 ≈ 5.7× OPT-1.3b, ≈ 9.7–10.8× Pythia), decaying to ~2× by K=256. Where the deposit
SITS = the curvature core.

## Thread 3 — vacuums: the rank read was confounded  **[SUPERSEDED by Thread 4]**
`vacuums_*.json` projected the childhood GRADIENT out of top-K during training → childhood learning (lossdrop) NOT matched
across K (OPT 6.66→15.41), confounding rank with learning-amount, co-directional with the OPT effect. Gave a (confounded)
"Pythia protection is out-of-V / K-invariant" read. Not bankable. → motivated Thread 4.

## Thread 4 — Probe 2 clean rank ablation: protection lives in the broad SHELL, not the core  **[BANKED, confound-killed]**
`cloud/probe2_rank_lambda.py`, `VERDICT_PROBE2_RANK_2026_06_23.md`, `PREREG_PROBE2_RANK_CLEAN_2026_06_23.md`. Childhood
run ONCE per seed, deposit truncated to top-k V-dims re-normalized to full in-V magnitude → **lossdrop identical across k
(OPT 6.68 / Pythia 8.78, flat) → confound dead by construction.** Anchors held (B−A k128: OPT +0.080/exp .089, Pythia
+0.062/exp .073). Rank ladder Bk−A:

| k | OPT | Pythia |
|---|---|---|
| 0 (out-of-V only) | +0.034 | +0.006 |
| core (14 / 9) | +0.021 (dip, hi-var) | +0.032 |
| 64 | +0.104 | +0.049 |
| 128 (full) | +0.080 | +0.062 |

- **Load-bearingness is NOT in the K_E core, both families.** Pythia: smooth monotone climb, core carries ~half, still
  rising at 128 → distributed across the shell. OPT: core is a weak high-variance dip, protection peaks ~k64 and saturates
  before 128 → carried by an intermediate (~top-32–64) shell.
- **basisinv enrichment is INCIDENTAL to protection:** the deposit concentrates in the core, but its load-bearing power
  comes from the broader shell. The Thread-2-vs-Thread-3 tension (tiny core vs load-in-tail) dissolves — different objects
  (where the deposit SITS vs where protection COMES FROM), now confound-free.
- **The confound-kill FLIPPED a result:** Pythia out-of-V deposit alone (k=0) protects ~nothing (+0.006); vacuums'
  "out-of-V" was the lossdrop confound. Pythia protection is in-V distributed. (OPT keeps real out-of-V protection,
  k0 +0.034 ≈ 40% of full — a genuine cross-arch difference.)
- **Core deposit-specificity (swap at core rank):** Pythia core IS deposit-specific (real +0.032 ≫ swap +0.006); OPT core
  is NOT (real +0.021 ≈ swap +0.031) — OPT's specificity (Thread 1, full-V) emerges only with more rank.

## The localized picture (cross-arch)
Load-bearing OPT: deposit sits in the curvature core but **concentrates its protection in a mid-rank Fisher shell**, where
specific carriers ABSORB the surge as non-destructive sinks (deposit-specific), plus a real out-of-V component (~40%);
overshoots the full deposit when energy is packed in that shell. Inert Pythia: deposit also sits in the core but its
protection is **thin, in-V, distributed across rank**, never concentrated; every carrier actively DEFLECTS the surge out
of V (deposit-specific) but that deflection buys no retention. The map is load-bearing where a mid-rank shell absorbs;
inert where the subspace only spreads and deflects.

## ★ MAGNITUDE CORRECTION 2026-06-25 — the floor battery re-sized the headline (`VERDICT_FLOOR_BATTERY_2026_06_25.md`)
A full per-cell B−D reproducibility-floor battery (bf16/fp32/seedsample/seedcurve, a100, N up to 100) corrected the
absolute numbers this whole consolidation quotes:
- **OPT B−D is ~+0.01, NOT +0.055.** Properly seeded (N=100 +0.006±0.010; 5×30 +0.0128), OPT is *barely* positive. The
  "+0.055" was a low-N seed-luck high draw (per-seed B−D SD = 0.10 → 30-seed SE ≈ 0.018).
- **The dissociation SURVIVES but at ~half magnitude:** OPT(+0.01) − Pythia(−0.016) = **+0.022 to +0.029, ~1.8–2.8σ** —
  real but modest, not the banked +0.061. Pythia is reliably negative, OPT reliably ≥0, so it is not noise.
- **Per-cell floor ~0.02** (OPT hardware 0.0165 bf16 / 0.0087 fp32 ⊕ seed 0.0168; Pythia hardware 0 ⊕ seed 0.0159). The
  famous "0.066" was a ~2σ tail pair-difference, NOT the 1σ floor. No unnamed source.
- **VINDICATION:** the fragile quantity was the *absolute* B−D level — exactly what the verdicts always demoted. The
  **paired within-run differentials** (DSI swap-gate; Probe-2 rank) are where the floor cancels, so they STAND. The
  mechanism program explains a real ~+0.03 dissociation, not a +0.06 effect — every absolute magnitude above must be
  re-read against the ~0.02 floor.
- **BINDING go-forward:** any absolute-B−D claim needs fp32 + N≥100 (or paired differentials); N=30 is under-powered below
  ~0.05. Probes 3/4/5 must be re-spec'd to N≥100 / paired before firing.

## ★ Probe-5 sequential residual 2026-06-25 — the eroded deposit is DEBRIS, not memory (`VERDICT_PROBE5_RESIDUAL_2026_06_25.md`)
First probe under the binding rule (N=100 + paired, fp32, gh200). Question: is the phase-2-eroded task-A deposit a usable
prior for re-learning task A (memory), or no better than a random rank-matched subspace (debris)?
- **DEBRIS, unambiguous.** resid−randproj = **−0.0014 ± 0.0057 (zero)** — the eroded deposit ≈ a random projection of equal
  rank. resid (0.637) ≈ A-baseline (0.628): carrying it forward buys nothing over training fresh.
- **What helps is RESTORATION, not reuse:** proj (snap task-A V-coords back to w_A) = 0.734, **+0.10 above all others**;
  resid−proj = **−0.0962 ± 0.0058 (~16σ)**. The damage is *coordinate displacement in V* — phase-2 moved the task-A-critical
  coords away from w_A; resetting them recovers task-A. (Not a free mitigation — presupposes you hold w_A and V; the value is
  mechanistic.) The *surviving displaced deposit* is debris; the *original w_A coords* are what carry task-A.
- Self-validating: phase-2 B−D = **+0.008 ± 0.011**, reproducing the floor-battery correction on a fresh run.
- Fits the program: complements Probe-2 (no compact reusable core to recover) and the floor reframe (mechanism is
  interference geometry — what to remove — not a durable stored prior). Load-bearingness = absorption/deflection (DSI), NOT
  retrievable memory. **The "is the residue a prior?" thread is closed: no.**
- **Pythia cross-arch = NO-GO/defer:** OPT being debris kills the memory/debris dichotomy; the residual Pythia contrast is a
  sub-0.02 sub-effect with a weak prior. Spend GPU on Probe-3/4 re-spec instead.

## ★ Probe-6 displacement-absorption 2026-06-25 — load-bearingness is NOT coordinate-pinning (`VERDICT_PROBE6_DISPLACEMENT_2026_06_25.md`)
Built to test the bridge DSI↔Probe-5: does the childhood's *absorption of the surge into V* actually *reduce the
displacement* of task-A coords (Probe-5's damage)? Paired NoCH vs CH off the same w_A, N=100. H1 = "absorption=pinning"
(ΔD>0, Δret>0, corr>0).
- **H1 FALSIFIED.** Δret = **+0.088 ± 0.012 (+7.6σ)** — childhood protects task-A (load-bearingness reproduces clean).
  BUT ΔD = **−0.129 (−17.5σ)** (childhood ends FARTHER from w_A in V) and r(ΔD,Δret) = **−0.322** (more in-V displacement →
  MORE protection). The pre-committed SURPRISE/REOPEN branch fired.
- **Confound named:** CH trains 400 extra steps so absolute ΔD sign is step-count-inflated. Confound-FREE conclusions:
  (1) the +0.088 protection is real; (2) the NEGATIVE correlation kills pinning (if proximity protected, less-displaced
  seeds would retain better — opposite holds).
- **Positive read:** protection is **reorganization/relocation, not suppression** — task-A survives LARGER excursions
  through its subspace ⇒ the robust region moved or the load spread (coheres with Probe-2 shell-not-core). Faint deflection
  signal (in-V fraction 0.516→0.476). Resolves the Probe-5 tension: distance-from-w_A predicts retention *within* a fixed
  path; childhood *changes the path*.
- **NEXT = Probe-7:** kill the step-count confound (phase-2-only displacement from each arm's pre-phase-2 anchor) + test
  suppression vs reorganization via forgetting-per-unit-displacement.

## ★ Probe-7 / 7b 2026-06-26 — REORGANIZATION confirmed dispositive (`VERDICT_PROBE7_REORGANIZATION_2026_06_26.md`)
Confound-kill of Probe-6 (phase-2-ONLY displacement from per-arm pre-phase-2 anchor) + non-destructive-childhood re-run.
- **Probe-7b (clean dose lr 1.5e-5, N=100):** Δforget +0.165/+14.8σ, Δrobust +0.356/+13σ, **ΔD2 −0.004 (≈0, suppression
  GONE)**, r_ch −0.04 (movement↔forget decoupled). Counterfactual = **~100% reorganization, 0% suppression.** Pre-committed
  REORG_if met outright. **Load-bearingness protects task-A by making the SAME movement less damaging (damage/unit ÷3:
  0.526→0.170), NOT by pinning/suppressing motion.** Probe-7's +2.7σ suppression sliver was a destructive-dose artifact.
- Calibration: post-childhood task-A 1e-5→0.792, 1.5e-5→0.776, 2e-5→0.758, 3e-5→0.738 (confirms Probe-7's 3e-5 destroyed).
- Caveat: 7b gate read False by 0.002 (calibration sampling noise, not destruction) — immaterial; 1e-5 confirmatory optional.
- **Lit (3 adversarial agents → PARTIALLY-KNOWN):** decomposition forgetting=movement×curvature & displacement-decoupling are
  KNOWN (Mirzadeh NeurIPS'20, Mehta JMLR'23, RL's Razor'25; EWC/OGD/GPM = suppression foils). NOVEL = a causal warm-up that
  rescales damage-per-unit ~3× at equal/larger movement (the slope-change), + targeting + cross-arch (Probe-8).

## ★ Probe-8 2026-06-27 — protection is DIRECTIONAL, not global flatness (`VERDICT_PROBE8_CURVATURE_ALIGNMENT_2026_06_27.md`)
Perturb task-A along the realized phase-2 in-V direction vs random in-V (magnitude-matched, m∈{1,2,3}), w/ vs w/o childhood.
Dual-cell N=100. (v1 estimator-VOID, v2 +14σ double-flagged; v3 LOCK 65ae39d4 fixed dose via calibration margin+K_CAL=20.)
- **★ FINDING 1 (clean, null-independent): NOT global flatness.** global g = drand_NoCH−drand_CH ≤ 0 at all m, both archs,
  10–16σ. Flat-minima account (Mirzadeh'20/Mehta'23) predicts g>0 (childhood softens random dirs too); observed opposite.
  **The protection is directional, not a uniform basin-widening.** The real step past the literature.
- **FINDING 2: TARGETED reorganization, graded.** Childhood collapses the phase-2-direction damage (OPT 0.147→0.005 ~97%,
  +11.6σ; Pythia 0.143→0.046 ~68%, +10.7σ) while random dirs untouched. NOT a binary OPT-yes/Pythia-no dissociation —
  both show it, load-bearing OPT more completely. Erodes at larger m (finite protection).
- **FINDING 3 + caveat:** random in-V doesn't dent task-A even at 3× (extreme directionality) → but the random-in-V NULL is
  WEAK (drand≈0), so Δaniso partly re-expresses Probe-7. Pre-reg sensitivity gate (drandN_m3>0.01) FAILED both → VOID-soft;
  FINDING 1 is independent of it. **Suggestive-strong, not fully dispositive.**
- **NEXT = Probe-8 v4:** sharper null (random within top-k≈8–16 Fisher dirs, not the 128-dim shell) so drand bites and
  Δaniso is a real discriminator. Then targeted-vs-global + OPT>Pythia grading become dispositive.

## ★★ Probe-8 v4 2026-06-28 — directional protection is SPECIFIC in OPT, GENERIC in Pythia (DISPOSITIVE) (`VERDICT_PROBE8_v4_SPECIFICITY_2026_06_28.md`)
Biting null = a DIFFERENT task's gradient (TREC) + randtop16. N(d)=childhood neutralization of dir d; S=N(phase2)−N(alttask).
Both cells passed ALL gates (nondestruct + bite) → dispositive.
- **OPT (load-bearing) = SPECIFIC:** N(phase2)=+0.153, N(alttask)=+0.031, S=**+0.122 (10.1σ)**. Childhood neutralizes the
  ACTUAL interference dir ~fully while barely touching a (stronger-damaging) other-task dir.
- **Pythia (inert) = GENERIC:** N(phase2)=+0.097≈N(alttask)=+0.082, S=**+0.014 (1.2σ, n.s.)**. Directional but not aimed.
- **Dissociation S_OPT−S_Pythia=+0.108 (~6.5σ).** Load-bearing arch aims its protection at the specific interference; inert
  arch confers broad directional robustness.
- Caveats: phase2(realized path) vs alttask(one-shot grad) asymmetry (contrast clean, construction fixed); WHY OPT is
  specific (childhood blind to phase-2) UNTESTED — hypothesis: specificity ∝ childhood↔phase-2 task-similarity; scope n=2.

## Probe 9 (aiming-axis screen) — DONE 2026-06-28: Ω NOT dialable by childhood stream → reframe + 2 anchors
*Question:* can you sweep the geometric overlap Ω (childhood-displacement subspace vs phase-2-interference subspace, in V) by
choosing the childhood stream? *Finding:* **No.** All 9 natural streams cluster at the same low Ω (OPT span 0.011, Pythia
0.025); the pre-committed abort-gate fired and the measure fan-out was skipped (~$120 saved). The childhood displacement is
~orthogonal to the interference yet still protects → Ω is set by the *interference* (phase-2 task), not the childhood. Pivot
to the phase-2 lever (9b). Two cross-substrate anchors banked alongside, STANDING as independent findings (NOT an Ω axis):
**Librarian** (free-lane retriever, protection holds, Ω 0.046, 138× swap-control) and **Deepfake** (frozen no-reorg backbone:
aimed GPM suppression protects 0.855 vs random 0.211 — a suppression-only foil to OPT reorganization; cos_meanshift −0.99
antagonism is domain-driven, caveat noted). Files `gateA/probe9_screen_*.json`, `gateA/anchor_*.json`.

## Probe 9b (phase-2 lever) — DONE 2026-06-29: the Ω-LAW is REFUTED; protection ∝ interference magnitude
*Question:* does displacement-overlap Ω predict protection magnitude P = forget_NoCH − forget_CH across phase-2 tasks? Sweep Ω
by varying the phase-2 task (fixed childhood = mix5), N=100 paired, both cells. Screen healthy (OPT span 0.171, Pythia 0.328).
*Finding:* **NO.** OPT-alone gave ρ(P,Ω) = −0.77 — which looked like a clean negative "free-lane" law (protection highest where
interference lands orthogonal to where the childhood moved). It was a **forget-magnitude confound + n=6 leverage:**
- OPT jackknife: raw ρ robust (−0.60..−0.90) but the partial ρ(P,Ω|forget) collapsed −0.79 → +0.00 dropping one point.
- Pythia replication (7 coverage tasks): ρ(P,Ω) = −0.18, partial ρ(P,Ω|forget) **FLIPS to +0.71** (opposite sign).
- Pooled within-cell z-scored OLS (n=13), P ~ Ω + forget: **β_Ω = +0.03 (≈0), β_forget = +0.92, R² = 0.80**; pooled partial
  ρ(P,Ω|forget) = +0.03. σ_sem (Ramasesh semantic similarity) dead throughout.
**The robust cross-task law: P ∝ forget_noch (interference magnitude), both architectures.** Geometry adds nothing.
*Pre-reg integrity:* the Pythia addendum (LOCK fa94862b) recorded P1 as a falsified directional prediction AND pre-committed
the two interpretations + named forget as a covariate BEFORE the result — so a false −0.77 law was caught before it became a
claim. **Walk-backs (banked):** "free-lane / orthogonal-lane" mechanism retired; no "3-substrate Ω law" (anchors stand alone).
*Does NOT touch* the upstream mechanism (7/7b, 8v3, 8v4) — 9b tested overlap-predicts-magnitude; the answer is no.
Verdict `VERDICT_PROBE9b_OMEGA_LAW_REFUTED_2026_06_29`.

## ★ THE ARC (mechanism CLOSED+dispositive; predictive-geometry CLOSED-negative): debris → not-pinning → reorganization → directional-not-global → specific(OPT)/generic(Pythia) → [overlap does NOT predict magnitude]
Catastrophic forgetting is directional damage in the task's curvature subspace; a non-destructive childhood protects by
REORGANIZING the task to be robust in those directions (not pinning, not global flattening); in a load-bearing architecture
that robustness is AIMED at the specific interference, in an inert one it is generic (8v4 = the capstone / claim to defend).
Refutes global-flatness (Mirzadeh'20/Mehta'23); adds an architecture-dependent specificity axis. Probe-9/9b add a clean
NEGATIVE: displacement-overlap geometry does NOT predict protection *magnitude* across tasks (that's just interference
magnitude); the aiming (8v4) is about *which direction is neutralized within a task*, not *how much* across tasks.
Write-up-ready core = 8v4; needs breadth (scale, 3rd arch) + the specificity-similarity sweep for a venue.

## Probe 10 (specificity ~ rehearsal-similarity) — CLOSED 2026-07-06 (LOCK c83884a7)
*Question:* WHY does OPT aim (8v4)? *Finding:* **specificity IS rehearsal-driven in OPT; the OPT>Pythia dissociation is
GRADED, not binary.** Constructed causal contrast (4 swap-childhoods {trec,subj}+one of {agnews,dbpedia,yahoo,emotion}, probe
agnews+yahoo realized dirs, DV = S = N(T-dir)−N(randtop16), magnitude-matched → not forget-confounded, N=80).
- **OPT SOLID:** ident−unrel gap agnews +0.0183/3.9σ, yahoo +0.0338/7.8σ; agnews ladder monotone ρ=1.0. Rehearsing the exact
  interference task ~doubles aimed protection over an unrelated childhood (generic floor S≈+0.02–0.05). → 8v4 "why" = rehearsal.
- **Pythia null (re-run @1e-5, clean; first attempt @3e-5 VOID-soft/destructive → killed):** PARTIAL. yahoo clean dissociation
  (OPT 7.8σ vs Pythia 1.2σ null; 5.4σ apart) but agnews n.s. (OPT 3.9σ vs Pythia 2.5σ; ~1σ apart). **P10-null (Pythia flat)
  PARTIALLY VIOLATED** → OPT *substantially more* rehearsal-specific than Pythia, NOT uniquely load-bearing (mirrors 8v3's
  graded targeting).
- **"Same-family" unreliable** (dbpedia weak sibling: OPT-yahoo below unrel, Pythia-yahoo negative) → operative similarity is
  interference-DIRECTION, not task-format. OWED: format-vs-direction post-hoc (small GPU; realized dirs not persisted).
Verdict `VERDICT_PROBE10_SPECIFICITY_REHEARSAL_2026_07_06`. Files `gateA/probe10_simrehearsal_{opt350m,pythia410m}.json`.
Tests *specificity S*, NOT magnitude P (9b showed P∝forget) — distinct from the refuted Ω-law.

## ★★ Probe-11/12/13 2026-07-09 — SPECIFICITY RECKONING: 8v4 dissociation RETRACTED; new dose-response-sign law (`VERDICT_PROBE11_12_13_SPECIFICITY_RECKONING_2026_07_09.md`)
Attempted to harden 8v4 (realized-path alttask, LOCK 332e7029) + extend to a 3rd arch (BLOOM-560m, w_A 0.84) and to scale
(OPT-1.3b, w_A 0.93). The hardening BACKFIRED on the capstone — in the good way (caught a fragile result):
- **HUNT-1 (local, $0):** N(d) ∝ threat-damage of d (r=0.81 over 12 direction×cell pairs); S = N(phase2)−N(alttask) ≈
  sign(dp2_n − dalt_n) (r=0.986). So S measures *which direction threatens more*, not aiming.
- **Metric-hardening:** on the realized-alt metric **Pythia S = +0.108** (was +0.014 GENERIC in 8v4) — the OPT/Pythia
  dissociation EVAPORATED. One-shot alttask (8v4) saturated; realized alttask is a weak null (bite<0.03). Neither isolates
  aiming → **S-vs-other-task retracted as an aiming measure.**
- **SURVIVES + STRENGTHENS:** the construction-stable measure **N_phase2 vs N_random** (matched magnitude) — childhood
  concentrates protection on the interference direction; robust across OPT/Pythia and **scale-stable to 1.3B**.
- **NEW LAW (Probe-13 dose-response, LOCK 49d14b48, N=50):** the **sign of the childhood→protection dose-response is
  architecture-dependent.** POSITIVE (more childhood → more protection): OPT-350M +0.075, Pythia-410M +0.044, OPT-1.3B +0.090
  (scales strongest — the DSI "dies at 1B" hint does NOT appear). NEGATIVE: **BLOOM-560M −0.049** — the lone eroder (more
  childhood → less, then harmful; at 2e-6 OPT +0.071 vs BLOOM −0.003). N_conc(=N_phase2−N_random) >0 for the three good
  protectors, ~0→negative for BLOOM.
- CAVEATS: BLOOM retA_drop non-monotone in lr → finding stated on the dose axis (clean); n=3 good + 1 outlier (BLOOM sole
  inverted case → needs a 2nd BLOOM-like arch); 1 task-pair. Hunt-2 (damage-matched fair null) not run — now moot (main claim
  uses N_phase2-vs-random). OPS BUG: launcher intermittently hangs on scp upload (needs timeout/retry + watchdog).
- **OPEN:** *why BLOOM erodes* (ALiBi? corpus? childhood↔task-A curvature interaction) + a 2nd BLOOM-like arch to test whether
  "inverted dose-response" is a class. Verbal shift: "specific" = concentrated on the interference vs random, NOT vs other tasks.

## Open / next (leverage-ranked; revised post-RECKONING 2026-07-09)
- **★ HIGHEST: why is BLOOM the lone eroder?** Add a 2nd BLOOM-like arch (ALiBi / non-OPT-Pythia family) to test whether
  "inverted childhood→protection dose-response" is a class or a one-off; + a mechanistic look at what BLOOM's childhood does
  to the task-A curvature subspace. This is the live thread the reckoning opened.
- **DONE (do NOT redo):** harden 8v4 (realized-alt) → RETRACTED the dissociation (Probe-11); 3rd arch (BLOOM) + scale (OPT-1.3b)
  → dose-response-sign law (Probe-13). The "SPECIFIC/GENERIC" framing is dead; use N_phase2-vs-random concentration + the
  dose-response sign.
- Optional Hunt-2 (damage-matched fair null) — moot for the main claim; only if resurrecting phase2-vs-other-task specifically.
- **specificity mechanism = Probe-10 (DONE 2026-07-06, above):** rehearsal-similarity drives OPT aiming (but note: on the
  reckoning, "aiming vs other tasks" is retracted — Probe-10's within-OPT rehearsal result stands, its cross-arch framing is
  now under the dose-response lens).
- Consolidate (DOING 2026-06-29: brief v2 + this narrative refreshed).
- Boundary probe (exploratory): find a regime where childhood fails (P≈0) or harms (P<0); all 9b P were positive.
- **DO NOT:** resurrect the Ω / displacement-overlap predictor or a "3-substrate Ω law" (refuted by 9b).
- Old backlog (low): Probe 3/4 re-spec, Thread-4 orphan, DSI #4.
- Probe 7/7b — **DONE 2026-06-26: REORGANIZATION (dispositive, ~100%/0%)**.
- Optional: 7b 1e-5 ultra-clean; Pythia 7b (reorg in inert arch?).
- Probe 6 (displacement-absorption) — **DONE 2026-06-25: H1 FALSIFIED, not pinning** (above).
- Probe 5 (sequential residual) — **DONE 2026-06-25: DEBRIS** (above). Pythia arm deferred (likely-null sub-0.02).
- Probe 3 (childhood dose-response), Probe 4 (disjoint dual deposits) — staged; **must be
  re-spec'd to N≥100 / paired differentials per the floor battery** before firing (Probe-4 also needs its disjointness-gate
  threshold fix). Probes 6/7 gated. **These are now the highest-value open GPU work.**
- Re-add DSI #4 damage-attribution non-perturbingly (was dropped on an eval() bug).
- The OPT k64>k128 overshoot is ~1 SE — a dedicated mid-rank-peak confirm would harden "load concentrates at intermediate
  rank" if pursued.
