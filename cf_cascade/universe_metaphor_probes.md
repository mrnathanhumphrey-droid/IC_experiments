# PREREG — Deposit–Surge Interaction (DSI) follow-on probe battery. 2026-06-22.

Pre-registered menu of mechanism probes extending the DSI (Deposit–Surge Interaction) trajectory prereg.
All probes target the **per-step interaction between the childhood deposit and the phase-2 task-B gradient** —
the mechanism that produces the cross-architecture map-load-bearingness dissociation. Written as straight
mechanism (no informal framing). Every probe is per-(layer,carrier), never pooled, curvature-controlled,
magnitude-matched, with a mandatory swap control on any positive and a reproduce-known anchor before banking.

---

## SHARED PRECONDITIONS (bind every probe below)

**Band (pre-committed):** OPT-350m and Pythia-410m ONLY. The behavioral dissociation is alive here
(OPT B−D = +0.055 sig; Pythia B−D = −0.006, CI excludes OPT) and **dead at 1B** (both families B−D → 0).
Mechanism questions are only askable where the families still differ. Do NOT run any DSI-family probe at 1B.

**Matched dispositive regime (reproduce exactly):** lr_depth 5e-5 / 250 steps, lr_child 3e-5 / 400 steps,
pinned w_A, held-out = subj, N = 30 paired seeds. HARD-STOP any cell that lands outside the matched
w_A accuracy band or matched A_forget severity.

**Banked anchors (the reproduce-known gate — the instrumented harness MUST reproduce these within noise
before any DSI quantity is read; if per-step logging perturbed the dynamics, the harness is invalid):**

| cell | B−A | D−A | B−D | A_forget | mapfrac_B |
|---|---|---|---|---|---|
| OPT-350m | +0.089 (sig) | +0.034 (ns) | **+0.055** | 0.237 | 0.219 |
| Pythia-410m | +0.073 (sig) | +0.079 (sig) | **−0.006** | 0.215 | 0.340 |

**Objects (per cell, per seed, per-(layer,carrier)):**
- **V** = top-128 task-A Fisher-curvature directions at w_A (the banked basis). Reproduce against the
  dispositive V before anything.
- **S_dep** = childhood weight-change projected into V, per carrier: d = P_V(Δw_child); S_dep = unit direction
  of d per carrier. **Rank-1 per carrier, pre-committed — do not search rank** (except Probe 2, which is the
  rank ablation itself).
- **g_t** = per-step task-B gradient per carrier during phase-2 (raw gradient, pre-lr), t = 1…250.

**Carriers (TRIPWIRE-4 — each family in its OWN carrier structure, never impose one on the other):**
- OPT: `q_proj, k_proj, v_proj, out_proj, fc1, fc2`
- Pythia: `attention.query_key_value (qkv), mlp.dense_h_to_4h (h4h), mlp.dense_4h_to_h (4hh)`

**The four per-step DSI measurements (arm B = deposit present, vs arm A = no-deposit baseline):**
1. surge–deposit alignment: cos(g_t, S_dep)
2. surge in-V fraction: ‖P_V(g_t)‖² / ‖g_t‖²
3. deposit survival: ‖P_{S_dep}(w_t − w_A)‖ normalized to t=0 value
4. task-A damage localization at checkpoints: Δ(task-A loss) decomposed into S_dep / (V∖S_dep) / complement

**The four pre-committed mechanism signatures (carry across all probes that read DSI):**
- **DEFLECT** — arm-B in-V fraction (#2) falls below arm-A AND cos (#1) drops; deposit survives (#3 ~flat);
  task-A damage low and NOT localized in S_dep.
- **ABSORB** — arm-B alignment (#1) and in-V (#2) ≥ arm-A; deposit survives (#3 ~flat); task-A damage low
  DESPITE surge landing in the deposit subspace.
- **ERASE** — deposit magnitude (#3) decays over phase-2; task-A damage accrues localized in S_dep (#4).
- **PASS-THROUGH (null)** — arm-B ≈ arm-A on #1–#4; deposit does not engage the surge.

**Cross-arch prediction (the headline contrast, all probes):** OPT (load-bearing) → DEFLECT or ABSORB;
Pythia (inert) → ERASE or PASS-THROUGH. Either resolution localizes *why* the map is load-bearing on one
family and not the other. The design separates the four signatures; it does not confirm a pre-chosen one.

**Universal controls (GATING, not optional — carry into every probe):**
- **MAGNITUDE-MATCH (this is where the rung-3.5 toggle died).** All comparisons via cosines / normalized
  fractions, never raw projections. Confirm arm A vs B matched on per-step ‖g_t‖; if B's surge norm is
  systematically smaller, that drop is itself a deflect signal — report it as such, do not let it masquerade
  as an alignment change.
- **PER-STEP TRAJECTORY, not endpoints.** Deflect (early) and erase (late) can share an endpoint for opposite
  reasons. Report the full t-trajectory and the step index where arms B and A diverge.
- **SWAP CONTROL (Cut-1 style, MANDATORY on any positive).** Replace S_dep with a curvature-matched random
  subspace in V (topcurv-derived, matched mean-λ, NOT deposit-derived). If the same signature appears, it is
  generic curvature routing, not deposit-specific → the deposit-specific reading dies. Run the moment a
  deflect/absorb signal appears; do not bank without it.
- **CURVATURE-CONTROLLED** (V at w_A) is the clean read; never raw weight space.
- **REPRODUCE-KNOWN ANCHOR** (table above) before banking any verdict.
- **STOP-on-missing → generate the artifact yourself, do not proxy.**

**Ops:** Lambda A100 fits both band cells fp32; per-step projections onto precomputed V/S_dep are cheap
(storage = per-carrier per-step scalars). Fire your own launchers, NO pipe to head/tail (SIGPIPE orphans),
finally-terminate, match orphans by printed instance_id.

---

# PROBE 1 — Single-carrier DSI isolation  [STATUS: IMMEDIATE EXTENSION]

**Tests:** whether the deflect/absorb/erase mechanism is carrier-universal or carrier-specific.

**Method:** run the four DSI measurements at per-direction, per-step resolution on ONE frozen carrier's S_dep
at a time, instead of the per-carrier-median pooled read. Sweep across all carriers separately
(OPT 6 carrier types, Pythia 3), one specimen at a time.

**Pre-committed outcomes:**
- **Carrier-universal mechanism** → every carrier shows the same signature (all DEFLECT, or all ERASE).
- **Carrier-specific split** → e.g. OPT qkv DEFLECT while fc1 ERASE; this is invisible in the pooled DSI and
  would be a genuine finding (sharpens the SPIRAL fc1/fc2 dissociation at the interaction layer).

**Confounds / controls:** n=1 carrier does not generalize — run across carriers separately and read the split,
never pool. Per-direction resolution within V can surface a single hot direction; report the within-carrier
direction distribution, not just the carrier median. Swap control per carrier.

**What it settles:** whether map-load-bearingness is a whole-model property or lives in specific carriers
(and if the latter, which) — directly extends DSI with no new infrastructure.

## User additions

**FLY-AROUND INTERFACE (out-of-V / geometry lane — corroboration cross-check):** the per-carrier DSI signature
should be PREDICTED by per-carrier static **G5** (deposit↔surge overlap at the dispositive endpoint) — the DSI
per-step collision is the *dynamic* version of G5. Already on disk in `geom_matrix.json` (band-cell medians:
OPT-350m G5≈0.410, Pythia-410m≈0.505 — Pythia higher = more collision). PRE-COMMIT: low-G5 carriers → DEFLECT/
ABSORB, high-G5 → ERASE/PASS-THROUGH. Read the per-carrier rank-correlation between static G5 and the dynamic
signature: positive → G5 is the static snapshot of the per-step collision, two lanes corroborate at different
time-resolutions; null/negative → one instrument is wrong, diagnostic either way. Costs nothing (G5 exists).


---

# PROBE 2 — Deposit-rank ablation  [STATUS: IMMEDIATE EXTENSION]

**Tests:** dose-response of load-bearingness on deposit rank — is the deposit monolithic or distributed?

**Method:** run arm B with S_dep truncated to its top-k Fisher directions, k ∈ {16, 32, 64, 128}. Measure B−D
and the DSI signature at each k. (This is the ONE probe that varies deposit rank; rank-1 pre-commit is lifted
here by design.)

**Pre-committed outcomes:**
- **Monolithic** → B−D collapses sharply once k drops below some threshold (load-bearingness needs the full
  deposit; truncation destroys it).
- **Distributed** → B−D falls smoothly / proportionally with k (each slice carries proportional load).

**Confounds / controls:** truncating to top-k changes the deposit norm — re-normalize so the k-truncated deposit
has matched magnitude to full (else a rank effect is confounded with a magnitude effect). Reproduce full-rank
B−D (+0.055 OPT) at k=128 as the anchor. Swap control at each k (curvature-matched random rank-k).

**What it settles:** the internal structure of the deposit — whether load-bearingness is an all-or-nothing
property of the full subspace or an additive property of its directions. Cheap, standard ablation.

## User additions

**K_E KNOT (comoving-basis lane):** the TRUE task-A curvature core is tiny — OPT-350m **K_E90 = 14** dims hold
90% of Fisher mass (Pythia-410m even lower), measured in the basis-invariance run (`basisinv_*.json`). The
ladder k∈{16,32,64,128} therefore mostly samples ABOVE the real core. ADD k = K_E (≈14 OPT) as a pre-committed
knot: B−D saturated by k≈14 → load-bearingness lives in the genuine 90%-mass core (monolithic-in-core); B−D
still climbing 16→128 → the load is in the near-flat tail OUTSIDE the core (a surprise worth its own probe).
The de-confounded enrichment (deposit stays curvature-locked at every scale, NO exodus — enrich@128 1.8–3.7×
across the ladder) predicts saturation near K_E. NOTE the magnitude-match control you already flagged is
load-bearing here: truncating to k≈14 changes deposit norm a lot relative to k=128 — re-normalize per k.


---

# PROBE 3 — Childhood dose-response  [STATUS: STAGED]

**Tests:** whether deposit *strength* maps monotonically onto the interaction mechanism (weak→erase,
medium→absorb, strong→deflect).

**Method:** vary childhood training strength (steps / epochs) at FIXED task-B stream; build deposits of
increasing magnitude; measure B−D, mapfrac_B, and the DSI signature at each dose. Dose ladder pre-committed
(e.g. childhood steps {100, 200, 400, 800}).

**Pre-committed outcomes:**
- **Monotone mechanism shift** → signature transitions erase → absorb → deflect as dose rises.
- **Threshold** → mechanism flips at a single dose (load-bearingness switches on).
- **Flat** → mechanism dose-independent (deposit magnitude irrelevant to interaction type).

**Confounds / controls:** more childhood training also moves w_post-child accuracy and can change A_forget
severity — log both per dose and HARD-STOP if a dose leaves the matched band (a dose effect must not be a
regime-severity effect). Hold task-B stream fixed across doses. Swap control at the dose where any mechanism
shift appears.

**What it settles:** whether the four signatures are distinct mechanisms or a continuum indexed by deposit
magnitude — and the dose at which OPT-style load-bearingness switches on.

## User additions


---

# PROBE 4 — Disjoint-subspace dual deposits  [STATUS: STAGED]

**Tests:** when two deposits occupy non-overlapping V-subspaces, do they reinforce or interfere when the
phase-2 surge arrives — measured per-step.

**Method:** two childhood streams (disjoint task-B sets selected to build deposits in non-overlapping
V-subspaces; verify disjointness by S_dep1 ⟂ S_dep2 before firing). Run arm B with both deposits present;
per-step, measure each deposit's survival (#3) and the surge's interaction (#1, #2) with each separately.
Compare protection (B−A) to the single-deposit arms.

**Pre-committed outcomes:**
- **Reinforce** → dual-deposit B−A > either single; both deposits survive; coherent deflection from two
  subspaces.
- **Interfere** → dual-deposit B−A < sum of singles; one or both deposits erode; the surge plays them against
  each other.
- **Independent** → dual ≈ sum of singles; deposits act without coupling.

**Confounds / controls:** the two childhoods together = more total childhood training → match total childhood
compute against a single-deposit arm with equal steps (so a reinforcement is not just "more training").
Verify subspace disjointness numerically. Per-step read is required — interference may appear at a specific
phase-2 step and wash out at the endpoint. Swap control on any coupling signal.

**What it settles:** whether deposits compose, and where in phase-2 coupling occurs — the per-step extension of
the cross-pair universality result (which is endpoint-only).

## User additions


---

# PROBE 5 — Sequential-phase residual  [STATUS: STAGED]

**Tests:** after the phase-2 surge, is the post-collision residual deposit a usable prior for a later phase, or
interference?

**Method:** standard arm B through phase-2 (task-B). Measure residual deposit (what survives in S_dep). Then
run phase-3 on a THIRD task (task-C); measure task-A protection in phase-3 WITH the residual present vs a
matched arm with the residual projected out. (task-C pre-committed, structurally distinct from task-B.)

**Pre-committed outcomes:**
- **Memory** → residual present → better task-A protection in phase-3 (a partially-eroded deposit still guides).
- **Debris** → residual present → worse or unchanged phase-3 protection (residual is interference, not prior).

**Confounds / controls:** projecting the residual out is itself a perturbation — match it against a
curvature-matched random projection of equal magnitude (so "residual helps" is not "any projection hurts").
Reproduce phase-2 B−D before reading phase-3. task-C must be distinct from both A and B (else it is replay).
Swap control on any phase-3 effect.

**What it settles:** whether deposits accumulate across phases (a continual-learning prior) or degrade into
interference — bears on whether the mechanism supports multi-phase protection.

## User additions


---

# PROBE 6 — Synthetic-task-B oriented deposit  [STATUS: STAGED — SYNTHETIC-DATA-ONLY, GATED]

**Tests:** whether a deposit pre-oriented relative to the incoming surge (e.g. orthogonal vs head-on) changes
the interaction mechanism.

**GATE (mandatory — this is the cleaned-up form of the rung-3.5 toggle, which DIED on a magnitude confound):**
the orientation MUST be carried by the natural gradient of a **constructed task-B dataset** whose gradient
geometry has the prescribed cosine to the held-out task-A gradient. **NO gradient surgery.** Surgery rotating
the in-core gradient component is the dead toggle — it confounds orientation with perturbation magnitude. If a
synthetic task-B with the prescribed natural cosine cannot be constructed, the probe does NOT fire.

**Method:** construct task-B variants whose natural phase-2 gradient carries cosine c ∈ {≈0 orthogonal,
≈+0.2 aligned, ≈−0.23 head-on} to the task-A gradient (cosine in the DATA, constant natural magnitude). The
c≈0 level MUST reproduce the natural-regime B−D anchor (the toggle's failed anchor — if c≈0 does not reproduce
+0.055 on OPT, the construction is invalid and the probe is void). Measure DSI signature at each orientation.

**Pre-committed outcomes:**
- Orientation modulates mechanism → e.g. a deposit oriented orthogonal to the surge DEFLECTS where head-on
  ERASES; load-bearingness becomes a function of pre-orientation.
- Orientation-independent → mechanism set by architecture/deposit, not by relative orientation.

**Confounds / controls:** the MAGNITUDE side-effect that killed the toggle is the primary risk — enforce
constant natural surge magnitude across orientation levels and verify it; ALSO confirm the map-independent
channel (D−A) does NOT co-vary with orientation (if D−A collapses in lockstep with B−D across orientation,
the driver is magnitude, not orientation — the exact toggle failure mode). Anchor reproduction at c≈0 is
gating. Swap control on any orientation effect.

**What it settles:** whether the deposit's protective mechanism can be *steered* by pre-orienting it — the
clean test the broken toggle was reaching for. Conditional on a valid synthetic construction.

## User additions


---

# PROBE 7 — Reconstructed-surge feedback  [STATUS: GATED — DO NOT FIRE WITHOUT A PRE-REGISTERED FALSIFIER]

**Tests:** whether a surge constructed to match a characterized collision profile protects differently.

**CIRCULARITY GATE (the reason this is flagged, not scheduled):** building a synthetic task-B whose gradient
matches the DSI-characterized collision profile and then asking whether it "protects differently" is
**circular** — it can confirm its own input. This probe does NOT fire unless a falsifier is pre-registered:
a specific, pre-committed result that would show the reconstruction is WRONG (e.g. the reconstructed surge
must reproduce the measured B−D within noise AND a deliberately mis-specified reconstruction must fail to —
a discrimination test, not a confirmation). Without that two-sided test, this is not evidence and stays
unfired.

**If (and only if) the falsifier is specified:** construct a surge matching the measured per-step collision
profile (angle / magnitude trajectory / step-timing), run as arm B′, and require (a) B′ reproduces the natural
B−D, and (b) a mis-specified control surge does NOT. Both must hold or the loop is uninformative.

**Confounds / controls:** circularity (above) is the dominant risk. Same magnitude-match and anchor-reproduction
discipline as Probe 6. Swap control. If (a) and (b) cannot both be pre-committed, leave this probe out of the
fire set entirely.

**What it settles:** IF the discrimination falsifier holds — whether the characterized collision profile is a
sufficient description of the mechanism (a reconstructed surge with the profile behaves like the real one).
Otherwise nothing; held as a flagged idea, not a scheduled probe.

## User additions


---

## FIRING ORDER (pre-committed; user-adjustable in the additions block)

1. **Probe 1 (single-carrier isolation)** + **Probe 2 (rank ablation)** — immediate DSI extensions, no new
   infrastructure, both at OPT-350m / Pythia-410m.
2. **Probe 3 (dose-response)** — staged after 1–2 confirm the band reproduces.
3. **Probe 4 (disjoint dual)** and **Probe 5 (sequential residual)** — staged.
4. **Probe 6 (synthetic oriented)** — only if a valid synthetic task-B construction with natural prescribed
   cosine exists.
5. **Probe 7 (reconstructed-surge)** — only if a two-sided discrimination falsifier is pre-registered;
   otherwise unfired.

All mechanism probes stay at the 350m/410m band where the dissociation is live. None run at 1B.
The scale-invariant / task-A-agnostic basis lane (D−A, A_forget — the genuinely uninstrumented out-of-basis
targets) is SEPARATE and not part of this battery.

## User additions

