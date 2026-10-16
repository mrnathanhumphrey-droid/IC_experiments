# BRIEF — Catastrophic-forgetting "childhood protection" arc. For a desktop assistant. 2026-07-09 (v4: +Probe-14/15 confound break).

> **★ v4 CONFOUND BREAK (2026-07-09, read first — resolves v3's "why BLOOM" open):** BLOOM's erosion is a **CONJUNCTION, not one
> knob.** Fired the 2×2 (positional × corpus): **Falcon-rw-1b** (ALiBi + **English**, 1.3B) PROTECTS (N_phase2 +0.04→+0.21) →
> ALiBi alone ≠ eroder; **XGLM-564M** (sinusoidal + **multilingual**) PROTECTS (+0.020→+0.149) → multilingual alone ≠ eroder.
> BLOOM (ALiBi **and** multilingual) is the **unique eroder** → cause = **ALiBi×multilingual interaction OR BLOOM's post-embed
> LayerNorm**. **Mechanism metric `frac_child_in_V` REFUTED (honest null):** all 5 cells reorganize the SAME energy into the
> Fisher subspace V (0.20–0.46, ≈3–4× chance) — protector and eroder alike → the lever is **DIRECTION within V, not energy**;
> registered "BLOOM-low" prediction wrong. Verdict `VERDICT_PROBE14_15_CONFOUND_BREAK_2026_07_09`. Open → Thread A: cosine(in-V
> childhood disp, interference dir) turns the null into a positive directional mechanism; disambiguate conjunction-vs-post-embed-LN;
> RoPE-eroder for reach. Thread B (application): CF-applicable surfaces = **the Librarian** (SportsThought reranker/adapter, already
> instrumented w/ `b2_cf` inV-vs-swap + a live rerank-antagonism) + physics-detector CNN. Lit-novelty check in flight.

> **★ v3 RECKONING (2026-07-09, read first — supersedes the §2 headline below):** The 8v4 "OPT-specific / Pythia-generic"
> architecture dissociation is **RETRACTED** — it was a null-threat-magnitude artifact (Hunt-1: S = N(phase2)−N(alttask) ≈
> sign(dp2−dalt), r=0.986; on the hardened realized-alt metric Pythia flips to "specific" too). **Survives + strengthens:**
> childhood reduces forgetting **concentrated on the interference direction** (N_phase2 ≫ N_random), **robust across OPT/Pythia
> families and scale-stable to 1.3B** (no DSI-1B-death). **NEW clean law:** the **SIGN of the childhood→protection
> dose-response is architecture-dependent** — POSITIVE OPT-350M/Pythia-410M/OPT-1.3B (more childhood → more protection),
> NEGATIVE **BLOOM-560M** (more childhood → less, then harmful; lone eroder outlier). Verdict:
> `VERDICT_PROBE11_12_13_SPECIFICITY_RECKONING_2026_07_09`. Open: *why BLOOM* (+ need a 2nd BLOOM-like arch). The §2/§3/§8v4
> text below is the pre-reckoning story — the *mechanism* spine (directional damage → reorganization → concentration) stands;
> the *"aimed-specificity vs other tasks / architecture dissociation"* claim does not. "Specific" now means "concentrated on
> the interference vs random," NOT "vs other real tasks."

This is an orientation brief, NOT a paper draft. It states the whole arc, the numbers, what's solid vs caveated, what's
open, the vocabulary, and where every artifact lives — so an assistant can help reason, plan, or eventually write. Project
root: `D:/IC_experiments/cf_cascade/`. GPU runs are on Lambda Cloud (lane = ssh_key `ML-CF`). Nothing is running now ($0).

---

## 1. THE QUESTION
When a model learns task A, then trains on a different task, it forgets A (catastrophic forgetting). A *non-destructive
"childhood"* — a varied multi-task warm-up phase inserted BEFORE the model learns task A — makes A survive that later
training much better. We knew the effect existed; this arc nailed **why**, with pre-registered, locked, reproducible probes.

## 2. THE HEADLINE (the whole thread, tied)
**Catastrophic forgetting is *directional* damage in the task's curvature subspace. A non-destructive childhood protects
task A by REORGANIZING it to be robust in those directions — not by holding the weights still, and not by globally
flattening the loss basin. In a load-bearing architecture (OPT-350m) that robustness is AIMED at the specific interference
direction the next task uses; in an inert architecture (Pythia-410m) the same childhood confers only GENERIC directional
robustness.**

This refutes the standing mechanism in the literature (global flat-minima) and adds an architecture-dependent *specificity*
axis the literature does not have.

**What does NOT predict protection (tested + refuted, Probe-9/9b):** the geometric *overlap* between where the childhood
moves the weights and the interference direction (Ω) does **not** predict how much protection you get across tasks. The only
robust cross-task predictor of protection *magnitude* is the **interference magnitude itself** (how much the next task would
forget A without a childhood). The aiming in 8v4 is real, but it does not show up as "more overlap → more protection."

## 3. THE ARC (each probe: question → finding → key numbers → status)
All on OPT-350m (task-A accuracy w_A = 0.823) and/or Pythia-410m (w_A = 0.753). task-A = SST-2; phase-2 ("depth") task =
subj; alttask = TREC; childhood stream = agnews/dbpedia/yahoo/newscat/emotion. V = top-128 task-A Fisher curvature
directions per carrier weight. fp32, N=100 unless noted.

1. **DSI swap-gate** (`VERDICT_DSI_2026_06_22`) — *What makes a childhood load-bearing?* Load-bearingness = deposit-specific
   **absorption** of the childhood gradient surge into V (OPT pulls it into v/k/fc1); inertness = deposit-specific
   **deflection** (Pythia). Paired within-run differential → floor-robust → **STANDS.**

2. **Probe-2 rank** (`VERDICT_PROBE2_RANK_2026_06_23`) — *Where does the protection live?* In the broad V **shell**, not a
   tiny high-curvature core. Confound-killed (childhood-once, magnitude-matched). Paired → **STANDS.**

3. **Floor battery** (`VERDICT_FLOOR_BATTERY_2026_06_25`) — *How big is the run-to-run noise, and is the headline real?*
   Corrected the magnitude: **OPT B−D ≈ +0.01, not the earlier +0.055** (low-N seed-luck). The OPT−Pythia dissociation
   survives but is modest (+0.022–0.029, ~2σ). Per-cell noise floor ≈ 0.02. **Binding rule for all later work: any absolute
   B−D claim needs fp32 + N≥100, OR paired within-run differentials.** Vindicated the program's discipline — the *paired*
   findings (DSI, Probe-2) are exactly where the floor cancels.

4. **Probe-5 residual** (`VERDICT_PROBE5_RESIDUAL_2026_06_25`) — *Is the eroded task-A deposit a reusable memory?* **No —
   DEBRIS.** The post-forgetting residue is statistically identical to a random subspace of equal rank (resid−randproj =
   −0.001); you're +0.10 better off *deleting* the childhood direction than reusing the residue (resid−proj = −0.096, ~16σ).

5. **Probe-6 displacement** (`VERDICT_PROBE6_DISPLACEMENT_2026_06_25`) — *Does the childhood protect by holding the
   important weights still (pinning)?* **No.** Childhood PROTECTS retention (Δret +0.088, +7.6σ) while the weights end up
   displaced *more* in V (ΔD −0.129), and per-seed more-displacement→more-protection (r −0.32). Pinning falsified. (Caveat:
   absolute displacement confounded by childhood's extra training steps; the protection + negative-correlation are clean.)

6. **Probe-7 / 7b reorganization** (`VERDICT_PROBE7_REORGANIZATION_2026_06_26`) — *Confound-killed: suppression vs
   reorganization?* Measuring phase-2-ONLY displacement from each arm's pre-phase-2 anchor: at a calibrated non-destructive
   childhood, **suppression is zero (ΔD2 ≈ 0) and protection is ~100% reorganization** — Δforget +0.165 (+14.8σ), Δrobust
   +0.356 (+13σ): the SAME movement does ~3× less damage. **Dispositive.** Load-bearingness = same movement, less damage.

7. **Probe-8 v3 directionality** (`VERDICT_PROBE8_CURVATURE_ALIGNMENT_2026_06_27`) — *Is the reorganization global flatness
   (literature) or directional?* **Directional, not global (clean).** The childhood does NOT make random in-V directions
   safer (global g ≤ 0, 10–16σ, both archs) — a flat-minima account predicts the opposite. It specifically softens the
   phase-2 direction. (Caveat: the random-in-full-V null was too weak/dead → v4 fixed it.)

8. **Probe-8 v4 specificity** (`VERDICT_PROBE8_v4_SPECIFICITY_2026_06_28`) — *Is the directional protection specific to the
   interference, or generic?* With a null that bites (a different task's gradient + top-16 random): **OPT = SPECIFIC**
   (S = N(phase2)−N(alttask) = **+0.122, 10.1σ**; neutralizes the actual interference ~fully, barely touches a
   stronger-damaging other-task direction). **Pythia = GENERIC** (S = +0.014, 1.2σ n.s.; neutralizes both equally).
   Dissociation ~6.5σ. **Both cells passed all validity gates → DISPOSITIVE.** ★ The capstone / the claim worth defending.

9. **Probe-9 screen** (`VERDICT_PROBE9_SCREEN_2026_06_28`) — *Can you dial the geometric overlap Ω by choosing the childhood
   stream?* **No.** All 9 natural childhood streams cluster at the same low Ω with the phase-2 interference (OPT span 0.011,
   Pythia 0.025) — the pre-committed abort-gate fired, the measure fan-out was skipped (~$120 saved). Reframe: the childhood
   displacement is ~orthogonal to the interference yet protects → Ω is set by the *interference* (phase-2), not the childhood
   → pivot to the phase-2 lever (9b). Two cross-substrate anchors banked here (`VERDICT_ANCHOR_LIBRARIAN_OMEGA`,
   `VERDICT_ANCHOR_DEEPFAKE_SEEDROBUST`) — they STAND as independent findings (see §5), but do NOT form an Ω axis (9b).

10. **Probe-9b phase-2 lever** (`VERDICT_PROBE9b_OMEGA_LAW_REFUTED_2026_06_29`) — *Does Ω (childhood↔interference
    displacement overlap) predict protection magnitude P across tasks?* **NO — Ω-law REFUTED.** Sweeping Ω via the phase-2
    task (screen healthy: OPT span 0.171, Pythia 0.328), measuring paired protection P = forget_NoCH − forget_CH:
    OPT-alone gave ρ(P,Ω) = −0.77 (looked like a clean negative "free-lane" law), but it was a **forget-magnitude confound +
    n=6 leverage.** Pythia replication: ρ(P,Ω) = −0.18, and the partial ρ(P,Ω|forget) **flips to +0.71** (opposite sign).
    **Pooled within-cell z-scored OLS (n=13), P ~ Ω + forget: β_Ω = +0.03 (≈0), β_forget = +0.92, R² = 0.80**; pooled partial
    ρ(P,Ω|forget) = +0.03. The **real, robust law: P ∝ interference magnitude (forget_noch), both archs.** σ_sem (Ramasesh
    semantic similarity) dead (ρ ≈ 0). Pre-reg addendum (LOCK fa94862b) locked the two interpretations + the forget covariate
    BEFORE the result → a false law was caught before it became a claim. **CLOSED.** Walk-backs (banked, §5): "free-lane"
    mechanism retired; no 3-substrate Ω law.

## 4. LITERATURE POSITIONING (from a 3-agent adversarial novelty check, 2026-06-28)
Verdict: **PARTIALLY-KNOWN.** Cite-and-distinguish, don't claim:
- *Forgetting ≈ movement × curvature; flat minima forget less per unit* — **Mirzadeh et al., NeurIPS 2020** (the exact
  decomposition). *Pretraining → flatter basins → less forgetting* — **Mehta et al., JMLR 2023.** *Parameter displacement
  is a weak forgetting predictor* — **"RL's Razor" (Shenfeld et al.), 2025.** *Pin/steer-off important weights* — **EWC**
  (Kirkpatrick 2017), **OGD/GPM** (the suppression foils we refute).
- **Novel (defensible):** (a) the protection is **directional, NOT global flattening** (childhood does not soften random
  directions; g ≤ 0) — refutes the flat-minima mechanism for this effect; (b) an **architecture-dependent specificity axis**
  (load-bearing → aimed, inert → generic) that no prior work has; (c) a causal warm-up that rescales damage-per-unit-
  displacement ~3× at equal/larger movement (the slope-change), localizing the damage to one direction.

## 5. CAVEATS / LIMITS (honest — these gate any write-up)
- **Scope: two small models, one task pairing.** Prior work (DSI) showed the load-bearingness effect **dies at 1B** — scale
  replication is the #1 external question. The OPT-vs-Pythia dissociation is **n=2** on the architecture axis.
- **Why is OPT specific?** The childhood is **blind to phase-2** (runs before it, never sees it), yet OPT protects the
  phase-2 direction over the alttask direction. UNEXPLAINED. Leading hypothesis: specificity tracks **task-similarity**
  (phase-2=subj is closer to the childhood stream than alttask=TREC). UNTESTED.
- **Direction-construction asymmetry (Probe-8 v4):** `phase2` is a realized 250-step displacement; `alttask` is a one-shot
  averaged gradient — different *kinds* of direction. The OPT-vs-Pythia contrast holds construction fixed (so the
  dissociation is clean), but the absolute specificity magnitude has this baked in. Fix before print: realized short-training
  displacement for alttask too.
- Magnitudes: re-read every absolute B−D against the ~0.02 floor (Floor Battery rule). The *paired* findings are the solid
  ones.
- **Geometry does NOT predict protection magnitude (Probe-9b, settled).** Displacement-overlap Ω fails as a cross-task
  predictor (pooled β≈0 once interference magnitude is controlled). Do not resurrect the "free-lane / orthogonal-lane" story
  without a design that isolates Ω from forget. The aiming claim (8v4) is about *which direction is neutralized within a
  task*, NOT about *how much* protection you get across tasks — keep them separate in any write-up.
- **The two cross-substrate anchors stand alone, not as an Ω axis.** *Librarian* (free-lane retriever, protection holds) and
  *Deepfake* (frozen no-reorg backbone: aimed suppression protects, random suppression doesn't — a suppression-only foil to
  the OPT reorganization result; antagonism is domain-driven, caveat noted). Each is a legitimate standalone data point;
  9b refuted the axis that would have joined them.

## 6. OPEN THREADS (leverage-ranked — what to do next; revised post-9b)
The geometric-predictor direction is **closed** (9b: Ω doesn't predict magnitude). The remaining leverage is all on the
*surviving* claim — 8v4 specificity — and on consolidation.
1. **★ HIGHEST — harden the capstone (8v4 specificity).** It's the novel, defensible result and it's **n=2 / 350–410M**.
   (a) Replicate the SPECIFIC/GENERIC dissociation at a 3rd architecture and/or ≥1B (DSI says load-bearingness may die at 1B
   — finding that boundary is itself publishable). (b) Fix the direction-construction asymmetry (realized-path `alttask`).
   This is where GPU spend now buys something that survives review.
2. **★ specificity mechanism = PROBE-10, CLOSED 2026-07-06** (`VERDICT_PROBE10_SPECIFICITY_REHEARSAL_2026_07_06`, LOCK
   c83884a7). *Why does OPT aim?* Constructed rehearsal-similarity contrast (4 swap-childhoods, probe agnews+yahoo dirs,
   DV = S = N(T-dir) − N(randtop16), N=80). **OPT: rehearsal-driven — SOLID** (ident−unrel gap agnews +0.0183/3.9σ,
   yahoo +0.0338/7.8σ; agnews ladder monotone ρ=1.0; rehearsing the exact task ~doubles aimed protection over an unrelated
   childhood). **Pythia null: PARTIAL — the OPT>Pythia dissociation is GRADED, not binary** (yahoo clean: OPT 7.8σ vs Pythia
   1.2σ null, 5.4σ apart; agnews n.s.: OPT 3.9σ vs Pythia 2.5σ, ~1σ apart). So P10-null (Pythia flat) partially violated →
   OPT *substantially more* rehearsal-specific than Pythia, not uniquely-load-bearing (mirrors 8v3's graded targeting).
   "Same-family" is unreliable (dbpedia weak sibling) → operative similarity is interference-DIRECTION, not task-format.
   OWED: format-vs-direction post-hoc (small GPU; dirs not persisted). Files `gateA/probe10_simrehearsal_{opt350m,pythia410m}.json`.
3. **Consolidate (DOING NOW).** This brief refreshed through 9/9b; next is updating `CONSOLIDATION_forgetting_flow.md` and
   deciding the single claim to defend (it's 8v4). No GPU.
4. **Boundary probe (exploratory).** Every P in 9b was positive — find a regime where childhood fails (P≈0) or harms (P<0)
   to bound the phenomenon.
5. **Hardening / backlog (low):** confirmatories (Pythia-7b reorg; asterisk-free 7b); 2nd task pair; Probe-3 (dose) / Probe-4
   (dual) re-spec to N≥100/paired (Probe-4 disjointness-gate bug); Thread-4 orphan readout; DSI #4 damage-attribution.
**Do NOT:** resurrect the Ω / displacement-overlap predictor, or assemble a "3-substrate Ω law" (both refuted by 9b).

## 7. VOCABULARY (so terms in the verdicts parse)
- **task A / w_A** — the skill we want to retain (SST-2 sentiment); w_A = the checkpoint after depositing it; its accuracy
  is the "anchor" (OPT 0.823 / Pythia 0.753), reproduced on every run as a validity gate.
- **childhood** — multi-task warm-up phase before task A (5 streams, 400 steps). "Non-destructive" = leaves task-A retention
  within 0.05 of w_A. Dose knob = lr_child.
- **phase-2 / depth task** — the later training (subj) that causes forgetting.
- **deposit / B−D** — task-A's footprint in weights; B = full deposit arm, D = childhood-gradient projected out of V. B−D =
  the retention difference (the original "load-bearingness" metric; now known small/fragile, use paired forms).
- **V** — top-128 eigenvectors of task-A's Fisher (curvature) per carrier weight = the subspace where task-A damage lives.
- **carriers** — the weight matrices probed (OPT: q/k/v/out_proj, fc1, fc2; Pythia: qkv, h_to_4h, 4h_to_h).
- **reorganization vs suppression vs pinning** — *pinning/suppression* = protect by reducing movement in V (EWC-style);
  *reorganization* = same movement, less damage (what we found). *global flatness* = robust to ANY movement (refuted).
- **targeted/specific vs generic** — protection aimed at the actual interference direction vs broad over real-task dirs.

## 8. HOW IT'S RUN (discipline + machinery — reproducibility)
- **Pre-registration discipline:** every probe gets a `PREREG_*.md` written and **SHA256-locked** (`*_LOCK.json`) BEFORE any
  GPU fires. The harness is then built to match the lock. Decision rules + validity gates are pre-committed. Results that
  trip a pre-committed gate are flagged **VOID-soft** (e.g., a too-destructive childhood) and re-run clean — this happened
  and is why v3→v4 exist. This is the source of the trustworthiness; honor it.
- **Lambda ops (`cloud/`):** `lambda_launch_generic.py <probe> <cell>` launches an instance, uploads the harness + ckpt +
  data, runs it, pulls the result JSON to `gateA/`, and self-terminates. Probes registered in its `PROBES` map. Auth = HTTP
  Bearer with the key in `D:/physics_detector/.env` (`CLOUD_LAMBDA=`); **use python `requests`, not urllib (urllib gets
  403'd by the WAF).** Lane identity = ssh_key `ML-CF`; never touch instances tagged with other keys
  (e.g. `physics-detector-runner`).
- **GPU sizing (hard rule):** NEVER a10 for heavy per-estimate work (it's CPU-launch-bound; once burned ~6h hung). Always
  `GPU_PREF="gpu_1x_a100_sxm4,gpu_1x_a100,gpu_1x_gh200,gpu_1x_h100_sxm5"` (a10 excluded). If only a10 is free, WAIT — the
  watcher `fire_when_capacity_probe7b.py` polls and fires the moment a real card frees.
- **Monitoring (hard rule):** every fire arms `killswitch_smart.py <id...>` (polls log mtime via SSH, kills on a confirmed
  10-min stall, 20h backstop). Read progress OFF the machine (`ssh ... tail pl.log` → `[hb]`/`[seed]`/`[cal]` lines) — NEVER
  fabricate progress. Per-step heartbeat is built into every harness.
- **Cost so far:** ~$270 across the arc (incl. dead ends, the a10 detour, and Probe-9/9b ≈ $70). Each clean dual-cell run
  ≈ $30, ~3–5h. Cost lesson (9b): N parallel single-task instances each re-train the shared childhood → wasteful; for a
  one-cell sweep, subset the tasks (you only need points that *span* the x-axis) and de-risk on the load-bearing cell first.

## 9. ARTIFACT INDEX (everything, with status)
Root `D:/IC_experiments/cf_cascade/`.
- **Master narrative:** `CONSOLIDATION_forgetting_flow.md` (the full chronological arc, all threads).
- **Verdicts (banked findings):** `VERDICT_DSI_2026_06_22`, `VERDICT_PROBE2_RANK_2026_06_23`,
  `VERDICT_FLOOR_BATTERY_2026_06_25`, `VERDICT_PROBE5_RESIDUAL_2026_06_25`, `VERDICT_PROBE6_DISPLACEMENT_2026_06_25`,
  `VERDICT_PROBE7_REORGANIZATION_2026_06_26`, `VERDICT_PROBE8_CURVATURE_ALIGNMENT_2026_06_27`,
  `VERDICT_PROBE8_v4_SPECIFICITY_2026_06_28` (★ the capstone), `VERDICT_PROBE9_SCREEN_2026_06_28` (Ω-not-stream-dialable),
  `VERDICT_PROBE9b_OMEGA_LAW_REFUTED_2026_06_29` (Ω-law refuted; P∝forget), `VERDICT_ANCHOR_LIBRARIAN_OMEGA_2026_06_28`,
  `VERDICT_ANCHOR_DEEPFAKE_SEEDROBUST_2026_06_28` (the two standalone cross-substrate anchors),
  `VERDICT_PROBE10_SPECIFICITY_REHEARSAL_2026_07_06` (rehearsal drives OPT specificity; OPT>Pythia graded). (Older verdicts predate this arc.)
- **Preregs + locks:** `PREREG_*_{,LOCK.json}` — Probe-8 chain: v1 b33ab105 (VOID), v2 93f26963, v3 65ae39d4, v4 0112d69b;
  Probe-7 7bdfcbc6, 7b 9d27939a; Probe-9 v1 11fe73b3 / v2 6e269459; Probe-9b 869ce94e + **Pythia addendum fa94862b**
  (locked pre-GPU, holds the P1-falsification + pre-committed interpretations); **Probe-10 c83884a7** (specificity~similarity,
  IN FLIGHT); anchors Librarian 129df97c, Deepfake a227f6a5; floor f1dfa005.
- **Result data:** `gateA/probe{5,6,7,7b,8,8v3,8v4}_*.json`, `gateA/probe9_screen_*.json`,
  `gateA/probe9b_{screen,measure}_*.json`, `gateA/anchor_*.json`, `gateA/floor_battery_*`, `gateA/dsi_*`.
- **Harnesses + ops (`cloud/`):** `probe{5,6,7,7b,8,8v3,8v4,9,9b,10}_*_lambda.py` (9b = `probe9b_phase2lever_lambda.py`,
  10 = `probe10_simrehearsal_lambda.py`), `probe9b_dryrun.py`, `probe10_dryrun.py`,
  `deepfake_anchor_seedsweep.py` (Anchor B, copy-not-edit), `lambda_launch_generic.py`,
  `killswitch_smart.py`, `fire_when_capacity_probe7b.py`. Anchor A = `D:/SportsThought/librarian/training/librarian_omega.py`.
  SSH key `D:/physics_detector/cloud/ssh/ml_cf_ed25519`.
- **Memory pointer:** `~/.claude/.../memory/project_cf_floor_probe5_live_2026_06_25.md` (one-screen live state).

## 10. ONE-LINE STATUS
Arc CLOSED and dispositive on MECHANISM: forgetting is directional damage; childhood reorganizes for robustness (not pinning,
not global flatness); load-bearing OPT aims it at the specific interference, inert Pythia is generic (8v4 = the capstone).
Probe-9/9b tested whether displacement-overlap geometry *predicts protection magnitude* and **refuted it** (P ∝ interference
magnitude, not Ω; the "free-lane" story and the 3-substrate Ω axis are retired — a false −0.77 law caught by pre-reg +
replication). **★ SUPERSEDED BY THE v3 RECKONING (top of file, 2026-07-09):** the 8v4 specificity dissociation is RETRACTED (threat-magnitude
artifact); write-up-ready CORE is now **(a) directional concentration** (N_phase2 ≫ N_random, cross-family + scale-robust to
1.3B) and **(b) the architecture-dependent SIGN of the childhood→protection dose-response** (OPT/Pythia +, BLOOM − eroder).
The one live thread: *why BLOOM* + a 2nd BLOOM-like arch. Nothing running, $0, no orphans. (Probe-10's within-OPT
rehearsal-similarity result stands; its cross-arch "aiming" framing is now under the dose-response lens.)
