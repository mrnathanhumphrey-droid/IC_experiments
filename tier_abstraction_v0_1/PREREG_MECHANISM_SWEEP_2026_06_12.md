# PRE-REGISTRATION — Tier-Abstraction Mechanism Sweep (bushclear for design principles)

**Date:** 2026-06-12. Locked BEFORE running Arms D/E/F and the transfer judge.
**Goal:** Identify *robust, confirmed* design principles that produce genuine tier abstraction.
A number is not a principle until it clears the gates below AND survives seeds.

## Established anchors (same session, do not re-litigate)
- **Arm A** (bottleneck alone, fine←z_trans): selectivity ≈ **0** (±0.005). The NULL.
- **Arm B** (fine primary + super aux, both on z_trans): selectivity ≈ **+0.02**, β-independent. Supervision flicker.
- **Arm C** (objectives split: fine←z_low, super←z_trans): selectivity **+0.125 ± 0.003** at β=0.1, seed-robust inverted-U. The POSITIVE.

All new arms are re-anchored against A (0) and C (+0.13).

## Primary metric (fixed)
**Selectivity index** = within-superclass error fraction on z_trans − same on z_low.
Reported as the per-arm β(/α)-curve, mean ± sd over 3 seeds. Headline = peak selectivity
at the operating point that ALSO passes the validity gates below.

## Validity / sanity gates (a cell is DEGENERATE — excluded from principle claims — if):
- **Super-collapse:** `sup←z_trans < 0.45` (Arm C super sits 0.55–0.60). Can't claim "super
  preserved" if super died. Such cells are reported but never support a principle.
- **Lower-tier degradation:** `fine←z_low < 0.42` (Arm C holds ~0.50). The z_low baseline
  must stay fine-rich or the selectivity contrast is broken.
- **Training divergence:** NaN / loss explosion → re-run; if persistent, mechanism flagged unstable.

## Robustness criteria (fixed)
- **CONFIRMED** label requires: all 3 seeds same sign, **seed σ ≤ 0.02**, and the effect is a
  *curve feature* (predicted inverted-U / monotone ladder), not a single lucky cell.
- **SUGGESTIVE** (needs more seeds): right sign but σ > 0.02 or only one cell.
- **NULL:** peak selectivity < +0.05 across the swept range (≈ Arm B flicker; not a principle).

## Confound controls (added post-lit-hunt; both hunts flagged independently)
- **G6 — matched-super-accuracy control (kills "selectivity = just good at superclass").** Selectivity
  is only credited if it exceeds the selectivity of a representation matched on `sup←z_trans` accuracy but
  built WITHOUT a fine-price differential. Operational proxy: regress selectivity on sup-acc across all
  arms/cells; a mechanism must sit ABOVE that regression line (positive residual), not just have high
  selectivity. Arm A vs Arm C at matched sup-acc is the minimal version.
- **G7 — post-hoc leakage probe for adversarial Arm D.** Beating the TRAINING adversary ≠ removing fine.
  Real test = a higher-capacity (2-layer MLP) probe trained post-hoc on frozen z_trans to recover fine.
  Report MLP-fine-recoverability alongside the linear probe; "purged" requires the STRONGER probe to fail too.
- **G8 — collapse check.** An "abstract" upper tier must have KL_rate > 0 and sup←z_trans above the floor;
  otherwise it is collapsed/inactive (trivially fine-poor), not abstract. Already partly covered by the
  super-floor gate; make explicit for Arms D (high α) and E (stacked).

## Multiple-comparisons discipline
We sweep β (and α for D). To avoid post-hoc cherry-picking: a "win" requires the gate-passing
peak to be (a) ≥ threshold, (b) seed-robust per above, AND (c) part of the predicted curve
shape. A lone high cell with neighbors at zero does NOT count.

---

## Principles under test (each falsifiable, each with a confirm/refute gate)

**P1 — Objective compartmentalization produces abstraction.**
Already supported by Arm C. Arms D/E extend it. (Refuted only if D/E with compartmentalization
fail to clear +0.05 — would undercut C.)

**P2 — Active purging beats passive omission. [Arm D, adversarial]**
- Wiring: Arm C split + fine-adversary on z_trans via gradient-reversal (×α). β fixed 0.1.
- Internal control: **α=0 must reproduce Arm C (+0.13)** — sanity that the harness is sound.
- Sweep α ∈ {0, 0.1, 0.5, 1.0, 2.0}.
- **CONFIRM P2** if peak selectivity ≥ **+0.15** (non-overlapping with C), super-floor held,
  seed-robust. **REFUTE** if ≤ C within noise (purging adds nothing over omission) or super collapses.

**P3 — Stacked bottlenecks build a monotone abstraction ladder. [Arm E, 3-tier]**
- Wiring: z_low→VIB→z_mid→VIB→z_trans; fine(100)←z_low, super(20)←z_mid, macro(6)←z_trans.
  (macro = hand-mapped grouping of the 20 superclasses, semantically clean.)
- Measure selectivity at BOTH interfaces: low→mid (within-super fine loss) and mid→top
  (within-macro super loss).
- **CONFIRM P3** if BOTH interfaces show selectivity ≥ +0.05, gate-passing, seed-robust
  (a genuine staircase). **REFUTE** if only the supervised-adjacent interface fires, or mid
  supervision starves the top.

**P4 — Cost-asymmetry alone induces selectivity, without objective-separation. [Arm F]**
- Wiring: Arm-A FAILURE setting (fine AND super both on z_trans). K=8 split into 4 groups ×2 dims
  with per-group KL multipliers [0.25,0.5,1.0,2.0]. Cheap channels pass freely; expensive squeezed.
- Internal control: **uniform multipliers [1,1,1,1]** at matched β (isolates asymmetry from
  merely-having-a-super-head).
- Sweep β ∈ {0.01,0.1,0.2,0.5} × {asym, uniform}.
- **CONFIRM P4** if asym selectivity ≥ +0.05 AND asym − uniform ≥ +0.03 (asymmetry is the
  cause), gate-passing, seed-robust. **REFUTE** if asym ≈ uniform (cost structure does nothing)
  or both ≈ 0 (can't rescue the Arm-A failure).

**P5 — Probe-selectivity corresponds to downstream usefulness. [Arm C5, transfer judge]**
The keystone. Hold out 15 fine classes during training (one from each of 15 superclasses, so
all 20 supers are still trained via siblings; held-out images fully excluded). Train Arm C
(beta=0.1). All comparisons are **PAIRED: z_trans vs z_low from the SAME trained model**, same
held-out set, same episodes/seeds — the only varying factor is which representation is queried.
A z_trans-vs-separate-model comparison is forbidden (confounds representation with architecture).

- **Why the obvious gate is rigged:** z_trans = f(z_low) and was trained to DISCARD within-super
  fine detail, so it *cannot* beat z_low at held-out FINE discrimination — failure there means
  abstraction worked, not that it's useless. The pay-gate must sit where a coarse abstraction is
  supposed to help: generalizing the superclass to UNSEEN fine classes.

- **PAY GATE (the falsifier):** fit an identical classifier on TRAINED-class features, predict the
  SUPERCLASS of held-out NOVEL-class query. **CONFIRM P5 iff
  `superacc(z_trans) - superacc(z_low) >= +2.0% top-1`, paired, with the 95% CI of the paired
  difference (over 3 seeds x bootstrap) excluding 0.** REFUTE otherwise (margin < 2% or CI
  straddles 0) -> Locatello-null: abstraction is "representational, not functional," and all
  selectivity findings are caveated accordingly.

- **Manipulation check (NOT a pay gate; expected z_low wins):** flat k-shot FINE at k=10,20.
  We PREDICT z_low > z_trans. If z_trans >= z_low here, z_trans did not actually drop fine detail
  and the premise is void.

- **Secondary (reported, not gating):** (a) k=1 flat fine — z_trans's 8-dim compactness may beat
  z_low's 64-dim via low-shot variance (abstraction-pays-via-sample-efficiency); (b) hierarchical
  classifier (z_trans super-gate -> z_low within-super) vs flat z_low at k=5 (does abstraction add
  usable structure).

- Run as the JUDGE on whichever mechanism(s) clear their selectivity gate.

---

## Design-principle graduation bar
A mechanism graduates to a **stated design principle** only if: gate-passing CONFIRMED +
seed-robust + mechanism-distinct (beats/differs from the simpler arm it's compared to) +
survives or is explicitly caveated by the P5 transfer judge.

## Run plan
1. Fast seed-0 pass: D + F (with their internal controls). Eyeball which mechanisms have a pulse.
2. Seed-fill (1,2) only the arms that clear NULL at seed 0.
3. Build + run E (3-tier) — the expensive faithful one.
4. C5 transfer judge on all gate-passers.
5. Synthesis memo: which principles graduated, which were refuted, what the bushclear cleared.
