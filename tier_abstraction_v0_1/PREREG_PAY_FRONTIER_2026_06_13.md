# PRE-REGISTRATION — The Pay Frontier ("when does tier abstraction pay?")

**Date:** 2026-06-13. Locked BEFORE running. Follows C5 (P5 refuted at large fit-pool, easy
coarse transfer). Question: does the abstraction pay in the regimes where theory says it should?

## EXECUTION ORDER (locked 2026-06-13)
**v0.next = Axis 1 (data scarcity) ONLY.** It is the cheapest (reuses the C5 harness, vary fit-pool),
the closest to the dominant theoretical claim (compact reps generalize from few examples — the whole
premise of few-shot/meta-learning), and has the most decisive null (a null here is a strong combined
finding, not a weak one). **Hold Axis 2 (distribution shift) and Axis 3 (task-pays-abstraction) for
v0.3** — Axis 3 especially carries the "I designed the task to favor my method" pathology Locatello
called out, so it runs LAST and only with a hardened task-construction protocol.

**Decisive-null logic:** if Axis 1 crosses over → regime-of-utility located, that's the contribution,
proceed to Axis 2. If Axis 1 nulls across ALL pool sizes → combined finding "tier-abstraction is
producible (Arm C, robust) but not functional across data scale (C5 + Axis 1)" — a sharp Locatello-style
result that says the program needs a STRUCTURAL reset (different architecture / task family), not more
tuning. E and Axes 2/3 become v0.3 only if Axis 1 finds something functional.

## Setup (shared, all tests)
- Same held-out Arm C model (β=0.1, 15 fine classes held out, images excluded), 3 seeds, retrained
  identically to C5. ALL comparisons PAIRED: z_trans vs z_low from the SAME model. Bootstrap CI over
  query, averaged across seeds. No architecture confound.
- "Pays" = a payoff mode CONFIRMS if z_trans beats z_low by **≥ +2.0% top-1, 95% CI of paired diff
  excludes 0**, in that mode. Each mode pre-registered separately so we learn WHICH regime pays.

## Why each regime is a fair test (the C5 null was none of these)
C5 used abundant fit-data (8000) on easy coarse transfer — by the data-processing inequality
(z_low ⊇ z_trans) that regime structurally favors the richer representation. Abstraction is
theorized to pay where the richer rep's extra info is a LIABILITY: scarce data (overfitting),
corruption (fragile detail), or tasks that need coarse scaffolding.

---

## Pay-mode 1 — LOW-DATA coarse transfer (sample scarcity)
*Theory: van Steenkiste 2019 — structured/compressed reps help in the few-label regime; the 8-dim
abstraction regularizes where the 64-dim rep overfits.*
- Fit the novel-super classifier on fit-pool sizes **{20, 50, 100, 300, 1000, 8000}**; predict
  superclass of novel-class query. Paired z_trans vs z_low at each size.
- **CONFIRM** if z_trans − z_low ≥ +2% (CI>0) at any fit-pool **≤ 100**.
- Also report the full curve + sample-complexity readout (labels to reach a fixed super-acc target).
- *Prediction (ours): pays — crossover at small fit-pool.*

## Pay-mode 2 — ROBUSTNESS under distribution shift
*Theory: Alemi 2017 — IB compression buys robustness; z_trans discarded the fragile fine-detail
that corruption attacks.*
- Evaluate novel-super accuracy under input corruption (Gaussian noise σ ∈ {0, 0.1, 0.2, 0.4},
  applied to query images post-normalization; fit classifier on clean trained features, 8000 pool).
- **CONFIRM** if z_trans − z_low ≥ +2% (CI>0) at σ ≥ 0.2, OR z_trans's degradation slope (acc vs σ)
  is significantly shallower than z_low's.
- *Prediction: pays — z_trans degrades slower.*

## Pay-mode 3 — ABSTRACTION AS SCAFFOLDING (hierarchical value-add)
*Theory: the abstraction is usable structure even if z_low alone is competent — use z_trans to
gate the coarse decision, z_low to discriminate within.*
- Hierarchical few-shot fine classifier: z_trans predicts the superclass, then z_low does NCM only
  among novel classes in that predicted super. Compare to flat z_low NCM. k=5, episodes=30.
- **CONFIRM** if hierarchical − flat_z_low ≥ +2% top-1 fine (CI>0) at k=5.
- *Prediction: pays if the super-gate is accurate enough to prune the within-super search.*

---

## Overall verdict logic
- **Abstraction PAYS** (functional) if ANY pay-mode confirms — report which regime(s).
- **Abstraction is REPRESENTATIONAL ONLY** if all three refute — strengthens the Locatello-null
  and bounds the principle to "produces structure, no demonstrated downstream value at this scale."
- Each mode reported independently (no fishing: a single confirming mode is a located payoff, not
  a multiple-comparisons artifact, because each mode is a distinct pre-registered mechanism).

## Build
`pay_frontier.py` — retrains the held-out Arm C (reuses HoldoutData + ArmCModel from transfer_judge,
does NOT modify C5), runs the three pay-modes, writes `pay_frontier_result.json`. ~10 min, 3 seeds.
