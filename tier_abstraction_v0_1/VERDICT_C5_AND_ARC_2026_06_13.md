# VERDICT — C5 Transfer Judge + Full Program Arc

**Date:** 2026-06-13. Pre-registered gate (PREREG_MECHANISM_SWEEP, P5) resolved.

## C5 result: P5 REFUTED (Locatello-null) — pre-registered, clean

Pay gate: novel-class **superclass** accuracy, PAIRED z_trans vs z_low from the SAME trained
Arm C model (β=0.1), 15 held-out fine classes (one per super 0–14, images fully excluded),
3 seeds, classifier fit on 8000 trained-class features, bootstrap CI over query.

| | z_low | z_trans | diff |
|---|---|---|---|
| novel-super acc (mean of 3 seeds) | 0.3524 | 0.3453 | **−0.0071** |
| 95% bootstrap CI of paired diff | | | **[−0.019, +0.006]** |
| threshold | | | +0.020 |

**REFUTED** — diff negative, CI straddles 0, below the +2% bar.

- **Manipulation check PASSED** (premise valid): flat fine k=10/20 → z_low 0.57 vs z_trans 0.46.
  z_trans genuinely discarded within-super fine detail; the refute is not a broken-model artifact.
- **Honest nuance:** it is a near-TIE, not a z_low blowout (seed 2 flipped to +0.007). An 8-dim
  abstraction nearly matches a 64-dim representation on coarse transfer — but "nearly" ≠ "beats."
- **Scope caveat (interpretation, not a re-run):** the gate ran at fit-pool=8000, the regime MOST
  favorable to z_low (data-processing inequality: z_low ⊇ z_trans in information, and abundant data
  lets a linear probe exploit it). The low-data regime where abstraction is theorized to pay
  (van Steenkiste 2019) was not tested here. → see the pay-frontier pre-reg.

**Reading:** the abstraction is REPRESENTATIONAL (it exists, is hierarchy-structured, seed-robust)
but NOT FUNCTIONAL on this transfer task at this scale — it transfers no better than the raw lower
tier. This replicates the field's dominant prior (Locatello et al. 2019 ICML) in our own setup.

## Full program arc

| Question | Arm(s) | Result |
|---|---|---|
| Does the bottleneck ALONE abstract? | A (v0.1) | **No** — uniform compression, selectivity ≈ 0 (H1 falsified) |
| Does compartmentalizing objectives abstract? | C | **Yes** — selectivity +0.13, seed-robust inverted-U |
| Does adversarial penalty add over omission? | D | **No** — redundant→super-collapse (P2 refuted) |
| Does channel-cost asymmetry substitute? | F/U | **No** — asym ≤ uniform, only the supervision flicker (P4 refuted) |
| Does the abstraction PAY downstream? | C5 | **No** — representational, not functional (P5 refuted) |

**Two real findings.** (1) POSITIVE & novel: a robust design principle for *producing* tier
abstraction — the **pricing account**: abstraction emerges when the upper tier's objective makes
fine-detail more expensive to keep than coarse-detail (cheapest via objective OMISSION); the
bottleneck converts the price into selective loss. Triangulated by 6 arms; measured by a new
instrument (the selectivity index). (2) NEGATIVE & expected: that abstraction does not, at this
scale on this task, translate to downstream usefulness — the Locatello caution, confirmed concretely.

## Confound controls that held (the "robust" tax)
- **G6 matched-super-accuracy:** Arm F (sup_trans ≈ 0.55, sel +0.03) vs Arm C (sup_trans ≈ 0.58,
  sel +0.13) — same super-accuracy, 4× selectivity → selectivity is the pricing mechanism, not
  superclass-accuracy-in-disguise. Arm C survived its sharpest threat.
- **C5 paired design:** same-model z_trans-vs-z_low, no architecture confound.
- **Pre-reg discipline:** every gate (+0.05 selectivity, +2% transfer) locked before running;
  refutes reported as refutes.

## What this does NOT close
The functional question. C5 tested ONE transfer task (coarse placement of novel classes) in the
ONE regime (abundant fit data) least favorable to abstraction. "Representational not functional"
is the verdict HERE; whether tier abstraction pays in low-data, under distribution shift, or on
tasks where coarse structure is load-bearing is open. → PREREG_PAY_FRONTIER.

## UPDATE (same day) — Axis 1 RESOLVES the functional question: abstraction PAYS in low-data

The C5 null was tested at fit_pool=8000 — by DPI the regime MOST favorable to z_low. Axis 1
(PREREG_PAY_FRONTIER) swept the fit-pool. Result, novel-class superclass acc, paired, 3 seeds:

| fit_pool | z_low | z_trans | diff | 95% CI |
|---|---|---|---|---|
| 50 | 0.174 | 0.220 | **+0.046** | [+0.034, +0.058] |
| 200 | 0.260 | 0.324 | **+0.063** | [+0.049, +0.077] |
| 1000 | 0.317 | 0.344 | **+0.027** | [+0.015, +0.040] |
| 4000 | 0.354 | 0.350 | −0.005 | [−0.018, +0.008] |
| 8000 | 0.362 | 0.352 | −0.010 | [−0.023, +0.003] |

**AXIS 1 CONFIRMED** (gate: +2%/CI>0 at pool ≤1000 — cleared at all three small pools, peak +6.3%
at fit_pool=200 ≈ 10 ex/super). The crossover is exactly van Steenkiste's regime. **C5 was not a
sterile-abstraction result; it tested the one regime (abundant data) where the richer rep wins.**

**Revised program conclusion: tier abstraction is producible (Arm C, robust) AND functional in the
sample-scarce regime — it buys sample efficiency for coarse transfer to novel classes. The fine
task still favors z_low at every k (DPI, by construction); the payoff is specifically COARSE +
LOW-DATA.** D/F seed-lock confirmed both refutations across 3 seeds (D: α=0 reproduces C ≈+0.126,
no gain from α; F: asym ≈ uniform ≈ +0.02–0.03). Next: Form 1 (hyperbolic geometry) — does the
FORM of the abstraction extend the crossover beyond fit_pool=1000?
