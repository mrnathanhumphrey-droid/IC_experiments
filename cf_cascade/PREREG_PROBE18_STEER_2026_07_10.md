# PRE-REG — Probe-18 (controllability / positive-direction): is the protect/erode effect CAUSALLY the childhood's movement along the interference direction? 2026-07-10.

SHA256-lock before GPU. Everything so far is ELIMINATION: the protect/erode sign is not scalar flatness (Probe-16), not energy
(frac-null), not solely the post-embed LN (Probe-17). All point to DIRECTION within V, but we have not shown it CAUSALLY. This
probe steers the childhood's gradient relative to the realized interference direction dVp and tests whether the effect nullifies
— turning "not X, not Y, not Z" into a positive causal statement, and demonstrating controllability (steer away → switch off).

## Design — 3 arms, paired vs the shared NoCH measure
Per cell, per dose:
- **baseline** — unsteered childhood (native N_phase2).
- **orth_dVp** — each step, project the childhood gradient ORTHOGONAL to the w_A interference direction dVp (per carrier, in V)
  → forbid the childhood from moving along the axis phase-2 attacks.
- **orth_rand** — same projection against a MATCHED RANDOM in-V direction. **The control** — rules out "any projection weakens it."
dVp = phase-2 (subj) realized from w_A, projected in-V, unit-normalized per carrier; realized ONCE per seed. Gradient-projection
hook unit-verified (after orth: `<g,dVp>`=0 exactly; orth vs an unrelated dir preserves `<g,dVp>` → control is specific).

**Cells / doses / N:** **OPT-350M (PRIMARY)** at its strong dose **3e-5** (baseline N_phase2 ≈ +0.157 — large effect → HIGH-POWER
nullify test); **BLOOM-560M (SECONDARY)** at **7e-6** (baseline ≈ −0.032 — weak eroder, lower power but included). N=15 paired seeds.

## Pre-registered read (locked before result)
- **CAUSAL-DIRECTIONAL (predicted):** orth_dVp collapses N_phase2 → ~0 (for OPT: +0.157 → ≈0; for BLOOM: −0.032 → ≈0), while
  orth_rand ≈ baseline. ⇒ the entire protect/erode effect IS the childhood moving along dVp (positive causal proof of the
  directional mechanism) AND it is CONTROLLABLE (steer away → nullify). The random-control being ≈baseline is what makes it airtight.
- **PARTIAL:** orth_dVp reduces N_phase2 but not to ~0 ⇒ dVp is a major but not sole channel (other in-V directions contribute).
- **CONFOUNDED / NULL:** orth_rand ALSO drops N_phase2 (≈ orth_dVp) ⇒ any gradient projection generically weakens the effect ⇒
  cannot attribute to dVp; reinterpret (the projection itself, not the direction, is doing the work).
- Report N_phase2 per arm with SEs + retA per arm (steering may change destructiveness).

## Scope of the claim (honest)
This tests **nullify** (+ the random control) — the crisp, achievable controllability + the causal directional proof. It does NOT
test **amplify** (concentrate onto dVp) or **flip** (make OPT erode) — deferred; a flip likely needs BLOOM's specific harmful
reorganization, which projection alone cannot inject. Claim = "the effect is causally the childhood's dVp-movement and can be
switched off," NOT "we can invert a protector."

## VOID / guards
anchor off > 0.02 → hard VOID. dataset fail → VOID. childhood-destructive arms are FINE (retA recorded; measurement still valid).
Incremental JSON dump per seed.

## Ops
Harness `cloud/probe18_steer_lambda.py` (probe13 spine + `steered_childhood` projection hook + `realize_dVp`/`rand_dir`) —
py_compile-clean; projection orthogonality + control-specificity unit-verified. Launcher: probe18 in PROBES + DOSES/NSEEDS/ANCHOR
pass-through. Reuses OPT/BLOOM w_A ckpts (local). a100-first (us-east-1 $1.99), a10 excluded; scp-hang fix live. Per cell ~$7
(3 arms × 1 dose × N=15 + NoCH/dVp realize per seed).

## Status
NOT fired (harness built + py_compile-clean + projection unit-verified). Next: LOCK → fire OPT-350m (primary, high-power) ±
BLOOM → read N_phase2 per arm (orth_dVp→0? orth_rand≈baseline?).
