# PREREG — DOMAIN-CONTROLLED re-test of the 4-op deepfake cosine "law"

**Locked 2026-07-26.** $0 local CPU, frozen backbone `prod_swap_seed0.pt` (128-d), corpus `D:/corpus/face_swap_corpus`. Reuses probe_detector_seq machinery byte-faithfully (mean-difference directions, residual-Q100 crops).

## Why
The arc claims ONE discriminative-direction cosine governs 4 ops (LEARN/FORGET route, UNLEARN collateral ∝ +cos, PROTECT leakage). Load-bearing fact: diffusion is anti-parallel to the swap family (cos ≈ −0.93). But all IC probes compute d_family = normalize(mean(fake_family) − mean(SHARED ffpp/real)). Diffusion fakes live in the IMDB-WIKI (deepfakeface) domain, so d_diffusion carries the wiki−ffpp DOMAIN offset. A one-off check (physics_detector/…/antagonism_confound_result.json) already found for the swap↔diffusion PAIR: cos_shared −0.954, cos_forgery **+0.669** (diffusion fake vs its OWN wiki real), cos_domain −0.909, ‖domain‖ 7.52 vs ‖forgery‖ 1.29 (5.8×), ref-sweep ffpp/shared/wiki = −0.937/−0.954/+0.669. This re-test EXTENDS that single pair to the FULL 4-op geometry with CIs + the operational routing test — which the one-off never did.

## Design (3 references, same frozen 128-d features)
For each paradigm p in [swap=ffpp, diffusion=deepfakeface, celebdf, dfdc]: extract features for fake_p, own-dataset real_p, and the shared ffpp/real pool.
- **d_shared[p]** = norm(mean(fake_p) − mean(ffpp_real))    [ORIGINAL confounded method]
- **d_forgery[p]** = norm(mean(fake_p) − mean(own_real_p))   [DOMAIN-CONTROLLED forgery direction]
- **r_domain[p]** = norm(mean(own_real_p) − mean(ffpp_real)) [PURE domain axis; ~0 for ffpp]
Build 4x4 cosine matrices C_shared, C_forgery, C_domain. Bootstrap over crops (B=500, resample fake+real sets) → 95% CI on every off-diagonal cosine.

## Part 2 — does the operational LAW track forgery or domain?
1. **Matrix source:** across the 6 off-diagonal pairs, corr(C_shared_off, C_forgery_off) vs corr(C_shared_off, C_domain_off). The operative cosine (C_shared, what the router/unlearn/protect USED) is explained by whichever it correlates with.
2. **Routing flip:** deterministic sequential router (paradigm joins nearest head if max cos ≥ τ, else spawns) over [swap,diffusion,celebdf,dfdc], τ-sweep. Does diffusion ISOLATE under C_shared but JOIN swap under C_forgery? Report cluster assignments both.
3. **Scope:** per-pair — does ONLY diffusion's antagonism flip (domain artifact), or do the swap-family internal cosines (celebdf/dfdc +0.96, swap +0.59/+0.43) ALSO change? Real forgery structure should be domain-invariant.

## Pre-registered decision (commit BEFORE numbers)
- **LAW SURVIVES (real 4-op forgery geometry):** C_forgery ≈ C_shared (corr ≥ 0.7 AND swap↔diffusion forgery-cos CI still < 0), i.e. domain control does NOT change the geometry; operations track forgery.
- **DOMAIN ARTIFACT (law falsified as FORGERY law):** swap↔diffusion forgery-cos CI > 0 (flips positive) AND corr(C_shared,C_domain) > corr(C_shared,C_forgery) AND diffusion stops isolating under C_forgery at the τ that isolated it under C_shared. Then: the operational wins (isolating out-of-domain generators) are REAL but the "one forgery cosine governs 4 ops" causal framing — and the audio_cf/text triangulation leaning on it — is a DOMAIN-DISTANCE phenomenon mislabeled as forgery geometry.
- **NUANCED (report honestly):** which pairs flip. If diffusion-only flips while swap-family internal cosines survive domain-control, the swap-family clustering is real forgery geometry and only the diffusion "antagonist" is domain.

## Falsifier-first
$0 CPU, cached features. Bootstrap CIs make every cosine falsifiable. Every outcome decision-relevant + honest. SMOKE (SUBSAMPLE small, 2 paradigms) before full. USER writes; Claude probes.
