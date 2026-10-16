# PRE-REG — Detector ROUTE + intra-cluster GPM: retain BOTH antagonists at their individual best. 2026-07-11.

SHA256-lock before the run. Direct follow-on to the ROUTE arm (PREREG 2dc474a2, VERDICT_DETECTOR_ROUTE). That run showed:
(a) isolating the anti-parallel antagonist (diffusion, cos −0.93) in its OWN head rescues it fully (0.68→0.99), label-free;
(b) BUT the swap-family {swap,celebdf,dfdc} is NOT self-aligned enough (cos 0.59/0.84) to coexist for free — swap still
erodes to 0.656 INSIDE the shared head (below GPM's 0.740). This arm closes (b) without reopening (a): keep diffusion in its
own head AND run GPM *inside* the swap-family head to protect swap from the celebdf/dfdc surge.

## HARD ISOLATION (unchanged, locked)
Same sandbox / frozen `prod_models/prod_swap_seed0.pt` (read-only) / crops globbed read-only from `D:/corpus/face_swap_corpus` /
writes ONLY to `./out/`. NEVER touches live prod ensemble or served detector. $0 LOCAL CPU. Same PARADIGMS / SUBSAMPLE=220 /
shared-reals / hash-split / seed / TAU=0 router as the ROUTE run → apples-to-apples vs naive / gpm / rand / route(plain).

## Design (new arm `route_gpm`, appended to probe_detector_seq.py)
- IDENTICAL data-driven router (TAU=0 on fake−real direction cosine) → same landed heads: H0={swap,celebdf,dfdc}, H1={diffusion}.
  (Router is unchanged; if it lands different heads, that itself is reported.)
- **The ONLY change vs plain route:** when a paradigm JOINS an existing head, its continue-train gradient is GPM-projected
  orthogonal to the head's ACCUMULATED-prior-members' memory subspace U = orthonormalize(∪ pca_basis(prior member's fake feats,
  K=PCA_K)) — exactly the surge-arm GPM construction, but applied WITHIN a head to protect that head's earlier paradigms.
  A SPAWNED (first-in-head) paradigm trains fresh (no priors to protect). Diffusion, alone in H1, is never projected → keeps 0.99.
- Scored the same two ways on the SAME retention test sets: `route_gpm/oracle` (head that owns p) and `route_gpm/maxens`
  (max sigmoid over heads, label-free deployable).

## Pre-registered read (locked before the run)
- **PRIMARY (retain BOTH antagonists at individual-best):** route_gpm/oracle FINAL retention predicted swap ≈ 0.74 (GPM-level,
  protected inside H0) AND diffusion ≈ 0.99 (own head) SIMULTANEOUSLY. Concretely: swap ≥ 0.72 (≥ plain-route's 0.656 AND
  ≥ within-noise of full-GPM's 0.740) AND diffusion ≥ 0.95. This is the combination NONE of naive (0.65/0.68), gpm-single-head
  (0.74/0.57 — diffusion starved), or plain-route (0.66/0.99 — swap not protected) achieves.
- **THE min-metric:** min(swap, diffusion) for route_gpm/oracle > all of: naive (0.652), gpm (0.575), plain-route (0.656).
  Predicted min ≈ 0.72–0.74 (swap-bound). Higher min = retains both.
- **maxens (label-free):** route_gpm/maxens swap improves over plain-route/maxens (0.639) toward ~0.72 while diffusion stays ≥0.95.
- **NULL / failure to report honestly:** if intra-cluster GPM does NOT lift swap above plain-route (0.656) — e.g. celebdf/dfdc
  surge doesn't actually live in swap's protected subspace, or projecting starves celebdf/dfdc instead — report the no-lift and/or
  the celebdf/dfdc cost. If swap rises but celebdf/dfdc fall below their plain-route values (0.894/0.936) by >0.03, GPM bought
  swap by starving its own cluster-mates → report that trade, don't hide it.

## Caveats (carried)
Filename-hash split (not identity-disjoint) → absolute AUCs inflated; DELTAS on fixed test sets are the claim. Head-level only
(frozen backbone). SUBSAMPLE=220, single seed, one order — don't over-read sub-0.03. oracle = upper bound (family detector at
inference); maxens = deployable floor. Intra-cluster GPM adds a projection cost inside H0; the celebdf/dfdc retention is the check.

## Ops
Append `intra_gpm` path to `run_route_arm()` (build U from prior members' fake PCA when a paradigm joins; unchanged elsewhere);
add a `route_gpm` call in main; extend HEADLINE + JSON with route_gpm/oracle + route_gpm/maxens + the min-metric comparison.
py_compile-clean + --smoke before the full run. $0 local CPU, minutes. Output `out/seq_detector.json` (route_gpm arm added).

## Status
PREREG written. Next: SHA256-lock → build intra_gpm path → --smoke → full run → read swap+diffusion joint retention vs all arms → bank.
