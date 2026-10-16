# PRE-REG — Detector ROUTE arm: does cluster-aware routing retain BOTH antagonistic paradigms where naive/GPM cannot? 2026-07-11.

SHA256-lock before the run. Phase-3 IMPROVEMENT on the sequential-surge result (out/seq_detector.json). That run established:
naive single head cannot hold swap AND diffusion (final swap 0.652 / diffusion 0.683); GPM protects swap (0.740) only by
STARVING diffusion (0.575) — the two are anti-parallel, no free lane; and the swap-family {swap, celebdf, dfdc} are mutually
compatible (they restore/coexist) while diffusion is the LONE antagonist. This arm tests the direct architectural consequence:
if antagonistic clusters get SEPARATE heads, we should retain BOTH — a concrete, measured improvement, not a projection.

## HARD ISOLATION (unchanged, locked)
Same sandbox. Frozen backbone `prod_models/prod_swap_seed0.pt` (read-only copy). Crops GLOBBED read-only from
`D:/corpus/face_swap_corpus`. Writes ONLY to `./out/`. NEVER touches live `prod_swap_seed*.pt` or the served detector.
$0 LOCAL CPU. Same PARADIGMS / SUBSAMPLE=220 / shared-reals / hash-split / seed as the surge run → apples-to-apples vs naive/gpm/rand.

## Design (route arm, appended to probe_detector_seq.py)
- **Data-driven router (train time).** Each arriving paradigm p gets a discriminative direction d_p = normalize(mean(Xfake_p) −
  mean(Xreal_shared)). It JOINS the existing head whose accumulated direction has cosine(d_p, d_head) > TAU (=0.0, i.e. same
  fake-vs-real direction), else SPAWNS a new head. The full cosine decision log is printed (must show swap/celebdf/dfdc cluster
  together and diffusion split off — the router is NOT hand-wired; TAU=0 separates aligned from anti-parallel).
- **Within-head training is sequential + faithful.** First paradigm of a head = fresh `train_head`; a paradigm joining an
  existing head = `continue_train` on that head (NO GPM — same-cluster paradigms are compatible by construction, so no
  interference to project away). Heads accumulate their cluster's paradigms in arrival order.
- **Two inference rules, both scored on the SAME retention test sets as the other arms:**
  - `route_oracle` — score paradigm-p's test set with the head that owns p. UPPER BOUND (assumes a family detector at inference).
  - `route_maxens` — score(x) = max over all heads of sigmoid(head(x)). LABEL-FREE, deployable (a fake trips ≥1 head; a real
    should trip none). The honest "what you can actually ship without knowing the generator."

## Pre-registered read (locked before the run)
- **PRIMARY (retains both):** route_oracle FINAL retention on swap AND diffusion simultaneously beats BOTH naive-final
  (0.652 / 0.683) and gpm-final (0.740 / 0.575). Concretely predicted: swap ≳ 0.88 (its own cluster head, undamaged by
  diffusion), diffusion ≳ 0.95 (its own head, undamaged by swaps). A single number can't be high for both in naive/gpm; route can.
- **DEPLOYABLE (label-free):** route_maxens beats naive-final on swap AND diffusion jointly (both clusters recovered without a
  family label). It MAY sit below oracle (cross-head false alarms on the shared reals cost some AUC) — quantify that gap; it is
  the price of label-free routing, reported honestly, not hidden.
- **ARCHITECTURE REFINEMENT:** the router lands 2 heads, not 4 — swap-family = ONE head (cos > TAU among swap/celebdf/dfdc),
  diffusion = its own. If the router instead splits swaps apart or merges diffusion in, the clean-cluster story is WRONG → report that.
- **NULL / failure:** if route_oracle does NOT beat gpm on both (e.g. swap-family head, holding 3 paradigms, dilutes swap below
  gpm's 0.740), routing didn't help → report the dilution, don't spin. If maxens ≈ naive, label-free routing is a wash.

## Caveats (carried + new)
- Split = filename-hash, NOT identity-disjoint → absolute AUCs inflated; the DELTAS vs naive/gpm on the SAME fixed test sets are
  the claim, robust to inflation. Head-level only (frozen backbone). SUBSAMPLE=220, single seed, one order — don't over-read sub-0.03.
- route_oracle is an UPPER BOUND: it presumes you can detect the generator family at inference. maxens is the realistic floor.
  The real deployable win lives between them; both are reported.
- The router uses the fake−real DIRECTION cosine (what a linear head actually learns), not raw feature centroids — this is the
  quantity whose anti-parallelism (cos ≈ −0.95 pairwise, cl_gpm_diffusion) is the whole mechanism. TAU=0 is the principled cut.

## Ops
Append `discriminative_dir()` + `run_route_arm()` to probe_detector_seq.py; wire into main after naive/gpm/rand; extend HEADLINE +
JSON with route_oracle / route_maxens rows and the router cosine-decision log. py_compile-clean + --smoke before the full run.
$0 local CPU, minutes. Output `out/seq_detector.json` (route arms added).

## Status
PREREG written. Next: SHA256-lock → build route arm → --smoke → full run → read oracle/maxens vs naive/gpm → bank.
