"""Gate-0(a) — structure-location check on a RAW PRETRAINED net (no task, no damage).

Settles, with data, the geometry question the knife (K7/K8) raised analytically:
  K7  index-space spatial autocorrelation of weight matrices  -> expect ~0
      (dense transformer matrices are permutation-symmetric on both axes; if
       autocorr ~ 0, matrix-layout box-averaging is VOID -> eigen-shell is
       correctly primary, not a default).
  K8  weight-ESD (Martin-Mahoney HT-SR) decade-span           -> expect >= 2 decades
      (eigen-shell geometry needs the WW^T / W^T W spectrum to span multiple
       decades with a power-law tail; if it does, eigen-shell is scale-rich
       ON THIS NET; if < 1 decade, scale-poor -> ladder up Pythia sizes).
  POP eigen-components per log-shell pooled across same-shape matrices
      -> vs c2 moment need (frugal) and KM 5000/bin floor (hungry secondary).

PRE-REGISTERED THRESHOLDS (declared before the numbers):
  K7 PASS  median |lag-1 autocorr| < 0.05   (>0.2 would REOPEN matrix-layout)
  K8 PASS  median bulk decade-span >= 2.0 AND a finite power-law tail exponent
  Report which net-size first clears K8 (the banked size-ladder finding).

This is read-only structural analysis of pretrained weights. $0 local.
No basis is frozen here, no damage measured, nothing locked -- this check is the
INPUT to the reconstruction pre-reg, run before any geometry/instrument is locked.
"""
import os, sys, json, time
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import numpy as np
import torch

MODEL = sys.argv[1] if len(sys.argv) > 1 else "EleutherAI/pythia-410m"
MIN_DIM = 256          # only analyze 2D matrices with both dims >= this
OUT = f"D:/IC_experiments/cf_cascade/gate0a_{MODEL.split('/')[-1]}.json"


def esd_eigs(W):
    """Martin-Mahoney ESD: eigenvalues of the smaller Gram matrix W^T W / n_cols
    (or W W^T). Returns sorted-descending positive eigenvalues."""
    W = W.astype(np.float64)
    m, n = W.shape
    # use the smaller Gram for speed; normalize by the contracted dimension
    if m >= n:
        G = (W.T @ W) / m
    else:
        G = (W @ W.T) / n
    ev = np.linalg.eigvalsh(G)
    ev = ev[ev > 0]
    return np.sort(ev)[::-1]


def decade_span(ev):
    """Bulk decade-span = log10(lambda_max / lambda_bulk_min), dropping the
    numerical-zero tail via a relative floor at 1e-6 * lambda_max."""
    if ev.size < 8:
        return None
    lmax = ev[0]
    floor = 1e-6 * lmax
    bulk = ev[ev > floor]
    lmin = bulk[-1]
    return {
        "lambda_max": float(lmax),
        "lambda_bulk_min": float(lmin),
        "decades_full": float(np.log10(lmax / lmin)),
        # robust span ignoring the very top/bottom 1% (less sensitive to a lone outlier)
        "decades_p1_p99": float(np.log10(np.percentile(bulk, 99) / np.percentile(bulk, 1))),
        "n_eig": int(ev.size),
        "n_bulk": int(bulk.size),
    }


def hill_alpha(ev, tail_frac=0.1):
    """Hill estimator of the power-law tail exponent alpha of the ESD
    (eigenvalue tail). HT-SR regime: alpha in ~[2,6]. Returns the
    eigenvalue-density tail exponent rho(l) ~ l^-alpha mapping."""
    ev = np.sort(ev)[::-1]
    k = max(8, int(tail_frac * ev.size))
    k = min(k, ev.size - 1)
    top = ev[:k]
    lk = ev[k]
    if lk <= 0:
        return None
    # Hill estimator for the tail index of the eigenvalue magnitudes
    xi = float(np.mean(np.log(top / lk)))
    if xi <= 0:
        return None
    tail_index = 1.0 / xi                 # P(lambda > l) ~ l^-tail_index
    alpha_density = tail_index + 1.0      # rho(l) ~ l^-alpha
    return {"tail_index": tail_index, "alpha_density": alpha_density, "k": int(k)}


def autocorr_2d(W):
    """Lag-1 spatial autocorrelation of the centered weight matrix along rows
    and columns (K7). Also a permuted control: shuffle rows+cols and recompute
    -- a genuine spatial signal would COLLAPSE under permutation; a structureless
    field is unchanged (~0 in both)."""
    def lag1(A):
        a = A - A.mean()
        v = a.var()
        if v <= 0:
            return 0.0
        # row-direction lag-1 (adjacent columns), averaged over rows
        rc = np.mean(a[:, :-1] * a[:, 1:]) / v
        # col-direction lag-1 (adjacent rows)
        cc = np.mean(a[:-1, :] * a[1:, :]) / v
        return 0.5 * (rc + cc)
    true_ac = lag1(W)
    rng = np.random.default_rng(0)
    Wp = W[rng.permutation(W.shape[0])][:, rng.permutation(W.shape[1])]
    perm_ac = lag1(Wp)
    return {"true_lag1": float(true_ac), "perm_lag1": float(perm_ac)}


def main():
    t0 = time.time()
    print(f"[gate0a] loading {MODEL} (HF_HOME={os.environ['HF_HOME']}) ...", flush=True)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    print(f"[gate0a] loaded. params = {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    rows = []
    for name, p in model.named_parameters():
        if p.ndim != 2:
            continue
        m, n = p.shape
        if min(m, n) < MIN_DIM:
            continue
        W = p.detach().cpu().numpy()
        ev = esd_eigs(W)
        ds = decade_span(ev)
        al = hill_alpha(ev)
        ac = autocorr_2d(W)
        rows.append({
            "name": name, "shape": [int(m), int(n)],
            "decade": ds, "alpha": al, "autocorr": ac,
        })
        if ds is not None:
            print(f"  {name:46s} {str([m,n]):14s} "
                  f"decades={ds['decades_full']:.2f} (p1-99 {ds['decades_p1_p99']:.2f})  "
                  f"alpha={al['alpha_density']:.2f}  "
                  f"ac_true={ac['true_lag1']:+.4f} ac_perm={ac['perm_lag1']:+.4f}", flush=True)

    # ---- aggregate by shape class ----
    from collections import defaultdict
    byshape = defaultdict(list)
    for r in rows:
        byshape[tuple(r["shape"])].append(r)

    print(f"\n[gate0a] === SHAPE-CLASS AGGREGATES ===", flush=True)
    agg = {}
    for shape, rs in sorted(byshape.items(), key=lambda kv: -kv[0][0]*kv[0][1]):
        dec = np.median([r["decade"]["decades_full"] for r in rs])
        decp = np.median([r["decade"]["decades_p1_p99"] for r in rs])
        alp = np.median([r["alpha"]["alpha_density"] for r in rs if r["alpha"]])
        act = np.median([abs(r["autocorr"]["true_lag1"]) for r in rs])
        acp = np.median([abs(r["autocorr"]["perm_lag1"]) for r in rs])
        ncomp = rs[0]["decade"]["n_eig"]
        pooled = ncomp * len(rs)
        agg[str(shape)] = {"n_matrices": len(rs), "n_eig_each": ncomp, "pooled_components": pooled,
                           "median_decades_full": float(dec), "median_decades_p1_p99": float(decp),
                           "median_alpha": float(alp), "median_abs_ac_true": float(act),
                           "median_abs_ac_perm": float(acp)}
        print(f"  shape {str(shape):14s} x{len(rs):3d}  decades={dec:.2f} (p1-99 {decp:.2f})  "
              f"alpha={alp:.2f}  |ac_true|={act:.4f} |ac_perm|={acp:.4f}  "
              f"pooled_eig={pooled}", flush=True)

    # ---- pre-registered verdicts ----
    all_ac = np.median([abs(r["autocorr"]["true_lag1"]) for r in rows])
    all_dec = np.median([r["decade"]["decades_full"] for r in rows])
    largest = max(byshape.keys(), key=lambda s: s[0]*s[1])
    largest_dec = np.median([r["decade"]["decades_full"] for r in byshape[largest]])

    k7_pass = all_ac < 0.05
    k8_pass = largest_dec >= 2.0
    print(f"\n[gate0a] === PRE-REGISTERED VERDICTS ===", flush=True)
    print(f"  K7 index-autocorr: median |lag-1| = {all_ac:.4f}  -> "
          f"{'PASS (matrix-layout VOID, eigen-shell correctly primary)' if k7_pass else 'FAIL/REOPEN matrix-layout' }", flush=True)
    print(f"  K8 ESD decade-span: largest-matrix median = {largest_dec:.2f} decades (shape {largest})  -> "
          f"{'PASS (eigen-shell scale-rich on THIS net)' if k8_pass else 'FAIL (scale-poor -> ladder up Pythia size)'}", flush=True)
    print(f"  net-wide median decade-span = {all_dec:.2f}", flush=True)

    out = {"model": MODEL,
           "params_M": float(sum(p.numel() for p in model.parameters())/1e6),
           "per_matrix": rows, "by_shape": agg,
           "verdict": {"K7_autocorr_median_abs": float(all_ac), "K7_pass": bool(k7_pass),
                       "K8_largest_decade_span": float(largest_dec), "K8_pass": bool(k8_pass),
                       "largest_shape": list(largest)},
           "wall_s": time.time() - t0}
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\n[gate0a] saved {OUT}  wall={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
