"""Clause 1 (GATE) of the boxed-shape confirmation: is the curvature-ordered damage field a
PERSISTENT (H>0.5) MONOFRACTAL? Load-bearing — if H<=0.5 the 'smoother-than-iid' reading was a
fat-tail artifact and the whole boxed shape breaks; stop and reconsider.

Two sub-tests, PRE-REGISTERED:
  (a) persistence: single-Hurst via DFA. Tail-robust PRIMARY = h(1) (first-moment, converges for
      alpha>1); standard = h(2). Predict BOTH > 0.5.
  (b) monofractal-in-correlation: the f(alpha) width is fat-tail-spurious (established last run:
      real < full-shuffle). Monofractal-correlation => real width <= full-shuffle width (no
      POSITIVE multifractal excess over the marginal). Predict real_width <= shuffle_width.

INTERNAL-CONSISTENCY GATE (false-CASCADE lesson): h(2)>0.5 must agree with real<full-shuffle
width. If DFA says persistent but widths disagree, one is mis-estimated -> DO NOT BANK.

CPU-only, thread-capped, on saved deltaw_agnews.npz + kfac_binding_basis.npz + w_A checkpoint.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "4"
import json, numpy as np, torch
torch.set_num_threads(4)

CKPT = "D:/IC_experiments/cf_cascade/gateA/pythia410m_sst2_subsetmin.pt"
DW_NPZ = "D:/IC_experiments/cf_cascade/gateA/deltaw_agnews.npz"
KFAC_NPZ = "D:/IC_experiments/cf_cascade/gateA/kfac_binding_basis.npz"
OUT = "D:/IC_experiments/cf_cascade/gateA/clause1_hurst.json"
QS = np.array([0.5, 1.0, 2.0, 3.0]); WQ = np.array([-3., -2., -1., -0.5, 0.5, 1., 2., 3.])
N_FULL = 50; ORDER = 1; rng = np.random.default_rng(20260618)


def fluct(x, scales, qs, order=ORDER):
    """generalized DFA fluctuation F_q(s) and per-q log-log slope h(q) + R^2."""
    N = len(x); Y = np.cumsum(x - x.mean()); Fq = np.full((len(qs), len(scales)), np.nan)
    for si, s in enumerate(scales):
        ns = N // s
        if ns < 4: continue
        F2 = []; t = np.arange(s)
        for v in range(ns):
            for seg in (Y[v * s:(v + 1) * s], Y[N - (v + 1) * s:N - v * s]):
                c = np.polyfit(t, seg, order); F2.append(np.mean((seg - np.polyval(c, t)) ** 2))
        F2 = np.array(F2); F2 = F2[F2 > 0]
        if len(F2) < 4: continue
        for qi, q in enumerate(qs):
            Fq[qi, si] = np.exp(0.5 * np.mean(np.log(F2))) if abs(q) < 1e-9 else (np.mean(F2 ** (q / 2))) ** (1 / q)
    ls = np.log(scales); h = np.full(len(qs), np.nan); r2 = np.zeros(len(qs))
    for qi in range(len(qs)):
        y = np.log(Fq[qi]); m = np.isfinite(y)
        if m.sum() < 4: continue
        A = np.polyfit(ls[m], y[m], 1); h[qi] = A[0]
        ss = np.sum((y[m] - y[m].mean()) ** 2); r2[qi] = 1 - np.sum((y[m] - np.polyval(A, ls[m])) ** 2) / ss if ss > 0 else 0
    return h, r2


def falpha_width(x, scales):
    h, r2 = fluct(x, scales, WQ); ok = (r2 > 0.9) & np.isfinite(h)
    if ok.sum() < 3: return np.nan
    qg, hg = WQ[ok], h[ok]; tau = qg * hg - 1; al = np.gradient(tau, qg)
    return float(al.max() - al.min())


def analyze(damage, eigval, scales, label):
    d = damage[np.argsort(eigval)]; N = len(d)
    h, r2 = fluct(d, scales, QS)                                  # h(0.5),h(1),h(2),h(3)
    W_real = falpha_width(d, scales)
    Wf = np.array([w for w in (falpha_width(d[rng.permutation(N)], scales) for _ in range(N_FULL)) if np.isfinite(w)])
    h1, h2 = h[1], h[2]; r2_1, r2_2 = r2[1], r2[2]
    persistent = bool(np.isfinite(h1) and np.isfinite(h2) and h1 > 0.5 and h2 > 0.5)
    monofractal_corr = bool(np.isfinite(W_real) and W_real <= Wf.mean())
    consistent = bool(persistent == monofractal_corr)             # H>0.5 <=> real<shuffle width
    print(f"\n  [{label}] N={N}", flush=True)
    print(f"    h(q): " + "  ".join(f"q={q}:{hh:.3f}(R2={rr:.2f})" for q, hh, rr in zip(QS, h, r2)), flush=True)
    print(f"    tail-robust h(1)={h1:.3f}  standard h(2)={h2:.3f}  -> persistent(>0.5)={persistent}", flush=True)
    print(f"    f(a) width: real={W_real:.3f}  full-shuffle mean={Wf.mean():.3f} (n={len(Wf)}) -> monofractal-corr(real<=shuffle)={monofractal_corr}", flush=True)
    print(f"    INTERNAL-CONSISTENCY (persistent <=> monofractal-corr): {consistent}", flush=True)
    return {"label": label, "N": N, "h_q": {str(q): (None if not np.isfinite(hh) else float(hh)) for q, hh in zip(QS, h)},
            "r2_q": {str(q): float(rr) for q, rr in zip(QS, r2)}, "h1_tailrobust": float(h1), "h2_standard": float(h2),
            "r2_h1": float(r2_1), "r2_h2": float(r2_2), "W_real": W_real, "W_full_shuffle_mean": float(Wf.mean()),
            "persistent": persistent, "monofractal_corr": monofractal_corr, "internal_consistent": consistent}


def main():
    dw = np.load(DW_NPZ); kf = np.load(KFAC_NPZ); names = sorted(dw.files); sd = torch.load(CKPT, map_location="cpu")
    kf_d, kf_e, esd_d, esd_e = [], [], [], []
    for n in names:
        d = torch.tensor(dw[n]).double()
        ev, U = torch.linalg.eigh(sd[n].detach().double().T @ sd[n].detach().double() / 1024)
        esd_d.append(((d @ U) ** 2).sum(0).numpy()); esd_e.append(ev.numpy())
        mn = n[:-len(".weight")]; UA = torch.tensor(kf[f"{mn}__UA"]).double()
        kf_d.append(((d @ UA) ** 2).sum(0).numpy()); kf_e.append(kf[f"{mn}__eA"])
    kf_d = np.concatenate(kf_d); kf_e = np.concatenate(kf_e); esd_d = np.concatenate(esd_d); esd_e = np.concatenate(esd_e)
    N = len(kf_d); scales = np.unique(np.round(np.logspace(np.log10(16), np.log10(N // 4), 12)).astype(int))
    print(f"[C1] N={N} scales={scales.tolist()} (DFA order={ORDER}, tail-robust primary h(1))", flush=True)
    res = {"curvature_PRIMARY": analyze(kf_d, kf_e, scales, "kfac_curvature"),
           "esd_SECONDARY": analyze(esd_d, esd_e, scales, "weight_esd")}
    c = res["curvature_PRIMARY"]
    gate = "PASS (persistent monofractal, internally consistent) -> proceed to Clause 2" if (c["persistent"] and c["monofractal_corr"] and c["internal_consistent"]) else "FAIL/INCONSISTENT -> STOP, reconsider boxed shape"
    print(f"\n[C1] === CLAUSE 1 GATE (curvature): {gate} ===", flush=True)
    json.dump({"N": N, "scales": scales.tolist(), "results": res, "gate": gate}, open(OUT, "w"), indent=2)
    print(f"[C1] saved clause1_hurst.json", flush=True)


if __name__ == "__main__":
    main()
