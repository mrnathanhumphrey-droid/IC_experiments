"""Yelp clause battery C1+C2 + tail-index (CPU, on deltaw_yelp.npz from Job 1). C3/C4 already in
yelp_diffusion.json. Measures Yelp damage tail-index alpha FIRST (route per decision tree; compare
to AG-News 2.62 — a big difference is itself a finding about overlap->damage-distribution).
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "4"
import json, numpy as np, torch
torch.set_num_threads(4)
from scipy.stats import spearmanr

CKPT = "D:/IC_experiments/cf_cascade/gateA/pythia410m_sst2_subsetmin.pt"
DW_NPZ = "D:/IC_experiments/cf_cascade/gateA/deltaw_yelp.npz"
KFAC_NPZ = "D:/IC_experiments/cf_cascade/gateA/kfac_binding_basis.npz"
OUT = "D:/IC_experiments/cf_cascade/gateA/yelp_clauses.json"
QS = np.array([0.5, 1.0, 2.0, 3.0]); WQ = np.array([-3., -2., -1., -0.5, 0.5, 1., 2., 3.])
N_FULL = 50; N_SHELLS = 32; ORDER = 1; rng = np.random.default_rng(20260618)


def fluct(x, scales, qs, order=ORDER):
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
    qg, hg = WQ[ok], h[ok]; tau = qg * hg - 1; al = np.gradient(tau, qg); return float(al.max() - al.min())


def hill_alpha(x, frac=0.1):
    x = np.sort(x[x > 0])[::-1]; k = max(8, int(frac * len(x))); k = min(k, len(x) - 1); xk = x[k]
    if xk <= 0: return None
    xi = float(np.mean(np.log(x[:k] / xk))); return 1.0 / xi if xi > 0 else None


def c1(damage, eigval, scales, label):
    d = damage[np.argsort(eigval)]; N = len(d)
    h, r2 = fluct(d, scales, QS); W_real = falpha_width(d, scales)
    Wf = np.array([w for w in (falpha_width(d[rng.permutation(N)], scales) for _ in range(N_FULL)) if np.isfinite(w)])
    persistent = bool(h[1] > 0.5 and h[2] > 0.5); mono = bool(W_real <= Wf.mean()); consistent = bool(persistent == mono)
    print(f"  [C1 {label}] h(1)={h[1]:.3f} h(2)={h[2]:.3f} persist={persistent} | width real={W_real:.3f}<=shuf={Wf.mean():.3f} mono={mono} | consistent={consistent}", flush=True)
    return {"h1": float(h[1]), "h2": float(h[2]), "h_q": [float(x) for x in h], "r2_q": [float(x) for x in r2],
            "W_real": W_real, "W_shuffle": float(Wf.mean()), "persistent": persistent, "monofractal_corr": mono, "consistent": consistent}


def c2(damage, eigval, label):
    rho_comp = float(spearmanr(eigval, damage).correlation)
    order = np.argsort(eigval); d = damage[order]; e = eigval[order]; N = len(d)
    edges = np.linspace(0, N, N_SHELLS + 1).astype(int); sh_e, sh_v = [], []
    for i in range(N_SHELLS):
        sd = d[edges[i]:edges[i + 1]]; se = e[edges[i]:edges[i + 1]]
        if len(sd) >= 8: sh_e.append(se.mean()); sh_v.append(sd.var())
    sh_e = np.array(sh_e); sh_v = np.array(sh_v); m = (sh_e > 0) & (sh_v > 0)
    A = np.polyfit(np.log10(sh_e[m]), np.log10(sh_v[m]), 1); pred = np.polyval(A, np.log10(sh_e[m]))
    ss = np.sum((np.log10(sh_v[m]) - np.log10(sh_v[m]).mean()) ** 2); r2 = 1 - np.sum((np.log10(sh_v[m]) - pred) ** 2) / ss
    shell_rho = float(spearmanr(sh_e[m], sh_v[m]).correlation)
    print(f"  [C2 {label}] var-vs-eig: slope={A[0]:.3f} R2={r2:.3f} shell-rho={shell_rho:+.3f} (per-comp rho={rho_comp:+.3f})", flush=True)
    return {"rho_component": rho_comp, "var_slope": float(A[0]), "var_r2": float(r2), "var_shell_rho": shell_rho}


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
    alpha = hill_alpha(kf_d)
    branch = ("alpha<3: tail-robust low-q (current DFA OK)" if alpha and alpha < 3 else "alpha>=3: moderate-p OK")
    print(f"\n[YELP-CLAUSES] N={N}", flush=True)
    print(f"  TAIL-INDEX (curvature damage) alpha = {alpha:.3f}  [AG-News was 2.62]  -> {branch}", flush=True)
    res = {"tail_index_alpha": alpha, "alpha_branch": branch,
           "C1": {"curvature": c1(kf_d, kf_e, scales, "curv"), "esd": c1(esd_d, esd_e, scales, "esd")},
           "C2": {"curvature": c2(kf_d, kf_e, "curv"), "esd": c2(esd_d, esd_e, "esd")}}
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"[YELP-CLAUSES] saved yelp_clauses.json", flush=True)


if __name__ == "__main__":
    main()
