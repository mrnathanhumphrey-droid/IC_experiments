"""Airtight stat for the banked REVERSAL of the script's printed CASCADE verdict.
real vs the FULL full-shuffle ensemble (N=200, the clean Kantelhardt marginal-matched null),
both bases. Cascade would need W_real in the UPPER tail (W_real > shuffle). We expect the
opposite (W_real below the ensemble) -> cascade refuted. Reports z-score + empirical position.
CPU, on the saved arrays. (within-shell kept at 30 for the record but it is the confounded leg.)
"""
import json, numpy as np, torch

CKPT = "D:/IC_experiments/cf_cascade/gateA/pythia410m_sst2_subsetmin.pt"
DW_NPZ = "D:/IC_experiments/cf_cascade/gateA/deltaw_agnews.npz"
KFAC_NPZ = "D:/IC_experiments/cf_cascade/gateA/kfac_binding_basis.npz"
OUT = "D:/IC_experiments/cf_cascade/gateA/surrogate_pvalue.json"
N_FULL = 200; QS = np.array([-3., -2., -1., -0.5, 0.5, 1., 2., 3.]); ORDER = 1
rng = np.random.default_rng(20260618)


def mfdfa_hq(x, scales, qs, order=ORDER):
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


def width(x, scales, qs):
    h, r2 = mfdfa_hq(x, scales, qs); ok = (r2 > 0.9) & np.isfinite(h)
    if ok.sum() < 3: return np.nan
    qg, hg = qs[ok], h[ok]; tau = qg * hg - 1; alpha = np.gradient(tau, qg)
    return float(alpha.max() - alpha.min())


def stat(damage, eigval, scales, label):
    d = damage[np.argsort(eigval)]; N = len(d)
    W_real = width(d, scales, QS)
    Wf = np.array([w for w in (width(d[rng.permutation(N)], scales, QS) for _ in range(N_FULL)) if np.isfinite(w)])
    mu, sd = float(Wf.mean()), float(Wf.std())
    z = (W_real - mu) / sd
    n_below = int((Wf < W_real).sum())                     # shuffles LESS multifractal than real
    p_cascade = float((Wf >= W_real).sum() + 1) / (len(Wf) + 1)   # one-sided: real in UPPER tail (cascade)
    print(f"\n  [{label}] W_real={W_real:.4f}", flush=True)
    print(f"    full-shuffle ens (n={len(Wf)}): mean={mu:.4f} std={sd:.4f} min={Wf.min():.4f} max={Wf.max():.4f}", flush=True)
    print(f"    z=(real-mean)/std = {z:+.2f}   shuffles below real: {n_below}/{len(Wf)}", flush=True)
    print(f"    p(cascade; real in upper tail) = {p_cascade:.4f}  -> {'CASCADE' if p_cascade < 0.05 else 'NO cascade (real below ensemble)'}", flush=True)
    return {"label": label, "W_real": W_real, "full_mean": mu, "full_std": sd, "full_min": float(Wf.min()),
            "full_max": float(Wf.max()), "n_surr": len(Wf), "z": z, "n_shuffles_below_real": n_below,
            "p_cascade_upper_tail": p_cascade, "cascade": bool(p_cascade < 0.05)}


def main():
    dw = np.load(DW_NPZ); kf = np.load(KFAC_NPZ); names = sorted(dw.files); sd = torch.load(CKPT, map_location="cpu")
    esd_d, esd_e, kf_d, kf_e = [], [], [], []
    for n in names:
        d = torch.tensor(dw[n]).double()
        ev, U = torch.linalg.eigh(sd[n].detach().double().T @ sd[n].detach().double() / 1024)
        esd_d.append(((d @ U) ** 2).sum(0).numpy()); esd_e.append(ev.numpy())
        mn = n[:-len(".weight")]; UA = torch.tensor(kf[f"{mn}__UA"]).double()
        kf_d.append(((d @ UA) ** 2).sum(0).numpy()); kf_e.append(kf[f"{mn}__eA"])
    esd_d = np.concatenate(esd_d); esd_e = np.concatenate(esd_e); kf_d = np.concatenate(kf_d); kf_e = np.concatenate(kf_e)
    N = len(esd_d); scales = np.unique(np.round(np.logspace(np.log10(16), np.log10(N // 4), 12)).astype(int))
    print(f"[PVAL] N={N} scales={scales.tolist()} full-shuffle ensemble n={N_FULL}", flush=True)
    res = {"curvature_PRIMARY": stat(kf_d, kf_e, scales, "kfac_curvature"),
           "esd_SECONDARY": stat(esd_d, esd_e, scales, "weight_esd")}
    json.dump({"N": N, "n_full_surr": N_FULL, "scales": scales.tolist(), "results": res}, open(OUT, "w"), indent=2)
    print(f"\n[PVAL] saved surrogate_pvalue.json", flush=True)


if __name__ == "__main__":
    main()
