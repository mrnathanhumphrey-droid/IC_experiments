"""Within-shell surrogate (route a) — the cascade-vs-diffusion-vs-noise test. CPU-only.

The damage field is DISTRIBUTED in both the weight-ESD and K-FAC curvature bases (gauge-robust),
and forgetting is a trajectory (localized start -> distributed end, the F2 finding). Question:
is the distributed damage a genuine MULTI-SCALE CASCADE (turbulence-transfer), or just the
DIFFUSION-from-the-localized-start, or diffuse NOISE?

MFDFA on damage ordered by eigenvalue (the spectral scale axis), tail-robust (low-q, R2-gated).
Three references discriminate (the cascade-vs-diffusion knife the F2 finding forces):
  real            : W_real (f(alpha) width + h(q) spread)
  within_shell    : shuffle damage WITHIN eigenvalue shells (preserves variance-profile + coarse
                    trend, destroys FINE cross-shell structure)  x N_SURR
  full_shuffle    : shuffle everything (keeps only the marginal = monofractal/iid baseline) x N_SURR

PRE-REGISTERED two-part verdict, PER BASIS (curvature primary/richer, ESD secondary/orthogonal):
  (1) scale-coupling present iff  W_real > p95(W_within_shell)        [destroyed by within-shell shuffle]
  (2) cascade vs diffusion vs noise:
        W_real ~ within ~ full              -> NOISE (width is just the heavy-tail marginal)
        W_real ~ within  >  full            -> DIFFUSION-from-localized-start (coarse trend, F2 restated)
        W_real  >  within  (and W_real wide)-> CASCADE (fine cross-shell coupling)
  "~" := within p95 band of that surrogate; "wide" := W_real > 0.15 (multifractal) vs <0.05 monofractal.
"""
import json, numpy as np, torch

CKPT = "D:/IC_experiments/cf_cascade/gateA/pythia410m_sst2_subsetmin.pt"
DW_NPZ = "D:/IC_experiments/cf_cascade/gateA/deltaw_agnews.npz"
KFAC_NPZ = "D:/IC_experiments/cf_cascade/gateA/kfac_binding_basis.npz"
OUT = "D:/IC_experiments/cf_cascade/gateA/surrogate_cascade.json"
N_SURR = 30; N_SHELLS = 32; QS = np.array([-3., -2., -1., -0.5, 0.5, 1., 2., 3.]); ORDER = 1
WIDE = 0.15; MONO = 0.05; rng_master = np.random.default_rng(20260618)


def mfdfa_hq(x, scales, qs, order=ORDER):
    """h(q) and log-log R^2 per q for series x via MFDFA (forward+backward segments)."""
    N = len(x); Y = np.cumsum(x - x.mean())
    Fq = np.full((len(qs), len(scales)), np.nan)
    for si, s in enumerate(scales):
        ns = N // s
        if ns < 4: continue
        F2 = []
        t = np.arange(s)
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
        pred = np.polyval(A, ls[m]); ss = np.sum((y[m] - y[m].mean()) ** 2)
        r2[qi] = 1 - np.sum((y[m] - pred) ** 2) / ss if ss > 0 else 0
    return h, r2


def width(x, scales, qs):
    """f(alpha) width and h(q) spread; restrict to q with clean (R2>0.9) log-log fit (tail-robust)."""
    h, r2 = mfdfa_hq(x, scales, qs)
    ok = (r2 > 0.9) & np.isfinite(h)
    if ok.sum() < 3: return np.nan, np.nan, int(ok.sum())
    qg, hg = qs[ok], h[ok]
    tau = qg * hg - 1; alpha = np.gradient(tau, qg); f = qg * alpha - tau
    return float(alpha.max() - alpha.min()), float(hg.max() - hg.min()), int(ok.sum())


def shell_shuffle(d, shell_id, rng):
    out = d.copy()
    for sh in np.unique(shell_id):
        idx = np.where(shell_id == sh)[0]; out[idx] = d[idx][rng.permutation(len(idx))]
    return out


def analyze(damage, eigval, label, scales):
    order = np.argsort(eigval)                      # scale axis = eigenvalue ascending
    d = damage[order]; N = len(d)
    shell_id = np.minimum((np.arange(N) * N_SHELLS) // N, N_SHELLS - 1)   # equal-count shells along eigval rank
    W_real, dh_real, nok = width(d, scales, QS)
    Wi, Wf = [], []
    for _ in range(N_SURR):
        Wi.append(width(shell_shuffle(d, shell_id, rng_master), scales, QS)[0])
        Wf.append(width(d[rng_master.permutation(N)], scales, QS)[0])
    Wi = np.array([w for w in Wi if np.isfinite(w)]); Wf = np.array([w for w in Wf if np.isfinite(w)])
    p95_i = float(np.percentile(Wi, 95)) if len(Wi) else np.nan
    p95_f = float(np.percentile(Wf, 95)) if len(Wf) else np.nan
    scale_coupling = bool(np.isfinite(W_real) and W_real > p95_i)
    diffusion_struct = bool(np.isfinite(p95_f) and np.median(Wi) > p95_f)   # within > full => coarse trend exists
    if not scale_coupling:
        verdict = ("NOISE (variance-profile/marginal) — within~full~real, no scale-coupling" if not diffusion_struct
                   else "DIFFUSION-from-localized-start — coarse trend (within>full) but no fine coupling (real~within); F2 restated, not a new cascade")
    else:  # real > within
        verdict = (f"CASCADE — fine cross-shell coupling (real>within) AND multifractal-wide (W={W_real:.3f}>{WIDE})" if W_real > WIDE
                   else f"WEAK/AMBIGUOUS — real>within (some fine coupling) but width {W_real:.3f} not wide (<{WIDE}); likely diffusion-dominated, not a clean cascade")
    return {"label": label, "N": N, "n_q_clean": nok, "W_real": W_real, "dh_real": dh_real,
            "W_within_mean": float(np.mean(Wi)) if len(Wi) else None, "W_within_p95": p95_i,
            "W_full_mean": float(np.mean(Wf)) if len(Wf) else None, "W_full_p95": p95_f,
            "scale_coupling": scale_coupling, "diffusion_struct": diffusion_struct, "verdict": verdict}


def main():
    dw_npz = np.load(DW_NPZ); kf = np.load(KFAC_NPZ); names = sorted(dw_npz.files)
    sd = torch.load(CKPT, map_location="cpu")
    esd_d, esd_e, kf_d, kf_e = [], [], [], []
    for n in names:
        d = torch.tensor(dw_npz[n]).double()
        ev, U = torch.linalg.eigh(sd[n].detach().double().T @ sd[n].detach().double() / 1024)
        esd_d.append(((d @ U) ** 2).sum(0).numpy()); esd_e.append(ev.numpy())
        mn = n[:-len(".weight")]
        UA = torch.tensor(kf[f"{mn}__UA"]).double(); eA = kf[f"{mn}__eA"]
        kf_d.append(((d @ UA) ** 2).sum(0).numpy()); kf_e.append(eA)
    esd_d = np.concatenate(esd_d); esd_e = np.concatenate(esd_e)
    kf_d = np.concatenate(kf_d); kf_e = np.concatenate(kf_e)
    N = len(esd_d); scales = np.unique(np.round(np.logspace(np.log10(16), np.log10(N // 4), 12)).astype(int))

    res = {"curvature_PRIMARY": analyze(kf_d, kf_e, "kfac_curvature", scales),
           "esd_SECONDARY": analyze(esd_d, esd_e, "weight_esd", scales)}
    print(f"\n[SURR] within-shell surrogate (N={N}, shells={N_SHELLS}, surr={N_SURR}, q-clean R2>0.9)", flush=True)
    print(f"  scales: {scales.tolist()}", flush=True)
    for k, v in res.items():
        print(f"\n  [{k}] (n_q_clean={v['n_q_clean']}/{len(QS)})", flush=True)
        print(f"    W_real={v['W_real']:.4f}  dh_real={v['dh_real']:.4f}", flush=True)
        print(f"    within-shell W: mean={v['W_within_mean']:.4f} p95={v['W_within_p95']:.4f}", flush=True)
        print(f"    full-shuffle W: mean={v['W_full_mean']:.4f} p95={v['W_full_p95']:.4f}", flush=True)
        print(f"    scale_coupling(real>within_p95)={v['scale_coupling']}  diffusion_struct(within>full_p95)={v['diffusion_struct']}", flush=True)
        print(f"    -> {v['verdict']}", flush=True)
    json.dump({"N": N, "n_shells": N_SHELLS, "n_surr": N_SURR, "scales": scales.tolist(),
               "qs": QS.tolist(), "results": res}, open(OUT, "w"), indent=2)
    print(f"\n[SURR] saved surrogate_cascade.json", flush=True)


if __name__ == "__main__":
    main()
