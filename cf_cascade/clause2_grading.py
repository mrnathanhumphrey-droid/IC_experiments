"""Clause 2 of the boxed-shape confirmation: is the diffusion CURVATURE-GRADED (not ESD-graded)?
Bin damage into eigenvalue shells; regress per-shell damage variance (and mean) on the shell's
eigenvalue, separately for the K-FAC curvature basis and the weight-ESD basis.

PRE-REGISTERED: curvature regression carries REAL grading (Spearman |rho|>0.5 AND log-log R^2>0.3);
ESD regression is WEAK (rho=0.11 per-component already showed damage barely tracks ESD spectrum),
and curvature must clearly exceed ESD. Both flat -> 'curvature-graded' breaks.
INTERNAL-CONSISTENCY: C1 gave curvature h(1)~0.98 (near-trend persistence) -> the curvature grading
should be STRONG to be consistent with that (a near-integrated DFA implies a smooth monotone trend).
If C2 curvature is flat while h~0.98, inconsistent -> flag.

CPU-only, thread-capped, on saved arrays.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "4"
import json, numpy as np, torch
torch.set_num_threads(4)
from scipy.stats import spearmanr

CKPT = "D:/IC_experiments/cf_cascade/gateA/pythia410m_sst2_subsetmin.pt"
DW_NPZ = "D:/IC_experiments/cf_cascade/gateA/deltaw_agnews.npz"
KFAC_NPZ = "D:/IC_experiments/cf_cascade/gateA/kfac_binding_basis.npz"
OUT = "D:/IC_experiments/cf_cascade/gateA/clause2_grading.json"
N_SHELLS = 32


def shell_regress(damage, eigval, label):
    # per-component Spearman first (robust, no binning choice)
    rho_comp = float(spearmanr(eigval, damage).correlation)
    order = np.argsort(eigval); d = damage[order]; e = eigval[order]; N = len(d)
    edges = np.linspace(0, N, N_SHELLS + 1).astype(int)
    sh_eig, sh_var, sh_mean = [], [], []
    for i in range(N_SHELLS):
        seg_d = d[edges[i]:edges[i + 1]]; seg_e = e[edges[i]:edges[i + 1]]
        if len(seg_d) < 8: continue
        sh_eig.append(seg_e.mean()); sh_var.append(seg_d.var()); sh_mean.append(seg_d.mean())
    sh_eig = np.array(sh_eig); sh_var = np.array(sh_var); sh_mean = np.array(sh_mean)
    pos = sh_eig > 0
    def loglog(y):
        m = pos & (y > 0)
        if m.sum() < 4: return None, None, None
        x = np.log10(sh_eig[m]); ly = np.log10(y[m])
        A = np.polyfit(x, ly, 1); pred = np.polyval(A, x)
        ss = np.sum((ly - ly.mean()) ** 2); r2 = 1 - np.sum((ly - pred) ** 2) / ss if ss > 0 else 0
        return float(A[0]), float(r2), float(spearmanr(sh_eig[m], y[m]).correlation)
    sv, sv_r2, sv_rho = loglog(sh_var)
    sm, sm_r2, sm_rho = loglog(sh_mean)
    print(f"\n  [{label}] (per-component Spearman rho(eig,damage)={rho_comp:+.3f}; {len(sh_eig)} shells)", flush=True)
    print(f"    per-shell VARIANCE vs eigval: log-log slope={sv:.3f} R2={sv_r2:.3f} shell-rho={sv_rho:+.3f}", flush=True)
    print(f"    per-shell MEAN     vs eigval: log-log slope={sm:.3f} R2={sm_r2:.3f} shell-rho={sm_rho:+.3f}", flush=True)
    return {"label": label, "rho_component": rho_comp, "n_shells": len(sh_eig),
            "var_slope": sv, "var_r2": sv_r2, "var_shell_rho": sv_rho,
            "mean_slope": sm, "mean_r2": sm_r2, "mean_shell_rho": sm_rho}


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
    res = {"curvature": shell_regress(kf_d, kf_e, "kfac_curvature"), "esd": shell_regress(esd_d, esd_e, "weight_esd")}
    cv, ev = res["curvature"], res["esd"]
    curv_graded = bool(abs(cv["var_shell_rho"]) > 0.5 and (cv["var_r2"] or 0) > 0.3)
    exceeds = bool((cv["var_r2"] or 0) > 2 * (ev["var_r2"] or 0) or abs(cv["var_shell_rho"]) > abs(ev["var_shell_rho"]) + 0.3)
    verdict = ("CURVATURE-GRADED confirmed (curvature strong, clearly exceeds ESD)" if (curv_graded and exceeds)
               else "PARTIAL: curvature grades but does not clearly exceed ESD" if curv_graded
               else "FAILS: curvature grading weak -> 'curvature-graded' clause breaks")
    print(f"\n[C2] === CLAUSE 2: {verdict} ===", flush=True)
    print(f"     curvature var R2={cv['var_r2']:.3f} rho={cv['var_shell_rho']:+.3f}  |  esd var R2={ev['var_r2']:.3f} rho={ev['var_shell_rho']:+.3f}", flush=True)
    json.dump({"results": res, "curvature_graded": curv_graded, "exceeds_esd": exceeds, "verdict": verdict}, open(OUT, "w"), indent=2)
    print(f"[C2] saved clause2_grading.json", flush=True)


if __name__ == "__main__":
    main()
