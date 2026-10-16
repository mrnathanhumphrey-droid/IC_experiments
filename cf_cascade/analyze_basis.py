"""CPU-only basis-consistency analysis on the SAVED arrays (the failed GPU run persisted both
Delta_w and the K-FAC basis before crashing in the analysis arithmetic). No GPU.

ESD basis recomputed from the A-min checkpoint (w_A) on CPU to guarantee it's the SAME w_A that
Delta_w = w_B - w_A was differenced against. Compares concentration of Delta_w in:
  esd_input            : input axis on weight-ESD eigvecs (the c2 lane basis)  [reproduces ~6857]
  kfac_curvature_input : input axis on K-FAC A eigvecs (curvature basis, same reduction) [PRIMARY]
  kfac_binding_2d      : full U_G^T dW U_A binding decomposition (subsampled)
  raw_element          : raw dW^2 per element (diffuse-noise reference, subsampled)
plus ESD-vs-curvature top-k subspace overlap.

PRE-REGISTERED VERDICT (cuts mirror concentration_check, applied in kfac_curvature_input):
  CONCENTRATED-IN-CURVATURE: top20 > 0.50 OR PR < 50   (-> ESD-distrib = basis artifact)
  DISTRIBUTED-IN-BOTH:       top20 < 0.20 AND PR > 500  (-> surrogate valid in ESD; F2 early-local->late-distrib)
  INTERMEDIATE:              otherwise
"""
import json, numpy as np, torch

CKPT = "D:/IC_experiments/cf_cascade/gateA/pythia410m_sst2_subsetmin.pt"
DW_NPZ = "D:/IC_experiments/cf_cascade/gateA/deltaw_agnews.npz"
KFAC_NPZ = "D:/IC_experiments/cf_cascade/gateA/kfac_binding_basis.npz"
OUT = "D:/IC_experiments/cf_cascade/gateA/basis_consistency.json"
SQUARE = "attention.dense"


def conc(damage):
    d = damage[damage > 0]; N = len(d); tot = d.sum()
    srt = np.sort(d)[::-1]; csum = np.cumsum(srt)
    def tk(k): return float(csum[min(k, N) - 1] / tot)
    PR = float((tot ** 2) / np.sum(d ** 2))
    gs = np.sort(d); gini = float((2 * np.sum(np.arange(1, N + 1) * gs) / (N * gs.sum())) - (N + 1) / N)
    return {"N": int(N), "PR": PR, "PR_over_N": PR / N, "gini": gini,
            "top20": tk(20), "top100": tk(100), "top1pct": tk(max(1, int(0.01 * N)))}


def main():
    dw_npz = np.load(DW_NPZ); kf = np.load(KFAC_NPZ)
    names = sorted(dw_npz.files)
    sd = torch.load(CKPT, map_location="cpu")
    dW = {n: torch.tensor(dw_npz[n]).double() for n in names}
    Uesd = {}; UA = {}; UG = {}
    for n in names:
        W = sd[n].detach().double()
        _, U = torch.linalg.eigh(W.T @ W / W.shape[0]); Uesd[n] = U          # ascending
        mn = n[:-len(".weight")]                                              # module name for kfac keys
        UA[n] = torch.tensor(kf[f"{mn}__UA"]).double(); UG[n] = torch.tensor(kf[f"{mn}__UG"]).double()

    esd_dmg = []; kfacA_dmg = []; bind2d_dmg = []; raw_dmg = []; ov50 = []; ov256 = []
    rng = np.random.default_rng(0)
    for n in names:
        d = dW[n]
        esd_dmg.append(((d @ Uesd[n]) ** 2).sum(0).numpy())
        kfacA_dmg.append(((d @ UA[n]) ** 2).sum(0).numpy())
        C = (UG[n].T @ d @ UA[n]); bind2d_dmg.append((C ** 2).flatten().numpy())
        raw_dmg.append((d ** 2).flatten().numpy())
        for k, store in ((50, ov50), (256, ov256)):
            Pe = Uesd[n][:, -k:]; Pa = UA[n][:, -k:]
            store.append(float((Pe.T @ Pa).pow(2).sum().item() / k))
    sub = lambda x: x[rng.integers(0, len(x), 40000)]
    rep = {"esd_input": conc(np.concatenate(esd_dmg)),
           "kfac_curvature_input": conc(np.concatenate(kfacA_dmg)),
           "kfac_binding_2d": conc(np.concatenate([sub(x) for x in bind2d_dmg])),
           "raw_element": conc(np.concatenate([sub(x) for x in raw_dmg]))}
    overlap = {"top50_esd_vs_kfac": float(np.mean(ov50)), "top256_esd_vs_kfac": float(np.mean(ov256))}

    kc = rep["kfac_curvature_input"]
    if kc["top20"] > 0.50 or kc["PR"] < 50:
        verdict = "CONCENTRATED-IN-CURVATURE -> ESD-distributed is a BASIS ARTIFACT; force-lane localization confirmed; run surrogate in CURVATURE basis"
    elif kc["top20"] < 0.20 and kc["PR"] > 500:
        verdict = "DISTRIBUTED-IN-BOTH -> distribution basis-robust; surrogate VALID in ESD basis; F2 = early-localized -> late-distributed"
    else:
        verdict = "INTERMEDIATE -> report, decide surrogate basis with user/Wilson"

    print("\n[AB] === BASIS-CONSISTENCY (CPU, on saved Delta_w; forget was +0.083 this run) ===", flush=True)
    for k, v in rep.items():
        print(f"  {k:22s} PR={v['PR']:8.1f} (PR/N={v['PR_over_N']:.3f}) top20={v['top20']:.3f} top100={v['top100']:.3f} gini={v['gini']:.3f}", flush=True)
    print(f"  basis overlap ESD-vs-KFAC: top50={overlap['top50_esd_vs_kfac']:.3f}  top256={overlap['top256_esd_vs_kfac']:.3f}  (1=coincide, 0=orthogonal)", flush=True)
    print(f"  -> VERDICT: {verdict}", flush=True)
    json.dump({"representations": rep, "basis_overlap": overlap, "verdict": verdict,
               "note": "forget +0.083 this run (bf16/early-stop wobble vs +0.110); ESD vs curvature compared from SAME Delta_w"},
              open(OUT, "w"), indent=2)
    print(f"[AB] saved basis_consistency.json", flush=True)


if __name__ == "__main__":
    main()
