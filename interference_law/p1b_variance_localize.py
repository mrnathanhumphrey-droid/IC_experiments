"""
p1b_variance_localize.py -- Transfer-geometry P1b tightening (2026-07-26). Where does the RELIABLE frozen-FT variance live?
Decompose FT(i<-j) = mu + row_i(target) + col_j(source) + resid_ij(interaction); per-component split-half reliability.
Plus NORM-CONTROL: re-measure FT with all B-inits scaled to a common norm -> does the source/scale signal collapse & does
reliability survive? $0 local, frozen resnet18 D_PROJ=24, CIFAR+DTD. Reuses P1 helpers. ASCII-only.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R
import probe_dirlaw_arch as A
import probe_transfer as XF
import p1_descriptor_bakeoff as P1   # build(), constants

DEV = A.DEV
NSEED = R._i("NSEED", 10); K = P1.K; STEPS_FULL = P1.STEPS_FULL; BOOT = R._i("BOOT", 2000)
DATASETS = os.environ.get("DATASETS", "cifar,dtd").split(",")
OUT = HERE / "out" / "p1b_localize.json"
torch.manual_seed(0); np.random.seed(0)


def spearman(a, b):
    return float(R.spearman(np.asarray(a, float), np.asarray(b, float)))


def scaled_state(sd, w, b):
    sd = {k: v.clone() for k, v in sd.items()}
    for k in sd:
        if sd[k].numel() == w.size: sd[k] = torch.tensor(w, dtype=sd[k].dtype, device=sd[k].device).view_as(sd[k])
        else: sd[k] = torch.tensor([b], dtype=sd[k].dtype, device=sd[k].device).view_as(sd[k])
    return sd


def ft_matrices(caps, n, dim, task, s, common_norm=None):
    """One seed: return raw-init FT and (if common_norm) norm-init FT. diag=nan."""
    cold_s = np.zeros(n); hB = {}; wnorm = np.zeros(n)
    for ci, i in enumerate(caps):
        hc = XF.train_head(dim, task[i]["Xfew"], task[i]["yfew"], K, seed=7000*s + ci)
        cold_s[ci] = XF.auc_head(hc, task[i]["Xte"], task[i]["yte"])
        hB[i] = XF.train_head(dim, task[i]["Xfull"], task[i]["yfull"], STEPS_FULL, seed=7000*s + ci)
        w = hB[i].weight.detach().cpu().numpy().ravel(); b = float(hB[i].bias.detach().cpu().numpy().ravel()[0])
        wnorm[ci] = np.sqrt(np.sum(w**2) + b*b)
    RAW = np.full((n, n), np.nan); NRM = np.full((n, n), np.nan) if common_norm else None
    for ci, i in enumerate(caps):
        for cj, j in enumerate(caps):
            if i == j: continue
            sd = hB[j].state_dict()
            init = {k: v.clone() for k, v in sd.items()}
            hbi = XF.train_head(dim, task[i]["Xfew"], task[i]["yfew"], K, seed=7000*s + ci, init=init)
            RAW[ci, cj] = XF.auc_head(hbi, task[i]["Xte"], task[i]["yte"]) - cold_s[ci]
            if common_norm:
                w = hB[j].weight.detach().cpu().numpy().ravel().astype(np.float64); b = float(hB[j].bias.detach().cpu().numpy().ravel()[0])
                pn = np.sqrt(np.sum(w**2) + b*b); sc = common_norm / (pn + 1e-12)
                init2 = scaled_state(sd, (w*sc).astype(np.float32), float(b*sc))
                hbi2 = XF.train_head(dim, task[i]["Xfew"], task[i]["yfew"], K, seed=7000*s + ci, init=init2)
                NRM[ci, cj] = XF.auc_head(hbi2, task[i]["Xte"], task[i]["yte"]) - cold_s[ci]
    return RAW, NRM, wnorm


def decomp(M, n):
    """FT = mu + row_i + col_j + resid. Return component vectors on off-diag cells + var fractions."""
    m = ~np.eye(n, dtype=bool)
    mu = np.nanmean(M[m])
    row = np.nanmean(M, axis=1, keepdims=True) - mu       # (n,1) target effect
    col = np.nanmean(M, axis=0, keepdims=True) - mu       # (1,n) source effect
    ROW = np.broadcast_to(row, (n, n)); COL = np.broadcast_to(col, (n, n))
    RES = M - mu - ROW - COL
    rv, cv, ev, tv = ROW[m], COL[m], RES[m], (M[m]-mu)
    frac = lambda x: float(np.nanvar(x) / (np.nanvar(tv) + 1e-12))
    return {"row": rv, "col": cv, "res": ev, "tot": tv,
            "var_row": frac(rv), "var_col": frac(cv), "var_res": frac(ev), "mu": float(mu)}


def rel(a, b):
    r = spearman(a, b); return 2*r/(1+r) if r > -1 else float("nan")


def run(name):
    t0 = time.time(); print(f"[P1b] === {name} ===", flush=True)
    caps, n, dim, task = P1.build(name)
    # pass 1: get typical norm from seed 0
    _, _, wnorm0 = ft_matrices(caps, n, dim, task, 0, common_norm=None)
    common = float(np.median(wnorm0)); print(f"  [{name}] common_norm={common:.3f}", flush=True)
    raws = np.zeros((NSEED, n, n)); nrms = np.zeros((NSEED, n, n)); wn = np.zeros(n)
    for s in range(NSEED):
        RAW, NRM, wnorm = ft_matrices(caps, n, dim, task, s, common_norm=common)
        raws[s] = RAW; nrms[s] = NRM; wn = wnorm
        print(f"  [{name}] seed {s+1}/{NSEED} ({time.time()-t0:.0f}s)", flush=True)
    m = ~np.eye(n, dtype=bool)
    out = {"n": n, "common_norm": round(common, 3)}
    for tag, arr in (("raw", raws), ("norm", nrms)):
        mean = np.nanmean(arr, axis=0)
        even = np.nanmean(arr[0::2], axis=0); odd = np.nanmean(arr[1::2], axis=0)
        r_sb = rel(even[m], odd[m])
        d = decomp(mean, n)
        de = decomp(even, n); do = decomp(odd, n)
        rel_row = rel(de["row"], do["row"]); rel_col = rel(de["col"], do["col"]); rel_res = rel(de["res"], do["res"])
        # reliable-variance share = var_frac * component_reliability, renormalized
        rr = max(d["var_row"]*max(rel_row, 0), 0); rc = max(d["var_col"]*max(rel_col, 0), 0); re_ = max(d["var_res"]*max(rel_res, 0), 0)
        tot = rr+rc+re_ + 1e-12
        out[tag] = {"FT_reliability": round(r_sb, 4),
                    "var_frac": {"row_target": round(d["var_row"], 3), "col_source": round(d["var_col"], 3), "resid_interaction": round(d["var_res"], 3)},
                    "component_reliability": {"row": round(rel_row, 3), "col": round(rel_col, 3), "resid": round(rel_res, 3)},
                    "reliable_var_share": {"row_target": round(rr/tot, 3), "col_source": round(rc/tot, 3), "resid_interaction": round(re_/tot, 3)}}
        print(f"  [{name}/{tag}] FT r_SB={r_sb:+.3f} | var_frac row={d['var_row']:.2f} col={d['var_col']:.2f} resid={d['var_res']:.2f}"
              f" | comp_rel row={rel_row:+.2f} col={rel_col:+.2f} resid={rel_res:+.2f}"
              f" | RELIABLE-share row={rr/tot:.2f} col={rc/tot:.2f} resid={re_/tot:.2f}", flush=True)
    # does ||w_j|| explain the SOURCE (col) effect? correlate col-effect (raw) with wnorm
    mean_raw = np.nanmean(raws, axis=0); col_raw = np.nanmean(mean_raw, axis=0) - np.nanmean(mean_raw[m])
    rho_wnorm_col = spearman(wn, col_raw)
    # norm-control: is col-effect variance reduced under normalization?
    mean_nrm = np.nanmean(nrms, axis=0); col_nrm = np.nanmean(mean_nrm, axis=0) - np.nanmean(mean_nrm[m])
    out["col_effect"] = {"rho_wnorm_vs_col_raw": round(rho_wnorm_col, 4),
                         "col_var_raw": round(float(np.var(col_raw)), 6), "col_var_norm": round(float(np.var(col_nrm)), 6),
                         "col_var_ratio_norm_over_raw": round(float(np.var(col_nrm)/(np.var(col_raw)+1e-12)), 3)}
    print(f"  [{name}] rho(||w_j||, col_source_effect)={rho_wnorm_col:+.3f} | col-var raw={np.var(col_raw):.5f} -> norm={np.var(col_nrm):.5f} "
          f"(ratio {np.var(col_nrm)/(np.var(col_raw)+1e-12):.2f}) ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    t0 = time.time(); print(f"[P1b] LOCALIZE reliable FT variance | nseed={NSEED} dev={DEV}", flush=True)
    results = {}
    for name in DATASETS:
        try: results[name] = run(name)
        except Exception as e:
            import traceback; traceback.print_exc(); results[name] = {"error": f"{type(e).__name__}: {str(e)[:180]}"}
        OUT.write_text(json.dumps({"config": {"nseed": NSEED, "note": "P1b variance localization + norm control"}, "results": results}, indent=1))
    print("\n=== P1b: reliable-FT variance localization ===", flush=True)
    for nme, r in results.items():
        if "error" in r: print(f"  {nme}: ERROR {r['error'][:60]}"); continue
        rs = r["raw"]["reliable_var_share"]
        print(f"  {nme:6} RAW reliable-share: target(row)={rs['row_target']:.2f} source(col)={rs['col_source']:.2f} interaction(resid)={rs['resid_interaction']:.2f}"
              f" | rho(||w||,col)={r['col_effect']['rho_wnorm_vs_col_raw']:+.2f} col-var norm/raw={r['col_effect']['col_var_ratio_norm_over_raw']:.2f}", flush=True)
    print(f"[P1b] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
