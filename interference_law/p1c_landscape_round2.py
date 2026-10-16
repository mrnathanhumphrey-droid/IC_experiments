"""
p1c_landscape_round2.py -- Transfer-geometry P1c (PREREG P1c_2026_07_26). ROUND-2 descriptor hunt for the PROVEN frozen-FT
INTERACTION (P1b: ~76% of reliable variance, scale-invariant, direction-blind). Optimization-landscape descriptors, unit-normalized
w_j, primary target = interaction RESIDUAL. Strict bar |rho|>=0.30 & cross-dataset & perm_p<0.01. $0 local. ASCII-only.
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
import p1_descriptor_bakeoff as P1

DEV = A.DEV
NSEED = R._i("NSEED", 10); K = P1.K; STEPS_FULL = P1.STEPS_FULL; KPC = R._i("KPC", 5); BOOT = R._i("BOOT", 2000); PERM = R._i("PERM", 2000)
DATASETS = os.environ.get("DATASETS", "cifar,dtd").split(",")
OUT = HERE / "out" / "p1c_landscape.json"
torch.manual_seed(0); np.random.seed(0)


def spearman(a, b):
    return float(R.spearman(np.asarray(a, float), np.asarray(b, float)))


def sigmoid(z): return 1.0/(1.0+np.exp(-z))


def ft_seed(caps, n, dim, task, s):
    cold = np.zeros(n); hB = {}
    for ci, i in enumerate(caps):
        hc = XF.train_head(dim, task[i]["Xfew"], task[i]["yfew"], K, seed=7000*s+ci)
        cold[ci] = XF.auc_head(hc, task[i]["Xte"], task[i]["yte"])
        hB[i] = XF.train_head(dim, task[i]["Xfull"], task[i]["yfull"], STEPS_FULL, seed=7000*s+ci)
    M = np.full((n, n), np.nan)
    for ci, i in enumerate(caps):
        for cj, j in enumerate(caps):
            if i == j: continue
            init = {k: v.clone() for k, v in hB[j].state_dict().items()}
            hbi = XF.train_head(dim, task[i]["Xfew"], task[i]["yfew"], K, seed=7000*s+ci, init=init)
            M[ci, cj] = XF.auc_head(hbi, task[i]["Xte"], task[i]["yte"]) - cold[ci]
    return M


def grad_span(dim, task, i, k):
    """top-k PCs of the cold few-shot gradient trajectory for task i (numpy logistic GD)."""
    Xf = task[i]["Xfew"].astype(np.float64); yf = task[i]["yfew"].astype(np.float64); nf = len(yf)
    w = np.zeros(dim); b = 0.0; lr = 0.1; gs = []
    for _ in range(K):
        p = sigmoid(Xf@w + b); g = Xf.T@(p-yf)/nf; gb = float(np.mean(p-yf))
        gs.append(g.copy()); w -= lr*g; b -= lr*gb
    G = np.array(gs)  # (K, dim)
    Gc = G - G.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Gc, full_matrices=False)
    return Vt[:k].T  # (dim,k)


def descriptors(caps, n, dim, task):
    # trained heads (deterministic seed 0), unit-normalized [w,b]
    W = {}; Bb = {}
    for i in caps:
        h = XF.train_head(dim, task[i]["Xfull"], task[i]["yfull"], STEPS_FULL, seed=0)
        w = h.weight.detach().cpu().numpy().ravel().astype(np.float64); b = float(h.bias.detach().cpu().numpy().ravel()[0])
        pn = np.sqrt(np.sum(w**2)+b*b)+1e-12; W[i] = w/pn; Bb[i] = b/pn
    D = {c: task[c]["d"].astype(np.float64) for c in caps}
    C = {c: np.cov(task[c]["pos"].astype(np.float64).T) for c in caps}          # (dim,dim)
    S = {c: grad_span(dim, task, c, KPC) for c in caps}                          # reachable subspace
    names = ["Y1_reach", "Y2_newton", "Y3_spread", "Y4_whiten", "Y5_usage", "Y6_curvtr"]
    mats = {nm: np.full((n, n), np.nan) for nm in names}
    for ci, i in enumerate(caps):
        Xf = task[i]["Xfew"].astype(np.float64); yf = task[i]["yfew"].astype(np.float64); nf = len(yf)
        Si = S[i]; Ci = C[i]; di = D[i]; adi = np.abs(di)
        for cj, j in enumerate(caps):
            if i == j: continue
            w = W[j]; b = Bb[j]
            z = Xf@w + b; p = sigmoid(z)
            g = Xf.T@(p-yf)/nf
            Hs = (Xf * (p*(1-p))[:, None]).T @ Xf / nf + 1e-4*np.eye(dim)
            mats["Y1_reach"][ci, cj] = float(np.sum((Si.T@w)**2))
            try: mats["Y2_newton"][ci, cj] = float(g @ np.linalg.solve(Hs, g))
            except Exception: mats["Y2_newton"][ci, cj] = np.nan
            mats["Y3_spread"][ci, cj] = float(np.var(z))
            mats["Y4_whiten"][ci, cj] = float(w @ Ci @ di)
            mats["Y5_usage"][ci, cj] = spearman(np.abs(w), adi)
            mats["Y6_curvtr"][ci, cj] = float(np.trace(Hs))
    return names, mats


def decomp_resid(M, n):
    m = ~np.eye(n, dtype=bool); mu = np.nanmean(M[m])
    row = np.nanmean(M, axis=1, keepdims=True) - mu; col = np.nanmean(M, axis=0, keepdims=True) - mu
    RES = M - mu - np.broadcast_to(row, (n, n)) - np.broadcast_to(col, (n, n))
    return RES


def corr_stats(x, y):
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]; nrep = len(x); rng = np.random.RandomState(11)
    rho = spearman(x, y)
    rb = np.array([spearman(x[idx], y[idx]) for idx in (rng.randint(0, nrep, nrep) for _ in range(BOOT))])
    ci = [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))]
    perm = np.array([spearman(x, rng.permutation(y)) for _ in range(PERM)]); p = float((np.abs(perm) >= abs(rho)).mean())
    return rho, ci, p


def run(name):
    t0 = time.time(); print(f"[P1c] === {name} ===", flush=True)
    caps, n, dim, task = P1.build(name)
    seeds = np.zeros((NSEED, n, n))
    for s in range(NSEED):
        seeds[s] = ft_seed(caps, n, dim, task, s)
        if (s+1) % 5 == 0: print(f"  [{name}] FT seed {s+1}/{NSEED} ({time.time()-t0:.0f}s)", flush=True)
    FT = np.nanmean(seeds, axis=0); RES = decomp_resid(FT, n)
    names, mats = descriptors(caps, n, dim, task); print(f"  [{name}] descriptors built ({time.time()-t0:.0f}s)", flush=True)
    m = ~np.eye(n, dtype=bool); resid = RES[m]; ftv = FT[m]
    res = {}
    for nm in names:
        xv = mats[nm][m]
        r_res, ci_res, p_res = corr_stats(xv, resid)
        r_ft, ci_ft, p_ft = corr_stats(xv, ftv)
        hit = bool(abs(r_res) >= 0.30 and (ci_res[0] > 0 or ci_res[1] < 0) and p_res < 0.01)
        res[nm] = {"rho_resid": round(r_res, 4), "ci_resid": [round(ci_res[0], 4), round(ci_res[1], 4)], "perm_p_resid": round(p_res, 4),
                   "rho_ft": round(r_ft, 4), "perm_p_ft": round(p_ft, 4), "hit": hit}
        print(f"  [{name}] {nm:12} resid rho={r_res:+.3f} CI[{ci_res[0]:+.3f},{ci_res[1]:+.3f}] p={p_res:.3f} | ft rho={r_ft:+.3f} p={p_ft:.3f} {'<< HIT' if hit else ''}", flush=True)
    return {"n": n, "descriptors": res}


def main():
    t0 = time.time(); print(f"[P1c] LANDSCAPE ROUND-2 | nseed={NSEED} kpc={KPC} boot={BOOT} perm={PERM} dev={DEV} | primary=RESID, bar |rho|>=.30 & perm_p<.01 & both-datasets", flush=True)
    results = {}
    for name in DATASETS:
        try: results[name] = run(name)
        except Exception as e:
            import traceback; traceback.print_exc(); results[name] = {"error": f"{type(e).__name__}: {str(e)[:180]}"}
        OUT.write_text(json.dumps({"config": {"nseed": NSEED, "kpc": KPC, "prereg": "P1c_2026_07_26", "primary": "interaction_residual"}, "results": results}, indent=1))
    ok = [nme for nme in DATASETS if "error" not in results.get(nme, {})]
    print("\n=== P1c ROUND-2 (rho(X, interaction-residual) by dataset) ===", flush=True)
    replicated = []
    if ok:
        allnames = results[ok[0]]["descriptors"].keys()
        print(f"  {'descriptor':12}" + "".join(f"{d:>14}" for d in ok) + "   replicated_HIT", flush=True)
        for nm in allnames:
            rr = {d: results[d]["descriptors"][nm]["rho_resid"] for d in ok}
            hitboth = all(results[d]["descriptors"][nm]["hit"] for d in ok) and len({np.sign(rr[d]) for d in ok}) == 1
            if hitboth: replicated.append(nm)
            print(f"  {nm:12}" + "".join(f"{rr[d]:>+14.3f}" for d in ok) + f"   {'YES' if hitboth else ''}", flush=True)
    verdict = (f"HIT: {replicated} predict the interaction residual -> powered P2" if replicated else
               "NULL: interaction resists optimization-landscape descriptors too -> bank strengthened negative + PIVOT TO PLASTIC")
    print(f"\n[P1c] replicated_HITs={replicated} | VERDICT: {verdict}", flush=True)
    d = json.loads(OUT.read_text()); d["replicated_hits"] = replicated; d["verdict"] = verdict; OUT.write_text(json.dumps(d, indent=1))
    print(f"[P1c] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
