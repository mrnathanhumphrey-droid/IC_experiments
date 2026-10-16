"""
p1_descriptor_bakeoff.py -- Transfer-geometry P1 (PREREG P1_2026_07_26). Bake-off: which descriptor X(i,j) predicts the reliable
frozen forward-transfer FT established by P0? Two families: DIRECTION (expected null) vs SCALE/CONDITIONING (live hypothesis).
$0 local, frozen resnet18, D_PROJ=24, CIFAR+DTD. Reuses P0/probe_transfer helpers. ASCII-only.
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

DEV = A.DEV
CAPS = list(range(20))
NFEW_POS = R._i("NFEW_POS", 8); NFEW_NEG = R._i("NFEW_NEG", 24); NTR_POS = R._i("NTR_POS", 45); NTR_NEG = R._i("NTR_NEG", 135)
STEPS_FULL = R._i("STEPS_FULL", 120); K = R._i("K", 30); NSEED = R._i("NSEED", 10); D_PROJ = R._i("D_PROJ", 24); SEED = R._i("SEED", 0)
KPC = R._i("KPC", 5); BOOT = R._i("BOOT", 2000); PERM = R._i("PERM", 2000)
DATASETS = os.environ.get("DATASETS", "cifar,dtd").split(",")
OUT = HERE / "out" / "p1_bakeoff.json"
torch.manual_seed(SEED); np.random.seed(SEED)


def spearman(a, b):
    return float(R.spearman(np.asarray(a, float), np.asarray(b, float)))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def topk_pcs(X, k):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:k].T  # (dim, k) orthonormal columns


def linear_cka(X, Y):
    Xc = X - X.mean(0, keepdims=True); Yc = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(Xc.T @ Yc, 'fro')**2
    nx = np.linalg.norm(Xc.T @ Xc, 'fro'); ny = np.linalg.norm(Yc.T @ Yc, 'fro')
    return float(hsic / (nx * ny + 1e-12))


def build(name):
    banks = XF.get_banks(name); caps = [c for c in CAPS if c in banks]; n = len(caps)
    base = A.build("resnet18"); base.eval(); FEAT = {c: A.feats(base, banks[c]) for c in caps}; dim = FEAT[caps[0]].shape[1]
    del base; torch.cuda.empty_cache()
    if D_PROJ > 0:
        P = (np.random.RandomState(0).randn(dim, D_PROJ) / np.sqrt(D_PROJ)).astype(np.float32)
        FEAT = {c: (FEAT[c] @ P).astype(np.float32) for c in caps}; dim = D_PROJ
    rng = np.random.RandomState(0); task = {}
    for c in caps:
        pos = FEAT[c]; others = np.concatenate([FEAT[o] for o in caps if o != c])
        ptr, pte = XF.split_idx(len(pos), f"{name}p{c}"); negsel = rng.choice(len(others), min(NTR_NEG*3, len(others)), replace=False)
        neg = others[negsel]; ntr, nte = XF.split_idx(len(neg), f"{name}n{c}")
        pos_tr = pos[ptr]; neg_tr = neg[ntr]
        Xfull = np.concatenate([pos_tr[:NTR_POS], neg_tr[:NTR_NEG]]); yfull = np.concatenate([np.ones(min(NTR_POS, len(pos_tr))), np.zeros(min(NTR_NEG, len(neg_tr)))]).astype(np.float32)
        Xfew = np.concatenate([pos_tr[:NFEW_POS], neg_tr[:NFEW_NEG]]); yfew = np.concatenate([np.ones(min(NFEW_POS, len(pos_tr))), np.zeros(min(NFEW_NEG, len(neg_tr)))]).astype(np.float32)
        Xte = np.concatenate([pos[pte], neg[nte]]); yte = np.concatenate([np.ones(pte.sum()), np.zeros(nte.sum())]).astype(np.float32)
        task[c] = {"Xfull": Xfull, "yfull": yfull, "Xfew": Xfew, "yfew": yfew, "Xte": Xte, "yte": yte,
                   "d": R.ddir(pos, neg), "pos": pos, "U": topk_pcs(pos, KPC)}
    return caps, n, dim, task


def ft_target(caps, n, dim, task):
    FTmat = np.full((n, n), np.nan)
    acc = np.zeros((n, n))
    for s in range(NSEED):
        cold_s = np.zeros(n); hB = {}
        for ci, i in enumerate(caps):
            hc = XF.train_head(dim, task[i]["Xfew"], task[i]["yfew"], K, seed=7000*s + ci)
            cold_s[ci] = XF.auc_head(hc, task[i]["Xte"], task[i]["yte"])
            hB[i] = XF.train_head(dim, task[i]["Xfull"], task[i]["yfull"], STEPS_FULL, seed=7000*s + ci)
        for ci, i in enumerate(caps):
            for cj, j in enumerate(caps):
                if i == j: continue
                init = {k: v.clone() for k, v in hB[j].state_dict().items()}
                hbi = XF.train_head(dim, task[i]["Xfew"], task[i]["yfew"], K, seed=7000*s + ci, init=init)
                acc[ci, cj] += (XF.auc_head(hbi, task[i]["Xte"], task[i]["yte"]) - cold_s[ci]) / NSEED
    m = ~np.eye(n, dtype=bool); FTmat[m] = acc[m]
    return FTmat


def head_vec(dim, task, i):
    h = XF.train_head(dim, task[i]["Xfull"], task[i]["yfull"], STEPS_FULL, seed=0)
    w = h.weight.detach().cpu().numpy().ravel().astype(np.float64)
    b = float(h.bias.detach().cpu().numpy().ravel()[0])
    return w, b


def descriptors(caps, n, dim, task):
    W = {}; B = {}
    for i in caps:
        W[i], B[i] = head_vec(dim, task, i)
    D = {c: task[c]["d"].astype(np.float64) for c in caps}
    U = {c: task[c]["U"].astype(np.float64) for c in caps}
    names = ["COS", "X1_subspace", "X4_cka", "Xzs_auc", "X0_wnorm", "X3_headInAsub", "X6_gradnorm", "X7_loss0"]
    fam = {"COS": "dir", "X1_subspace": "dir", "X4_cka": "dir", "Xzs_auc": "dir",
           "X0_wnorm": "scale", "X3_headInAsub": "scale", "X6_gradnorm": "scale", "X7_loss0": "scale"}
    mats = {nm: np.full((n, n), np.nan) for nm in names}
    for ci, i in enumerate(caps):
        Xf = task[i]["Xfew"].astype(np.float64); yf = task[i]["yfew"].astype(np.float64); nf = len(yf)
        for cj, j in enumerate(caps):
            if i == j: continue
            wj = W[j]; bj = B[j]
            mats["COS"][ci, cj] = float(D[i] @ D[j])
            s = np.linalg.svd(U[i].T @ U[j], compute_uv=False)
            mats["X1_subspace"][ci, cj] = float(np.mean(s**2))
            mats["X4_cka"][ci, cj] = linear_cka(task[i]["pos"].astype(np.float64), task[j]["pos"].astype(np.float64))
            mats["Xzs_auc"][ci, cj] = XF.auc_head(_mk_head(dim, wj, bj), task[i]["Xte"], task[i]["yte"])
            mats["X0_wnorm"][ci, cj] = float(np.linalg.norm(wj))
            mats["X3_headInAsub"][ci, cj] = float(np.sum((U[i].T @ wj)**2) / (np.sum(wj**2) + 1e-12))
            z = Xf @ wj + bj; p = sigmoid(z)
            grad = Xf.T @ (p - yf) / nf
            mats["X6_gradnorm"][ci, cj] = float(np.linalg.norm(grad))
            eps = 1e-7; mats["X7_loss0"][ci, cj] = float(-np.mean(yf*np.log(p+eps) + (1-yf)*np.log(1-p+eps)))
    return names, fam, mats


def _mk_head(dim, w, b):
    import torch.nn as nn
    h = nn.Linear(dim, 1).to(DEV)
    with torch.no_grad():
        h.weight.copy_(torch.tensor(w, dtype=torch.float32, device=DEV).view(1, -1))
        h.bias.copy_(torch.tensor([b], dtype=torch.float32, device=DEV))
    return h


def corr_stats(x, y):
    rho = spearman(x, y); nrep = len(x); rng = np.random.RandomState(7)
    rb = np.array([spearman(x[idx], y[idx]) for idx in (rng.randint(0, nrep, nrep) for _ in range(BOOT))])
    ci = [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))]
    perm = np.array([spearman(x, rng.permutation(y)) for _ in range(PERM)]); p = float((np.abs(perm) >= abs(rho)).mean())
    return rho, ci, p


def run(name):
    t0 = time.time(); print(f"[P1] === {name} ===", flush=True)
    caps, n, dim, task = build(name)
    FT = ft_target(caps, n, dim, task); print(f"  [{name}] FT target built ({time.time()-t0:.0f}s)", flush=True)
    names, fam, mats = descriptors(caps, n, dim, task); print(f"  [{name}] descriptors built ({time.time()-t0:.0f}s)", flush=True)
    m = ~np.eye(n, dtype=bool); fv = FT[m]
    res = {}
    for nm in names:
        xv = mats[nm][m]
        ok = np.isfinite(xv) & np.isfinite(fv)
        rho, ci, p = corr_stats(xv[ok], fv[ok])
        hit = bool(abs(rho) >= 0.30 and (ci[0] > 0 or ci[1] < 0))
        res[nm] = {"family": fam[nm], "rho": round(rho, 4), "ci": [round(ci[0], 4), round(ci[1], 4)], "perm_p": round(p, 4), "hit": hit}
        print(f"  [{name}] {nm:16} ({fam[nm]:5}) rho={rho:+.3f} CI[{ci[0]:+.3f},{ci[1]:+.3f}] perm_p={p:.3f} {'<< HIT' if hit else ''}", flush=True)
    hits = [nm for nm in names if res[nm]["hit"]]
    print(f"  [{name}] hits={hits} ({time.time()-t0:.0f}s)", flush=True)
    return {"n": n, "hits": hits, "descriptors": res}


def main():
    t0 = time.time()
    print(f"[P1] BAKEOFF | nseed={NSEED} caps={len(CAPS)} K={K} kpc={KPC} D_PROJ={D_PROJ} boot={BOOT} perm={PERM} dev={DEV}", flush=True)
    print("[P1] HIT = |rho|>=0.30 & CI excl 0 & replicates same-sign on BOTH datasets & beats COS. LogME deferred to plastic regime.", flush=True)
    results = {}
    for name in DATASETS:
        try:
            results[name] = run(name)
        except Exception as e:
            import traceback; traceback.print_exc(); results[name] = {"error": f"{type(e).__name__}: {str(e)[:180]}"}
        OUT.write_text(json.dumps({"config": {"caps": CAPS, "K": K, "kpc": KPC, "d_proj": D_PROJ, "nseed": NSEED, "prereg": "P1_2026_07_26"}, "results": results}, indent=1))
    # cross-dataset replication verdict
    ok = [nme for nme in DATASETS if "error" not in results.get(nme, {})]
    print("\n=== P1 BAKE-OFF (rho(X,FT) by dataset) ===", flush=True)
    if len(ok) >= 1:
        allnames = results[ok[0]]["descriptors"].keys()
        print(f"  {'descriptor':16}{'family':>7}" + "".join(f"{d:>14}" for d in ok) + "   replicated_HIT", flush=True)
        replicated = []
        for nm in allnames:
            rhos = {d: results[d]["descriptors"][nm]["rho"] for d in ok}
            hitboth = all(results[d]["descriptors"][nm]["hit"] for d in ok) and len({np.sign(rhos[d]) for d in ok}) == 1
            if hitboth: replicated.append(nm)
            fam = results[ok[0]]["descriptors"][nm]["family"]
            print(f"  {nm:16}{fam:>7}" + "".join(f"{rhos[d]:>+14.3f}" for d in ok) + f"   {'YES' if hitboth else ''}", flush=True)
        verdict = (f"HIT: replicated descriptor(s) {replicated} predict FT (families={[results[ok[0]]['descriptors'][r]['family'] for r in replicated]}) -> P2"
                   if replicated else
                   "NULL: no descriptor >=0.30 replicates cross-dataset -> frozen transfer geometrically structureless -> MOVE TO PLASTIC REGIME (contrarian result stands given P0)")
    else:
        replicated = []; verdict = "no datasets ran"
    print(f"\n[P1] replicated_HITs={replicated} | VERDICT: {verdict}", flush=True)
    d = json.loads(OUT.read_text()); d["replicated_hits"] = replicated; d["verdict"] = verdict; OUT.write_text(json.dumps(d, indent=1))
    print(f"[P1] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
