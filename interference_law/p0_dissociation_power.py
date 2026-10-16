"""
p0_dissociation_power.py -- Transfer-geometry P0 (PREREG 2026-07-26). POWER the transfer/forgetting DISSOCIATION.
Reuses probe_forward machinery. Frozen resnet18, CIFAR+DTD, cached feats, D_PROJ=24 headroom. $0 local. ASCII-only.

Adds over probe_forward: per-seed FT matrices -> (1) split-half reliability (Spearman-Brown), (2) rho(cos,FT) bootstrap 95% CI,
(3) permutation null. Pre-registered decision bands printed at start and evaluated at end.
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
STEPS_FULL = R._i("STEPS_FULL", 120); K = R._i("K", 30); NSEED = R._i("NSEED", 12); D_PROJ = R._i("D_PROJ", 24); SEED = R._i("SEED", 0)
BOOT = R._i("BOOT", 2000); PERM = R._i("PERM", 2000)
DATASETS = os.environ.get("DATASETS", "cifar,dtd").split(",")
OUT = HERE / "out" / "p0_dissociation.json"
torch.manual_seed(SEED); np.random.seed(SEED)


def spearman(a, b):
    return float(R.spearman(np.asarray(a, float), np.asarray(b, float)))


def build_tasks(name):
    banks = XF.get_banks(name); caps = [c for c in CAPS if c in banks]; n = len(caps)
    base = A.build("resnet18"); base.eval(); FT = {c: A.feats(base, banks[c]) for c in caps}; dim = FT[caps[0]].shape[1]
    del base; torch.cuda.empty_cache()
    if D_PROJ > 0:
        P = (np.random.RandomState(0).randn(dim, D_PROJ) / np.sqrt(D_PROJ)).astype(np.float32)
        FT = {c: (FT[c] @ P).astype(np.float32) for c in caps}; dim = D_PROJ
    rng = np.random.RandomState(0); task = {}
    for c in caps:
        pos = FT[c]; others = np.concatenate([FT[o] for o in caps if o != c])
        ptr, pte = XF.split_idx(len(pos), f"{name}p{c}"); negsel = rng.choice(len(others), min(NTR_NEG*3, len(others)), replace=False)
        neg = others[negsel]; ntr, nte = XF.split_idx(len(neg), f"{name}n{c}")
        pos_tr = pos[ptr]; neg_tr = neg[ntr]
        Xfull = np.concatenate([pos_tr[:NTR_POS], neg_tr[:NTR_NEG]]); yfull = np.concatenate([np.ones(min(NTR_POS, len(pos_tr))), np.zeros(min(NTR_NEG, len(neg_tr)))]).astype(np.float32)
        Xfew = np.concatenate([pos_tr[:NFEW_POS], neg_tr[:NFEW_NEG]]); yfew = np.concatenate([np.ones(min(NFEW_POS, len(pos_tr))), np.zeros(min(NFEW_NEG, len(neg_tr)))]).astype(np.float32)
        Xte = np.concatenate([pos[pte], neg[nte]]); yte = np.concatenate([np.ones(pte.sum()), np.zeros(nte.sum())]).astype(np.float32)
        task[c] = {"Xfull": Xfull, "yfull": yfull, "Xfew": Xfew, "yfew": yfew, "Xte": Xte, "yte": yte, "d": R.ddir(pos, neg)}
    cosM = np.array([[float(task[i]["d"] @ task[j]["d"]) for j in caps] for i in caps])
    return caps, n, dim, task, cosM


def ft_per_seed(caps, n, dim, task, s):
    """Return n x n FT matrix for one seed (diag = nan)."""
    FTs = np.full((n, n), np.nan)
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
            FTs[ci, cj] = XF.auc_head(hbi, task[i]["Xte"], task[i]["yte"]) - cold_s[ci]
    return FTs, cold_s


def offdiag(M, n):
    m = ~np.eye(n, dtype=bool)
    return M[m]


def run(name):
    t0 = time.time(); print(f"[P0] === {name} ===", flush=True)
    caps, n, dim, task, cosM = build_tasks(name)
    perseed = np.zeros((NSEED, n, n)); cold0 = None
    for s in range(NSEED):
        FTs, cold_s = ft_per_seed(caps, n, dim, task, s)
        perseed[s] = FTs
        if s == 0: cold0 = cold_s.copy()
        print(f"  [{name}] seed {s+1}/{NSEED} done ({time.time()-t0:.0f}s)", flush=True)

    FTmean = np.nanmean(perseed, axis=0)
    cv = offdiag(cosM, n); fv = offdiag(FTmean, n)
    med_cold = float(np.median(cold0)); headroom = bool(0.60 <= med_cold <= 0.90)

    # (1) split-half reliability: odd vs even seeds
    even = np.nanmean(perseed[0::2], axis=0); odd = np.nanmean(perseed[1::2], axis=0)
    fe = offdiag(even, n); fo = offdiag(odd, n)
    r_half = spearman(fe, fo)
    r_sb = (2*r_half/(1+r_half)) if r_half > -1 else float("nan")   # Spearman-Brown
    # bootstrap CI on reliability (resample cell-pairs)
    ncell = len(fv); rng = np.random.RandomState(123); rb = []
    for _ in range(BOOT):
        idx = rng.randint(0, ncell, ncell)
        rr = spearman(fe[idx], fo[idx]); rb.append(2*rr/(1+rr) if rr > -1 else np.nan)
    rb = np.array([x for x in rb if np.isfinite(x)]); rel_ci = [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))]

    # (2) rho(cos,FT) + bootstrap CI
    rho = spearman(cv, fv); pear = float(np.corrcoef(cv, fv)[0, 1])
    rb2 = []
    for _ in range(BOOT):
        idx = rng.randint(0, ncell, ncell)
        rb2.append(spearman(cv[idx], fv[idx]))
    rb2 = np.array(rb2); rho_ci = [float(np.percentile(rb2, 2.5)), float(np.percentile(rb2, 97.5))]

    # (3) permutation null on rho
    perm = np.array([spearman(cv, rng.permutation(fv)) for _ in range(PERM)])
    p_perm = float((np.abs(perm) >= abs(rho)).mean())

    xfer_frac = float((fv >= 0.01).mean()); hurt_frac = float((fv <= -0.01).mean())

    # decision bands
    rel_ok = bool(r_sb >= 0.30 and rel_ci[0] > 0)
    excl_coupling = bool(rho_ci[0] > -0.30 and rho_ci[1] < 0.30)
    reaches_coupling = bool(rho_ci[0] <= -0.30 or rho_ci[1] >= 0.30)
    if rel_ok and excl_coupling:
        band = "CONFIRMED"
    elif reaches_coupling:
        band = "REJECTED"
    elif not rel_ok:
        band = "SAND"
    else:
        band = "INCONCLUSIVE"

    print(f"  [{name}] cold_med={med_cold:.3f} headroom={headroom} n={n} | xfer={xfer_frac:.2f} hurt={hurt_frac:.2f} FT[{fv.min():+.3f},{fv.max():+.3f}]", flush=True)
    print(f"  [{name}] FT reliability: r_half={r_half:+.3f} r_SB={r_sb:+.3f} CI[{rel_ci[0]:+.3f},{rel_ci[1]:+.3f}]  rel_ok={rel_ok}", flush=True)
    print(f"  [{name}] rho(cos,FT)={rho:+.3f} CI[{rho_ci[0]:+.3f},{rho_ci[1]:+.3f}] pearson={pear:+.3f} perm_p={p_perm:.3f} | excl_coupling={excl_coupling}", flush=True)
    print(f"  [{name}] >>> BAND = {band} ({time.time()-t0:.0f}s)", flush=True)
    return {"n": n, "cold_auc_median": round(med_cold, 4), "headroom": headroom, "nseed": NSEED,
            "transfer_frac": round(xfer_frac, 3), "hurt_frac": round(hurt_frac, 3),
            "FT_range": [round(float(fv.min()), 4), round(float(fv.max()), 4)],
            "FT_reliability": {"r_half": round(r_half, 4), "r_SB": round(r_sb, 4), "ci": [round(rel_ci[0], 4), round(rel_ci[1], 4)], "rel_ok": rel_ok},
            "rho_cos_FT": round(rho, 4), "rho_ci": [round(rho_ci[0], 4), round(rho_ci[1], 4)], "pearson": round(pear, 4),
            "perm_p": round(p_perm, 4), "excl_coupling": excl_coupling, "reaches_coupling": reaches_coupling, "band": band}


def main():
    t0 = time.time()
    print(f"[P0] DISSOCIATION POWER | nseed={NSEED} caps={len(CAPS)} K={K} few={NFEW_POS}/{NFEW_NEG} D_PROJ={D_PROJ} boot={BOOT} perm={PERM} dev={DEV}", flush=True)
    print("[P0] BANDS: CONFIRMED = r_SB>=0.30 & relCI_lo>0 & rho_CI in (-0.30,0.30) | REJECTED = rho_CI reaches +/-0.30 | SAND = reliability weak", flush=True)
    results = {}
    for name in DATASETS:
        try:
            results[name] = run(name)
        except Exception as e:
            import traceback; traceback.print_exc()
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:180]}"}
            print(f"  [{name}] ERROR: {type(e).__name__}: {str(e)[:180]}", flush=True)
        OUT.write_text(json.dumps({"config": {"caps": CAPS, "few": [NFEW_POS, NFEW_NEG], "K": K, "d_proj": D_PROJ,
                                   "nseed": NSEED, "boot": BOOT, "perm": PERM, "prereg": "P0_2026_07_26"},
                                   "results": results}, indent=1))
    print("\n=== P0 DISSOCIATION (frozen, powered) ===", flush=True)
    print(f"  {'dataset':8}{'coldAUC':>8}{'r_SB':>7}{'relCI':>16}{'rho':>8}{'rhoCI':>16}{'perm_p':>8}   BAND", flush=True)
    for nme, r in results.items():
        if "error" in r: print(f"  {nme:8} ERROR {r['error'][:50]}", flush=True); continue
        rel = r["FT_reliability"]
        print(f"  {nme:8}{r['cold_auc_median']:>8.3f}{rel['r_SB']:>7.2f}  [{rel['ci'][0]:+.2f},{rel['ci'][1]:+.2f}]{r['rho_cos_FT']:>8.3f}  [{r['rho_ci'][0]:+.2f},{r['rho_ci'][1]:+.2f}]{r['perm_p']:>8.3f}   {r['band']}", flush=True)
    ok = [n for n in DATASETS if "error" not in results.get(n, {})]
    bands = [results[n]["band"] for n in ok]
    overall = ("CONFIRMED" if bands and all(b == "CONFIRMED" for b in bands) else
               "REJECTED" if any(b == "REJECTED" for b in bands) else
               "SAND" if any(b == "SAND" for b in bands) else "MIXED")
    print(f"\n[P0] per-dataset bands={dict(zip(ok, bands))} | OVERALL={overall}", flush=True)
    d = json.loads(OUT.read_text()); d["overall_band"] = overall; OUT.write_text(json.dumps(d, indent=1))
    print(f"[P0] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
