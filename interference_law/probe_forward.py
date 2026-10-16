"""
probe_forward.py -- NEW-3b (PREREG lock bb5bba73): does FORWARD transfer exist? Frozen resnet18, few-shot A, learn A from a B-full-head
init vs cold init (same K steps). FT(i<-j) = AUC_A(from B-init) - AUC_A(cold). Cosine-governed? CIFAR + DTD. $0 local. ASCII-only prints.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R
import probe_dirlaw_arch as A
import probe_transfer as XF                  # get_banks / train_head / auc_head / split_idx

DEV = A.DEV
CAPS = list(range(20)); PER_CLASS = R._i("PER_CLASS", 100)
NFEW_POS = R._i("NFEW_POS", 8); NFEW_NEG = R._i("NFEW_NEG", 24); NTR_POS = R._i("NTR_POS", 45); NTR_NEG = R._i("NTR_NEG", 135)
STEPS_FULL = R._i("STEPS_FULL", 120); K = R._i("K", 30); NSEED = R._i("NSEED", 3); D_PROJ = R._i("D_PROJ", 0); SEED = R._i("SEED", 0)
DATASETS = os.environ.get("DATASETS", "cifar,dtd").split(",")
OUT = HERE / "out" / "forward.json"; LOCK = "bb5bba73"
torch.manual_seed(SEED); np.random.seed(SEED)


def run(name):
    t0 = time.time(); print(f"[FWD] === {name} ===", flush=True)
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

    FTmat = np.zeros((n, n)); cold_auc = np.zeros(n)
    for s in range(NSEED):
        cold_s = np.zeros(n); hB = {}
        for ci, i in enumerate(caps):
            hc = XF.train_head(dim, task[i]["Xfew"], task[i]["yfew"], K, seed=7000*s + ci)
            cold_s[ci] = XF.auc_head(hc, task[i]["Xte"], task[i]["yte"])
            hB[i] = XF.train_head(dim, task[i]["Xfull"], task[i]["yfull"], STEPS_FULL, seed=7000*s + ci)
        if s == 0: cold_auc = cold_s.copy()
        for ci, i in enumerate(caps):
            for cj, j in enumerate(caps):
                if i == j: continue
                init = {k: v.clone() for k, v in hB[j].state_dict().items()}
                hbi = XF.train_head(dim, task[i]["Xfew"], task[i]["yfew"], K, seed=7000*s + ci, init=init)
                FTmat[ci, cj] += (XF.auc_head(hbi, task[i]["Xte"], task[i]["yte"]) - cold_s[ci]) / NSEED
        print(f"  [{name}] seed {s} done ({time.time()-t0:.0f}s)", flush=True)

    m = ~np.eye(n, dtype=bool); fv = FTmat[m]; cv = cosM[m]
    med_cold = float(np.median(cold_auc)); headroom = bool(0.60 <= med_cold <= 0.90)
    xfer_frac = float((fv >= 0.01).mean()); hurt_frac = float((fv <= -0.01).mean())
    rho = float(R.spearman(cv, fv)); pear = float(np.corrcoef(cv, fv)[0, 1])
    ed = np.quantile(cv, np.linspace(0, 1, 9)); curve = []
    for b in range(8):
        mm = (cv >= ed[b]) & (cv <= ed[b+1] if b == 7 else cv < ed[b+1])
        if mm.sum(): curve.append((round(float((ed[b]+ed[b+1])/2), 3), round(float(fv[mm].mean()), 4)))
    c0 = None
    for k in range(1, len(curve)):
        if curve[k-1][1] < 0 <= curve[k][1]: c0 = round((curve[k-1][0]+curve[k][0])/2, 3); break
    top_bin_pos = bool(curve and curve[-1][1] > 0)

    p1 = bool(headroom and xfer_frac >= 0.15); p2 = bool(rho >= 0.30); p3 = top_bin_pos
    print(f"  [{name}] cold_auc_med={med_cold:.3f} headroom={headroom} | FT: transfer_frac(>=.01)={xfer_frac:.2f} hurt_frac={hurt_frac:.2f} "
          f"range[{fv.min():+.3f},{fv.max():+.3f}]", flush=True)
    print(f"  [{name}] rho(cos,FT)={rho:+.3f} pearson={pear:+.3f} | c0={c0} top_bin_FT>0={top_bin_pos} | curve: {curve}", flush=True)
    print(f"  [{name}] 3b-P1(fwd transfer exists)={p1} 3b-P2(cosine-gov)={p2} 3b-P3(aligned+)={p3} ({time.time()-t0:.0f}s)", flush=True)
    return {"n": n, "cold_auc_median": round(med_cold, 4), "headroom": headroom, "transfer_frac": round(xfer_frac, 3),
            "hurt_frac": round(hurt_frac, 3), "FT_range": [round(float(fv.min()), 4), round(float(fv.max()), 4)],
            "rho_cos_FT": round(rho, 3), "pearson": round(pear, 3), "zero_crossing_c0": c0, "top_bin_positive": p3,
            "FT_vs_cos_curve": curve, "checks": {"P1_fwd_transfer": p1, "P2_cosine_gov": p2, "P3_aligned_positive": p3}}


def main():
    t0 = time.time(); print(f"[FWD] lock={LOCK} datasets={DATASETS} caps={len(CAPS)} K={K} few={NFEW_POS}/{NFEW_NEG} D_PROJ={D_PROJ} dev={DEV}", flush=True)
    results = {}
    for name in DATASETS:
        try:
            results[name] = run(name)
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:180]}"}
            print(f"  [{name}] ERROR: {type(e).__name__}: {str(e)[:180]}", flush=True)
        OUT.write_text(json.dumps({"config": {"caps": CAPS, "few": [NFEW_POS, NFEW_NEG], "K": K, "d_proj": D_PROJ, "nseed": NSEED,
                                   "seed": SEED, "prereg_lock": LOCK}, "results": results}, indent=1))
    print("\n=== FORWARD TRANSFER (frozen, B-init few-shot A, FT vs cos) ===", flush=True)
    print(f"  {'dataset':8}{'coldAUC':>8}{'xferFrac':>9}{'hurtFrac':>9}{'rho(cos,FT)':>12}{'c0':>7}{'topBin+':>8}   P1P2P3", flush=True)
    for nme, r in results.items():
        if "error" in r: print(f"  {nme:8} ERROR {r['error'][:50]}", flush=True); continue
        c = r["checks"]
        print(f"  {nme:8}{r['cold_auc_median']:>8.3f}{r['transfer_frac']:>9.2f}{r['hurt_frac']:>9.2f}{r['rho_cos_FT']:>12.3f}"
              f"{str(r['zero_crossing_c0']):>7}{str(r['top_bin_positive']):>8}   {int(c['P1_fwd_transfer'])}{int(c['P2_cosine_gov'])}{int(c['P3_aligned_positive'])}", flush=True)
    ok = [n for n in DATASETS if "error" not in results.get(n, {}) and results[n].get("headroom")]
    p1 = all(results[n]["checks"]["P1_fwd_transfer"] for n in ok) if ok else False
    p2 = all(results[n]["checks"]["P2_cosine_gov"] for n in ok) if ok else False
    p3 = all(results[n]["checks"]["P3_aligned_positive"] for n in ok) if ok else False
    verdict = ("FORWARD TRANSFER is REAL + cosine-governed: signed law COMPLETE (aligned->transfer, anti->interference)" if p1 and p2 and p3 else
               "forward transfer exists, not cosine-gov" if p1 else
               "NO forward transfer either -> phenomenon genuinely ONE-SIDED" if ok else "no headroom -> retune")
    print(f"\n[FWD] headroom={ok} P1={p1} P2={p2} P3={p3} | VERDICT: {verdict}", flush=True)
    d = json.loads(OUT.read_text()); d["verdict"] = verdict; OUT.write_text(json.dumps(d, indent=1))
    print(f"[FWD] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
