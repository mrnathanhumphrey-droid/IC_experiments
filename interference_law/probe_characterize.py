"""
probe_characterize.py — NEW-1a''''-d (PREREG lock 40287dfa): what DESCRIBES the real low-rank interaction? Test candidate descriptors
(feature cosine / Jacobian cosine / gradient cosine / centroid cosine) vs the double-centered directed matrix on CIFAR + DTD, + SVD the
interaction subspace. Lead: FEATURE cosine −cos(d_i,d_j) (frozen-law variable) is the general descriptor. $0 local.
"""
from __future__ import annotations
import os, sys, json, time, copy
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R
import probe_dirlaw_arch as A
import probe_shape as SH                     # get_banks / dcenter_off / off / CAPS / NEG_BY

DEV = A.DEV
FT_STEPS = R._i("FT_STEPS", 300); FT_LR = R._f("FT_LR", 5e-4); NJ = R._i("NJ", 128); SEED = R._i("SEED", 0)
DATASETS = os.environ.get("DATASETS", "cifar,dtd").split(",")
OUT = HERE / "out" / "characterize.json"; LOCK = "40287dfa"
torch.manual_seed(SEED); np.random.seed(SEED)


def cosnp(a, b): return float(a @ b / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
def cost(a, b): return float(a @ b / (float(a.norm())*float(b.norm()) + 1e-12))
def spoff(M, F, m): return float(R.spearman(M[m], F[m]))


def run(name):
    t0 = time.time(); print(f"[CHAR] === {name} ===", flush=True)
    banks = SH.get_banks(name); neg = SH.NEG_BY[name]; Xneg_u8 = torch.cat([banks[c] for c in neg])
    caps = [c for c in SH.CAPS if c in banks]; n = len(caps)
    base = A.build("resnet18"); base.eval(); dim = A.feats(base, banks[caps[0]][:2]).shape[1]
    Fneg0 = A.feats(base, Xneg_u8); heads, auc0, dvec, mu = {}, {}, {}, {}
    for c in caps:
        FA = A.feats(base, banks[c]); heads[c] = A.train_head1(FA, Fneg0, dim); auc0[c] = A.head_auc(heads[c], FA, Fneg0)
        dvec[c] = R.ddir(FA, Fneg0); mu[c] = FA.mean(0)
    print(f"  [{name}] n={n} dim={dim} baseAUC={np.median(list(auc0.values())):.3f} ({time.time()-t0:.0f}s)", flush=True)

    theta0 = A.theta(base); dth = {}; F = np.zeros((n, n))
    for cj, j in enumerate(caps):
        Xb = torch.cat([banks[j], Xneg_u8]); yb = torch.cat([torch.ones(len(banks[j])), torch.zeros(len(Xneg_u8))]).long()
        mm = copy.deepcopy(base); A.full_ft(mm, dim, Xb, yb, FT_STEPS, FT_LR)
        dth[j] = (A.theta(mm) - theta0).float(); Fneg1 = A.feats(mm, Xneg_u8)
        for ci, i in enumerate(caps): F[ci, cj] = (auc0[i] - A.head_auc(heads[i], A.feats(mm, banks[i]), Fneg1)) if i != j else 0.0
        del mm; torch.cuda.empty_cache()
        print(f"    perturb {j}: colFgt={np.median([F[ci,cj] for ci in range(n) if caps[ci]!=j]):+.3f} ({time.time()-t0:.0f}s)", flush=True)

    gvec = {}; Xn_g = A.prep(Xneg_u8[:NJ]).to(DEV)
    for i in caps:
        base.zero_grad(set_to_none=True)
        mrg = heads[i](base(A.prep(banks[i][:NJ]).to(DEV))).mean() - heads[i](base(Xn_g)).mean(); mrg.backward()
        gvec[i] = torch.cat([pp.grad.detach().flatten() for pp in base.parameters()]).float().cpu()
    base.zero_grad(set_to_none=True)

    FC = np.zeros((n, n)); JC = np.zeros((n, n)); GC = np.zeros((n, n)); DS = np.zeros((n, n))
    for ci, i in enumerate(caps):
        for cj, j in enumerate(caps):
            if i == j: continue
            FC[ci, cj] = -cosnp(dvec[i], dvec[j]); DS[ci, cj] = -cosnp(mu[i], mu[j])
            JC[ci, cj] = -cost(gvec[i], dth[j]); GC[ci, cj] = -cost(gvec[i], gvec[j])

    m = SH.off(F); med = float(np.median(F[m])); DCf = SH.dcenter_off(F)
    cands = {"FC_feature": FC, "JC_jacobian": JC, "GC_gradient": GC, "DS_centroid": DS}
    rho = {k: round(spoff(SH.dcenter_off(v), DCf, m), 3) for k, v in cands.items()}
    best = max(rho, key=lambda k: abs(rho[k]))

    # SVD of the interaction
    s = np.linalg.svd(DCf, compute_uv=True); U, sig, Vt = s
    varfrac = (sig**2 / (sig**2).sum())[:4]
    u1 = U[:, 0]; v1 = Vt[0]
    top_v = sorted(range(n), key=lambda k: -abs(u1[k]))[:4]; top_p = sorted(range(n), key=lambda k: -abs(v1[k]))[:4]
    uv_align = float(abs(np.corrcoef(u1, v1)[0, 1]))

    print(f"  [{name}] medFgt={med:+.3f} | DESCRIPTORS rho(dcenter,DC): " +
          " ".join(f"{k.split('_')[1]}={rho[k]:+.3f}" for k in cands) + f" | BEST={best}", flush=True)
    print(f"  [{name}] SVD top-4 var%={[round(float(x)*100,1) for x in varfrac]} | mode1 victims(u1)={[caps[k] for k in top_v]} "
          f"perturbers(v1)={[caps[k] for k in top_p]} |u1·v1|corr={uv_align:.2f} ({time.time()-t0:.0f}s)", flush=True)
    del base, gvec, dth; torch.cuda.empty_cache()
    return {"n": n, "median_forget": round(med, 4), "descriptor_rho": rho, "best_descriptor": best,
            "svd_var_frac_top4": [round(float(x), 3) for x in varfrac], "mode1_victims": [int(caps[k]) for k in top_v],
            "mode1_perturbers": [int(caps[k]) for k in top_p], "mode1_u1v1_corr": round(uv_align, 3)}


def main():
    t0 = time.time(); print(f"[CHAR] lock={LOCK} datasets={DATASETS} dev={DEV}", flush=True)
    results = {}
    for name in DATASETS:
        try:
            results[name] = run(name)
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:180]}"}
            print(f"  [{name}] ERROR: {type(e).__name__}: {str(e)[:180]}", flush=True)
        OUT.write_text(json.dumps({"config": {"caps": SH.CAPS, "ft": [FT_STEPS, FT_LR], "seed": SEED, "prereg_lock": LOCK},
                                   "results": results}, indent=1))
    print("\n=== WHAT DESCRIBES THE INTERACTION (dcenter rho vs DC) ===", flush=True)
    print(f"  {'dataset':8}{'feature':>9}{'jacobian':>10}{'gradient':>10}{'centroid':>10}   best", flush=True)
    for nme, r in results.items():
        if "error" in r: print(f"  {nme:8} ERROR {r['error'][:50]}", flush=True); continue
        d = r["descriptor_rho"]
        print(f"  {nme:8}{d['FC_feature']:>9.3f}{d['JC_jacobian']:>10.3f}{d['GC_gradient']:>10.3f}{d['DS_centroid']:>10.3f}   {r['best_descriptor']}", flush=True)
    fc_dtd = results.get("dtd", {}).get("descriptor_rho", {}).get("FC_feature", 0)
    fc_cif = results.get("cifar", {}).get("descriptor_rho", {}).get("FC_feature", 0)
    p1 = bool(fc_dtd >= 0.20 and results.get("dtd", {}).get("best_descriptor") == "FC_feature")
    p2 = bool(fc_cif >= 0.20)
    verdict = ("FEATURE COSINE is the GENERAL descriptor (fits CIFAR+DTD); Jacobian −cos was CIFAR-coincidental" if p1 and p2 else
               "feature cosine fits DTD not CIFAR -> per-dataset descriptor, no single law" if p1 else
               "no candidate fits DTD -> the low-rank interaction descriptor is outside {feature,jacobian,gradient,centroid}")
    print(f"\n[CHAR] CHAR-P1(FC fits DTD & best)={p1} CHAR-P2(FC fits CIFAR)={p2} | VERDICT: {verdict}", flush=True)
    d = json.loads(OUT.read_text()); d["verdict"] = verdict; OUT.write_text(json.dumps(d, indent=1))
    print(f"[CHAR] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
