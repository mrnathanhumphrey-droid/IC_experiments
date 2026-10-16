"""
probe_dirlaw_data.py — NEW-1a''''-b (PREREG lock ea04454c): does the plastic directional law survive a DATASET switch? ResNet18 + class-
vs-neg + 224px HELD; data domain swapped CIFAR-100 -> DTD (textures) and GTSRB (traffic signs). Tracks the invariant align_dc. $0 local.
"""
from __future__ import annotations
import os, sys, json, time, copy
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R
import probe_dirlaw_arch as A               # build/prep/feats/theta/train_head1/head_auc/full_ft/sp/dcenter/DEV

DEV = A.DEV
VICT = list(range(24)); PERT = list(range(24, 30)); NEGC = list(range(30, 40))
FT_STEPS = R._i("FT_STEPS", 300); FT_LR = R._f("FT_LR", 5e-4); PER_CLASS = R._i("PER_CLASS", 100)
NJ = R._i("NJ", 128); SEED = R._i("SEED", 0); SIZE = 128
DATASETS = os.environ.get("DATASETS", "dtd,gtsrb").split(",")
ROOT = Path(os.environ.get("TVROOT", r"C:\Users\Nate\AppData\Local\Temp\claude\c--As-Above-So-Below-Master\12e9afc0-72b4-4b00-81e3-2fc4d7b8a97f\scratchpad\tvdata"))
OUT = HERE / "out" / "dirlaw_data.json"; LOCK = "ea04454c"
torch.manual_seed(SEED); np.random.seed(SEED)


def load_banks_tv(name):
    import torchvision.datasets as tvd
    import torchvision.transforms.functional as TF
    ROOT.mkdir(parents=True, exist_ok=True)
    if name == "dtd":
        srcs = [tvd.DTD(str(ROOT), split=s, partition=1, download=True) for s in ("train", "val", "test")]
    elif name == "gtsrb":
        srcs = [tvd.GTSRB(str(ROOT), split="train", download=True)]
    else:
        raise ValueError(name)
    needed = set(VICT) | set(PERT) | set(NEGC); banks = {c: [] for c in needed}
    for ds in srcs:
        for i in range(len(ds)):
            img, lab = ds[i]
            lab = int(lab)
            if lab in needed and len(banks[lab]) < PER_CLASS:
                t = TF.pil_to_tensor(TF.resize(img.convert("RGB"), [SIZE, SIZE]))
                banks[lab].append(t)
        if all(len(banks[c]) >= PER_CLASS for c in needed): break
    out = {}
    for c in needed:
        if len(banks[c]) < 20: raise RuntimeError(f"{name} class {c}: only {len(banks[c])} imgs")
        out[c] = torch.stack(banks[c])
    return out


def run_ds(name):
    t0 = time.time(); print(f"[DATA] === {name} (FT {FT_STEPS}/{FT_LR}) ===", flush=True)
    banks = load_banks_tv(name); Xneg_u8 = torch.cat([banks[c] for c in NEGC])
    print(f"  [{name}] banks: {len(banks)} classes, neg={len(Xneg_u8)} ({time.time()-t0:.0f}s)", flush=True)
    base = A.build("resnet18"); base.eval(); dim = A.feats(base, banks[VICT[0]][:2]).shape[1]
    Fneg0 = A.feats(base, Xneg_u8); heads, auc0 = {}, {}
    for i in VICT:
        FA0 = A.feats(base, banks[i]); heads[i] = A.train_head1(FA0, Fneg0, dim); auc0[i] = A.head_auc(heads[i], FA0, Fneg0)
    print(f"  [{name}] dim={dim} heads ready baseAUC={np.median(list(auc0.values())):.3f} ({time.time()-t0:.0f}s)", flush=True)

    theta0 = A.theta(base); nV, nP = len(VICT), len(PERT)
    dthetas, dmag, FORGET = [], np.zeros(nP), np.zeros((nV, nP))
    for pj, p in enumerate(PERT):
        Xb = torch.cat([banks[p], Xneg_u8]); yb = torch.cat([torch.ones(len(banks[p])), torch.zeros(len(Xneg_u8))]).long()
        mm = copy.deepcopy(base); A.full_ft(mm, dim, Xb, yb, FT_STEPS, FT_LR)
        dth = (A.theta(mm) - theta0).float(); dthetas.append(dth); dmag[pj] = float(dth.norm())
        Fneg1 = A.feats(mm, Xneg_u8)
        for ci, i in enumerate(VICT): FORGET[ci, pj] = auc0[i] - A.head_auc(heads[i], A.feats(mm, banks[i]), Fneg1)
        del mm; torch.cuda.empty_cache()
        print(f"    perturb {p}: |dth|={dmag[pj]:.2f} medFgt_col={np.median(FORGET[:,pj]):+.3f} ({time.time()-t0:.0f}s)", flush=True)

    gmag = np.zeros(nV); DOT = np.zeros((nV, nP)); COS = np.zeros((nV, nP))
    Xn_g = A.prep(Xneg_u8[:NJ]).to(DEV)
    for ci, i in enumerate(VICT):
        base.zero_grad(set_to_none=True)
        m = heads[i](base(A.prep(banks[i][:NJ]).to(DEV))).mean() - heads[i](base(Xn_g)).mean(); m.backward()
        g = torch.cat([pp.grad.detach().flatten() for pp in base.parameters()]).float().cpu(); gmag[ci] = float(g.norm())
        for pj in range(nP): DOT[ci, pj] = float(g @ dthetas[pj]); COS[ci, pj] = DOT[ci, pj]/(gmag[ci]*dmag[pj]+1e-12)
    base.zero_grad(set_to_none=True)

    ALIGN = -COS; GMAGb = np.repeat(gmag[:, None], nP, 1); med = float(np.median(FORGET)); inband = bool(0.03 <= med <= 0.45)
    tbl = {"fo_full": A.sp(-DOT, FORGET), "align": A.sp(ALIGN, FORGET), "align_dcenter": A.sp(A.dcenter(ALIGN), A.dcenter(FORGET)),
           "gmag_broadcast": A.sp(GMAGb, FORGET)}
    p1 = bool(tbl["align_dcenter"] >= 0.15); p2 = bool((tbl["align_dcenter"] > tbl["gmag_broadcast"]) and (tbl["gmag_broadcast"] <= 0.10))
    print(f"  [{name}] medFgt={med:+.3f} inband={inband} | align={tbl['align']:+.3f} align_dc={tbl['align_dcenter']:+.3f} "
          f"fo={tbl['fo_full']:+.3f} gmag={tbl['gmag_broadcast']:+.3f} cos[{COS.min():+.4f},{COS.max():+.4f}] | "
          f"DS-P1={p1} DS-P2={p2} ({time.time()-t0:.0f}s)", flush=True)
    del base, dthetas; torch.cuda.empty_cache()
    return {"dim": dim, "median_forget": round(med, 4), "inband": inband, "dmag_range": [round(float(dmag.min()), 2), round(float(dmag.max()), 2)],
            "cos_range": [round(float(COS.min()), 4), round(float(COS.max()), 4)], "decomp": {k: round(v, 3) for k, v in tbl.items()},
            "checks": {"DS_P1_interaction": p1, "DS_P2_magpattern": p2}}


def main():
    t0 = time.time(); print(f"[DATA] lock={LOCK} datasets={DATASETS} victims={len(VICT)} perturbers={len(PERT)} dev={DEV}", flush=True)
    results = {}
    for name in DATASETS:
        try:
            results[name] = run_ds(name)
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:180]}"}
            print(f"  [{name}] ERROR: {type(e).__name__}: {str(e)[:180]}", flush=True)
        OUT.write_text(json.dumps({"config": {"victims": VICT, "perturbers": PERT, "ft": [FT_STEPS, FT_LR], "per_class": PER_CLASS,
                                   "seed": SEED, "prereg_lock": LOCK}, "results": results}, indent=1))

    print("\n=== DIRECTIONAL LAW ACROSS DATASETS (resnet18, class-vs-neg held) ===", flush=True)
    print(f"  {'dataset':10}{'medFgt':>8}{'align':>8}{'align_dc':>10}{'gmag':>8}  P1P2", flush=True)
    for n, r in results.items():
        if "error" in r: print(f"  {n:10} ERROR {r['error'][:60]}", flush=True); continue
        d = r["decomp"]; c = r["checks"]; flag = "" if r["inband"] else " [OOB]"
        print(f"  {n:10}{r['median_forget']:>8.3f}{d['align']:>8.3f}{d['align_dcenter']:>10.3f}{d['gmag_broadcast']:>8.3f}  "
              f"{int(c['DS_P1_interaction'])}{int(c['DS_P2_magpattern'])}{flag}", flush=True)
    ok = [n for n in DATASETS if "error" not in results.get(n, {}) and results[n].get("inband") and results[n]["checks"]["DS_P1_interaction"]]
    verdict = ("DATASET-GENERAL: interaction replicates off CIFAR (incl. textures)" if "dtd" in ok and len(ok) >= 1 else
               "PARTIAL: holds on some datasets not others" if ok else
               "CIFAR-SPECIFIC: interaction does NOT replicate on other datasets -> scope correction")
    print(f"\n[DATA] holds_on={ok} | VERDICT: {verdict}", flush=True)
    d = json.loads(OUT.read_text()); d["verdict"] = verdict; d["holds_on"] = ok; OUT.write_text(json.dumps(d, indent=1))
    print(f"[DATA] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
