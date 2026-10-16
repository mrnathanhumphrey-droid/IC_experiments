"""
probe_cyclic.py — NEW-1b (PREREG lock 0f75676e): is the interference asymmetry a PECKING ORDER or a CYCLE? Helmholtz-Hodge decompose the
directed forgetting tournament K = gradient(pecking-order) + curl(cyclic); SPLIT-HALF reproducibility separates real cyclic structure from
noise. CIFAR + DTD, resnet18. $0 local.
"""
from __future__ import annotations
import os, sys, json, time, copy
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R
import probe_dirlaw_arch as A

DEV = A.DEV
CAPS = list(range(16)); NEG_BY = {"cifar": list(range(90, 100)), "dtd": list(range(30, 40))}
FT_STEPS = R._i("FT_STEPS", 300); FT_LR = R._f("FT_LR", 5e-4); PER_CLASS = R._i("PER_CLASS", 100)
SEED = R._i("SEED", 0); SIZE = 128
DATASETS = os.environ.get("DATASETS", "cifar,dtd").split(",")
ROOT = Path(r"C:\Users\Nate\AppData\Local\Temp\claude\c--As-Above-So-Below-Master\12e9afc0-72b4-4b00-81e3-2fc4d7b8a97f\scratchpad\tvdata")
OUT = HERE / "out" / "cyclic.json"; LOCK = "0f75676e"
torch.manual_seed(SEED); np.random.seed(SEED)


def get_banks(name):
    needed = set(CAPS) | set(NEG_BY[name])
    if name == "cifar":
        b = R.load_banks(); return {c: b[c] for c in needed}
    import torchvision.datasets as tvd
    import torchvision.transforms.functional as TF
    ROOT.mkdir(parents=True, exist_ok=True)
    srcs = [tvd.DTD(str(ROOT), split=s, partition=1, download=True) for s in ("train", "val", "test")]
    banks = {c: [] for c in needed}
    for ds in srcs:
        for i in range(len(ds)):
            img, lab = ds[i]; lab = int(lab)
            if lab in needed and len(banks[lab]) < PER_CLASS:
                banks[lab].append(TF.pil_to_tensor(TF.resize(img.convert("RGB"), [SIZE, SIZE])))
        if all(len(banks[c]) >= PER_CLASS for c in needed): break
    return {c: torch.stack(v) for c, v in banks.items() if len(v) >= 40}


def build_F(base, banks, caps, neg, sl):
    Xneg = torch.cat([banks[c][sl] for c in neg]); dim = A.feats(base, banks[caps[0]][:2]).shape[1]
    Fneg0 = A.feats(base, Xneg); heads, auc0 = {}, {}
    for c in caps:
        FA = A.feats(base, banks[c][sl]); heads[c] = A.train_head1(FA, Fneg0, dim); auc0[c] = A.head_auc(heads[c], FA, Fneg0)
    n = len(caps); F = np.zeros((n, n))
    for cj, j in enumerate(caps):
        Xb = torch.cat([banks[j][sl], Xneg]); yb = torch.cat([torch.ones(len(banks[j][sl])), torch.zeros(len(Xneg))]).long()
        mm = copy.deepcopy(base); A.full_ft(mm, dim, Xb, yb, FT_STEPS, FT_LR); Fneg1 = A.feats(mm, Xneg)
        for ci, i in enumerate(caps): F[ci, cj] = (auc0[i] - A.head_auc(heads[i], A.feats(mm, banks[i][sl]), Fneg1)) if i != j else 0.0
        del mm; torch.cuda.empty_cache()
    return F


def hodge(F):
    n = F.shape[0]; m = ~np.eye(n, dtype=bool); K = (F - F.T)/2
    h = np.array([K[i, [j for j in range(n) if j != i]].mean() for i in range(n)])
    grad = h[:, None] - h[None, :]; curl = K - grad
    grad = grad*m; curl = curl*m
    kf = float((K[m]**2).sum()) + 1e-12
    return h, curl, float((grad[m]**2).sum())/kf, float((curl[m]**2).sum())/kf


def run(name):
    t0 = time.time(); print(f"[CYC] === {name} ===", flush=True)
    banks = get_banks(name); neg = NEG_BY[name]; caps = [c for c in CAPS if c in banks]
    half = len(banks[caps[0]]) // 2
    base = A.build("resnet18"); base.eval()
    Fa = build_F(base, banks, caps, neg, slice(0, half)); print(f"  [{name}] half A built ({time.time()-t0:.0f}s)", flush=True)
    Fb = build_F(base, banks, caps, neg, slice(half, 2*half)); print(f"  [{name}] half B built ({time.time()-t0:.0f}s)", flush=True)
    del base; torch.cuda.empty_cache()

    n = len(caps); m = ~np.eye(n, dtype=bool)
    ha, curla, gfa, cfa = hodge(Fa); hb, curlb, gfb, cfb = hodge(Fb)
    meda = float(np.median(Fa[m])); medb = float(np.median(Fb[m])); inband = bool(0.03 <= (meda+medb)/2 <= 0.55)
    rho_F = float(R.spearman(Fa[m], Fb[m])); rho_h = float(R.spearman(ha, hb)); rho_curl = float(R.spearman(curla[m], curlb[m]))
    cyc_frac = (cfa + cfb) / 2; grad_frac = (gfa + gfb) / 2

    real_cycle = bool(rho_curl >= 0.30); noise_cycle = bool(rho_curl < 0.15); dom_real = bool(rho_h >= 0.50)
    print(f"  [{name}] medFgt {meda:.3f}/{medb:.3f} inband={inband} | Hodge: gradient(pecking)={grad_frac:.2f} curl(cyclic)={cyc_frac:.2f}", flush=True)
    print(f"  [{name}] split-half repro: rho(F)={rho_F:+.3f} rho(dominance h)={rho_h:+.3f} rho(curl)={rho_curl:+.3f} "
          f"| dominance-real={dom_real} cycle-real={real_cycle} cycle-noise={noise_cycle} ({time.time()-t0:.0f}s)", flush=True)
    order = sorted(range(n), key=lambda k: -((ha[k]+hb[k])/2))
    print(f"  [{name}] net-dominance h (predator=low prey=high): prey={[caps[k] for k in order[-4:]]} predator={[caps[k] for k in order[:4]]}", flush=True)
    return {"n": n, "median_forget": [round(meda, 4), round(medb, 4)], "inband": inband, "grad_frac": round(grad_frac, 3),
            "cyclic_frac": round(cyc_frac, 3), "rho_F": round(rho_F, 3), "rho_dominance": round(rho_h, 3),
            "rho_curl": round(rho_curl, 3), "dominance_real": dom_real, "cycle_real": real_cycle, "cycle_noise": noise_cycle,
            "caps": [int(c) for c in caps]}


def main():
    t0 = time.time(); print(f"[CYC] lock={LOCK} datasets={DATASETS} caps={len(CAPS)} dev={DEV}", flush=True)
    results = {}
    for name in DATASETS:
        try:
            results[name] = run(name)
        except Exception as e:
            results[name] = {"error": f"{type(e).__name__}: {str(e)[:180]}"}
            print(f"  [{name}] ERROR: {type(e).__name__}: {str(e)[:180]}", flush=True)
        OUT.write_text(json.dumps({"config": {"caps": CAPS, "ft": [FT_STEPS, FT_LR], "seed": SEED, "prereg_lock": LOCK},
                                   "results": results}, indent=1))
    print("\n=== PECKING ORDER vs CYCLE (Hodge + split-half) ===", flush=True)
    print(f"  {'dataset':8}{'grad%':>7}{'curl%':>7}{'rhoF':>7}{'rho_dom':>9}{'rho_curl':>10}   verdict", flush=True)
    for nme, r in results.items():
        if "error" in r: print(f"  {nme:8} ERROR {r['error'][:50]}", flush=True); continue
        v = ("CYCLES-REAL" if r["cycle_real"] else "cycle=NOISE" if r["cycle_noise"] else "cycle=ambiguous") + (" dom-real" if r["dominance_real"] else " dom-WEAK")
        print(f"  {nme:8}{r['grad_frac']*100:>6.0f}%{r['cyclic_frac']*100:>6.0f}%{r['rho_F']:>7.2f}{r['rho_dominance']:>9.2f}{r['rho_curl']:>10.2f}   {v}", flush=True)
    ok = [n for n in DATASETS if "error" not in results.get(n, {})]
    dom = all(results[n]["dominance_real"] for n in ok) if ok else False
    cyc = all(results[n]["cycle_real"] for n in ok) if ok else False
    noise = all(results[n]["cycle_noise"] for n in ok) if ok else False
    verdict = ("REAL NON-TRANSITIVE cycles (rock-paper-scissors) + stable dominance" if dom and cyc else
               "STABLE TRANSITIVE DOMINANCE + noise (no real cycles)" if dom and noise else
               "mixed/ambiguous — see per-dataset")
    print(f"\n[CYC] dominance-real={dom} cycles-real={cyc} cycle-noise={noise} | VERDICT: {verdict}", flush=True)
    d = json.loads(OUT.read_text()); d["verdict"] = verdict; OUT.write_text(json.dumps(d, indent=1))
    print(f"[CYC] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
