"""
stream_xdata_lambda.py -- cross-dataset generality check for Direction C (runs on Lambda GPU).
Extracts frozen ResNet18 (ImageNet) features for DTD (textures) + GTSRB (signs), builds (X,y,Xneg) banks in the
SAME format as the cached CIFAR bank, and runs the identical stream-composition analysis (SC.analyze) on each.
Tests whether the recency-weighted / path-dependent composition (found on CIFAR) GENERALIZES (the SS12 lesson).
"""
from __future__ import annotations
import os, sys, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as Fn
import torchvision as tv
from torchvision.models import resnet18, ResNet18_Weights

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_stream_compose as SC

OUT = HERE / "out" / "stream_xdata.json"
OUT.parent.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
PER = int(os.environ.get("XD_PER", "160"))
MEANd = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(DEV)
STDd = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(DEV)
_tt = tv.transforms.ToTensor()

net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); net.fc = torch.nn.Identity(); net = net.eval().to(DEV)


@torch.no_grad()
def feats(pils, bs=128):
    out = []
    for i in range(0, len(pils), bs):
        xs = [Fn.interpolate(_tt(p.convert("RGB")).unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
              for p in pils[i:i+bs]]
        x = torch.cat(xs).to(DEV); x = (x - MEANd) / STDd
        out.append(net(x).float().cpu().numpy())
    return np.concatenate(out)


def collect(ds, per):
    buf = defaultdict(list)
    for img, lab in ds:
        lab = int(lab)
        if len(buf[lab]) < per: buf[lab].append(img)
    return buf


def make_bank(buf, ncap, nres, minn=60):
    labs = [l for l in sorted(buf.keys()) if len(buf[l]) >= minn]
    caps = labs[:ncap]; res = labs[ncap:ncap+nres]
    X = []; y = []
    for ci, l in enumerate(caps):
        f = feats(buf[l]); X.append(f); y += [ci]*len(f)
    Xneg = np.concatenate([feats(buf[l]) for l in res])
    return np.concatenate(X), np.array(y), Xneg


def main():
    t0 = time.time(); print(f"[XD] dev={DEV} PER={PER}", flush=True)
    results = {}
    # DTD (47 texture classes) -- combine splits for volume
    try:
        parts = [tv.datasets.DTD(root="/home/ubuntu/data", split=s, partition=1, download=True) for s in ("train", "val", "test")]
        dtd = torch.utils.data.ConcatDataset(parts)
        buf = collect(dtd, PER); print(f"[XD] DTD collected {len(buf)} classes ({time.time()-t0:.0f}s)", flush=True)
        X, y, Xn = make_bank(buf, 40, 7); print(f"[XD] DTD bank X={X.shape} caps={len(set(y.tolist()))} neg={Xn.shape} ({time.time()-t0:.0f}s)", flush=True)
        results["dtd"] = SC.analyze(X, y, Xn, "dtd")
    except Exception as e:
        results["dtd"] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}; print(f"[XD] DTD ERROR {results['dtd']['error']}", flush=True)
    OUT.write_text(json.dumps(results, indent=1))
    # GTSRB (43 sign classes)
    try:
        g = tv.datasets.GTSRB(root="/home/ubuntu/data", split="train", download=True)
        buf = collect(g, PER); print(f"[XD] GTSRB collected {len(buf)} classes ({time.time()-t0:.0f}s)", flush=True)
        X, y, Xn = make_bank(buf, 36, 7); print(f"[XD] GTSRB bank X={X.shape} caps={len(set(y.tolist()))} neg={Xn.shape} ({time.time()-t0:.0f}s)", flush=True)
        results["gtsrb"] = SC.analyze(X, y, Xn, "gtsrb")
    except Exception as e:
        results["gtsrb"] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}; print(f"[XD] GTSRB ERROR {results['gtsrb']['error']}", flush=True)
    OUT.write_text(json.dumps(results, indent=1))

    print("\n=== CROSS-DATASET Direction C (does recency-weighted/path-dependent GENERALIZE?) ===", flush=True)
    for nm, r in results.items():
        if "error" in r: print(f"  {nm:6} ERROR {r['error'][:60]}", flush=True); continue
        q1 = r["Q1_compose"]; q2 = r["Q2_markov"]
        print(f"  {nm:6} gate_rho={r['gate']['rho_pair']:+.3f} | Q1 winner={q1['winner']} (recwt={q1['recency_weighted']:+.3f} vs add={q1['additive']:+.3f}) "
              f"| order_frac={q2['order_frac']:.2f} within-set_last_rho={q2['within_set_last_rho']:+.3f} markovian={q2['markovian']}", flush=True)
    print(f"[XD] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
