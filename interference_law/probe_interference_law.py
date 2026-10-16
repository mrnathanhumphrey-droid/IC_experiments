"""
probe_interference_law.py — Thread #1: calibrate the cosine->forgetting INTERFERENCE LAW into a predictive curve.

Substrate: CIFAR-100, frozen ImageNet ResNet18 (512-d). Each fine class = a "paradigm" (positives) vs a SHARED negative pool
(images from 10 reserved classes). Per-class discriminative direction d_c = normalize(mean feat_c - mean feat_neg). Sweep ordered
class pairs: train head on A, continue-train on B, measure A's AUC drop = forgetting. Fit forget ~ f(cos(d_A,d_B)); held-out R^2;
decision boundary c*. Overlay the audio pairs (#5). $0 local GPU.

ISOLATION: reads HF-cached CIFAR + cached ResNet18 weights; writes only ./out/.

Run:  python probe_interference_law.py --smoke   # 6 classes
      python probe_interference_law.py           # full
"""
from __future__ import annotations
import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SMOKE = "--smoke" in sys.argv
N_CLASSES = 6 if SMOKE else int(os.environ.get("N_CLASSES", "50"))
PER_CLASS = 40 if SMOKE else int(os.environ.get("PER_CLASS", "160"))
NEG_POOL = 120 if SMOKE else 500
HEAD_EPOCHS = int(os.environ.get("HEAD_EPOCHS", "120"))
RESIZE = 160 if not SMOKE else 96
SEED = 0
OUT = HERE / "out" / ("interference_law_smoke.json" if SMOKE else "interference_law.json")
CACHE = HERE / "out" / f"feat_C{N_CLASSES}_P{PER_CLASS}_R{RESIZE}.npz"
np.random.seed(0)

import torch, torch.nn as nn
torch.manual_seed(0); DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── frozen ResNet18 features on CIFAR-100 (cached) ────────────────────────────
def build_features():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        return {"X": z["X"], "y": z["y"], "Xneg": z["Xneg"]}
    from datasets import load_dataset
    import torchvision as tv
    from torchvision.models import resnet18, ResNet18_Weights
    from torchvision import transforms as T
    print("[LAW] loading CIFAR-100 + ResNet18 ...", flush=True)
    ds = load_dataset("uoft-cs/cifar100")["train"]
    fine = np.array(ds["fine_label"])
    pos_classes = list(range(N_CLASSES)); neg_classes = list(range(90, 100))
    tf = T.Compose([T.Resize((RESIZE, RESIZE)), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net = nn.Sequential(*list(net.children())[:-1]).to(DEVICE).eval()
    imgs = ds["img"]

    def feats_for(idxs):
        out = []
        for i in range(0, len(idxs), 128):
            batch = torch.stack([tf(imgs[j].convert("RGB")) for j in idxs[i:i + 128]]).to(DEVICE)
            with torch.no_grad():
                out.append(net(batch).flatten(1).cpu().numpy().astype(np.float32))
            if (i + 128) % 1024 == 0: print(f"    feats {i+128}/{len(idxs)}", flush=True)
        return np.concatenate(out, 0) if out else np.zeros((0, 512), np.float32)

    rng = np.random.RandomState(SEED)
    X, y = [], []
    for c in pos_classes:
        idx = np.where(fine == c)[0]; rng.shuffle(idx); idx = idx[:PER_CLASS]
        X.append(feats_for(list(idx))); y.append(np.full(len(idx), c))
    X = np.concatenate(X, 0); y = np.concatenate(y, 0)
    negidx = np.concatenate([np.where(fine == c)[0] for c in neg_classes]); rng.shuffle(negidx)
    Xneg = feats_for(list(negidx[:NEG_POOL]))
    np.savez(CACHE, X=X, y=y, Xneg=Xneg)
    print(f"[LAW] cached feats X={X.shape} Xneg={Xneg.shape}", flush=True)
    return {"X": X, "y": y, "Xneg": Xneg}


def _split(n, tag, holdout=0.3):
    te = np.array([(int(hashlib.md5(f"{tag}{i}".encode()).hexdigest(), 16) % 1000) < holdout * 1000 for i in range(n)])
    return ~te, te


def auc(scores, y):
    y = np.asarray(y); pos = scores[y == 1]; neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0: return 0.5
    alls = np.concatenate([pos, neg]); order = alls.argsort(); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(alls) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Head(nn.Module):
    def __init__(self, d): super().__init__(); self.fc = nn.Linear(d, 1)
    def forward(self, x): return self.fc(x).squeeze(1)


def _fit(head, X, y, epochs, opt):
    Xt = torch.from_numpy(X); yt = torch.from_numpy(y)
    pw = torch.tensor([float((y == 0).sum()) / max(float(y.sum()), 1)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True); lossf(head(Xt), yt).backward(); opt.step()
    return head


def train_head(X, y, epochs=HEAD_EPOCHS, lr=0.05, seed=0):
    torch.manual_seed(seed); h = Head(X.shape[1]); opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=1e-4)
    _fit(h, X, y, epochs, opt); h.eval(); return h


def continue_train(state, X, y, epochs=HEAD_EPOCHS, lr=0.05, seed=0):
    torch.manual_seed(seed); h = Head(X.shape[1]); h.load_state_dict(state)
    opt = torch.optim.SGD(h.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
    _fit(h, X, y, epochs, opt); h.eval(); return h


@torch.no_grad()
def scores(h, X): return torch.sigmoid(h(torch.from_numpy(X))).numpy()


def ddir(Xp, Xn):
    d = Xp.mean(0) - Xn.mean(0); nn_ = np.linalg.norm(d)
    return (d / nn_).astype(np.float32) if nn_ > 0 else d.astype(np.float32)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / (np.sqrt(rx @ rx) * np.sqrt(ry @ ry) + 1e-30))


def main():
    t0 = time.time()
    print(f"[LAW] C={N_CLASSES} per_class={PER_CLASS} neg={NEG_POOL} epochs={HEAD_EPOCHS} resize={RESIZE} dev={DEVICE}", flush=True)
    F = build_features(); X, y, Xneg = F["X"], F["y"], F["Xneg"]
    classes = sorted(set(int(c) for c in y))
    ntr, nte = _split(len(Xneg), "neg"); Xn_tr, Xn_te = Xneg[ntr], Xneg[nte]

    # per-class task matrices + discriminative directions
    P = {}
    for c in classes:
        Xc = X[y == c]; ctr, cte = _split(len(Xc), f"c{c}")
        P[c] = {"dir": ddir(Xc[ctr], Xn_tr),
                "Xtr": np.concatenate([Xc[ctr], Xn_tr]), "ytr": np.concatenate([np.ones(ctr.sum()), np.zeros(ntr.sum())]).astype(np.float32),
                "Xte": np.concatenate([Xc[cte], Xn_te]), "yte": np.concatenate([np.ones(cte.sum()), np.zeros(nte.sum())]).astype(np.float32)}

    # base heads + base AUC (non-saturation guard)
    base_auc, heads, states = {}, {}, {}
    for c in classes:
        h = train_head(P[c]["Xtr"], P[c]["ytr"], seed=SEED); heads[c] = h
        states[c] = {k: v.clone() for k, v in h.state_dict().items()}
        base_auc[c] = auc(scores(h, P[c]["Xte"]), P[c]["yte"])
    ba = np.array(list(base_auc.values()))
    print(f"[LAW] base AUC: median={np.median(ba):.3f} min={ba.min():.3f} max={ba.max():.3f} "
          f"(non-sat target median in [0.75,0.97])", flush=True)

    # cosine matrix
    M = np.array([[float(P[a]["dir"] @ P[b]["dir"]) for b in classes] for a in classes])
    off = M[~np.eye(len(classes), dtype=bool)]
    print(f"[LAW] cosine off-diagonal: min={off.min():+.2f} max={off.max():+.2f} mean={off.mean():+.2f}", flush=True)

    # pairwise forgetting sweep
    pairs = []
    for a in classes:
        for b in classes:
            if a == b: continue
            h2 = continue_train(states[a], P[b]["Xtr"], P[b]["ytr"], seed=SEED)
            a_after = auc(scores(h2, P[a]["Xte"]), P[a]["yte"])
            pairs.append({"A": a, "B": b, "cos": round(float(P[a]["dir"] @ P[b]["dir"]), 4),
                          "forget": round(base_auc[a] - a_after, 4), "A_base": round(base_auc[a], 4)})
    cs = np.array([p["cos"] for p in pairs]); fg = np.array([p["forget"] for p in pairs])
    rho = spearman(cs, fg); pear = float(np.corrcoef(cs, fg)[0, 1])
    print(f"\n[LAW] n_pairs={len(pairs)}  Spearman rho(cos,forget)={rho:+.3f}  Pearson={pear:+.3f}", flush=True)

    # binned curve (cosine deciles -> mean forgetting)
    edges = np.quantile(cs, np.linspace(0, 1, 11)); bins = []
    for i in range(10):
        m = (cs >= edges[i]) & (cs <= edges[i + 1]) if i == 9 else (cs >= edges[i]) & (cs < edges[i + 1])
        if m.sum(): bins.append({"cos_lo": round(float(edges[i]), 3), "cos_hi": round(float(edges[i + 1]), 3),
                                 "n": int(m.sum()), "mean_forget": round(float(fg[m].mean()), 4), "sd": round(float(fg[m].std()), 4)})
    print("  binned curve (cos bin -> mean forget):", flush=True)
    for bdct in bins: print(f"    cos[{bdct['cos_lo']:+.2f},{bdct['cos_hi']:+.2f}] n={bdct['n']:3d}  forget={bdct['mean_forget']:+.4f} ±{bdct['sd']:.3f}", flush=True)

    # calibrated predictor: fit linear forget~cos on half, predict other half (held-out R^2, MAE)
    idx = np.arange(len(pairs)); rng = np.random.RandomState(1); rng.shuffle(idx)
    tr, te = idx[:len(idx) // 2], idx[len(idx) // 2:]
    A = np.polyfit(cs[tr], fg[tr], 1); pred = np.polyval(A, cs[te])
    ss_res = float(((fg[te] - pred) ** 2).sum()); ss_tot = float(((fg[te] - fg[te].mean()) ** 2).sum())
    r2 = 1 - ss_res / (ss_tot + 1e-30); mae = float(np.abs(fg[te] - pred).mean())
    print(f"  held-out predictor: slope={A[0]:+.4f} intercept={A[1]:+.4f}  R^2={r2:+.3f}  MAE={mae:.4f}", flush=True)

    # decision boundary c*: cosine where the binned mean_forget crosses 0.05 / 0.10
    def cstar(thr):
        xs = [(b["cos_lo"] + b["cos_hi"]) / 2 for b in bins]; ys = [b["mean_forget"] for b in bins]
        for i in range(len(xs) - 1, -1, -1):
            if ys[i] >= thr: return round(xs[i], 3)
        return None
    c05, c10 = cstar(0.05), cstar(0.10)
    print(f"  decision boundary: c*(forget>=0.05)={c05}  c*(forget>=0.10)={c10}  (below c* -> route)", flush=True)

    # cross-modal overlay: audio pairs from #5
    audio_over = None
    ap = Path("D:/IC_experiments/audio_cf/out/audio_seq.json")
    if ap.exists():
        aj = json.loads(ap.read_text()); apairs = aj["D2_law"]["pairs"]
        acs = np.array([p["cos"] for p in apairs]); afg = np.array([p["forget"] for p in apairs])
        audio_over = {"n": len(apairs), "spearman": round(spearman(acs, afg), 3),
                      "slope": round(float(np.polyfit(acs, afg, 1)[0]), 4)}
        print(f"  cross-modal (audio #5): rho={audio_over['spearman']:+.3f} slope={audio_over['slope']:+.4f} "
              f"| vision slope={A[0]:+.4f}  -> sign match: {np.sign(audio_over['slope'])==np.sign(A[0])}", flush=True)

    OUT.write_text(json.dumps({
        "config": {"n_classes": N_CLASSES, "per_class": PER_CLASS, "neg_pool": NEG_POOL, "head_epochs": HEAD_EPOCHS,
                   "resize": RESIZE, "smoke": SMOKE, "prereg_lock": "4f76b55d"},
        "base_auc": {"median": float(np.median(ba)), "min": float(ba.min()), "max": float(ba.max())},
        "cosine_offdiag": {"min": float(off.min()), "max": float(off.max()), "mean": float(off.mean())},
        "law": {"spearman_rho": rho, "pearson": pear, "heldout_r2": r2, "heldout_mae": mae,
                "slope": float(A[0]), "intercept": float(A[1]), "cstar_05": c05, "cstar_10": c10},
        "binned_curve": bins, "cross_modal_audio": audio_over, "pairs": pairs}, indent=1))
    print(f"\nwrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
