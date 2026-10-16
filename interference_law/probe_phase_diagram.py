"""
probe_phase_diagram.py — Thread #5: PHASE DIAGRAM of the interference law. Sweep the FROZEN backbone (scale / architecture /
pretraining objective); per backbone run #1's forget-vs-cosine sweep; tabulate slope / rho / held-out R^2 / c*. $0 local background.
"""
from __future__ import annotations
import os, sys, json, time, hashlib
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
def _i(k, d): return int(os.environ.get(k, d))
SMOKE = os.environ.get("MODE", "full") == "smoke"
C = _i("C", 6 if SMOKE else 40)
PER_CLASS = _i("PER_CLASS", 40 if SMOKE else 120)
NEG_POOL = _i("NEG_POOL", 120 if SMOKE else 500)
HEAD_EPOCHS = _i("HEAD_EPOCHS", 120)
BACKBONES = (["resnet18", "resnet50"] if SMOKE else
             ["resnet18", "resnet50", "resnet152", "wide_resnet50_2", "vit_b_16", "vit_l_16", "convnext_small", "clip_vitb32", "dinov2_vits14"])
POS_CLASSES = list(range(C)); NEG_CLASSES = list(range(90, 100))
OUT = HERE / "out" / ("phase_diagram_smoke.json" if SMOKE else "phase_diagram.json")
torch.manual_seed(0); np.random.seed(0)
IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLIP_MS = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])


# ── generic backbone loader → (forward_fn, prep_fn, dim) ─────────────────────
def make_prep(mean, std, size):
    m = torch.tensor(mean).view(1, 3, 1, 1); s = torch.tensor(std).view(1, 3, 1, 1)
    def prep(u8):
        x = u8.float() / 255.0
        x = Fun.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return (x - m) / s
    return prep


def get_backbone(name):
    import torchvision.models as tvm
    if name in ("resnet18", "resnet50", "resnet152", "wide_resnet50_2"):
        m = getattr(tvm, name)(weights="IMAGENET1K_V1"); dim = m.fc.in_features
        body = nn.Sequential(*list(m.children())[:-1]).to(DEV).eval()
        return (lambda x: body(x).flatten(1)), make_prep(*IMAGENET, 224), dim
    if name in ("vit_b_16", "vit_l_16"):
        m = getattr(tvm, name)(weights="IMAGENET1K_V1"); dim = m.hidden_dim; m.heads = nn.Identity()
        m = m.to(DEV).eval(); return (lambda x: m(x)), make_prep(*IMAGENET, 224), dim
    if name == "convnext_small":
        m = tvm.convnext_small(weights="IMAGENET1K_V1"); dim = m.classifier[2].in_features
        m.classifier[2] = nn.Identity(); m = m.to(DEV).eval()
        return (lambda x: m(x)), make_prep(*IMAGENET, 224), dim
    if name == "clip_vitb32":
        from transformers import CLIPModel
        m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEV).eval()
        return (lambda x: m.get_image_features(pixel_values=x)), make_prep(*CLIP_MS, 224), 512
    if name == "dinov2_vits14":
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEV).eval()
        return (lambda x: m(x)), make_prep(*IMAGENET, 224), 384
    raise ValueError(name)


@torch.no_grad()
def extract(fwd, prep, u8bank, bs=128):
    outs = []
    for i in range(0, len(u8bank), bs):
        outs.append(fwd(prep(u8bank[i:i+bs]).to(DEV)).float().cpu().numpy())
    return np.concatenate(outs).astype(np.float32)


# ── #1's sweep machinery ──────────────────────────────────────────────────────
def auc(sc, y):
    y = np.asarray(y); pos = sc[y == 1]; neg = sc[y == 0]
    if len(pos) == 0 or len(neg) == 0: return 0.5
    a = np.concatenate([pos, neg]); o = a.argsort(); r = np.empty_like(o, float); r[o] = np.arange(1, len(a)+1)
    return float((r[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg)))
class Head(nn.Module):
    def __init__(s, d): super().__init__(); s.fc = nn.Linear(d, 1)
    def forward(s, x): return s.fc(x).squeeze(1)
def _fit(h, X, y, ep, opt):
    Xt = torch.from_numpy(X); yt = torch.from_numpy(y)
    pw = torch.tensor([float((y == 0).sum())/max(float(y.sum()), 1)]); lf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(ep): opt.zero_grad(set_to_none=True); lf(h(Xt), yt).backward(); opt.step()
    return h
def train_head(X, y, ep=HEAD_EPOCHS, seed=0):
    torch.manual_seed(seed); h = Head(X.shape[1]); _fit(h, X, y, ep, torch.optim.AdamW(h.parameters(), lr=0.05, weight_decay=1e-4)); h.eval(); return h
def cont_head(state, X, y, ep=HEAD_EPOCHS, seed=0):
    torch.manual_seed(seed); h = Head(X.shape[1]); h.load_state_dict(state); _fit(h, X, y, ep, torch.optim.SGD(h.parameters(), lr=0.05, weight_decay=1e-4, momentum=0.9)); h.eval(); return h
@torch.no_grad()
def scores(h, X): return torch.sigmoid(h(torch.from_numpy(X))).numpy()
def ddir(Xp, Xn):
    d = Xp.mean(0) - Xn.mean(0); n = np.linalg.norm(d); return (d/n).astype(np.float32) if n > 0 else d
def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean(); return float((rx@ry)/(np.sqrt(rx@rx)*np.sqrt(ry@ry)+1e-30))
def _split(n, tag, ho=0.3):
    te = np.array([(int(hashlib.md5(f"{tag}{i}".encode()).hexdigest(),16)%1000) < ho*1000 for i in range(n)]); return ~te, te


def sweep(F):
    X, y, Xneg = F["X"], F["y"], F["Xneg"]
    classes = sorted(set(int(c) for c in y)); ntr, nte = _split(len(Xneg), "neg"); Xn_tr, Xn_te = Xneg[ntr], Xneg[nte]
    P = {}
    for c in classes:
        Xc = X[y == c]; ctr, cte = _split(len(Xc), f"c{c}")
        P[c] = {"dir": ddir(Xc[ctr], Xn_tr), "Xtr": np.concatenate([Xc[ctr], Xn_tr]),
                "ytr": np.concatenate([np.ones(ctr.sum()), np.zeros(ntr.sum())]).astype(np.float32),
                "Xte": np.concatenate([Xc[cte], Xn_te]), "yte": np.concatenate([np.ones(cte.sum()), np.zeros(nte.sum())]).astype(np.float32)}
    base, states = {}, {}
    for c in classes:
        h = train_head(P[c]["Xtr"], P[c]["ytr"]); states[c] = {k: v.clone() for k, v in h.state_dict().items()}
        base[c] = auc(scores(h, P[c]["Xte"]), P[c]["yte"])
    pts = []
    for a in classes:
        for b in classes:
            if a == b: continue
            h2 = cont_head(states[a], P[b]["Xtr"], P[b]["ytr"])
            pts.append((float(P[a]["dir"] @ P[b]["dir"]), base[a] - auc(scores(h2, P[a]["Xte"]), P[a]["yte"])))
    cs = np.array([p[0] for p in pts]); fg = np.array([p[1] for p in pts])
    idx = np.arange(len(pts)); rng = np.random.RandomState(1); rng.shuffle(idx); tr, te = idx[:len(idx)//2], idx[len(idx)//2:]
    A = np.polyfit(cs[tr], fg[tr], 1); pred = np.polyval(A, cs[te])
    r2 = 1 - float(((fg[te]-pred)**2).sum())/(float(((fg[te]-fg[te].mean())**2).sum())+1e-30)
    # c*(0.10): highest cos-bin whose mean forget >= 0.10
    ed = np.quantile(cs, np.linspace(0, 1, 11)); cstar = None
    for i in range(9, -1, -1):
        m = (cs >= ed[i]) & (cs <= ed[i+1]) if i == 9 else (cs >= ed[i]) & (cs < ed[i+1])
        if m.sum() and fg[m].mean() >= 0.10: cstar = round(float((ed[i]+ed[i+1])/2), 3); break
    ba = np.array(list(base.values()))
    return {"n_pairs": len(pts), "spearman": round(spearman(cs, fg), 3), "slope": round(float(A[0]), 4),
            "intercept": round(float(A[1]), 4), "heldout_r2": round(r2, 3), "cstar_10": cstar,
            "base_auc_median": round(float(np.median(ba)), 3), "median_forget": round(float(np.median(fg)), 4),
            "cos_range": [round(float(cs.min()), 2), round(float(cs.max()), 2)]}


def main():
    t0 = time.time()
    from datasets import load_dataset
    import torchvision.transforms.functional as TF
    print(f"[PHASE] C={C} per_class={PER_CLASS} backbones={BACKBONES} dev={DEV}", flush=True)
    ds = load_dataset("uoft-cs/cifar100", split="train"); fine = np.array(ds["fine_label"]); imgs = ds["img"]
    rng = np.random.RandomState(0); banks = {}
    for c in POS_CLASSES + NEG_CLASSES:
        idx = np.where(fine == c)[0]; rng.shuffle(idx); idx = idx[:(PER_CLASS if c in POS_CLASSES else NEG_POOL//len(NEG_CLASSES)+5)]
        banks[c] = torch.stack([TF.pil_to_tensor(imgs[int(j)].convert("RGB")) for j in idx])
    Xneg_u8 = torch.cat([banks[c] for c in NEG_CLASSES])[:NEG_POOL]
    print(f"[PHASE] CIFAR banks loaded ({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for name in BACKBONES:
        try:
            fwd, prep, dim = get_backbone(name)
            Xneg = extract(fwd, prep, Xneg_u8)
            X = np.concatenate([extract(fwd, prep, banks[c]) for c in POS_CLASSES])
            yv = np.concatenate([np.full(len(banks[c]), c) for c in POS_CLASSES])
            r = sweep({"X": X, "y": yv, "Xneg": Xneg}); r["dim"] = dim
            r["saturated"] = not (0.05 <= r["median_forget"] <= 0.45)
            results[name] = r
            print(f"  {name:16} dim={dim:5} rho={r['spearman']:+.3f} slope={r['slope']:+.4f} R2={r['heldout_r2']:+.3f} "
                  f"c*={r['cstar_10']} baseAUC={r['base_auc_median']:.3f} medFgt={r['median_forget']:.3f} "
                  f"{'[SATURATED]' if r['saturated'] else ''} ({time.time()-t0:.0f}s)", flush=True)
        except Exception as e:
            results[name] = {"error": str(e)[:200]}
            print(f"  {name:16} SKIP ({type(e).__name__}: {str(e)[:100]})", flush=True)
        OUT.write_text(json.dumps({"config": {"C": C, "per_class": PER_CLASS, "smoke": SMOKE, "prereg_lock": "dcf864d2"},
                                   "backbones": results}, indent=1))

    print("\n=== PHASE DIAGRAM (unsaturated backbones) ===", flush=True)
    print(f"  {'backbone':16}{'dim':>6}{'slope':>9}{'rho':>8}{'R2':>7}{'c*':>7}{'medFgt':>8}", flush=True)
    for n, r in results.items():
        if "error" in r: continue
        flag = " SAT" if r.get("saturated") else ""
        print(f"  {n:16}{r['dim']:>6}{r['slope']:>9.4f}{r['spearman']:>8.3f}{r['heldout_r2']:>7.3f}{str(r['cstar_10']):>7}{r['median_forget']:>8.3f}{flag}", flush=True)
    print(f"\n[PHASE] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
