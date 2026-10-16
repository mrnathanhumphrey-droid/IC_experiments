"""DEEP-TREE test (v0.3) — does hyperbolic geometry beat Euclidean for tier abstraction once the
hierarchy is DEEP? CIFAR-100 is a 2-level tree (Form-1 null). iNaturalist is 7 levels
(kingdom>phylum>class>order>family>genus>species). Theory (Nickel&Kiela): hyperbolic's
low-distortion edge needs depth. If hyp beats euc HERE where it lost on CIFAR, depth was the key.

Design (frozen-backbone, isolates the GEOMETRY question, fast):
  z_low = frozen ImageNet ResNet-18 features (512d). translate (VIB) -> z_trans (euc OR hyperbolic).
  Train z_trans to predict a mid taxonomic level; hold out species; test NCM transfer of several
  levels (class/order/family) to NOVEL species across fit-pools, hyp vs euc, native geometry.
  KEY READOUT: does hyp-euc advantage GROW as the target level gets higher (deeper abstraction)?
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet18, ResNet18_Weights
import hyperbolic as H

# iNat = 100k tiny files; num_workers shared-memory FDs exhaust -> use file_system strategy
torch.multiprocessing.set_sharing_strategy("file_system")

LEVELS = ["class", "order", "family"]          # coarse->fine taxonomic levels to test transfer at
TRAIN_LEVEL = "order"                            # level z_trans is trained to predict
FIT_POOLS = [50, 200, 1000, 4000]


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def extract_features(device, data_root, version, cache):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["z"], {k: d[k] for k in d.files if k != "z"}
    tf = ResNet18_Weights.IMAGENET1K_V1.transforms()
    tgt = ["kingdom", "phylum", "class", "order", "family", "genus", "full"]
    print("  [phase] downloading+extracting iNat (slow, one-time)...", flush=True)
    ds = torchvision.datasets.INaturalist(root=data_root, version=version, target_type=tgt,
                                          transform=tf, download=True)
    print(f"  [phase] iNat ready: {len(ds)} images -> extracting ResNet-18 features", flush=True)
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); net.fc = nn.Identity(); net.eval().to(device)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=8)
    zs = []; labs = {k: [] for k in tgt}
    with torch.no_grad():
        for i, (x, ys) in enumerate(loader):
            zs.append(net(x.to(device)).cpu().numpy())
            for k, v in zip(tgt, ys):
                labs[k].append(np.asarray(v))
            if i % 50 == 0:
                print(f"  feat batch {i} ({i*256} imgs)", flush=True)
    z = np.concatenate(zs).astype(np.float32)
    labs = {k: np.concatenate(v).astype(np.int64) for k, v in labs.items()}
    np.savez_compressed(cache, z=z, **labs)
    return z, labs


class Translate(nn.Module):
    def __init__(self, d_in=512, hidden=256, K=16, n_coarse=51, hyp=False):
        super().__init__()
        self.hyp = hyp
        self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(True))
        self.head = nn.Linear(hidden, 2 * K)
        if hyp:
            self.proto = nn.Parameter(torch.randn(n_coarse, K) * 0.01)
        else:
            self.clf = nn.Linear(K, n_coarse)

    def forward(self, z_low):
        mu, logvar = self.head(self.body(z_low)).chunk(2, -1)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if self.training else mu
        if self.hyp:
            zt = H.expmap0(z)
            logits = -H.dist(zt.unsqueeze(1), H.expmap0(self.proto).unsqueeze(0))
        else:
            zt = z
            logits = self.clf(zt)
        return zt, mu, logvar, logits


def kl(mu, logvar):
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, -1).mean()


def train_translate(zl, y_coarse, device, hyp, seed, K, beta=0.1, epochs=60, bs=512):
    set_seed(seed)
    n_coarse = int(y_coarse.max()) + 1
    m = Translate(zl.shape[1], 256, K, n_coarse, hyp).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Z = torch.tensor(zl, device=device); Y = torch.tensor(y_coarse, device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    for ep in range(epochs):
        m.train()
        order = torch.randperm(Z.size(0), generator=g, device=device)
        for i in range(0, Z.size(0), bs):
            idx = order[i:i + bs]
            zt, mu, lv, logits = m(Z[idx])
            loss = F.cross_entropy(logits, Y[idx]) + beta * kl(mu, lv)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    return m


@torch.no_grad()
def ztrans(m, zl, device, bs=4096):
    m.eval(); out = []
    Z = torch.tensor(zl, device=device)
    for i in range(0, Z.size(0), bs):
        out.append(m(Z[i:i + bs])[0].cpu().numpy())
    return np.concatenate(out)


def project_np(x, e=1e-5):
    n = np.maximum(np.linalg.norm(x, -1, keepdims=True), e); return np.where(n > 1 - e, x / n * (1 - e), x)


def poincare_np(x, y, e=1e-5):
    d2 = ((x - y) ** 2).sum(-1); xn = (x ** 2).sum(-1); yn = (y ** 2).sum(-1)
    return np.arccosh(np.maximum(1 + 2 * d2 / (np.maximum(1 - xn, e) * np.maximum(1 - yn, e)), 1 + e))


def einstein(X, e=1e-9):
    x2 = (X ** 2).sum(-1, keepdims=True); Kc = 2 * X / (1 + x2); k2 = (Kc ** 2).sum(-1)
    g = 1 / np.sqrt(np.maximum(1 - k2, e)); mk = (g[:, None] * Kc).sum(0) / np.maximum(g.sum(), e)
    return mk / (1 + np.sqrt(np.maximum(1 - (mk ** 2).sum(), e)))


def ncm(feat_fit, y_fit, idx, feat_q, y_q, hyp):
    P, L = [], []
    yf = y_fit[idx]; ff = feat_fit[idx]
    for c in np.unique(y_fit):
        m = yf == c
        if m.sum() == 0:
            continue
        P.append(einstein(ff[m]) if hyp else ff[m].mean(0)); L.append(c)
    P = np.stack(P); L = np.array(L)
    d = poincare_np(feat_q[:, None, :], P[None, :, :]) if hyp else ((feat_q[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    return (L[d.argmin(1)] == y_q).astype(np.float64)


def boot(a, b, B=2000, seed=7):
    am = np.mean(a, 0); bm = np.mean(b, 0); pt = float(am.mean() - bm.mean()); n = am.shape[0]
    r = np.random.default_rng(seed); bs = [am[r.integers(0, n, n)].mean() - bm[r.integers(0, n, n)].mean() for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); return pt, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--version", default="2021_valid")
    ap.add_argument("--data_root", default="./inat"); ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    print("extracting frozen ResNet-18 features (downloads iNat on first run)...", flush=True)
    z, labs = extract_features(device, args.data_root, args.version, os.path.join(args.out_dir, "inat_feats.npz"))
    sp = labs["full"]                                  # species id
    rng = np.random.default_rng(0)
    species = np.unique(sp); held = set(rng.choice(species, size=max(1, len(species) // 7), replace=False))
    tr = np.array([s not in held for s in sp]); te = ~tr
    print(f"N={len(z)} species={len(species)} held_species={len(held)} train={tr.sum()} novel={te.sum()}", flush=True)

    store = {lv: {p: {"euc": [], "hyp": []} for p in FIT_POOLS} for lv in LEVELS}
    for seed in args.seeds:
        me = train_translate(z[tr], labs[TRAIN_LEVEL][tr], device, False, seed, args.K)
        mh = train_translate(z[tr], labs[TRAIN_LEVEL][tr], device, True, seed, args.K)
        ze_fit, ze_q = ztrans(me, z[tr], device), ztrans(me, z[te], device)
        zh_fit, zh_q = ztrans(mh, z[tr], device), ztrans(mh, z[te], device)
        r = np.random.default_rng(seed)
        for lv in LEVELS:
            yf, yq = labs[lv][tr], labs[lv][te]
            for p in FIT_POOLS:
                idx = r.choice(ze_fit.shape[0], size=min(p, ze_fit.shape[0]), replace=False)
                store[lv][p]["euc"].append(ncm(ze_fit, yf, idx, ze_q, yq, False))
                store[lv][p]["hyp"].append(ncm(zh_fit, yf, idx, zh_q, yq, True))
        print(f"[seed {seed}] done", flush=True)

    print("\n=== DEEP-TREE: hyp-euc transfer to NOVEL species, by taxonomic level x fit-pool ===", flush=True)
    res = {}
    for lv in LEVELS:
        res[lv] = {}
        print(f"-- level={lv} (n_classes={int(labs[lv].max())+1}) --")
        for p in FIT_POOLS:
            dx, lo, hi = boot(store[lv][p]["hyp"], store[lv][p]["euc"])
            ec = float(np.mean(store[lv][p]["euc"], 0).mean()); hc = float(np.mean(store[lv][p]["hyp"], 0).mean())
            res[lv][p] = {"euc": ec, "hyp": hc, "hyp_minus_euc": [dx, lo, hi]}
            flag = "  <-- HYP WINS" if (dx >= 0.02 and lo > 0) else ""
            print(f"  pool={p:>5} euc={ec:.3f} hyp={hc:.3f} hyp-euc={dx:+.3f}[{lo:+.3f},{hi:+.3f}]{flag}")
    json.dump({"train_level": TRAIN_LEVEL, "by_level": res}, open(os.path.join(args.out_dir, "deeptree_inat.json"), "w"), indent=2)
    print("\nDONE deeptree", flush=True)


if __name__ == "__main__":
    main()
