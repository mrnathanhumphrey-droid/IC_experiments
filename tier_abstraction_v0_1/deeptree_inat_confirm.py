"""CONFIRMATORY iNat deep-tree (v0.3-confirm) — locked per PREREG_INAT_DEEPTREE_2026_06_13.

Closes the geometry question the scout could only lead on. Two locked upgrades over the scout:
  (1) 3-TIER STACKED bottleneck (Arm-E): z_low -> T1(predict class) -> T2(predict order)
      -> T3(predict family). Each tier is a VIB; next tier consumes prev tier's mu.
  (2) BALANCED-k transfer: NCM prototypes built from exactly k examples PER CLASS at the
      target level (k in {1,2,5,10,20}), NOT a raw pool — removes the populous-class confound.

Two arms trained separately (as in scout): EUC (L2 mean prototypes) vs HYP (Poincare ball,
expmap0 latents, Einstein-midpoint prototypes, Poincare distance). Readout per tier at its own
taxonomic level, hyp vs euc, bootstrap 95% CI on the paired diff.

LOCKED GATES (cannot move post-hoc):
  CONFIRM hyperbolic-needs-depth  := hyp-euc >= +0.02 AND CI excludes 0 at low balanced-k (k<=2),
                                     AND the hyp-euc advantage GROWS across tiers (class->order->family).
  REFUTE                          := hyp <= euc at low-k across tiers (no cell clears the gate),
                                     or no monotone depth trend -> hyperbolic doesn't help tier
                                     abstraction even on the 6-level tree.

Reuses cached features (/lambda/nfs/iNat/inat_feats.npz) — no download/extract.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import hyperbolic as H

torch.multiprocessing.set_sharing_strategy("file_system")

TIERS = [("class", 0), ("order", 1), ("family", 2)]   # stack order: coarse->fine (locked)
KS = [1, 2, 5, 10, 20]                                  # balanced k per class
K = 16
BETA = 0.1
EPOCHS = 60


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class StackVIB(nn.Module):
    """3 stacked VIB tiers; tier i predicts level i. Each tier reparam'd; next consumes prev mu."""
    def __init__(self, d_in, ncls, hyp, K=16, hidden=256):
        super().__init__()
        self.hyp = hyp
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(True))
        self.heads = nn.ModuleList()      # produce 2K (mu, logvar) per tier
        d = hidden
        protos, clfs = [], []
        for n in ncls:
            self.heads.append(nn.Linear(d, 2 * K))
            if hyp:
                protos.append(nn.Parameter(torch.randn(n, K) * 0.01))
            else:
                clfs.append(nn.Linear(K, n))
            d = K
        self.cls = nn.ParameterList(protos) if hyp else nn.ModuleList(clfs)

    def forward(self, z_low):
        h = self.trunk(z_low)
        mus, lvs, zs, logits = [], [], [], []
        x = h
        for i, head in enumerate(self.heads):
            mu, lv = head(x).chunk(2, -1)
            z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu) if self.training else mu
            if self.hyp:
                zt = H.expmap0(z)
                lg = -H.dist(zt.unsqueeze(1), H.expmap0(self.cls[i]).unsqueeze(0))
            else:
                zt = z
                lg = self.cls[i](zt)
            mus.append(mu); lvs.append(lv); zs.append(zt); logits.append(lg)
            x = mu        # next tier consumes this tier's mean (stacked)
        return zs, mus, lvs, logits


def kl(mu, lv):
    return 0.5 * torch.sum(mu.pow(2) + lv.exp() - lv - 1.0, -1).mean()


def train_stack(zl, ys_by_tier, ncls, device, hyp, seed):
    set_seed(seed)
    m = StackVIB(zl.shape[1], ncls, hyp, K).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Z = torch.tensor(zl, device=device)
    Ys = [torch.tensor(y, device=device).long() for y in ys_by_tier]
    g = torch.Generator(device=device).manual_seed(seed)
    bs = 512
    for ep in range(EPOCHS):
        m.train()
        order = torch.randperm(Z.size(0), generator=g, device=device)
        for i in range(0, Z.size(0), bs):
            idx = order[i:i + bs]
            zs, mus, lvs, logits = m(Z[idx])
            loss = sum(F.cross_entropy(logits[t], Ys[t][idx]) for t in range(len(ncls)))
            loss = loss + BETA * sum(kl(mus[t], lvs[t]) for t in range(len(ncls)))
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    return m


@torch.no_grad()
def encode(m, zl, device, bs=4096):
    m.eval(); outs = None
    Z = torch.tensor(zl, device=device)
    for i in range(0, Z.size(0), bs):
        zs, _, _, _ = m(Z[i:i + bs])
        zs = [z.cpu().numpy() for z in zs]
        outs = [[z] for z in zs] if outs is None else [o + [z] for o, z in zip(outs, zs)]
    return [np.concatenate(o) for o in outs]   # list over tiers


def project_np(x, e=1e-5):
    n = np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), e); return np.where(n > 1 - e, x / n * (1 - e), x)


def poincare_np(x, y, e=1e-5):
    d2 = ((x - y) ** 2).sum(-1); xn = (x ** 2).sum(-1); yn = (y ** 2).sum(-1)
    return np.arccosh(np.maximum(1 + 2 * d2 / (np.maximum(1 - xn, e) * np.maximum(1 - yn, e)), 1 + e))


def einstein(Xp, e=1e-9):
    x2 = (Xp ** 2).sum(-1, keepdims=True); Kc = 2 * Xp / (1 + x2); k2 = (Kc ** 2).sum(-1)
    g = 1 / np.sqrt(np.maximum(1 - k2, e)); mk = (g[:, None] * Kc).sum(0) / np.maximum(g.sum(), e)
    return mk / (1 + np.sqrt(np.maximum(1 - (mk ** 2).sum(), e)))


def balanced_idx(y_fit, k, rng):
    out = []
    for c in np.unique(y_fit):
        ci = np.where(y_fit == c)[0]
        out.append(rng.choice(ci, size=min(k, len(ci)), replace=False))
    return np.concatenate(out)


def ncm(feat_fit, y_fit, idx, feat_q, y_q, hyp):
    yf = y_fit[idx]; ff = feat_fit[idx]
    P, L = [], []
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
    r = np.random.default_rng(seed)
    bs = [am[r.integers(0, n, n)].mean() - bm[r.integers(0, n, n)].mean() for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); return pt, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cache", default="/lambda/nfs/iNat/inat_feats.npz")
    ap.add_argument("--out_dir", default="/lambda/nfs/iNat")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    d = np.load(args.cache, allow_pickle=True)
    z = d["z"].astype(np.float32); labs = {k: d[k] for k in d.files if k != "z"}
    sp = labs["full"]
    rng0 = np.random.default_rng(0)
    species = np.unique(sp); held = set(rng0.choice(species, size=max(1, len(species) // 7), replace=False))
    tr = np.array([s not in held for s in sp]); te = ~tr
    ncls = [int(labs[name].max()) + 1 for name, _ in TIERS]
    print(f"N={len(z)} tiers={[(n, c) for (n, _), c in zip(TIERS, ncls)]} train={tr.sum()} novel={te.sum()}", flush=True)

    ys_tr = [labs[name][tr] for name, _ in TIERS]
    store = {name: {k: {"euc": [], "hyp": []} for k in KS} for name, _ in TIERS}
    for seed in args.seeds:
        me = train_stack(z[tr], ys_tr, ncls, device, False, seed)
        mh = train_stack(z[tr], ys_tr, ncls, device, True, seed)
        ze_tr = encode(me, z[tr], device); ze_te = encode(me, z[te], device)
        zh_tr = encode(mh, z[tr], device); zh_te = encode(mh, z[te], device)
        r = np.random.default_rng(seed)
        for ti, (name, _) in enumerate(TIERS):
            yf, yq = labs[name][tr], labs[name][te]
            for k in KS:
                idx = balanced_idx(yf, k, r)
                store[name][k]["euc"].append(ncm(ze_tr[ti], yf, idx, ze_te[ti], yq, False))
                store[name][k]["hyp"].append(ncm(zh_tr[ti], yf, idx, zh_te[ti], yq, True))
        print(f"[seed {seed}] done", flush=True)

    print("\n=== CONFIRMATORY: hyp-euc by TIER x balanced-k (CONFIRM=+0.02 & CI>0 at k<=2, grows w/ depth) ===", flush=True)
    res = {}
    for ti, (name, _) in enumerate(TIERS):
        res[name] = {}
        print(f"-- tier{ti+1}={name} (n={ncls[ti]}) --")
        for k in KS:
            dx, lo, hi = boot(store[name][k]["hyp"], store[name][k]["euc"])
            ec = float(np.mean(store[name][k]["euc"], 0).mean()); hc = float(np.mean(store[name][k]["hyp"], 0).mean())
            res[name][k] = {"euc": ec, "hyp": hc, "hyp_minus_euc": [dx, lo, hi]}
            flag = "  <-- HYP WINS" if (dx >= 0.02 and lo > 0) else ""
            print(f"  k={k:>2} euc={ec:.3f} hyp={hc:.3f} hyp-euc={dx:+.3f}[{lo:+.3f},{hi:+.3f}]{flag}")
    json.dump({"design": "3tier_stacked_balanced_k", "tiers": [n for n, _ in TIERS], "ks": KS, "by_tier": res},
              open(os.path.join(args.out_dir, "deeptree_inat_confirm.json"), "w"), indent=2)
    print("\nDONE confirmatory", flush=True)


if __name__ == "__main__":
    main()
