"""Form-3: Compositional vs Holistic abstraction. LOCKED per PREREG_FORM3_COMPOSITIONAL_2026_06_13.

Q: at MATCHED capacity + supervision, does a FACTORED abstraction (one z_trans module per
ground-truth factor) beat a HOLISTIC one at transfer to attribute COMBINATIONS never seen together?

Substrate: 3DShapes (6 independent factors). Two arms, identical trunk/VIB/optim, only factoring differs:
  COMPOSITIONAL: trunk -> F VIB modules (dim d) -> F heads (head f predicts factor f from module f)
  HOLISTIC:      trunk -> 1 VIB module (dim F*d) -> F heads (all predict from the single latent)
Hardened holdout: novel PAIRINGS of individually-seen factor values; >=5 RANDOM splits (not hand-picked).
Gate: compositional - holistic >= +3% top-1 on held-out-combo (paired CI>0) in >=3/5 splits.
Secondary: gap must concentrate on held-out combos vs in-distribution (else "better classifier").

--synthetic runs the full pipeline on fake data (CPU smoke-test) before paying for a GPU.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy("file_system")
CARD = [10, 10, 10, 8, 4, 15]   # 3DShapes factor cardinalities (floor/wall/obj hue, scale, shape, orient)
NF = len(CARD)


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_3dshapes(h5_path, n_sub, seed=0):
    import h5py
    # contiguous full read (fast), then random subsample in RAM. fancy-indexing the h5
    # directly (f["images"][idx]) is pathologically slow on the chunked dataset; a
    # contiguous slice would be biased since 3DShapes is ordered by factor.
    print("  [load] reading full 3DShapes h5 into RAM...", flush=True)
    with h5py.File(h5_path, "r") as f:
        images = f["images"][:]                       # full 480k, contiguous -> fast
        labels_all = f["labels"][:]
    rng = np.random.default_rng(seed)
    idx = rng.choice(images.shape[0], size=min(n_sub, images.shape[0]), replace=False)
    imgs = images[idx]; labels = labels_all[idx]
    del images, labels_all
    print(f"  [load] subsampled {len(imgs)} of {idx.max()+1}-range", flush=True)
    # map each factor column to integer class index
    y = np.zeros((labels.shape[0], NF), dtype=np.int64)
    for j in range(NF):
        vals = np.unique(labels[:, j]); lut = {v: i for i, v in enumerate(vals)}
        y[:, j] = np.array([lut[v] for v in labels[:, j]])
    x = (imgs.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)   # NCHW
    return x, y


def synth(n_sub, seed=0):
    rng = np.random.default_rng(seed)
    y = np.stack([rng.integers(0, c, n_sub) for c in CARD], 1).astype(np.int64)
    # images weakly encode factors so the smoke test exercises real learning signal
    x = rng.standard_normal((n_sub, 3, 16, 16)).astype(np.float32) * 0.1
    for j in range(NF):
        x[:, j % 3, j, :] += (y[:, j][:, None] / CARD[j])
    return x, y


def make_split(y, split_seed, hold_frac=0.10):
    """Hold out novel pairings of 2 random factor axes; every value still appears in train."""
    rng = np.random.default_rng(split_seed)
    a, b = rng.choice(NF, size=2, replace=False)
    pairs = [(va, vb) for va in range(CARD[a]) for vb in range(CARD[b])]
    n_hold = max(1, int(round(hold_frac * len(pairs))))
    held = set(map(tuple, np.array(pairs)[rng.choice(len(pairs), n_hold, replace=False)]))
    is_test = np.array([(int(y[i, a]), int(y[i, b])) in held for i in range(len(y))])
    return a, b, is_test


class Trunk(nn.Module):
    def __init__(self, hw, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, hidden), nn.ReLU(True))

    def forward(self, x):
        return self.net(x)


class Arm(nn.Module):
    """compositional=True -> F modules dim d; else 1 module dim F*d. Both -> F heads. Matched capacity."""
    def __init__(self, hw, d=4, compositional=True, hidden=256):
        super().__init__()
        self.comp = compositional; self.d = d
        self.trunk = Trunk(hw, hidden)
        if compositional:
            self.enc = nn.ModuleList([nn.Linear(hidden, 2 * d) for _ in range(NF)])
            self.heads = nn.ModuleList([nn.Linear(d, CARD[j]) for j in range(NF)])
        else:
            self.enc = nn.Linear(hidden, 2 * d * NF)
            self.heads = nn.ModuleList([nn.Linear(d * NF, CARD[j]) for j in range(NF)])

    def forward(self, x):
        h = self.trunk(x); kl = 0.0; logits = []
        if self.comp:
            for j in range(NF):
                mu, lv = self.enc[j](h).chunk(2, -1)
                z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu) if self.training else mu
                kl = kl + 0.5 * torch.sum(mu.pow(2) + lv.exp() - lv - 1.0, -1).mean()
                logits.append(self.heads[j](z))
        else:
            mu, lv = self.enc(h).chunk(2, -1)
            z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu) if self.training else mu
            kl = 0.5 * torch.sum(mu.pow(2) + lv.exp() - lv - 1.0, -1).mean()
            for j in range(NF):
                logits.append(self.heads[j](z))
        return logits, kl


def train_arm(x, y, tr, device, compositional, seed, beta=0.01, epochs=8, bs=512):
    set_seed(seed)
    hw = x.shape[-1]
    m = Arm(hw, 4, compositional).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr = torch.tensor(x[tr], device=device); Ytr = torch.tensor(y[tr], device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    for ep in range(epochs):
        m.train(); order = torch.randperm(Xtr.size(0), generator=g, device=device)
        for i in range(0, Xtr.size(0), bs):
            idx = order[i:i + bs]
            logits, kl = m(Xtr[idx])
            loss = sum(F.cross_entropy(logits[j], Ytr[idx, j]) for j in range(NF)) + beta * kl
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return m


@torch.no_grad()
def per_factor_correct(m, x, y, sel, device, bs=2048):
    """returns (n_sel, NF) bool correctness array over selected samples."""
    m.eval(); ii = np.where(sel)[0]; outs = []
    for s in range(0, len(ii), bs):
        idx = ii[s:s + bs]
        logits, _ = m(torch.tensor(x[idx], device=device))
        pred = np.stack([lg.argmax(1).cpu().numpy() for lg in logits], 1)
        outs.append(pred == y[idx])
    return np.concatenate(outs) if outs else np.zeros((0, NF), bool)


def boot_gap(comp_ok, hol_ok, B=2000, seed=7):
    """paired bootstrap over held-out images of (comp - hol) mean accuracy on the held-out axes."""
    rng = np.random.default_rng(seed); n = comp_ok.shape[0]
    pt = comp_ok.mean() - hol_ok.mean()
    bs = [comp_ok[rng.integers(0, n, n)].mean() - hol_ok[rng.integers(0, n, n)].mean() for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); return float(pt), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default="/lambda/nfs/iNat/3dshapes.h5")
    ap.add_argument("--n_sub", type=int, default=120000)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out_dir", default="/lambda/nfs/iNat")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    if args.synthetic:
        x, y = synth(args.n_sub)
    else:
        x, y = load_3dshapes(args.h5, args.n_sub)
    print(f"N={len(x)} img={x.shape[1:]} factors={CARD}", flush=True)

    results = []
    for sp in range(args.splits):
        a, b, is_test = make_split(y, 1000 + sp)
        tr = ~is_test
        print(f"[split {sp}] held-out axes=({a},{b}) test={is_test.sum()} train={tr.sum()}", flush=True)
        comp_ho, hol_ho, comp_id, hol_id = [], [], [], []
        for seed in args.seeds:
            mc = train_arm(x, y, tr, device, True, seed)
            mh = train_arm(x, y, tr, device, False, seed)
            # held-out-combo correctness on the 2 held-out axes only
            cc = per_factor_correct(mc, x, y, is_test, device)[:, [a, b]]
            hc = per_factor_correct(mh, x, y, is_test, device)[:, [a, b]]
            comp_ho.append(cc.mean()); hol_ho.append(hc.mean())
            # in-distribution (train rows) on same axes -> secondary guard
            ci = per_factor_correct(mc, x, y, tr, device)[:, [a, b]]
            hi = per_factor_correct(mh, x, y, tr, device)[:, [a, b]]
            comp_id.append(ci.mean()); hol_id.append(hi.mean())
            print(f"  [split {sp} seed {seed}] comp_ho={cc.mean():.3f} hol_ho={hc.mean():.3f}", flush=True)
        # pooled paired bootstrap over held-out images (last seed's correctness arrays, axes pooled)
        cc = per_factor_correct(mc, x, y, is_test, device)[:, [a, b]].reshape(-1)
        hc = per_factor_correct(mh, x, y, is_test, device)[:, [a, b]].reshape(-1)
        gap, lo, hi = boot_gap(cc.astype(float), hc.astype(float))
        win = gap >= 0.03 and lo > 0
        id_gap = float(np.mean(comp_id) - np.mean(hol_id))
        results.append({"split": sp, "axes": [int(a), int(b)],
                        "comp_ho": float(np.mean(comp_ho)), "hol_ho": float(np.mean(hol_ho)),
                        "ho_gap": gap, "ho_gap_ci": [lo, hi], "id_gap": id_gap, "win": bool(win)})
        print(f"[split {sp}] comp_ho={np.mean(comp_ho):.3f} hol_ho={np.mean(hol_ho):.3f} "
              f"gap={gap:+.3f}[{lo:+.3f},{hi:+.3f}] id_gap={id_gap:+.3f} {'WIN' if win else ''}", flush=True)

    nwin = sum(r["win"] for r in results)
    verdict = "CONFIRM" if nwin >= 3 else "REFUTE"
    print(f"\n=== Form-3: {nwin}/{args.splits} splits show comp-hol>=+3% (CI>0) -> {verdict} ===", flush=True)
    json.dump({"gate": "+3% & CI>0 in >=3/5 splits", "n_win": nwin, "verdict": verdict, "splits": results},
              open(os.path.join(args.out_dir, "form3_compositional.json"), "w"), indent=2)
    print("DONE form3", flush=True)


if __name__ == "__main__":
    main()
