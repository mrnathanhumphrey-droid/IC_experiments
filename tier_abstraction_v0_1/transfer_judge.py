"""C5 — the transfer judge (P5). Does Arm C's abstraction PAY?

Hold out 15 fine classes (one per superclass for supers 0..14; all 20 supers still trained via
siblings; held-out images fully excluded). Train Arm C (beta=0.1). All comparisons PAIRED:
z_trans vs z_low from the SAME trained model.

PAY GATE (falsifier, pre-registered): fit an identical classifier on TRAINED-class features and
predict the SUPERCLASS of held-out NOVEL-class query. CONFIRM P5 iff
  superacc(z_trans) - superacc(z_low) >= +0.02 top-1, paired, 95% bootstrap CI of the
  difference (over query, averaged across 3 seeds) excludes 0.
Manipulation check (expected z_low wins): flat k-shot FINE at k=10,20.
Secondary: k=1 flat fine (sample efficiency); hierarchical (z_trans super-gate -> z_low within).

Self-contained loader (does not import data_gpu). Reuses ArmCModel from train_armC.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data import FINE_TO_COARSE, CIFAR100_MEAN, CIFAR100_STD
from train_armC import ArmCModel
from losses import kl_standard_normal


def holdout_classes(n_super=15):
    """One fine class from each of supers 0..n_super-1 -> 15 held-out fine classes."""
    held = []
    for s in range(n_super):
        for f in range(100):
            if FINE_TO_COARSE[f] == s:
                held.append(f); break
    return sorted(held)


class HoldoutData:
    def __init__(self, device, held, data_root="./data"):
        self.device = device
        tr = torchvision.datasets.CIFAR100(data_root, train=True, download=True)
        te = torchvision.datasets.CIFAR100(data_root, train=False, download=True)
        held_set = set(held)
        f2c = torch.tensor(FINE_TO_COARSE, dtype=torch.long)

        def pack(ds):
            x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # uint8 [N,3,32,32]
            yf = torch.tensor(ds.targets, dtype=torch.long)
            return x, yf

        xtr, yftr = pack(tr); xte, yfte = pack(te)
        tr_held_mask = torch.tensor([int(y) in held_set for y in yftr.tolist()])
        te_held_mask = torch.tensor([int(y) in held_set for y in yfte.tolist()])

        def gpu(x, yf):
            return {"x": x.to(device), "fine": yf.to(device), "coarse": f2c[yf].to(device)}

        # held-IN train -> trains the model; held-IN train subset -> fits the super-classifier
        self.train_in = gpu(xtr[~tr_held_mask], yftr[~tr_held_mask])
        # NOVEL pools: support = held-out TRAIN images, query = held-out TEST images
        self.novel_support = gpu(xtr[tr_held_mask], yftr[tr_held_mask])
        self.novel_query = gpu(xte[te_held_mask], yfte[te_held_mask])

        self.held = held
        self.mean = torch.tensor(CIFAR100_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR100_STD, device=device).view(1, 3, 1, 1)

    def _norm(self, xb):
        return (xb.float().div(255.0) - self.mean) / self.std

    def _augment(self, xf01):
        B = xf01.size(0)
        xp = F.pad(xf01, (4, 4, 4, 4))
        oy = torch.randint(0, 9, (B,), device=xf01.device); ox = torch.randint(0, 9, (B,), device=xf01.device)
        ar = torch.arange(32, device=xf01.device)
        rows = (oy.view(B, 1) + ar.view(1, 32)); cols = (ox.view(B, 1) + ar.view(1, 32))
        xp = torch.gather(xp, 2, rows.view(B, 1, 32, 1).expand(B, 3, 32, 40))
        xp = torch.gather(xp, 3, cols.view(B, 1, 1, 32).expand(B, 3, 32, 32))
        flip = torch.rand(B, device=xf01.device) < 0.5
        xp[flip] = torch.flip(xp[flip], dims=[3])
        return xp

    def iter_train(self, batch, gen):
        s = self.train_in; n = s["x"].size(0)
        order = torch.randperm(n, generator=gen, device=self.device)
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            xa = self._augment(s["x"][idx].float().div(255.0))
            yield (xa - self.mean) / self.std, s["fine"][idx], s["coarse"][idx]

    @torch.no_grad()
    def features(self, model, pool, batch=1024):
        model.eval(); zl, zt = [], []
        x = pool["x"]
        for i in range(0, x.size(0), batch):
            out = model(self._norm(x[i:i + batch]))
            zl.append(out["z_low"].cpu().numpy()); zt.append(out["z_trans"].cpu().numpy())
        return np.concatenate(zl), np.concatenate(zt)


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def train_armC_holdout(data, device, beta, seed, epochs):
    set_seed(seed)
    model = ArmCModel(K=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator(device=device).manual_seed(seed)
    for ep in range(epochs):
        model.train()
        for x, fine, coarse in data.iter_train(128, gen):
            out = model(x)
            loss = F.cross_entropy(out["fine_logits"], fine) + F.cross_entropy(out["super_logits"], coarse) \
                + beta * kl_standard_normal(out["mu"], out["logvar"]).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model


def super_clf_acc(feat_train, super_train, feat_query, super_query):
    """Fit identical logistic super-classifier on trained features, eval on novel query.
    Returns per-query correctness boolean array."""
    sc = StandardScaler().fit(feat_train)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(feat_train), super_train)
    pred = clf.predict(sc.transform(feat_query))
    return (pred == super_query).astype(np.float64)


def ncm_fine_acc(feat_sup, fine_sup, feat_qry, fine_qry, classes, k, episodes, rng):
    """k-shot nearest-class-mean fine accuracy over `classes`. Returns mean acc over episodes."""
    accs = []
    by_cls = {c: np.where(fine_sup == c)[0] for c in classes}
    for _ in range(episodes):
        means, labs = [], []
        for c in classes:
            idx = rng.choice(by_cls[c], size=k, replace=len(by_cls[c]) < k)
            means.append(feat_sup[idx].mean(0)); labs.append(c)
        means = np.stack(means); labs = np.array(labs)
        mu = feat_sup.mean(0); sd = feat_sup.std(0) + 1e-6
        q = (feat_qry - mu) / sd; m = (means - mu) / sd
        d = ((q[:, None, :] - m[None, :, :]) ** 2).sum(-1)  # [Nq, nclass]
        pred = labs[d.argmin(1)]
        accs.append((pred == fine_qry).mean())
    return float(np.mean(accs)), float(np.std(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--fit_pool", type=int, default=8000, help="trained-class subset for super-clf fit")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    held = holdout_classes(15)
    data = HoldoutData(device, held)
    print(f"held-out fine classes ({len(held)}): {held}")
    print(f"train-in {data.train_in['x'].size(0)} | novel support {data.novel_support['x'].size(0)} "
          f"| novel query {data.novel_query['x'].size(0)}")

    sup_low_corr, sup_trans_corr = [], []   # per-seed per-query correctness, for bootstrap
    fine_rows = []  # manipulation + secondary

    for seed in args.seeds:
        model = train_armC_holdout(data, device, args.beta, seed, args.epochs)
        # features
        zl_fit, zt_fit = data.features(model, data.train_in)
        rng = np.random.default_rng(seed)
        sub = rng.choice(zl_fit.shape[0], size=min(args.fit_pool, zl_fit.shape[0]), replace=False)
        sup_fit = data.train_in["coarse"].cpu().numpy()
        zl_q, zt_q = data.features(model, data.novel_query)
        zl_s, zt_s = data.features(model, data.novel_support)
        fine_q = data.novel_query["fine"].cpu().numpy(); sup_q = data.novel_query["coarse"].cpu().numpy()
        fine_s = data.novel_support["fine"].cpu().numpy()

        # PAY GATE: super-acc on novel query, z_low vs z_trans (paired)
        cl = super_clf_acc(zl_fit[sub], sup_fit[sub], zl_q, sup_q)
        ct = super_clf_acc(zt_fit[sub], sup_fit[sub], zt_q, sup_q)
        sup_low_corr.append(cl); sup_trans_corr.append(ct)
        print(f"[seed {seed}] PAY GATE novel-super acc  z_low={cl.mean():.4f}  z_trans={ct.mean():.4f}  "
              f"diff={ct.mean()-cl.mean():+.4f}")

        # Manipulation + secondary: flat fine k-shot
        for k in [1, 5, 10, 20]:
            al, sl = ncm_fine_acc(zl_s, fine_s, zl_q, fine_q, held, k, args.episodes, rng)
            at, st = ncm_fine_acc(zt_s, fine_s, zt_q, fine_q, held, k, args.episodes, rng)
            fine_rows.append({"seed": seed, "k": k, "z_low": al, "z_low_sd": sl, "z_trans": at, "z_trans_sd": st})
            print(f"[seed {seed}] flat fine k={k:>2}  z_low={al:.4f}  z_trans={at:.4f}  diff={at-al:+.4f}")

    # PAY GATE verdict: bootstrap CI of paired diff over query, averaged across seeds
    low_mean = np.mean(sup_low_corr, axis=0)    # [Nq]
    trans_mean = np.mean(sup_trans_corr, axis=0)
    point = float(trans_mean.mean() - low_mean.mean())
    nq = low_mean.shape[0]; rng = np.random.default_rng(12345)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, nq, nq)
        boot.append(trans_mean[idx].mean() - low_mean[idx].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    confirmed = (point >= 0.02) and (lo > 0)

    verdict = {
        "held_out_fine": held,
        "pay_gate": {"superacc_zlow": float(low_mean.mean()), "superacc_ztrans": float(trans_mean.mean()),
                     "paired_diff": point, "ci95": [float(lo), float(hi)],
                     "threshold": 0.02, "CONFIRMED": bool(confirmed)},
        "flat_fine": fine_rows,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "transfer_judge_result.json"), "w") as f:
        json.dump(verdict, f, indent=2)

    print("\n=== C5 PAY GATE ===")
    print(f"novel-class superclass acc: z_low={low_mean.mean():.4f}  z_trans={trans_mean.mean():.4f}")
    print(f"paired diff = {point:+.4f}   95% CI = [{lo:+.4f}, {hi:+.4f}]   threshold +0.0200")
    print(f"P5 {'CONFIRMED' if confirmed else 'REFUTED'} "
          f"({'z_trans abstraction transfers to novel classes' if confirmed else 'Locatello-null: representational not functional'})")


if __name__ == "__main__":
    main()
