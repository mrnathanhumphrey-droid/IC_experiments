"""Arm F/U — asymmetric channel capacity (P4). Tested in the Arm-A FAILURE setting (fine AND
super both predicted from z_trans, uniform-beta gave selectivity~0). Split K=8 into 4 groups
x2 dims with per-group KL multipliers. Cheap channels pass freely; expensive get squeezed.

  --mult asym    -> normalized [0.25,0.5,1.0,2.0] (mean 1)   -> arm label F
  --mult uniform -> [1,1,1,1]                                 -> arm label U  (internal control)

Multipliers normalized to mean 1 so that at matched beta the total KL BUDGET is matched and
only the channel-cost SHAPE differs (isolates asymmetry). Run: arm{F|U}_beta{b}_K8_seed{s}.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_gpu import GPUData
from models import LowerTierCNN, TranslationVIB


def make_mults(kind, n_groups=4):
    if kind == "uniform":
        m = np.ones(n_groups)
    elif kind == "asym":
        m = np.array([0.25, 0.5, 1.0, 2.0])
    else:
        raise ValueError(kind)
    return (m / m.mean()).astype(np.float32)  # normalize to mean 1 -> matched total budget


def grouped_kl(mu, logvar, mults, group_size=2):
    """Per-group-weighted KL. mu/logvar [B,K]; K = len(mults)*group_size. Returns scalar mean."""
    B, K = mu.shape
    per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)          # [B,K]
    groups = per_dim.view(B, len(mults), group_size).sum(-1)          # [B,n_groups]
    weighted = (groups * mults.view(1, -1)).sum(-1)                   # [B]
    return weighted.mean()


class ArmFModel(nn.Module):
    def __init__(self, z_low_dim=64, K=8, n_fine=100, n_coarse=20):
        super().__init__()
        self.lower = LowerTierCNN(z_low_dim)
        self.translate = TranslationVIB(z_low_dim, 128, K)
        self.fine_head = nn.Sequential(nn.Linear(K, 64), nn.ReLU(inplace=True), nn.Linear(64, n_fine))
        self.super_head = nn.Sequential(nn.Linear(K, 64), nn.ReLU(inplace=True), nn.Linear(64, n_coarse))

    def forward(self, x):
        z_low = self.lower(x)
        z_trans, mu, logvar = self.translate(z_low)
        return {"z_low": z_low, "z_trans": z_trans, "mu": mu, "logvar": logvar,
                "fine_logits": self.fine_head(z_trans), "super_logits": self.super_head(z_trans)}


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate_split(model, data, split, mults):
    model.eval(); fc = sc = n = 0; kl_sum = 0.0
    for x, fine, coarse in data.iter_eval(split):
        out = model(x)
        fc += int((out["fine_logits"].argmax(1) == fine).sum())
        sc += int((out["super_logits"].argmax(1) == coarse).sum())
        kl_sum += float(grouped_kl(out["mu"], out["logvar"], mults)) * x.size(0); n += x.size(0)
    return {"fine_acc": fc / n, "super_acc": sc / n, "kl_rate": kl_sum / n}


@torch.no_grad()
def extract_features(model, data, split):
    model.eval(); zl, zt, ys, cs = [], [], [], []
    for x, fine, coarse in data.iter_eval(split):
        out = model(x)
        zl.append(out["z_low"].cpu().numpy()); zt.append(out["z_trans"].cpu().numpy())
        ys.append(fine.cpu().numpy()); cs.append(coarse.cpu().numpy())
    return (np.concatenate(zl), np.concatenate(zt), np.concatenate(ys), np.concatenate(cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--mult", type=str, choices=["asym", "uniform"], required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--out_dir", type=str, default="results"); ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    ap.add_argument("--data_root", type=str, default="./data")
    args = ap.parse_args()

    arm = "F" if args.mult == "asym" else "U"
    run = f"arm{arm}_beta{args.beta}_K{args.K}_seed{args.seed}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True); os.makedirs(args.ckpt_dir, exist_ok=True)

    data = GPUData(device, args.data_root)
    mults = torch.tensor(make_mults(args.mult), device=device)
    model = ArmFModel(K=args.K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    history = []; t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for x, fine, coarse in data.iter_train(args.batch_size, gen):
            out = model(x)
            l_fine = F.cross_entropy(out["fine_logits"], fine)
            l_super = F.cross_entropy(out["super_logits"], coarse)
            kl = grouped_kl(out["mu"], out["logvar"], mults)
            loss = l_fine + l_super + args.beta * kl
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            val = evaluate_split(model, data, "val", mults)
            history.append({"epoch": ep, "l_fine": float(l_fine.detach()), "l_super": float(l_super.detach()),
                            "kl": float(kl.detach()), "val": val})
            print(f"[{run}/{args.mult}] ep{ep:02d} fine={float(l_fine):.3f} super={float(l_super):.3f} "
                  f"kl={float(kl):.3f} val_fine={val['fine_acc']:.3f} val_super={val['super_acc']:.3f}")

    test = evaluate_split(model, data, "test", mults)
    elapsed = time.time() - t0
    zl_tr, zt_tr, fine_tr, coarse_tr = extract_features(model, data, "train")
    zl_te, zt_te, fine_te, coarse_te = extract_features(model, data, "test")
    np.savez_compressed(os.path.join(args.out_dir, f"features_{run}.npz"),
                        z_low_tr=zl_tr, z_trans_tr=zt_tr, fine_tr=fine_tr, coarse_tr=coarse_tr,
                        z_low_te=zl_te, z_trans_te=zt_te, fine_te=fine_te, coarse_te=coarse_te)
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"{run}.pt"))
    meta = {"run": run, "arm": arm, "beta": args.beta, "mult": args.mult, "K": args.K,
            "seed": args.seed, "epochs": args.epochs, "test": test, "elapsed_sec": elapsed, "history": history}
    with open(os.path.join(args.out_dir, f"train_{run}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{run}] DONE test_fine={test['fine_acc']:.3f} test_super={test['super_acc']:.3f} "
          f"test_kl={test['kl_rate']:.3f} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
