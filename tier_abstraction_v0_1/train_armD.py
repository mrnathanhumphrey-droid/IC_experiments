"""Arm D — adversarial disentanglement (P2). Arm C split + a fine-adversary on z_trans via
gradient reversal. The adversary learns to predict fine from z_trans; the GRL flips its
gradient (x alpha) into z_trans, pushing z_trans to be HOSTILE to fine while super stays easy.

Wiring: fine<-z_low, super<-z_trans, fine_adv<-GRL(z_trans). beta fixed at 0.1 (Arm C sweet
spot). alpha=0 must reproduce Arm C (internal sanity). Run: armD_beta0.1_a{alpha}_K8_seed{s}.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_gpu import GPUData
from models import LowerTierCNN, TranslationVIB
from losses import kl_standard_normal


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.alpha * g, None


def grad_reverse(x, alpha):
    return GradReverse.apply(x, alpha)


class ArmDModel(nn.Module):
    def __init__(self, z_low_dim=64, K=8, n_fine=100, n_coarse=20):
        super().__init__()
        self.lower = LowerTierCNN(z_low_dim)
        self.translate = TranslationVIB(z_low_dim, 128, K)
        self.fine_head = nn.Sequential(nn.Linear(z_low_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, n_fine))
        self.super_head = nn.Sequential(nn.Linear(K, 64), nn.ReLU(inplace=True), nn.Linear(64, n_coarse))
        self.fine_adv_head = nn.Sequential(nn.Linear(K, 64), nn.ReLU(inplace=True), nn.Linear(64, n_fine))

    def forward(self, x, alpha):
        z_low = self.lower(x)
        z_trans, mu, logvar = self.translate(z_low)
        return {
            "z_low": z_low, "z_trans": z_trans, "mu": mu, "logvar": logvar,
            "fine_logits": self.fine_head(z_low),
            "super_logits": self.super_head(z_trans),
            "fine_adv_logits": self.fine_adv_head(grad_reverse(z_trans, alpha)),
        }


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate_split(model, data, split, alpha):
    model.eval()
    fc = sc = n = 0; kl_sum = 0.0
    for x, fine, coarse in data.iter_eval(split):
        out = model(x, alpha)
        fc += int((out["fine_logits"].argmax(1) == fine).sum())
        sc += int((out["super_logits"].argmax(1) == coarse).sum())
        kl_sum += float(kl_standard_normal(out["mu"], out["logvar"]).sum()); n += x.size(0)
    return {"fine_acc": fc / n, "super_acc": sc / n, "kl_rate": kl_sum / n}


@torch.no_grad()
def extract_features(model, data, split, alpha):
    model.eval(); zl, zt, ys, cs = [], [], [], []
    for x, fine, coarse in data.iter_eval(split):
        out = model(x, alpha)
        zl.append(out["z_low"].cpu().numpy()); zt.append(out["z_trans"].cpu().numpy())
        ys.append(fine.cpu().numpy()); cs.append(coarse.cpu().numpy())
    return (np.concatenate(zl), np.concatenate(zt), np.concatenate(ys), np.concatenate(cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--out_dir", type=str, default="results"); ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    ap.add_argument("--data_root", type=str, default="./data")
    args = ap.parse_args()

    run = f"armD_beta{args.beta}_a{args.alpha}_K{args.K}_seed{args.seed}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True); os.makedirs(args.ckpt_dir, exist_ok=True)

    data = GPUData(device, args.data_root)
    model = ArmDModel(K=args.K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    history = []; t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for x, fine, coarse in data.iter_train(args.batch_size, gen):
            out = model(x, args.alpha)
            l_fine = F.cross_entropy(out["fine_logits"], fine)
            l_super = F.cross_entropy(out["super_logits"], coarse)
            l_adv = F.cross_entropy(out["fine_adv_logits"], fine)
            kl = kl_standard_normal(out["mu"], out["logvar"]).mean()
            loss = l_fine + l_super + args.beta * kl + l_adv
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            val = evaluate_split(model, data, "val", args.alpha)
            history.append({"epoch": ep, "l_fine": float(l_fine.detach()), "l_super": float(l_super.detach()),
                            "l_adv": float(l_adv.detach()), "kl": float(kl.detach()), "val": val})
            print(f"[{run}] ep{ep:02d} fine={float(l_fine):.3f} super={float(l_super):.3f} "
                  f"adv={float(l_adv):.3f} kl={float(kl):.3f} val_fine={val['fine_acc']:.3f} val_super={val['super_acc']:.3f}")

    test = evaluate_split(model, data, "test", args.alpha)
    elapsed = time.time() - t0
    zl_tr, zt_tr, fine_tr, coarse_tr = extract_features(model, data, "train", args.alpha)
    zl_te, zt_te, fine_te, coarse_te = extract_features(model, data, "test", args.alpha)
    np.savez_compressed(os.path.join(args.out_dir, f"features_{run}.npz"),
                        z_low_tr=zl_tr, z_trans_tr=zt_tr, fine_tr=fine_tr, coarse_tr=coarse_tr,
                        z_low_te=zl_te, z_trans_te=zt_te, fine_te=fine_te, coarse_te=coarse_te)
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"{run}.pt"))
    meta = {"run": run, "arm": "D", "beta": args.beta, "alpha": args.alpha, "K": args.K,
            "seed": args.seed, "epochs": args.epochs, "test": test, "elapsed_sec": elapsed, "history": history}
    with open(os.path.join(args.out_dir, f"train_{run}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{run}] DONE test_fine={test['fine_acc']:.3f} test_super={test['super_acc']:.3f} "
          f"test_kl={test['kl_rate']:.3f} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
