"""Arm C — superclass as the PRIMARY upper-tier objective (thread #2 from v0.1 verdict).

Wiring (IC-shaped):
  fine label  <- head on z_low        (lower tier keeps fine detail; bypasses bottleneck)
  super label <- head on z_trans       (upper tier, THROUGH the VIB bottleneck, is the
                                         coarse adjudicator)
  L = L_fine(z_low) + L_super(z_trans) + beta*KL    (equal weight on the two tasks)

z_trans receives gradient ONLY from the superclass loss -> it is pressured to be
superclass-SUFFICIENT. The pre-registered question: does that make z_trans selectively
discard within-superclass fine structure (selectivity index climbs with beta), i.e. does
DIRECTING the upper objective at the hierarchy produce the abstraction the bottleneck alone
did not? Contrast vs v0.1 Arm A (selectivity pinned at 0) and Arm B (+0.02, beta-independent).

Saves features in the SAME npz format as train.py so evaluate.py/plot.py/aggregate.py work
unchanged. Run label: armC_beta{b}_K{K}_seed{s}.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_gpu import GPUData
from models import LowerTierCNN, TranslationVIB
from losses import kl_standard_normal


class ArmCModel(nn.Module):
    def __init__(self, z_low_dim=64, K=8, n_fine=100, n_coarse=20):
        super().__init__()
        self.lower = LowerTierCNN(z_low_dim)
        self.translate = TranslationVIB(z_low_dim, 128, K)
        self.fine_head = nn.Sequential(nn.Linear(z_low_dim, 64), nn.ReLU(inplace=True),
                                       nn.Linear(64, n_fine))     # fine from z_low (direct)
        self.super_head = nn.Sequential(nn.Linear(K, 64), nn.ReLU(inplace=True),
                                        nn.Linear(64, n_coarse))  # super from z_trans (bottleneck)

    def forward(self, x):
        z_low = self.lower(x)
        z_trans, mu, logvar = self.translate(z_low)
        return {
            "z_low": z_low, "z_trans": z_trans, "mu": mu, "logvar": logvar,
            "fine_logits": self.fine_head(z_low),
            "super_logits": self.super_head(z_trans),
        }


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate_split(model, data, split):
    model.eval()
    fine_correct = sup_correct = n = 0
    kl_sum = 0.0
    for x, fine, coarse in data.iter_eval(split):
        out = model(x)
        fine_correct += int((out["fine_logits"].argmax(1) == fine).sum())
        sup_correct += int((out["super_logits"].argmax(1) == coarse).sum())
        kl_sum += float(kl_standard_normal(out["mu"], out["logvar"]).sum())
        n += x.size(0)
    return {"fine_acc": fine_correct / n, "super_acc": sup_correct / n, "kl_rate": kl_sum / n}


@torch.no_grad()
def extract_features(model, data, split):
    model.eval()
    zl, zt, ys, cs = [], [], [], []
    for x, fine, coarse in data.iter_eval(split):
        out = model(x)
        zl.append(out["z_low"].cpu().numpy()); zt.append(out["z_trans"].cpu().numpy())
        ys.append(fine.cpu().numpy()); cs.append(coarse.cpu().numpy())
    return (np.concatenate(zl), np.concatenate(zt), np.concatenate(ys), np.concatenate(cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    ap.add_argument("--data_root", type=str, default="./data")
    args = ap.parse_args()

    run = f"armC_beta{args.beta}_K{args.K}_seed{args.seed}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True); os.makedirs(args.ckpt_dir, exist_ok=True)

    data = GPUData(device, args.data_root)
    model = ArmCModel(K=args.K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for x, fine, coarse in data.iter_train(args.batch_size, gen):
            out = model(x)
            l_fine = F.cross_entropy(out["fine_logits"], fine)
            l_super = F.cross_entropy(out["super_logits"], coarse)
            kl = kl_standard_normal(out["mu"], out["logvar"]).mean()
            loss = l_fine + l_super + args.beta * kl
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            val = evaluate_split(model, data, "val")
            lf, ls, lk = float(l_fine.detach()), float(l_super.detach()), float(kl.detach())
            history.append({"epoch": ep, "l_fine": lf, "l_super": ls, "kl": lk, "val": val})
            print(f"[{run}] ep{ep:02d} fine={lf:.3f} super={ls:.3f} "
                  f"kl={lk:.3f} val_fine={val['fine_acc']:.3f} val_super={val['super_acc']:.3f}")

    test = evaluate_split(model, data, "test")
    elapsed = time.time() - t0
    zl_tr, zt_tr, fine_tr, coarse_tr = extract_features(model, data, "train")
    zl_te, zt_te, fine_te, coarse_te = extract_features(model, data, "test")
    np.savez_compressed(os.path.join(args.out_dir, f"features_{run}.npz"),
                        z_low_tr=zl_tr, z_trans_tr=zt_tr, fine_tr=fine_tr, coarse_tr=coarse_tr,
                        z_low_te=zl_te, z_trans_te=zt_te, fine_te=fine_te, coarse_te=coarse_te)
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"{run}.pt"))
    meta = {"run": run, "arm": "C", "beta": args.beta, "K": args.K, "seed": args.seed,
            "epochs": args.epochs, "test": test, "elapsed_sec": elapsed, "history": history}
    with open(os.path.join(args.out_dir, f"train_{run}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{run}] DONE test_fine={test['fine_acc']:.3f} test_super={test['super_acc']:.3f} "
          f"test_kl={test['kl_rate']:.3f} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
