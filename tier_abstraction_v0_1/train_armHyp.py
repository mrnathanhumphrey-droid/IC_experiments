"""Form 1 — hyperbolic z_trans. Identical to Arm C EXCEPT z_trans lives in the Poincare ball
and the superclass head is hyperbolic-distance-to-prototypes. fine<-z_low stays Euclidean.

Wiring: encoder -> tangent (mu,logvar) -> sample (VIB, wrapped-normal) -> expmap0 -> z_hyp;
super logit_s = -dist(z_hyp, expmap0(proto_s)); fine from z_low. L = L_fine + L_super + beta*KL.
Saves features (z_low, z_trans=ball coords) in the standard npz format. Run: armHyp_beta{b}_K8_seed{s}.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_gpu import GPUData
from models import LowerTierCNN, TranslationVIB
from losses import kl_standard_normal
import hyperbolic as H


class ArmHypModel(nn.Module):
    def __init__(self, z_low_dim=64, K=8, n_fine=100, n_coarse=20):
        super().__init__()
        self.lower = LowerTierCNN(z_low_dim)
        self.translate = TranslationVIB(z_low_dim, 128, K)          # outputs tangent mu/logvar + sample
        self.fine_head = nn.Sequential(nn.Linear(z_low_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, n_fine))
        self.proto = nn.Parameter(torch.randn(n_coarse, K) * 0.01)  # tangent-space super prototypes

    def forward(self, x):
        z_low = self.lower(x)
        z_tan, mu, logvar = self.translate(z_low)          # sampled tangent (train) / mu (eval)
        z_hyp = H.expmap0(z_tan)                            # point in the ball
        proto = H.expmap0(self.proto)                      # [C,K] prototypes in the ball
        super_logits = -H.dist(z_hyp.unsqueeze(1), proto.unsqueeze(0))  # [B,C]
        return {"z_low": z_low, "z_trans": z_hyp, "mu": mu, "logvar": logvar,
                "fine_logits": self.fine_head(z_low), "super_logits": super_logits}


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate_split(model, data, split):
    model.eval(); fc = sc = n = 0; kl_sum = 0.0
    for x, fine, coarse in data.iter_eval(split):
        out = model(x)
        fc += int((out["fine_logits"].argmax(1) == fine).sum())
        sc += int((out["super_logits"].argmax(1) == coarse).sum())
        kl_sum += float(kl_standard_normal(out["mu"], out["logvar"]).sum()); n += x.size(0)
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
    ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4); ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints"); ap.add_argument("--data_root", type=str, default="./data")
    args = ap.parse_args()

    run = f"armHyp_beta{args.beta}_K{args.K}_seed{args.seed}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True); os.makedirs(args.ckpt_dir, exist_ok=True)

    data = GPUData(device, args.data_root)
    model = ArmHypModel(K=args.K).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    history = []; t0 = time.time(); nan_hits = 0
    for ep in range(args.epochs):
        model.train()
        for x, fine, coarse in data.iter_train(args.batch_size, gen):
            out = model(x)
            l_fine = F.cross_entropy(out["fine_logits"], fine)
            l_super = F.cross_entropy(out["super_logits"], coarse)
            kl = kl_standard_normal(out["mu"], out["logvar"]).mean()
            loss = l_fine + l_super + args.beta * kl
            if not torch.isfinite(loss):
                nan_hits += 1; opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # hyperbolic stability
            opt.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            val = evaluate_split(model, data, "val")
            history.append({"epoch": ep, "l_fine": float(l_fine.detach()), "l_super": float(l_super.detach()),
                            "kl": float(kl.detach()), "val": val, "nan_hits": nan_hits})
            print(f"[{run}] ep{ep:02d} fine={float(l_fine):.3f} super={float(l_super):.3f} kl={float(kl):.3f} "
                  f"val_fine={val['fine_acc']:.3f} val_super={val['super_acc']:.3f} nan={nan_hits}")

    test = evaluate_split(model, data, "test")
    elapsed = time.time() - t0
    zl_tr, zt_tr, fine_tr, coarse_tr = extract_features(model, data, "train")
    zl_te, zt_te, fine_te, coarse_te = extract_features(model, data, "test")
    np.savez_compressed(os.path.join(args.out_dir, f"features_{run}.npz"),
                        z_low_tr=zl_tr, z_trans_tr=zt_tr, fine_tr=fine_tr, coarse_tr=coarse_tr,
                        z_low_te=zl_te, z_trans_te=zt_te, fine_te=fine_te, coarse_te=coarse_te)
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"{run}.pt"))
    meta = {"run": run, "arm": "Hyp", "beta": args.beta, "K": args.K, "seed": args.seed,
            "epochs": args.epochs, "test": test, "elapsed_sec": elapsed, "nan_hits": nan_hits, "history": history}
    with open(os.path.join(args.out_dir, f"train_{run}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{run}] DONE test_fine={test['fine_acc']:.3f} test_super={test['super_acc']:.3f} "
          f"test_kl={test['kl_rate']:.3f} nan_hits={nan_hits} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
