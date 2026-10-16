"""Train one (beta, K, seed, arm) run. Saves checkpoint, frozen features (npz) for the
probe stage, and a metrics json including the test KL rate (the MI proxy).

Arm A: lam=0, use_super=False   (abstraction from the bottleneck alone)
Arm B: lam=0.5, use_super=True  (Wilson's original; superclass head present)
"""
import argparse, json, os, time
import numpy as np
import torch

from data_gpu import GPUData
from models import FullModel
from losses import combined_loss, kl_standard_normal


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def evaluate_split(model, data, split, use_super):
    model.eval()
    fine_correct = sup_correct = n = 0
    kl_sum = 0.0
    for x, fine, coarse in data.iter_eval(split):
        out = model(x)
        fine_correct += int((out["fine_logits"].argmax(1) == fine).sum())
        if use_super and out["super_logits"] is not None:
            sup_correct += int((out["super_logits"].argmax(1) == coarse).sum())
        kl_sum += float(kl_standard_normal(out["mu"], out["logvar"]).sum())
        n += x.size(0)
    return {"fine_acc": fine_correct / n,
            "super_acc": (sup_correct / n) if use_super else None,
            "kl_rate": kl_sum / n}


@torch.no_grad()
def extract_features(model, data, split):
    """Frozen-backbone features at eval (z_trans = mu, deterministic)."""
    model.eval()
    zl, zt, ys, cs = [], [], [], []
    for x, fine, coarse in data.iter_eval(split):
        out = model(x)
        zl.append(out["z_low"].cpu().numpy())
        zt.append(out["z_trans"].cpu().numpy())  # == mu in eval mode
        ys.append(fine.cpu().numpy()); cs.append(coarse.cpu().numpy())
    return (np.concatenate(zl), np.concatenate(zt),
            np.concatenate(ys), np.concatenate(cs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--use_super", type=int, required=True)  # 1=Arm B, 0=Arm A
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--arm", type=str, required=True)        # "A" or "B" (label only)
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--ckpt_dir", type=str, default="checkpoints")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    use_super = bool(args.use_super)
    run = f"arm{args.arm}_beta{args.beta}_K{args.K}_seed{args.seed}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    data = GPUData(device, args.data_root)
    model = FullModel(z_low_dim=64, K=args.K, use_super=use_super).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    train_gen = torch.Generator(device=device).manual_seed(args.seed)

    history = []
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        for x, fine, coarse in data.iter_train(args.batch_size, train_gen):
            out = model(x)
            loss, parts = combined_loss(out, fine, coarse, args.beta, args.lam, use_super)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep % 10 == 0 or ep == args.epochs - 1:
            val = evaluate_split(model, data, "val", use_super)
            history.append({"epoch": ep, **parts, "val": val})
            print(f"[{run}] ep{ep:02d} L={parts['total']:.3f} "
                  f"fine={parts['l_fine']:.3f} kl={parts['kl']:.3f} "
                  f"val_fine={val['fine_acc']:.3f} val_kl={val['kl_rate']:.3f}")

    test = evaluate_split(model, data, "test", use_super)
    elapsed = time.time() - t0

    # frozen features for the probe stage
    zl_tr, zt_tr, fine_tr, coarse_tr = extract_features(model, data, "train")
    zl_te, zt_te, fine_te, coarse_te = extract_features(model, data, "test")
    np.savez_compressed(
        os.path.join(args.out_dir, f"features_{run}.npz"),
        z_low_tr=zl_tr, z_trans_tr=zt_tr, fine_tr=fine_tr, coarse_tr=coarse_tr,
        z_low_te=zl_te, z_trans_te=zt_te, fine_te=fine_te, coarse_te=coarse_te,
    )
    torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"{run}.pt"))

    meta = {"run": run, "arm": args.arm, "beta": args.beta, "lam": args.lam,
            "K": args.K, "seed": args.seed, "epochs": args.epochs,
            "use_super": use_super, "test": test, "elapsed_sec": elapsed,
            "history": history}
    with open(os.path.join(args.out_dir, f"train_{run}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{run}] DONE test_fine={test['fine_acc']:.3f} "
          f"test_kl={test['kl_rate']:.3f} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
