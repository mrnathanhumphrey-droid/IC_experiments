"""§6.B locked test (pre-reg bbb91aeb). Fresh held-out training (existing checkpoints saw all 100 classes
-> can't do NOVEL-class transfer; fresh is more rigorous). Local, CUDA.

PRIMARY: novel-class SUPERCLASS transfer, Arm-C z_trans vs Arm-B z_trans (BOTH K=8 bottlenecked = matched
dim+exposure; diff = dedicated vs shared bottleneck). MDE pre-reg ~0.03 (N=1500 query); <0.02 = NO_VERDICT.
SECONDARY: fine acc, Arm-C vs matched-readout-uniform (fine<-z_low, super-wt 0) on held-IN test = NET
super-objective effect (tax vs regularization, named), not clean compartmentalization.
"""
import json, numpy as np, torch, torch.nn.functional as F
import torchvision
from transfer_judge import holdout_classes, HoldoutData, super_clf_acc, set_seed
from train_armC import ArmCModel
from models import FullModel
from losses import kl_standard_normal
from data import FINE_TO_COARSE, CIFAR100_MEAN, CIFAR100_STD

BETA, SEEDS, EPOCHS = 0.1, [0, 1, 2], 50
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def _train(model, data, beta, seed, epochs, arm):
    set_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    for ep in range(epochs):
        model.train()
        for x, fine, coarse in data.iter_train(128, gen):
            out = model(x)
            kl = kl_standard_normal(out["mu"], out["logvar"]).mean()
            if arm == "C":      # fine<-z_low, super<-z_trans (equal weight)
                loss = F.cross_entropy(out["fine_logits"], fine) + F.cross_entropy(out["super_logits"], coarse) + beta * kl
            elif arm == "B":    # fine<-z_trans + super<-z_trans (shared bottleneck, lam=0.5)
                loss = F.cross_entropy(out["fine_logits"], fine) + 0.5 * F.cross_entropy(out["super_logits"], coarse) + beta * kl
            else:               # uniform: fine<-z_low ONLY, super-weight 0 (z_trans dead)
                loss = F.cross_entropy(out["fine_logits"], fine) + beta * kl
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model

def heldin_test(device, held):
    """held-IN test images (the 85 trained classes) for the fine secondary."""
    te = torchvision.datasets.CIFAR100("./data", train=False, download=True)
    held_set = set(held)
    x = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous()
    yf = torch.tensor(te.targets, dtype=torch.long)
    mask = torch.tensor([int(y) not in held_set for y in yf.tolist()])
    mean = torch.tensor(CIFAR100_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR100_STD, device=device).view(1, 3, 1, 1)
    xb = ((x[mask].float().div(255.0).to(device)) - mean) / std
    return xb, yf[mask].to(device)

@torch.no_grad()
def fine_acc(model, xb, yf, bs=1024):
    model.eval(); c = n = 0
    for i in range(0, xb.size(0), bs):
        c += int((model(xb[i:i+bs])["fine_logits"].argmax(1) == yf[i:i+bs]).sum()); n += xb[i:i+bs].size(0)
    return c / n

def main():
    held = holdout_classes(15)
    data = HoldoutData(DEV, held)
    xin, yin = heldin_test(DEV, held)
    sup_fit_labels = data.train_in["coarse"].cpu().numpy()
    sup_q = data.novel_query["coarse"].cpu().numpy()
    print(f"held-out 15 classes | train-in {data.train_in['x'].size(0)} | novel query {len(sup_q)} | held-IN test {xin.size(0)}")
    print(f"beta={BETA} epochs={EPOCHS} seeds={SEEDS}\n")

    cC, cB = [], []     # per-query superclass-correctness for Arm-C / Arm-B z_trans
    fineC, fineU = [], []
    for s in SEEDS:
        mC = _train(ArmCModel(K=8).to(DEV), data, BETA, s, EPOCHS, "C")
        mB = _train(FullModel(z_low_dim=64, K=8, use_super=True).to(DEV), data, BETA, s, EPOCHS, "B")
        mU = _train(ArmCModel(K=8).to(DEV), data, BETA, s, EPOCHS, "U")
        rng = np.random.default_rng(s)
        # PRIMARY: super-clf on trained-class z_trans -> predict super of NOVEL query z_trans
        _, ztC_fit = data.features(mC, data.train_in); _, ztC_q = data.features(mC, data.novel_query)
        _, ztB_fit = data.features(mB, data.train_in); _, ztB_q = data.features(mB, data.novel_query)
        sub = rng.choice(ztC_fit.shape[0], size=8000, replace=False)
        corrC = super_clf_acc(ztC_fit[sub], sup_fit_labels[sub], ztC_q, sup_q)
        corrB = super_clf_acc(ztB_fit[sub], sup_fit_labels[sub], ztB_q, sup_q)
        cC.append(corrC); cB.append(corrB)
        # SECONDARY: fine acc on held-IN test, Arm-C vs matched-readout-uniform (both fine<-z_low)
        aC, aU = fine_acc(mC, xin, yin), fine_acc(mU, xin, yin)
        fineC.append(aC); fineU.append(aU)
        print(f"[seed {s}] novel-super  Arm-C z_trans={corrC.mean():.4f}  Arm-B z_trans={corrB.mean():.4f}  "
              f"diff={corrC.mean()-corrB.mean():+.4f} | fine(heldIN) C={aC:.4f} U={aU:.4f} dC-U={aC-aU:+.4f}")

    # PRIMARY decision: paired diff over query, mean across seeds, bootstrap over query
    mC_q = np.mean(cC, axis=0); mB_q = np.mean(cB, axis=0)
    point = float(mC_q.mean() - mB_q.mean())
    nq = mC_q.shape[0]; rng = np.random.default_rng(12345)
    boot = [mC_q[idx].mean() - mB_q[idx].mean() for idx in (rng.integers(0, nq, nq) for _ in range(2000))]
    lo, hi = np.percentile(boot, [2.5, 97.5]); MDE = 0.03
    if point >= MDE and lo > 0:      verdict = "COMPARTMENTALIZATION-TRANSFERS"
    elif hi < MDE:                   verdict = "NO_EFFECT (powered null: meaningful effect ruled out)"
    else:                            verdict = "NO_VERDICT (underpowered: CI spans 0 and MDE)"

    print(f"\n=== PRIMARY (Arm-C z_trans vs Arm-B z_trans, novel-class superclass transfer) ===")
    print(f"  Arm-C={mC_q.mean():.4f}  Arm-B={mB_q.mean():.4f}  paired diff={point:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  MDE={MDE}")
    print(f"  -> {verdict}")
    print(f"\n=== SECONDARY (fine, Arm-C vs matched-readout-uniform; NET super effect, NOT compartmentalization) ===")
    print(f"  Arm-C fine={np.mean(fineC):.4f}  uniform fine={np.mean(fineU):.4f}  net diff={np.mean(fineC)-np.mean(fineU):+.4f}")
    print(f"  (>0 = super regularizes z_low; <0 = super taxes z_low capacity; not separable here)")

    json.dump({"beta": BETA, "epochs": EPOCHS, "seeds": SEEDS,
               "primary": {"armC_superacc": float(mC_q.mean()), "armB_superacc": float(mB_q.mean()),
                           "paired_diff": point, "ci95": [float(lo), float(hi)], "MDE": MDE, "verdict": verdict},
               "secondary_fine": {"armC": float(np.mean(fineC)), "uniform": float(np.mean(fineU)),
                                  "net_diff": float(np.mean(fineC)-np.mean(fineU))}},
              open("results/run_6b_transfer_result.json", "w"), indent=2)
    print("\nwrote results/run_6b_transfer_result.json")

if __name__ == "__main__":
    main()
