"""Form 1 transfer test — does hyperbolic z_trans EXTEND/AMPLIFY the low-data crossover that
Euclidean z_trans showed in Axis 1? Apples-to-apples: prototype (NCM) classifier scored in each
representation's NATIVE geometry (Poincare distance for hyperbolic, Euclidean otherwise), all vs
the same z_low reference, comparing crossover CURVES across the fit-pool sweep.

Per seed: train BOTH a Euclidean Arm C (reuse) and a hyperbolic Arm-Hyp holdout model on the same
held-in 85 fine classes; sweep fit-pool for novel-class SUPER transfer. Report three curves:
euc_ztrans - zlow, hyp_ztrans - zlow, hyp_ztrans - euc_ztrans (the direct Form-1 contrast).

FORM-1 CONFIRMS iff hyp_ztrans - euc_ztrans >= +0.02 (CI>0) at any fit_pool <= 1000, OR hyp extends
the crossover (hyp_ztrans - zlow >= +0.02, CI>0) at fit_pool >= 4000 where Euclidean died.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F

from transfer_judge import HoldoutData, holdout_classes, train_armC_holdout
from train_armHyp import ArmHypModel
from losses import kl_standard_normal

FIT_POOLS = [50, 200, 1000, 4000, 8000]


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def train_hyp_holdout(data, device, beta, seed, epochs):
    set_seed(seed)
    model = ArmHypModel(K=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    gen = torch.Generator(device=device).manual_seed(seed)
    for ep in range(epochs):
        model.train()
        for x, fine, coarse in data.iter_train(128, gen):
            out = model(x)
            loss = F.cross_entropy(out["fine_logits"], fine) + F.cross_entropy(out["super_logits"], coarse) \
                + beta * kl_standard_normal(out["mu"], out["logvar"]).mean()
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
    return model


def project_np(x, eps=1e-5):
    n = np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)
    return np.where(n > 1 - eps, x / n * (1 - eps), x)


def poincare_np(x, y, eps=1e-5):
    diff2 = ((x - y) ** 2).sum(-1); xn = (x ** 2).sum(-1); yn = (y ** 2).sum(-1)
    denom = np.maximum(1 - xn, eps) * np.maximum(1 - yn, eps)
    return np.arccosh(np.maximum(1 + 2 * diff2 / denom, 1 + eps))


def ncm_super(feat_fit, sup_fit, idx, feat_q, sup_q, metric):
    """Prototype (class-mean) super-classifier from fit_fit[idx]; classify novel query in `metric`.
    Returns per-query correctness array."""
    protos, labs = [], []
    sf = sup_fit[idx]; ff = feat_fit[idx]
    for s in range(20):
        m = sf == s
        if m.sum() == 0:
            continue
        protos.append(ff[m].mean(0)); labs.append(s)
    P = np.stack(protos); labs = np.array(labs)
    if metric == "euclidean":
        d = ((feat_q[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    else:  # hyperbolic
        Pb = project_np(P); Qb = project_np(feat_q)
        d = poincare_np(Qb[:, None, :], Pb[None, :, :])
    pred = labs[d.argmin(1)]
    return (pred == sup_q).astype(np.float64)


def boot_diff(a_seeds, b_seeds, B=2000, seed=777):
    """paired diff (a-b) per-query, averaged across seeds, bootstrap CI over query."""
    am = np.mean(a_seeds, axis=0); bm = np.mean(b_seeds, axis=0)
    point = float(am.mean() - bm.mean()); nq = am.shape[0]; rng = np.random.default_rng(seed); bt = []
    for _ in range(B):
        i = rng.integers(0, nq, nq); bt.append(am[i].mean() - bm[i].mean())
    lo, hi = np.percentile(bt, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--out_dir", type=str, default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    held = holdout_classes(15)
    data = HoldoutData(device, held)

    store = {p: {"zlow": [], "euc": [], "hyp": []} for p in FIT_POOLS}
    for seed in args.seeds:
        euc = train_armC_holdout(data, device, args.beta, seed, args.epochs)
        hyp = train_hyp_holdout(data, device, args.beta, seed, args.epochs)
        zl_e, zt_e = data.features(euc, data.train_in)
        zl_eq, zt_eq = data.features(euc, data.novel_query)
        _, zt_h = data.features(hyp, data.train_in)
        _, zt_hq = data.features(hyp, data.novel_query)
        sup_fit = data.train_in["coarse"].cpu().numpy(); sup_q = data.novel_query["coarse"].cpu().numpy()
        rng = np.random.default_rng(seed)
        for p in FIT_POOLS:
            idx = rng.choice(zl_e.shape[0], size=min(p, zl_e.shape[0]), replace=False)
            store[p]["zlow"].append(ncm_super(zl_e, sup_fit, idx, zl_eq, sup_q, "euclidean"))
            store[p]["euc"].append(ncm_super(zt_e, sup_fit, idx, zt_eq, sup_q, "euclidean"))
            store[p]["hyp"].append(ncm_super(zt_h, sup_fit, idx, zt_hq, sup_q, "hyperbolic"))
        print(f"[seed {seed}] done")

    print("\n=== FORM 1: hyperbolic vs euclidean z_trans, native-geometry NCM super-transfer ===")
    print(f"{'pool':>5} | {'zlow':>6} {'euc':>6} {'hyp':>6} | {'euc-zlow':>16} {'hyp-zlow':>16} {'hyp-euc':>16}")
    results = {}; confirm = False
    for p in FIT_POOLS:
        zlow = float(np.mean(store[p]["zlow"], axis=0).mean())
        euc = float(np.mean(store[p]["euc"], axis=0).mean())
        hyp = float(np.mean(store[p]["hyp"], axis=0).mean())
        de, le, he = boot_diff(store[p]["euc"], store[p]["zlow"])
        dh, lh, hh = boot_diff(store[p]["hyp"], store[p]["zlow"])
        dx, lx, hx = boot_diff(store[p]["hyp"], store[p]["euc"])
        gate_he = (dx >= 0.02 and lx > 0 and p <= 1000)
        gate_ext = (dh >= 0.02 and lh > 0 and p >= 4000)
        if gate_he or gate_ext:
            confirm = True
        results[p] = {"zlow": zlow, "euc": euc, "hyp": hyp,
                      "euc_minus_zlow": [de, le, he], "hyp_minus_zlow": [dh, lh, hh], "hyp_minus_euc": [dx, lx, hx]}
        flag = "  <-- FORM1" if (gate_he or gate_ext) else ""
        print(f"{p:>5} | {zlow:.3f} {euc:.3f} {hyp:.3f} | "
              f"{de:+.3f}[{le:+.3f},{he:+.3f}] {dh:+.3f}[{lh:+.3f},{hh:+.3f}] {dx:+.3f}[{lx:+.3f},{hx:+.3f}]{flag}")

    verdict = {"held_out_fine": held, "by_pool": results, "FORM1_CONFIRMED": bool(confirm),
               "gate": "hyp-euc>=+0.02&CI>0 at pool<=1000 OR hyp-zlow>=+0.02&CI>0 at pool>=4000"}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "pay_frontier_form1.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nFORM 1 {'CONFIRMED — geometry extends/amplifies the crossover' if confirm else 'NULL — hyperbolic does not beat Euclidean z_trans'}")


if __name__ == "__main__":
    main()
