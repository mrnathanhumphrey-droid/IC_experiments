"""Axis-G SUPERSESSION (v1.R). Tests the CORRECTION half on granularity.

v1 conjunction = drop-then-FLAT: prevention arrested serial decay but didn't make granularity pay
(k3-k1 still -0.027). Pre-registered "Flat" reading: prevention necessary-not-sufficient -> the
ACTIVE half (supersession) is required. This tests it.

Lesson from v1: DON'T contaminate the abstract tier with finer objectives (accumulation hurt).
Instead: keep abstract tier CLEAN (M_class, = k1) + add finer objective as a SEPARATE correction
store (M_order) that SUPERSEDES the abstract tier's cross-class mistakes, precondition-conditional.
= separation (Arm-C) + correction (rapide_store), not accumulation.

Supersession rule: base order = order-NCM on M_class rep. PRECONDITION = M_class's order-call
crosses M_class's own class-call (internal inconsistency = abstract tier flags it's confused). On
those, OVERRIDE with M_order's order-call. Else keep base.

Gates: G_super sel(super)-sel(base k1) >= +0.02 CI>0 ; G_disc super(conditional) >= blanket(always M_order).
--synthetic = CPU smoke before GPU.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy("file_system")
TOP, REF = "class", "order"
K, BETA, EPOCHS = 16, 0.1, 60


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class SingleVIB(nn.Module):
    def __init__(self, d_in, n, K=16, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(True))
        self.head = nn.Linear(hidden, 2 * K); self.clf = nn.Linear(K, n)

    def forward(self, x):
        mu, lv = self.head(self.trunk(x)).chunk(2, -1)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu) if self.training else mu
        return z, mu, lv, self.clf(z)


def kl(mu, lv): return 0.5 * torch.sum(mu.pow(2) + lv.exp() - lv - 1.0, -1).mean()


def train(zl, y, n, device, seed):
    set_seed(seed)
    m = SingleVIB(zl.shape[1], n, K).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Z = torch.tensor(zl, device=device); Y = torch.tensor(y, device=device).long()
    g = torch.Generator(device=device).manual_seed(seed); bs = 512
    for ep in range(EPOCHS):
        m.train(); order = torch.randperm(Z.size(0), generator=g, device=device)
        for i in range(0, Z.size(0), bs):
            idx = order[i:i + bs]
            z, mu, lv, lg = m(Z[idx])
            loss = F.cross_entropy(lg, Y[idx]) + BETA * kl(mu, lv)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    return m


@torch.no_grad()
def rep(m, zl, device, bs=4096):
    m.eval(); out = []
    Z = torch.tensor(zl, device=device)
    for i in range(0, Z.size(0), bs):
        out.append(m(Z[i:i + bs])[0].cpu().numpy())
    return np.concatenate(out)


def ncm(feat_fit, y_fit, feat_q):
    P, L = [], []
    for c in np.unique(y_fit):
        P.append(feat_fit[y_fit == c].mean(0)); L.append(c)
    P = np.stack(P); L = np.array(L)
    return L[(((feat_q[:, None, :] - P[None, :, :]) ** 2).sum(-1)).argmin(1)]


def sel_from_pred(pred, true, ref_to_top, ref_fit):
    err = pred != true
    if err.sum() == 0: return 0.0
    same = (ref_to_top[pred[err]] == ref_to_top[true[err]]).mean()
    refs = np.unique(ref_fit); tops = ref_to_top[refs]; M = len(refs); chance = 0.0
    for c in np.unique(tops):
        m_c = (tops == c).sum(); chance += (m_c / M) * ((m_c - 1) / max(M - 1, 1))
    return float(same - chance)


def boot_diff(a, b, B=2000, seed=7):
    a = np.asarray(a); b = np.asarray(b); pt = float(a.mean() - b.mean()); r = np.random.default_rng(seed); n = len(a)
    bs = [a[r.integers(0, n, n)].mean() - b[r.integers(0, n, n)].mean() for _ in range(B)]
    return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--cache", default="/lambda/nfs/iNat/inat_feats.npz")
    ap.add_argument("--out_dir", default="/lambda/nfs/iNat")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    if args.synthetic:
        n = 6000; z = np.random.RandomState(0).randn(n, 64).astype(np.float32)
        cls = np.random.RandomState(1).randint(0, 6, n)
        order = cls * 4 + np.random.RandomState(2).randint(0, 4, n)
        z[:, :6] += np.eye(6)[cls] * 2.0
        labs = {"class": cls, "order": order}
    else:
        d = np.load(args.cache, allow_pickle=True); z = d["z"].astype(np.float32)
        labs = {k: d[k].astype(np.int64) for k in ("class", "order")}
    for k in list(labs):
        _, labs[k] = np.unique(labs[k], return_inverse=True)
    ref_to_top = np.zeros(int(labs[REF].max()) + 1, dtype=np.int64)
    for o in np.unique(labs[REF]):
        ref_to_top[o] = labs[TOP][labs[REF] == o][0]

    rng = np.random.default_rng(0); N = len(z); fit = rng.random(N) < 0.7; q = ~fit
    cf, of = labs[TOP][fit], labs[REF][fit]; oq = labs[REF][q]
    print(f"N={N} class={cf.max()+1} order={of.max()+1} fit={fit.sum()} q={q.sum()}", flush=True)

    base_s, super_s, blank_s, fire_rate = [], [], [], []
    for seed in args.seeds:
        Mc = train(z[fit], cf, int(cf.max()) + 1, device, seed)          # clean abstract tier
        Mo = train(z[fit], of, int(of.max()) + 1, device, seed)          # separate correction store
        rc_f, rc_q = rep(Mc, z[fit], device), rep(Mc, z[q], device)
        ro_f, ro_q = rep(Mo, z[fit], device), rep(Mo, z[q], device)
        c_hat = ncm(rc_f, cf, rc_q)                                      # M_class class call
        o_base = ncm(rc_f, of, rc_q)                                     # M_class order call (base)
        o_corr = ncm(ro_f, of, ro_q)                                     # M_order order call (corrector)
        fire = ref_to_top[o_base] != c_hat                              # precondition: order call crosses class call
        o_super = np.where(fire, o_corr, o_base)
        base_s.append(sel_from_pred(o_base, oq, ref_to_top, of))
        super_s.append(sel_from_pred(o_super, oq, ref_to_top, of))
        blank_s.append(sel_from_pred(o_corr, oq, ref_to_top, of))        # blanket override (always M_order)
        fire_rate.append(float(fire.mean()))
        print(f"[seed {seed}] base={base_s[-1]:+.4f} super={super_s[-1]:+.4f} blanket={blank_s[-1]:+.4f} fire={fire_rate[-1]:.3f}", flush=True)

    print("\n=== Axis-G SUPERSESSION: clean tier + separate correction store ===", flush=True)
    dsb, lsb, hsb = boot_diff(super_s, base_s)        # super vs base(k1)
    dsk, lsk, hsk = boot_diff(super_s, blank_s)       # conditional vs blanket (discipline)
    g_super = dsb >= 0.02 and lsb > 0
    g_disc = lsk > -0.005                              # conditional at least as good as blanket
    print(f"  base(k1)={np.mean(base_s):+.4f}  supersession={np.mean(super_s):+.4f}  blanket={np.mean(blank_s):+.4f}  fire_rate={np.mean(fire_rate):.3f}")
    print(f"  G_super super-base={dsb:+.4f}[{lsb:+.4f},{hsb:+.4f}] -> {g_super}")
    print(f"  G_disc  super-blanket={dsk:+.4f}[{lsk:+.4f},{hsk:+.4f}] -> {g_disc}")
    verdict = "SUPERSESSION_PULLS" if g_super else "CORRECTION_NULL"
    res = {"base_k1": float(np.mean(base_s)), "supersession": float(np.mean(super_s)),
           "blanket": float(np.mean(blank_s)), "fire_rate": float(np.mean(fire_rate)),
           "G_super": {"diff": dsb, "ci": [lsb, hsb], "pass": bool(g_super)},
           "G_disc": {"diff": dsk, "ci": [lsk, hsk], "pass": bool(g_disc)},
           "verdict": verdict, "raw": {"base": base_s, "super": super_s, "blanket": blank_s}}
    json.dump(res, open(os.path.join(args.out_dir, "granularity_axisG_supersession.json"), "w"), indent=2)
    print(f"  VERDICT: {verdict}\nDONE axisG-supersession", flush=True)


if __name__ == "__main__":
    main()
