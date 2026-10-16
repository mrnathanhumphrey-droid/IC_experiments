"""Axis-G CONJUNCTION re-test. LOCKED per PREREG_AXISG_GRANULARITY_2026_06_14 (amendment 2026-06-14).

Objectives must price ONE representation IN CONJUNCTION, not pass through each other as serial gates
(serial gates = serial misclassification by priors; the sequential run inverted -0.045). Realized as:
SHARED TRUNK = the conjunction (all k objectives jointly shape it) + separate bottlenecked VIB heads
= the compartmentalization (each level its own readout off the SAME trunk; no head consumes another).
Vary k = number of conjoined objectives; measure CLASS-head selectivity. Read the k-ladder as one
joint shape (rises with conjunction = lever real), not 3 independent gates.

--synthetic = CPU smoke-test before paying for GPU.
"""
import argparse, json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.multiprocessing.set_sharing_strategy("file_system")

ARMS = {1: ["class"], 2: ["class", "order"], 3: ["class", "order", "family"]}  # class head always idx 0
TOP = "class"; REF = "order"
K = 16; BETA = 0.1; EPOCHS = 60


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class ConjunctionVIB(nn.Module):
    """shared trunk -> k PARALLEL bottlenecked heads, ALL consuming the trunk (conjunction)."""
    def __init__(self, d_in, ncls, K=16, hidden=256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(True))
        self.heads = nn.ModuleList([nn.Linear(hidden, 2 * K) for _ in ncls])
        self.clf = nn.ModuleList([nn.Linear(K, n) for n in ncls])

    def forward(self, z_low):
        h = self.trunk(z_low); mus, lvs, zs, logits = [], [], [], []
        for i, head in enumerate(self.heads):
            mu, lv = head(h).chunk(2, -1)                       # ALL heads consume trunk h (conjunction)
            z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu) if self.training else mu
            mus.append(mu); lvs.append(lv); zs.append(z); logits.append(self.clf[i](z))
        return zs, mus, lvs, logits


def kl(mu, lv): return 0.5 * torch.sum(mu.pow(2) + lv.exp() - lv - 1.0, -1).mean()


def train(zl, ys, ncls, device, seed):
    set_seed(seed)
    m = ConjunctionVIB(zl.shape[1], ncls, K).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    Z = torch.tensor(zl, device=device); Y = [torch.tensor(y, device=device).long() for y in ys]
    g = torch.Generator(device=device).manual_seed(seed); bs = 512
    for ep in range(EPOCHS):
        m.train(); order = torch.randperm(Z.size(0), generator=g, device=device)
        for i in range(0, Z.size(0), bs):
            idx = order[i:i + bs]
            zs, mus, lvs, logits = m(Z[idx])
            loss = sum(F.cross_entropy(logits[t], Y[t][idx]) for t in range(len(ncls)))
            loss = loss + BETA * sum(kl(mus[t], lvs[t]) for t in range(len(ncls)))
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    return m


@torch.no_grad()
def class_repr(m, zl, device, bs=4096):
    m.eval(); out = []
    Z = torch.tensor(zl, device=device)
    for i in range(0, Z.size(0), bs):
        out.append(m(Z[i:i + bs])[0][0].cpu().numpy())          # head 0 = class head
    return np.concatenate(out)


def ncm_predict(feat_fit, y_fit, feat_q):
    P, L = [], []
    for c in np.unique(y_fit):
        P.append(feat_fit[y_fit == c].mean(0)); L.append(c)
    P = np.stack(P); L = np.array(L)
    d = ((feat_q[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    return L[d.argmin(1)]


def selectivity(zt_fit, zt_q, ref_fit, ref_q, ref_to_top):
    pred = ncm_predict(zt_fit, ref_fit, zt_q); err = pred != ref_q
    if err.sum() == 0: return 0.0
    same = (ref_to_top[pred[err]] == ref_to_top[ref_q[err]]).mean()
    refs = np.unique(ref_fit); tops = ref_to_top[refs]; M = len(refs); chance = 0.0
    for c in np.unique(tops):
        m_c = (tops == c).sum(); chance += (m_c / M) * ((m_c - 1) / max(M - 1, 1))
    return float(same - chance)


def boot_diff(a, b, B=2000, seed=7):
    a = np.asarray(a); b = np.asarray(b); pt = float(a.mean() - b.mean()); r = np.random.default_rng(seed); n = len(a)
    bs = [a[r.integers(0, n, n)].mean() - b[r.integers(0, n, n)].mean() for _ in range(B)]
    return pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_pos(a, B=2000, seed=7):
    a = np.asarray(a); pt = float(a.mean()); r = np.random.default_rng(seed); n = len(a)
    bs = [a[r.integers(0, n, n)].mean() for _ in range(B)]
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
        family = order * 3 + np.random.RandomState(3).randint(0, 3, n)
        z[:, :6] += np.eye(6)[cls] * 2.0
        labs = {"class": cls, "order": order, "family": family}
    else:
        d = np.load(args.cache, allow_pickle=True); z = d["z"].astype(np.float32)
        labs = {k: d[k].astype(np.int64) for k in ("class", "order", "family")}
    for k in list(labs):
        _, labs[k] = np.unique(labs[k], return_inverse=True)
    ref_to_top = np.zeros(int(labs[REF].max()) + 1, dtype=np.int64)
    for o in np.unique(labs[REF]):
        ref_to_top[o] = labs[TOP][labs[REF] == o][0]

    rng = np.random.default_rng(0); n = len(z); fit = rng.random(n) < 0.7; q = ~fit
    print(f"N={n} class={labs['class'].max()+1} order={labs['order'].max()+1} family={labs['family'].max()+1} fit={fit.sum()} q={q.sum()}", flush=True)

    store = {k: [] for k in ARMS}
    for seed in args.seeds:
        for k, levels in ARMS.items():
            ncls = [int(labs[l].max()) + 1 for l in levels]
            ys = [labs[l][fit] for l in levels]
            m = train(z[fit], ys, ncls, device, seed)
            ztf = class_repr(m, z[fit], device); ztq = class_repr(m, z[q], device)
            s = selectivity(ztf, ztq, labs[REF][fit], labs[REF][q], ref_to_top)
            store[k].append(s)
            print(f"[seed {seed}] k={k} conjoin({'+'.join(levels)}) class-selectivity={s:+.4f}", flush=True)

    print("\n=== Axis-G CONJUNCTION: class-tier selectivity vs # conjoined objectives ===", flush=True)
    res = {}
    for k in ARMS:
        pt, lo, hi = boot_pos(store[k]); res[f"k{k}"] = {"sel": pt, "ci": [lo, hi], "raw": store[k]}
        print(f"  k={k}: selectivity={pt:+.4f} [{lo:+.4f},{hi:+.4f}]")
    d31, l31, h31 = boot_diff(store[3], store[1]); d32, l32, h32 = boot_diff(store[3], store[2])
    g0 = res["k1"]["ci"][0] > 0; g1 = d31 >= 0.02 and l31 > 0; g2 = l32 > 0
    res["joint_read"] = {"k1>0": bool(g0), "k3-k1": [d31, l31, h31], "k3-k2": [d32, l32, h32],
                         "rises_with_conjunction": bool(g1)}
    verdict = "PULL_LEVER" if g1 else "ARCHITECT"
    res["verdict"] = verdict
    print(f"\n  k1>0={g0} | k3-k1={d31:+.4f}[{l31:+.4f},{h31:+.4f}] | k3-k2={d32:+.4f}[{l32:+.4f},{h32:+.4f}]")
    print(f"  rises-with-conjunction={g1} -> VERDICT: {verdict}")
    json.dump(res, open(os.path.join(args.out_dir, "granularity_axisG_parallel.json"), "w"), indent=2)
    print("DONE axisG-conjunction", flush=True)


if __name__ == "__main__":
    main()
