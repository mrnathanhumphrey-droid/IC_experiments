"""
probe_stream_compose.py -- DIRECTION C: does the pairwise frozen-readout forgetting law COMPOSE over a task STREAM?
Substrate = the clean SS2 regime: cached CIFAR-100 ResNet18 frozen features (feat_C50_P160_R160.npz), capability = class-vs-neg
LINEAR readout, continual = train head on victim A, continue-train same head sequentially on B1..Bk, measure A's surviving AUC.
CPU-light pure-numpy logistic (no GPU, no model). ASCII-only prints.

GATE   : reproduce pairwise forget ~ -cos(d_A,d_B) (rho ~ -0.7) + PARTIAL forgetting (headroom) on this bank.
C-Q1   : COMPOSITION -- does stream forget follow from pairwise cosines? additive Sum(-cos) vs mean(-cos) vs last(-cos) vs recency-weighted.
C-Q2   : MARKOVIAN vs PATH-DEPENDENT -- fixed SET, many ORDERINGS: is surviving-A set-governed (order-var << set-var) or recency-driven?
"""
from __future__ import annotations
import os, json, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BANK = HERE / "out" / "feat_C50_P160_R160.npz"
OUT = HERE / "out" / "stream_compose.json"
SEED = int(os.environ.get("SC_SEED", "0"))
STEPS_A = int(os.environ.get("SC_STEPS_A", "300"))     # fit victim head
PSTEP = int(os.environ.get("SC_PSTEP", "40"))          # continue-train steps per perturber (tune for PARTIAL forget)
LR = float(os.environ.get("SC_LR", "0.5"))
K = int(os.environ.get("SC_K", "4"))                   # stream length
NVIC = int(os.environ.get("SC_NVIC", "16"))            # victims
NSET = int(os.environ.get("SC_NSET", "12"))            # perturber-sets per victim
NORD = int(os.environ.get("SC_NORD", "6"))             # orderings per set
GAMMA = float(os.environ.get("SC_GAMMA", "0.5"))       # recency decay: later perturbers weighted more
rng = np.random.RandomState(SEED)


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int)
    p = s[y == 1]; n = s[y == 0]
    if len(p) == 0 or len(n) == 0: return float("nan")
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s)+1)
    return float((r[y == 1].sum() - len(p)*(len(p)+1)/2) / (len(p)*len(n)))


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean(); d = np.sqrt((rx**2).sum()*(ry**2).sum())
    return float((rx*ry).sum()/d) if d > 0 else float("nan")


def sig(z): return 1.0/(1.0+np.exp(-np.clip(z, -30, 30)))


def train_step(w, b, Xp, Xn, steps, lr):
    """continue-train binary logistic (pos=Xp vs neg=Xn) `steps` GD steps from (w,b). returns new (w,b)."""
    X = np.concatenate([Xp, Xn]); y = np.concatenate([np.ones(len(Xp)), np.zeros(len(Xn))])
    m = len(y)
    for _ in range(steps):
        g = sig(X @ w + b) - y
        w = w - lr * (X.T @ g) / m
        b = b - lr * g.mean()
    return w, b


def analyze(X, y, Xneg, name="cifar"):
    t0 = time.time()
    X = np.asarray(X, np.float64); Xneg = np.asarray(Xneg, np.float64); y = np.asarray(y)
    # standardize features (helps logistic conditioning)
    mu = X.mean(0); sd = X.std(0) + 1e-6
    X = (X - mu)/sd; Xneg = (Xneg - mu)/sd
    caps = sorted(np.unique(y).tolist()); C = len(caps)
    print(f"[SC:{name}] bank C={C} X={X.shape} neg={Xneg.shape} | STEPS_A={STEPS_A} PSTEP={PSTEP} LR={LR} K={K}", flush=True)

    # per-cap pos split (train/test) + discriminative direction + cosine matrix
    pos = {}; d_dir = {}
    negtr, negte = Xneg[:350], Xneg[350:]
    for c in caps:
        Xc = X[y == c]; ntr = int(0.6*len(Xc)); pos[c] = (Xc[:ntr], Xc[ntr:])
        d_dir[c] = X[y == c].mean(0) - Xneg.mean(0); d_dir[c] /= (np.linalg.norm(d_dir[c])+1e-9)
    cos = {(i, j): float(d_dir[i] @ d_dir[j]) for i in caps for j in caps if i != j}

    def victim_head(a):
        w = np.zeros(X.shape[1]); b = 0.0
        w, b = train_step(w, b, pos[a][0], negtr, STEPS_A, LR)
        return w, b

    def aucA(a, w, b):
        Xte = np.concatenate([pos[a][1], negte]); yte = np.concatenate([np.ones(len(pos[a][1])), np.zeros(len(negte))])
        return auc(Xte @ w + b, yte)

    victims = list(rng.choice(caps, min(NVIC, C), replace=False))
    others = {a: [c for c in caps if c != a] for a in victims}

    # ---- GATE: pairwise forget vs -cos + partial-forgetting headroom ----
    pf_cos = []; pf_forget = []
    for a in victims:
        w0, b0 = victim_head(a); a0 = aucA(a, w0, b0)
        for bcap in rng.choice(others[a], 12, replace=False):
            w1, b1 = train_step(w0.copy(), b0, pos[bcap][0], negtr, PSTEP, LR)
            pf_cos.append(cos[(a, bcap)]); pf_forget.append(a0 - aucA(a, w1, b1))
    pf_cos = np.array(pf_cos); pf_forget = np.array(pf_forget)
    rho_pair = spearman(pf_cos, -np.ones_like(pf_cos)*0 + pf_forget)  # rho(cos, forget)
    rho_pair = spearman(pf_cos, pf_forget)
    med_f = float(np.median(pf_forget)); partial = bool(0.03 <= med_f <= 0.45)
    print(f"  [GATE] pairwise rho(cos,forget)={rho_pair:+.3f} | median forget={med_f:.3f} partial={partial} "
          f"| forget range[{pf_forget.min():.3f},{pf_forget.max():.3f}] ({time.time()-t0:.0f}s)", flush=True)

    # ---- C-Q1 + C-Q2: streams ----
    obs = []; add = []; meanc = []; lastc = []; recwt = []  # per (victim,set,ordering)
    setvar = []; ordvar = []; last_within = []             # markovian bookkeeping
    for a in victims:
        w0, b0 = victim_head(a); a0 = aucA(a, w0, b0)
        set_means = []
        for _ in range(NSET):
            S = list(rng.choice(others[a], K, replace=False))
            coss = np.array([cos[(a, s)] for s in S])
            ord_forgets = []; ord_lastcos = []
            for _o in range(NORD):
                order = list(rng.permutation(S))
                w, b = w0.copy(), b0
                for s in order:
                    w, b = train_step(w, b, pos[s][0], negtr, PSTEP, LR)
                f = a0 - aucA(a, w, b)
                ord_forgets.append(f); ord_lastcos.append(cos[(a, order[-1])])
                obs.append(f)
                add.append(float((-coss).sum()))                 # additive Sum(-cos) (order-blind)
                meanc.append(float((-coss).mean()))               # mean(-cos) (order-blind)
                lastc.append(-cos[(a, order[-1])])                # last(-cos) recency
                wts = np.array([GAMMA**(K-1-i) for i in range(K)])  # later steps heavier
                recwt.append(float(np.sum(wts * np.array([-cos[(a, s)] for s in order]))))  # recency-weighted
            ord_forgets = np.array(ord_forgets)
            set_means.append(ord_forgets.mean())
            ordvar.append(float(ord_forgets.var()))               # variance across orderings of SAME set
            last_within.append(spearman(ord_lastcos, ord_forgets))  # does last-perturber cos explain within-set order variance?
        setvar.append(float(np.var(set_means)))                   # variance across sets (same victim)

    obs = np.array(obs); add = np.array(add); meanc = np.array(meanc); lastc = np.array(lastc); recwt = np.array(recwt)
    r_add = spearman(add, obs); r_mean = spearman(meanc, obs); r_last = spearman(lastc, obs); r_rec = spearman(recwt, obs)
    mean_ordvar = float(np.mean(ordvar)); mean_setvar = float(np.mean(setvar))
    order_frac = mean_ordvar / (mean_ordvar + mean_setvar + 1e-12)
    lw = np.array([x for x in last_within if not np.isnan(x)]); mean_last_within = float(lw.mean()) if len(lw) else float("nan")

    print(f"  [stream forget] median={np.median(obs):.3f} range[{obs.min():.3f},{obs.max():.3f}]", flush=True)
    print(f"  [C-Q1 COMPOSE] rho(pred,forget): additive={r_add:+.3f}  mean={r_mean:+.3f}  last/recency={r_last:+.3f}  recency-weighted(g={GAMMA})={r_rec:+.3f}", flush=True)
    print(f"  [C-Q2 MARKOV ] order-var={mean_ordvar:.5f} set-var={mean_setvar:.5f} order_frac={order_frac:.3f} "
          f"| within-set rho(last-cos,forget)={mean_last_within:+.3f} ({time.time()-t0:.0f}s)", flush=True)

    markovian = bool(order_frac < 0.25)
    winner = max([("additive", r_add), ("mean", r_mean), ("last/recency", r_last), ("recency-weighted", r_rec)], key=lambda kv: kv[1])
    print(f"\n=== DIRECTION C [{name}]: stream composition ===", flush=True)
    print(f"  GATE pairwise law reproduces      : rho={rho_pair:+.3f} partial={partial}", flush=True)
    print(f"  Q1 best composition predictor     : {winner[0]} (rho={winner[1]:+.3f})", flush=True)
    print(f"  Q2 forgetting is                  : {'MARKOVIAN (set-governed)' if markovian else 'PATH-DEPENDENT (order matters)'} "
          f"(order_frac={order_frac:.2f}; recency within-set rho={mean_last_within:+.3f})", flush=True)
    return {"name": name, "gate": {"rho_pair": round(rho_pair, 3), "median_forget": round(med_f, 3), "partial": partial},
            "Q1_compose": {"additive": round(r_add, 3), "mean": round(r_mean, 3), "last_recency": round(r_last, 3),
                           "recency_weighted": round(r_rec, 3), "gamma": GAMMA, "winner": winner[0]},
            "Q2_markov": {"order_var": round(mean_ordvar, 5), "set_var": round(mean_setvar, 5), "order_frac": round(order_frac, 3),
                          "within_set_last_rho": round(mean_last_within, 3), "markovian": markovian},
            "config": {"STEPS_A": STEPS_A, "PSTEP": PSTEP, "LR": LR, "K": K, "NVIC": NVIC, "NSET": NSET, "NORD": NORD, "seed": SEED}}


def main():
    d = np.load(BANK)
    res = analyze(d["X"], d["y"], d["Xneg"], "cifar")
    OUT.write_text(json.dumps(res, indent=1))
    print(f"[SC] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
