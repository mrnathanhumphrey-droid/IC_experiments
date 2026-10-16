"""
lowrank_adjudication_lambda.py — #4 LOW-RANK ADJUDICATION, powered vision run (Lambda).
Pre-reg: interference_law/PREREG_LOWRANK_ADJUDICATION_2026_07_27.md  (LOCK b5c3929a).
Local smoke (probe_lowrank_dirsmoke.py, N=12 CIFAR) PASSED: per-task eff-rank 2.37 << interaction 4.61 << pooled 13.9, mean|cos| 0.17.

Powered test of the DICHOTOMY + its mechanism:
  each TASK's forgetting operator is low-rank, but the pairwise interference MAP is high-rank
  because per-pair forgetting DIRECTIONS are DISPERSED across source->target pairs.

Design:
  * N=24 caps, CIFAR-100 + DTD, SEEDS seeds.
  * Build the full N=24 forgetting matrix F + per-pair feature-space forgetting directions d_ij ONCE per (dataset,seed);
    subsample to N in {6,12,24} for the SCALING test (no rebuild).
  * Metrics per (dataset,seed,N): interaction eff-rank (dcenter F), per-task operator eff-rank (median + per-cap),
    pooled direction eff-rank, mean|cos| across pair-directions.
  * HAAR null: random-direction ceiling for per-task / pooled eff-rank (concentration real vs random?).
  * PERM null: shuffle which direction belongs to which (i,j) -> destroys the per-task grouping; observed per-task must beat permuted.
  * SCALING prediction (the mechanism's signature): interaction eff-rank GROWS with N while per-task stays FLAT.
  * Aggregate across seeds: mean + bootstrap CI (resample seeds); honest that seeds are few.

Reuses LOCKED probe_dirlaw_arch (A.build/feats/full_ft/train_head1/head_auc) UNCHANGED. torchvision loaders (CIFAR100+DTD) for Lambda robustness.
SMOKE=1 -> N=6, 1 seed, cifar, FT_STEPS=150 (Lambda plumbing de-risk).
"""
from __future__ import annotations
import os, sys, json, time, copy
from pathlib import Path
import numpy as np, torch
import torchvision.datasets as tvd
import torchvision.transforms.functional as TF

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R          # A.head_auc uses R.auc
import probe_dirlaw_arch as A

DEV = A.DEV
SMOKE = int(os.environ.get("SMOKE", "0"))
CAPS_FULL = list(range(24)); NEG_BY = {"cifar": list(range(90, 100)), "dtd": list(range(30, 40))}
PER_CLASS = 100; SIZE = 128
FT_STEPS = int(os.environ.get("FT_STEPS", "150" if SMOKE else "300")); FT_LR = 5e-4
SEEDS = [0] if SMOKE else [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
DATASETS = ["cifar"] if SMOKE else os.environ.get("DATASETS", "cifar,dtd").split(",")
NCAPS_LIST = [6] if SMOKE else [int(x) for x in os.environ.get("NCAPS", "6,12,24").split(",")]
NCAPS_MAX = max(NCAPS_LIST)
HAAR = 50 if SMOKE else int(os.environ.get("HAAR", "300"))
PERM = 50 if SMOKE else int(os.environ.get("PERM", "2000"))
BOOT = 200 if SMOKE else int(os.environ.get("BOOT", "2000"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", str(HERE / "_data" / "tvdata")))
OUT = HERE / "out" / os.environ.get("OUTNAME", "lowrank_adjudication.json")
ROWS = HERE / "out" / os.environ.get("ROWSNAME", "lowrank_rows.jsonl")
LOCK = "b5c3929a"


# ---------- data ----------
# CIFAR: HF datasets (fast CDN; known-working locally + in plastic runs) w/ torchvision fallback.
# torchvision CIFAR100 download on Lambda is pathologically slow (smoke 2026-07-27: 66% of 169MB in 1800s -> watchdog).
def _cifar_torchvision(needed, rng):
    ds = tvd.CIFAR100(str(DATA_ROOT), train=True, download=True); labels = np.array(ds.targets); banks = {}
    for c in needed:
        idx = np.where(labels == c)[0]; rng.shuffle(idx); idx = idx[:PER_CLASS]
        banks[c] = torch.stack([TF.pil_to_tensor(ds[int(j)][0].convert("RGB")) for j in idx])
    return banks

def load_banks(name, needed):
    DATA_ROOT.mkdir(parents=True, exist_ok=True); rng = np.random.RandomState(0)
    if name == "cifar":
        try:
            b = R.load_banks()   # HF uoft-cs/cifar100, PER_CLASS=100, 32x32 uint8 (exact code that ran locally)
            miss = [c for c in needed if c not in b or len(b[c]) < 20]
            if miss: raise RuntimeError(f"HF banks short: {miss[:5]}")
            print("[data] HF cifar100 banks OK", flush=True); return {c: b[c] for c in needed}
        except Exception as e:
            print(f"[data] HF loader failed ({type(e).__name__}: {str(e)[:80]}) -> torchvision", flush=True)
            return _cifar_torchvision(needed, rng)
    # dtd: HF tanganke/dtd (fast CDN; train3760+test1880=120/class, labels 0-46 alphabetical == torchvision) w/ torchvision fallback
    try:
        from datasets import load_dataset
        banks = {c: [] for c in needed}
        for split in ("train", "test"):
            ds = load_dataset("tanganke/dtd", split=split); labs = np.array(ds["label"])
            for c in needed:
                if len(banks[c]) >= PER_CLASS: continue
                for j in np.where(labs == c)[0]:
                    if len(banks[c]) >= PER_CLASS: break
                    banks[c].append(TF.pil_to_tensor(TF.resize(ds[int(j)]["image"].convert("RGB"), [SIZE, SIZE])))
        out = {c: torch.stack(v) for c, v in banks.items() if len(v) >= 20}
        miss = [c for c in needed if c not in out]
        if miss: raise RuntimeError(f"HF dtd short: {miss[:5]}")
        print(f"[data] HF tanganke/dtd banks OK ({min(len(v) for v in out.values())}-{max(len(v) for v in out.values())}/class)", flush=True)
        return out
    except Exception as e:
        print(f"[data] HF dtd failed ({type(e).__name__}: {str(e)[:80]}) -> torchvision (slow)", flush=True)
    srcs = [tvd.DTD(str(DATA_ROOT), split=s, partition=1, download=True) for s in ("train", "val", "test")]
    banks = {c: [] for c in needed}
    for dsx in srcs:
        for i in range(len(dsx)):
            img, lab = dsx[i]; lab = int(lab)
            if lab in needed and len(banks[lab]) < PER_CLASS:
                banks[lab].append(TF.pil_to_tensor(TF.resize(img.convert("RGB"), [SIZE, SIZE])))
        if all(len(banks[c]) >= PER_CLASS for c in needed): break
    return {c: torch.stack(v) for c, v in banks.items() if len(v) >= 20}


# ---------- metrics ----------
def eff_rank(s):
    s = np.asarray(s, float); s2 = s**2
    return float((s2.sum()**2) / (np.square(s2).sum() + 1e-30))

def dcenter(M):
    n = M.shape[0]; D = M.astype(float).copy()
    rm = np.array([D[i, [j for j in range(n) if j != i]].mean() for i in range(n)])
    cm = np.array([D[[i for i in range(n) if i != j], j].mean() for j in range(n)])
    g = D[~np.eye(n, dtype=bool)].mean(); out = np.zeros_like(D)
    for i in range(n):
        for j in range(n):
            if i != j: out[i, j] = D[i, j] - rm[i] - cm[j] + g
    return out

def build_matrix_and_dirs(name, banks, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    neg = NEG_BY[name]; Xneg = torch.cat([banks[c] for c in neg])
    caps = [c for c in CAPS_FULL if c in banks and len(banks[c]) >= 20][:NCAPS_MAX]; n = len(caps)
    base = A.build("resnet18"); base.eval(); dim = A.feats(base, banks[caps[0]][:2]).shape[1]
    Fneg0 = A.feats(base, Xneg); heads, auc0, sep0 = {}, {}, {}
    for c in caps:
        FA = A.feats(base, banks[c]); heads[c] = A.train_head1(FA, Fneg0, dim); auc0[c] = A.head_auc(heads[c], FA, Fneg0)
        sep0[c] = FA.mean(0) - Fneg0.mean(0)
    F = np.zeros((n, n)); Dd = {}
    for cj, j in enumerate(caps):
        Xb = torch.cat([banks[j], Xneg]); yb = torch.cat([torch.ones(len(banks[j])), torch.zeros(len(Xneg))]).long()
        mm = copy.deepcopy(base); A.full_ft(mm, dim, Xb, yb, FT_STEPS, FT_LR); Fneg1 = A.feats(mm, Xneg)
        for ci, i in enumerate(caps):
            if i == j: continue
            FAi = A.feats(mm, banks[i]); F[ci, cj] = auc0[i] - A.head_auc(heads[i], FAi, Fneg1)
            Dd[(ci, cj)] = ((FAi.mean(0) - Fneg1.mean(0)) - sep0[i]).astype(np.float64)
        del mm; torch.cuda.empty_cache()
        print(f"    [{name} s{seed}] perturb {j} ({cj+1}/{n}) colFgt={np.median([F[ci,cj] for ci in range(n) if caps[ci]!=j]):+.3f}", flush=True)
    del base; torch.cuda.empty_cache()
    return F, Dd, caps

def metrics_at_N(F, Dd, k, haar=0, perm=0):
    """F,Dd built at NCAPS_MAX; restrict to first k caps."""
    sub = list(range(k)); Fk = F[np.ix_(sub, sub)]
    DC = dcenter(Fk); s_int = np.linalg.svd(DC, compute_uv=False); inter_er = eff_rank(s_int)
    vf = (s_int**2 / (s_int**2).sum())
    # per-task
    pt = []
    for ci in sub:
        M = np.stack([Dd[(ci, cj)] for cj in sub if cj != ci]); pt.append(eff_rank(np.linalg.svd(M, compute_uv=False)))
    pt = np.array(pt)
    # pooled
    alld = np.stack([Dd[(ci, cj)] for ci in sub for cj in sub if ci != cj]); d = alld.shape[1]
    nrm = alld / (np.linalg.norm(alld, axis=1, keepdims=True) + 1e-30)
    pool_er = eff_rank(np.linalg.svd(nrm, compute_uv=False))
    G = nrm @ nrm.T; iu = np.triu_indices(len(nrm), 1); mean_abs_cos = float(np.abs(G[iu]).mean())
    out = {"N": k, "median_forget": float(np.median(Fk[~np.eye(k, dtype=bool)])),
           "interaction_eff_rank": round(inter_er, 3), "interaction_var_top6": [round(float(x), 4) for x in vf[:6]],
           "per_task_eff_rank_median": round(float(np.median(pt)), 3),
           "per_task_eff_rank_iqr": [round(float(np.percentile(pt, 25)), 3), round(float(np.percentile(pt, 75)), 3)],
           "pooled_direction_eff_rank": round(pool_er, 3), "mean_abs_cos": round(mean_abs_cos, 4)}
    # Haar null (random directions, matched count/dim): ceiling for per-task & pooled eff-rank
    if haar:
        rng = np.random.RandomState(1234 + k); pt_h, pool_h = [], []
        for _ in range(haar):
            g = rng.randn(k - 1, d); pt_h.append(eff_rank(np.linalg.svd(g, compute_uv=False)))
            gp = rng.randn(k * (k - 1), d); gp /= (np.linalg.norm(gp, axis=1, keepdims=True) + 1e-30)
            pool_h.append(eff_rank(np.linalg.svd(gp, compute_uv=False)))
        out["haar_per_task_eff_rank"] = round(float(np.mean(pt_h)), 3)
        out["haar_pooled_eff_rank"] = round(float(np.mean(pool_h)), 3)
    # Perm null: shuffle direction<->pair assignment, recompute per-task median; observed must be BELOW permuted (grouping is real)
    if perm:
        rng = np.random.RandomState(777 + k); keys = [(ci, cj) for ci in sub for cj in sub if ci != cj]
        vecs = np.stack([Dd[key] for key in keys]); permvals = []
        for _ in range(perm):
            pv = vecs[rng.permutation(len(vecs))]; look = {keys[t]: pv[t] for t in range(len(keys))}
            pm = [eff_rank(np.linalg.svd(np.stack([look[(ci, cj)] for cj in sub if cj != ci]), compute_uv=False)) for ci in sub]
            permvals.append(np.median(pm))
        permvals = np.array(permvals)
        out["perm_per_task_median_mean"] = round(float(permvals.mean()), 3)
        out["perm_p_pt_below"] = round(float((permvals <= np.median(pt)).mean()), 4)  # frac permuted <= observed (want ~0)
    return out


def boot_ci(vals, reps, seed=0):
    v = np.asarray(vals, float); rng = np.random.RandomState(seed)
    if len(v) < 2: return [round(float(v.mean()), 3), round(float(v.mean()), 3)]
    bs = [v[rng.randint(0, len(v), len(v))].mean() for _ in range(reps)]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def main():
    t0 = time.time()
    print(f"[ADJ] #4 powered | SMOKE={SMOKE} datasets={DATASETS} seeds={SEEDS} Ncaps={NCAPS_LIST} ft=({FT_STEPS},{FT_LR}) dev={DEV}", flush=True)
    ROWS.write_text(""); percell = {}
    for name in DATASETS:
        needed = set(CAPS_FULL) | set(NEG_BY[name]); banks = load_banks(name, needed)
        print(f"[ADJ] {name}: banks loaded ({len([c for c in CAPS_FULL if c in banks])} caps)", flush=True)
        for seed in SEEDS:
            try:
                F, Dd, caps = build_matrix_and_dirs(name, banks, seed)
                for k in NCAPS_LIST:
                    m = metrics_at_N(F, Dd, k, haar=HAAR if k == NCAPS_MAX else 0, perm=PERM if k == NCAPS_MAX else 0)
                    row = {"dataset": name, "seed": seed, **m}
                    percell.setdefault(name, {}).setdefault(k, []).append(m)
                    with open(ROWS, "a") as f: f.write(json.dumps(row) + "\n")
                    print(f"  [{name} s{seed} N={k}] interER={m['interaction_eff_rank']} perTask={m['per_task_eff_rank_median']} "
                          f"pooled={m['pooled_direction_eff_rank']} cos={m['mean_abs_cos']}"
                          + (f" | haarPT={m.get('haar_per_task_eff_rank')} permP={m.get('perm_p_pt_below')}" if k == NCAPS_MAX else ""), flush=True)
            except Exception as e:
                print(f"  [{name} s{seed}] ERROR {type(e).__name__}: {str(e)[:160]}", flush=True)
    # aggregate
    agg = {}
    for name, byk in percell.items():
        agg[name] = {}
        for k, rows in byk.items():
            g = lambda key: [r[key] for r in rows]
            agg[name][str(k)] = {
                "n_seeds": len(rows),
                "interaction_eff_rank": [round(float(np.mean(g("interaction_eff_rank"))), 3), boot_ci(g("interaction_eff_rank"), BOOT)],
                "per_task_eff_rank_median": [round(float(np.mean(g("per_task_eff_rank_median"))), 3), boot_ci(g("per_task_eff_rank_median"), BOOT)],
                "pooled_direction_eff_rank": [round(float(np.mean(g("pooled_direction_eff_rank"))), 3), boot_ci(g("pooled_direction_eff_rank"), BOOT)],
                "mean_abs_cos": round(float(np.mean(g("mean_abs_cos"))), 4),
                "median_forget": round(float(np.mean(g("median_forget"))), 4),
            }
            if k == NCAPS_MAX and "haar_per_task_eff_rank" in rows[0]:
                agg[name][str(k)]["haar_per_task_eff_rank"] = round(float(np.mean(g("haar_per_task_eff_rank"))), 3)
                agg[name][str(k)]["haar_pooled_eff_rank"] = round(float(np.mean(g("haar_pooled_eff_rank"))), 3)
                agg[name][str(k)]["perm_p_pt_below"] = round(float(np.mean(g("perm_p_pt_below"))), 4)
        # scaling slopes (per-task should be ~flat; interaction should grow)
        ks = sorted(byk.keys())
        if len(ks) >= 2:
            it = [np.mean([r["interaction_eff_rank"] for r in byk[k]]) for k in ks]
            ptv = [np.mean([r["per_task_eff_rank_median"] for r in byk[k]]) for k in ks]
            agg[name]["scaling"] = {"N": ks, "interaction": [round(float(x), 3) for x in it],
                                    "per_task": [round(float(x), 3) for x in ptv],
                                    "interaction_slope": round(float(np.polyfit(ks, it, 1)[0]), 4),
                                    "per_task_slope": round(float(np.polyfit(ks, ptv, 1)[0]), 4)}
    result = {"prereg_lock": LOCK, "smoke": bool(SMOKE), "datasets": DATASETS, "seeds": SEEDS, "Ncaps": NCAPS_LIST,
              "ft": [FT_STEPS, FT_LR], "aggregate": agg, "wall_s": round(time.time() - t0)}
    OUT.write_text(json.dumps(result, indent=1))
    print("\n=== #4 POWERED AGGREGATE ===", flush=True)
    for name, byk in agg.items():
        sc = byk.get("scaling", {})
        print(f"[{name}] scaling N={sc.get('N')} interaction={sc.get('interaction')} (slope {sc.get('interaction_slope')}) "
              f"per_task={sc.get('per_task')} (slope {sc.get('per_task_slope')})", flush=True)
        top = byk.get(str(NCAPS_MAX), {})
        print(f"  N={NCAPS_MAX}: perTask={top.get('per_task_eff_rank_median')} vs haar={top.get('haar_per_task_eff_rank')} "
              f"| interaction={top.get('interaction_eff_rank')} pooled={top.get('pooled_direction_eff_rank')} "
              f"cos={top.get('mean_abs_cos')} permP={top.get('perm_p_pt_below')}", flush=True)
    print(f"\n[ADJ] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
