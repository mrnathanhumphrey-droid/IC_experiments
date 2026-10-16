"""
probe_lowrank_dirsmoke.py — #4 LOW-RANK ADJUDICATION, reduced direction-capturing SMOKE.
Pre-reg: interference_law/PREREG_LOWRANK_ADJUDICATION_2026_07_27.md  (LOCK b5c3929a).

Question the SMOKE answers (CIFAR only, N=12 caps, ~12 full-FTs):
  Is each TASK's forgetting operator low-rank while the pairwise-interaction MAP is high-rank
  because per-pair forgetting DIRECTIONS are dispersed?

Reuses the LOCKED probe_lowrank machinery (A.build/feats/full_ft/train_head1/head_auc, R.load_banks)
UNCHANGED; only ADDS capture of the per-pair feature-space forgetting direction
  d_ij = [ mean feats(mm_j, banks[i]) - mean feats(mm_j, Xneg) ]  -  [ (pos-neg) sep at base ]
i.e. the shift of task-i's (pos - neg) centroid separation caused by training on j (512-d).
This is the object head_i reads out; its projection onto head_i drives the AUC drop F[i,j].

Falsifier readings (from the locked pre-reg):
  * KILL (no dichotomy): per-task operator eff_rank ALSO high (>= 6, ~ interaction eff_rank).
  * KILL (mechanism wrong): per-pair directions ALIGNED (low dispersion) yet interaction high-rank.
$0 local (RTX 5070 Ti). NOT the full test — CIFAR only, N=12, 1 seed. Smoke to see if the mechanism has legs.
"""
from __future__ import annotations
import os, sys, json, time, copy
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R
import probe_dirlaw_arch as A

DEV = A.DEV
CAPS = list(range(12))            # reduced from 24
NEG  = list(range(90, 100))       # cifar negatives (same as locked probe_lowrank)
FT_STEPS = 300; FT_LR = 5e-4; PER_CLASS = 100; SEED = 0
OUT = HERE / "out" / "lowrank_dirsmoke.json"
torch.manual_seed(SEED); np.random.seed(SEED)


def load_cifar_banks(needed):
    """Try the locked HF loader; fall back to torchvision CIFAR100 (memory: HF/PIL bug risk)."""
    try:
        b = R.load_banks()
        miss = [c for c in needed if c not in b or len(b[c]) < 20]
        if miss: raise RuntimeError(f"HF banks missing/short: {miss[:5]}")
        print("[data] HF cifar100 banks OK", flush=True)
        return {c: b[c] for c in needed}
    except Exception as e:
        print(f"[data] HF loader failed ({type(e).__name__}: {str(e)[:80]}) -> torchvision CIFAR100", flush=True)
        import torchvision.datasets as tvd
        import torchvision.transforms.functional as TF
        root = str(HERE / "_data" / "tvdata")
        ds = tvd.CIFAR100(root, train=True, download=True)
        fine = np.array(ds.targets); rng = np.random.RandomState(0)
        banks = {}
        for c in needed:
            idx = np.where(fine == c)[0]; rng.shuffle(idx); idx = idx[:PER_CLASS]
            banks[c] = torch.stack([TF.pil_to_tensor(ds[int(j)][0].convert("RGB")) for j in idx])
        print("[data] torchvision cifar100 banks OK", flush=True)
        return banks


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


def main():
    t0 = time.time()
    print(f"[DIRSMOKE] #4 low-rank adjudication smoke | CAPS={CAPS} NEG={NEG} ft=({FT_STEPS},{FT_LR}) dev={DEV}", flush=True)
    needed = set(CAPS) | set(NEG)
    banks = load_cifar_banks(needed)
    caps = [c for c in CAPS if c in banks and len(banks[c]) >= 20]; n = len(caps)
    Xneg = torch.cat([banks[c] for c in NEG])
    base = A.build("resnet18"); base.eval()
    dim = A.feats(base, banks[caps[0]][:2]).shape[1]
    Fneg0 = A.feats(base, Xneg)
    heads, auc0, sep0 = {}, {}, {}
    for c in caps:
        FA = A.feats(base, banks[c]); heads[c] = A.train_head1(FA, Fneg0, dim); auc0[c] = A.head_auc(heads[c], FA, Fneg0)
        sep0[c] = FA.mean(0) - Fneg0.mean(0)            # base (pos - neg) separation, 512-d
    print(f"[DIRSMOKE] base heads: mean auc0={np.mean(list(auc0.values())):.3f} dim={dim} n={n}", flush=True)

    F = np.zeros((n, n))
    D = {}                                              # D[(i,j)] = forgetting direction (512-d), i != j
    for cj, j in enumerate(caps):
        Xb = torch.cat([banks[j], Xneg]); yb = torch.cat([torch.ones(len(banks[j])), torch.zeros(len(Xneg))]).long()
        mm = copy.deepcopy(base); A.full_ft(mm, dim, Xb, yb, FT_STEPS, FT_LR)
        Fneg1 = A.feats(mm, Xneg)
        for ci, i in enumerate(caps):
            if i == j: continue
            FAi = A.feats(mm, banks[i])
            F[ci, cj] = auc0[i] - A.head_auc(heads[i], FAi, Fneg1)
            sep1 = FAi.mean(0) - Fneg1.mean(0)          # perturbed (pos - neg) separation
            D[(ci, cj)] = (sep1 - sep0[i]).astype(np.float64)   # shift of task-i separation caused by j
        del mm; torch.cuda.empty_cache()
        med = np.median([F[ci, cj] for ci in range(n) if caps[ci] != j])
        print(f"    perturb {j}: colFgt={med:+.3f}  ({cj+1}/{n})", flush=True)
    del base; torch.cuda.empty_cache()

    # --- interaction eff-rank (reproduce probe_lowrank's object) ---
    DC = dcenter(F); s_int = np.linalg.svd(DC, compute_uv=False)
    vf = (s_int**2 / (s_int**2).sum()); inter_er = eff_rank(s_int)

    # --- per-task forgetting operator rank: stack {d_ij : j != i} for fixed target i ---
    per_task_er = {}
    for ci, i in enumerate(caps):
        M = np.stack([D[(ci, cj)] for cj in range(n) if cj != ci])   # (n-1, 512)
        s = np.linalg.svd(M, compute_uv=False); per_task_er[i] = eff_rank(s)
    pt_vals = np.array(list(per_task_er.values()))

    # --- pooled direction dispersion across ALL pairs ---
    alld = np.stack([D[(ci, cj)] for ci in range(n) for cj in range(n) if ci != cj])   # (n*(n-1), 512)
    nrm = alld / (np.linalg.norm(alld, axis=1, keepdims=True) + 1e-30)
    s_pool = np.linalg.svd(nrm, compute_uv=False); pool_er = eff_rank(s_pool)          # effective # distinct directions
    G = nrm @ nrm.T; iu = np.triu_indices(len(nrm), 1); mean_abs_cos = float(np.abs(G[iu]).mean())

    med_fgt = float(np.median(F[~np.eye(n, dtype=bool)]))
    print("\n=== #4 SMOKE RESULT (CIFAR, N={}) ===".format(n), flush=True)
    print(f"  median off-diag forgetting      = {med_fgt:+.3f}", flush=True)
    print(f"  INTERACTION eff-rank (dcenter F)= {inter_er:.2f}   top6 var% = {[round(float(x)*100,1) for x in vf[:6]]}", flush=True)
    print(f"  PER-TASK operator eff-rank      = median {np.median(pt_vals):.2f}  range [{pt_vals.min():.2f}, {pt_vals.max():.2f}]", flush=True)
    print(f"  POOLED direction eff-rank       = {pool_er:.2f}   (effective # distinct forgetting directions, max={min(len(nrm),alld.shape[1])})", flush=True)
    print(f"  mean |cos| across pair-dirs     = {mean_abs_cos:.3f}   (low = dispersed)", flush=True)
    print(f"\n  READ: dichotomy holds if PER-TASK eff-rank LOW while INTERACTION + POOLED eff-rank HIGH,", flush=True)
    print(f"        and mean|cos| LOW (dispersed). KILL if per-task eff-rank ~ interaction, or mean|cos| high.", flush=True)

    res = {
        "smoke": True, "dataset": "cifar", "n": n, "caps": [int(c) for c in caps],
        "ft": [FT_STEPS, FT_LR], "seed": SEED, "prereg_lock": "b5c3929a",
        "median_forget": round(med_fgt, 4),
        "interaction_eff_rank": round(inter_er, 3),
        "interaction_var_top6": [round(float(x), 4) for x in vf[:6]],
        "per_task_eff_rank_median": round(float(np.median(pt_vals)), 3),
        "per_task_eff_rank_min": round(float(pt_vals.min()), 3),
        "per_task_eff_rank_max": round(float(pt_vals.max()), 3),
        "per_task_eff_rank_by_cap": {int(c): round(float(v), 3) for c, v in per_task_er.items()},
        "pooled_direction_eff_rank": round(float(pool_er), 3),
        "pooled_dim": int(min(len(nrm), alld.shape[1])),
        "mean_abs_cos_pairdirs": round(mean_abs_cos, 4),
    }
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\n[DIRSMOKE] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
