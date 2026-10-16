"""
probe_phase_continuum.py -- FACE-2: is the frozen<->plastic transition a DISCONTINUITY or a genuine phase transition with a
linear-response window? SS17 swept LoRA RANK (discrete, capacity-entangled, rank-1 floored) and found the cosine DIRECTION law
(frozen rho_cos=-0.74) already dead by rank-2 -> called it a "sharp boundary." But rank can't resolve BELOW rank-2, where the
transition actually lives. Here we FIX rank and use ||dtheta|| (travel-norm) as a CONTINUOUS order parameter, walking the REAL GD
trajectory via SNAPSHOTS at increasing step counts. This maps the sub-rank-2 region SS17 could not see.

H_linear (reframe): the cosine law is the LINEAR-RESPONSE limit of forgetting. At small ||dtheta|| ->  rho_cos revives toward -0.74;
    as ||dtheta|| grows, disruptiveness (2nd order) takes over -> rho_cos -> 0. Crossover at ||dtheta||_c. SS17's rank-2 (||dtheta||~34)
    was already in the nonlinear regime.
H_discontinuity (null): no revival -- rho_cos ~ 0 the instant forgetting is nonzero. SS17's discontinuity holds at fine resolution;
    the frozen cosine law is a measure-zero (exact-frozen) property.

cache->grid: TRAINING each (rank,seed,B) LoRA is the cost; snapshot re-probes are cheap forward passes. Multi-seed -> CIs on
rho_cos(||dtheta||) (fixes SS17's single-seed hole). Reuses probe_representational_law: build_model, features, probe_auc, ddir,
lora_params, prep, load_banks, spearman, A_CLASSES/NEG_CLASSES/B_POOL/BCLS. Vision (CIFAR-100/ResNet18 conv-LoRA) -- same substrate as SS17.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R

OUT = HERE / "out" / os.environ.get("OUTNAME", "phase_continuum.json"); OUT.parent.mkdir(parents=True, exist_ok=True)
DEV = R.DEV
SMOKE = os.environ.get("SMOKE", "0") == "1"
RANKS = [int(x) for x in os.environ.get("PC_RANKS", ("2" if SMOKE else "2,8")).split(",")]
SNAPS = sorted(set(int(x) for x in os.environ.get("PC_SNAPS", ("32,256,1000" if SMOKE else "2,4,8,16,32,64,128,256,1000")).split(",")))
N_B = int(os.environ.get("PC_NB", "2" if SMOKE else "6"))
SEEDS = [int(x) for x in os.environ.get("PC_SEEDS", ("0" if SMOKE else "0,1,2")).split(",")]
B_LR = float(os.environ.get("PC_BLR", "8e-3"))
BAND = (0.05, 0.40)


def load_banks_tv():
    # torchvision CIFAR-100 loader (bypasses HF datasets/PIL ExifTags trap; Face-1b lesson). Same output
    # format as R.load_banks: {class: uint8 tensor [n,3,32,32]}, prep() normalizes later. Fine labels 0-99 == HF fine_label.
    from torchvision.datasets import CIFAR100
    ds = CIFAR100(root="/home/ubuntu/cifar_data", train=True, download=True)
    data = ds.data  # [50000,32,32,3] uint8 HWC
    targets = np.array(ds.targets)
    rng = np.random.RandomState(0); banks = {}
    need = set(R.A_CLASSES) | set(R.NEG_CLASSES) | set(R.B_POOL)
    for c in need:
        idx = np.where(targets == c)[0]; rng.shuffle(idx); idx = idx[:R.PER_CLASS]
        banks[c] = torch.from_numpy(data[idx]).permute(0, 3, 1, 2).contiguous()  # [n,3,32,32] uint8
    return banks


def dtheta_norm(model):
    return float(np.sqrt(sum(float((p.detach()**2).sum()) for p in R.lora_params(model))))


@torch.no_grad()
def drift(F0, F1):
    return float(np.mean(np.linalg.norm(F1 - F0, axis=1)))


def main():
    t0 = time.time()
    print(f"[PC] SMOKE={SMOKE} ranks={RANKS} snaps={SNAPS} N_B={N_B} seeds={SEEDS} B_lr={B_LR} dev={DEV}", flush=True)
    banks = load_banks_tv(); print(f"[PC] loaded {len(banks)} banks (torchvision) ({time.time()-t0:.0f}s)", flush=True)
    Xneg_u8 = torch.cat([banks[c] for c in R.NEG_CLASSES])

    # base (frozen) reference -- LoRA init update = 0, so rank-independent; compute ONCE
    base = R.build_model(RANKS[0])
    Fneg0 = R.features(base, Xneg_u8)
    FA0 = {a: R.features(base, banks[a]) for a in R.A_CLASSES}
    accA0 = {a: R.probe_auc(FA0[a], Fneg0) for a in R.A_CLASSES}
    dA = {a: R.ddir(FA0[a], Fneg0) for a in R.A_CLASSES}
    print(f"[PC] base accA0 median={np.median(list(accA0.values())):.3f} ({time.time()-t0:.0f}s)", flush=True)

    # fixed B-tasks + their base directions (fair across ranks/seeds/snapshots)
    rng = np.random.RandomState(1)
    BTASKS = [list(rng.choice(R.B_POOL, R.BCLS, replace=False)) for _ in range(N_B)]
    dB = {bi: R.ddir(np.concatenate([R.features(base, banks[c]) for c in bt]), Fneg0) for bi, bt in enumerate(BTASKS)}

    # accumulators keyed (rank, snap): parallel arrays over seed x B x A
    acc = {(r, s): {"cos": [], "fg": [], "seed": [], "B": []} for r in RANKS for s in SNAPS}
    meta = {(r, s): {"dtheta": [], "drift": []} for r in RANKS for s in SNAPS}   # per (seed,B) scalars

    for r in RANKS:
        for seed in SEEDS:
            for bi, bt in enumerate(BTASKS):
                torch.manual_seed(seed * 1000 + bi); np.random.seed(seed * 1000 + bi)
                model = R.build_model(r)
                head = nn.Linear(512, len(bt)).to(DEV)
                params = R.lora_params(model) + list(head.parameters())
                opt = torch.optim.AdamW(params, lr=B_LR, weight_decay=1e-4)
                X = torch.cat([banks[c] for c in bt]); y = torch.cat([torch.full((len(banks[c]),), i) for i, c in enumerate(bt)])
                prev = 0
                for target in SNAPS:
                    model.train()
                    for st in range(prev, target):
                        idx = torch.randint(0, len(y), (128,)); feat = model(R.prep(X[idx]).to(DEV))
                        loss = Fun.cross_entropy(head(feat), y[idx].to(DEV))
                        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    prev = target
                    model.eval()
                    Fneg1 = R.features(model, Xneg_u8)
                    dth = dtheta_norm(model); dr = drift(Fneg0, Fneg1)
                    for a in R.A_CLASSES:
                        acc1 = R.probe_auc(R.features(model, banks[a]), Fneg1)
                        acc[(r, target)]["cos"].append(float(dA[a] @ dB[bi]))
                        acc[(r, target)]["fg"].append(float(accA0[a] - acc1))
                        acc[(r, target)]["seed"].append(seed); acc[(r, target)]["B"].append(bi)
                    meta[(r, target)]["dtheta"].append(dth); meta[(r, target)]["drift"].append(dr)
                del model, head, opt
                if DEV == "cuda": torch.cuda.empty_cache()
            print(f"[PC] rank={r} seed={seed} done ({time.time()-t0:.0f}s)", flush=True)

    # ---- aggregate: rho_cos(||dtheta||) with per-seed CIs ----
    curves = {r: [] for r in RANKS}
    for r in RANKS:
        for s in SNAPS:
            d = acc[(r, s)]; cos = np.array(d["cos"]); fg = np.array(d["fg"]); sd = np.array(d["seed"]); B = np.array(d["B"])
            rho_pool = R.spearman(cos, fg) if len(cos) > 5 else float("nan")
            # within-B partial, averaged per seed -> CI across seeds
            per_seed = []
            for se in SEEDS:
                wl = [R.spearman(cos[(sd == se) & (B == bi)], fg[(sd == se) & (B == bi)])
                      for bi in range(N_B) if ((sd == se) & (B == bi)).sum() > 5]
                if wl: per_seed.append(float(np.mean(wl)))
            within_mean = float(np.mean(per_seed)) if per_seed else float("nan")
            within_sd = float(np.std(per_seed)) if len(per_seed) > 1 else 0.0
            med_fg = float(np.median(fg)); inband = bool(BAND[0] <= med_fg <= BAND[1])
            row = {"rank": r, "snap_steps": s, "mean_dtheta": round(float(np.mean(meta[(r, s)]["dtheta"])), 3),
                   "mean_drift": round(float(np.mean(meta[(r, s)]["drift"])), 3),
                   "rho_cos_pool": round(rho_pool, 3), "rho_cos_within_B": round(within_mean, 3),
                   "within_B_sd_across_seeds": round(within_sd, 3), "median_forget": round(med_fg, 4),
                   "inband": inband, "n_seeds": len(per_seed)}
            curves[r].append(row)
            print(f"  [PC] r={r} steps={s:4d} ||dth||={row['mean_dtheta']:6.2f} med_fgt={med_fg:.4f} band={inband} "
                  f"rho_cos_within={within_mean:+.3f}+/-{within_sd:.3f} pool={rho_pool:+.3f}", flush=True)

    # ---- verdict: does rho_cos revive at small ||dtheta|| (H_linear) or stay dead (H_discontinuity)? ----
    def verdict_for(r):
        rows = [x for x in curves[r] if not np.isnan(x["rho_cos_within_B"])]
        if not rows: return "no measurable snapshots"
        rows_s = sorted(rows, key=lambda x: x["mean_dtheta"])
        small = rows_s[0]; large = rows_s[-1]
        # H_linear signature: rho_cos MORE negative at small ||dtheta|| than large, and small clearly negative
        revives = (small["rho_cos_within_B"] <= -0.30) and (small["rho_cos_within_B"] < large["rho_cos_within_B"] - 0.15)
        dead_everywhere = all(x["rho_cos_within_B"] > -0.20 for x in rows)
        if revives: return (f"H_LINEAR SUPPORTED -- rho_cos revives to {small['rho_cos_within_B']:+.3f} at ||dth||={small['mean_dtheta']:.2f} "
                            f"(vs {large['rho_cos_within_B']:+.3f} at ||dth||={large['mean_dtheta']:.2f}); linear-response window exists.")
        if dead_everywhere: return (f"H_DISCONTINUITY SUPPORTED -- rho_cos > -0.20 at every measurable ||dtheta|| "
                                    f"(smallest={small['rho_cos_within_B']:+.3f} @ ||dth||={small['mean_dtheta']:.2f}); dead as soon as motion is nonzero. SS17 holds at fine resolution.")
        return (f"AMBIGUOUS -- small ||dth|| rho_cos={small['rho_cos_within_B']:+.3f}, large={large['rho_cos_within_B']:+.3f}; "
                f"no clean revival, not uniformly dead.")

    verdicts = {r: verdict_for(r) for r in RANKS}
    res = {"config": {"ranks": RANKS, "snaps": SNAPS, "N_B": N_B, "seeds": SEEDS, "B_lr": B_LR, "smoke": SMOKE},
           "frozen_ref_rho": -0.74, "curves": curves, "verdicts": verdicts}
    OUT.write_text(json.dumps(res, indent=1))
    print("\n[PC] === rho_cos vs ||dtheta|| (per rank) ===", flush=True)
    for r in RANKS:
        print(f"  rank {r}: {verdicts[r]}", flush=True)
    print(f"[PC] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
