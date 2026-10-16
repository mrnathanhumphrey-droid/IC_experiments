"""
probe_directional_unlearn.py — Thread #3: DIRECTIONAL UNLEARNING. Erase SWAP detection by removing the head's response to the
part of swap's direction UNIQUE to swap (orthogonal to the other paradigms), sparing diffusion/celebdf/dfdc. vs NAIVE relabel-and-
finetune and a matched RANDOM control. Closed-form, $0 local CPU. Reuses probe_detector_seq machinery.

ISOLATION: frozen backbone (read-only), crops read-only, writes only ./out/. Never touches live prod / served detector.
"""
from __future__ import annotations
import sys, json, time, os
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_detector_seq as ds   # reuse load_backbone/extract/_glob_sub/_hash_split/auc/Head/train_head/discriminative_dir

SMOKE = "--smoke" in sys.argv
SUB = 60 if SMOKE else 220
OUT = HERE / "out" / (f"directional_unlearn_{os.environ.get('TARGET','swap')}{'_smoke' if SMOKE else ''}.json")
np.random.seed(0); torch.manual_seed(0)
PARAS = [p[0] for p in ds.PARADIGMS]          # swap, diffusion, celebdf, dfdc
TARGET = os.environ.get("TARGET", "swap")     # capability to erase


def scores(head, X): return ds.scores(head, X)


def main():
    t0 = time.time()
    ds.SUBSAMPLE = SUB
    bb = ds.load_backbone(); print("[UNLEARN] backbone loaded", flush=True)
    rall = ds._glob_sub(ds.SHARED_REAL_GLOB, SUB * 2, ds.SEED); r_tr, r_te = ds._hash_split(rall)
    Xr_tr, yr_tr = ds.extract(bb, r_tr, 0); Xr_te, yr_te = ds.extract(bb, r_te, 0)

    P = {}
    for name, d in ds.PARADIGMS:
        f_tr, f_te = ds._hash_split(ds._glob_sub(str(ds.CORPUS / d / "fake" / "*.png"), SUB, ds.SEED))
        Xf_tr, _ = ds.extract(bb, f_tr, 1); Xf_te, _ = ds.extract(bb, f_te, 1)
        P[name] = {"Xf_tr": Xf_tr, "Xte": np.concatenate([Xf_te, Xr_te]),
                   "yte": np.concatenate([np.ones(len(f_te)), np.zeros(len(r_te))]).astype(np.float32),
                   "dir": ds.discriminative_dir(Xf_tr, Xr_tr)}
        print(f"  {name:10} fake tr={len(f_tr)} te={len(f_te)}", flush=True)

    # universal FAKE-vs-REAL head (all paradigms pooled)
    Xall = np.concatenate([P[p]["Xf_tr"] for p in PARAS] + [Xr_tr])
    yall = np.concatenate([np.ones(sum(len(P[p]["Xf_tr"]) for p in PARAS)), np.zeros(len(Xr_tr))]).astype(np.float32)
    H = ds.train_head(Xall, yall, seed=0)
    def per_para(head): return {p: round(ds.auc(scores(head, P[p]["Xte"]), P[p]["yte"]), 4) for p in PARAS}
    base = per_para(H)
    print(f"\n[UNLEARN] base universal head AUC: {base}", flush=True)

    w = H.fc.weight.detach().numpy().reshape(-1).astype(np.float64)   # (128,)
    b = float(H.fc.bias.detach().numpy()[0])
    others = [p for p in PARAS if p != TARGET]

    # û = part of d_swap orthogonal to span{other paradigm dirs}
    U = np.stack([P[o]["dir"] for o in others], 1).astype(np.float64)   # (128, n_others)
    Q, _ = np.linalg.qr(U)                                              # orthonormal basis of others' span
    d_t = P[TARGET]["dir"].astype(np.float64)
    proj = Q @ (Q.T @ d_t); unique = d_t - proj; uhat = unique / (np.linalg.norm(unique) + 1e-12)
    removed_mag = float(w @ uhat)
    cos_unique_swap = float(uhat @ d_t)
    print(f"[UNLEARN] ||d_swap unique component||={np.linalg.norm(unique):.3f}  cos(uhat,d_swap)={cos_unique_swap:.3f}  "
          f"w·uhat(removed)={removed_mag:+.3f}", flush=True)

    def head_from_w(wv):
        h = ds.Head(128);
        with torch.no_grad():
            h.fc.weight.copy_(torch.from_numpy(wv.astype(np.float32)).view(1, -1)); h.fc.bias.copy_(torch.tensor([b], dtype=torch.float32))
        h.eval(); return h

    # 1) DIRECTIONAL: remove w's response to uhat
    w_dir = w - removed_mag * uhat
    res_dir = per_para(head_from_w(w_dir))

    # 2) NAIVE: relabel swap -> real, continue-train a few steps
    Xn = np.concatenate([P[TARGET]["Xf_tr"], Xr_tr, np.concatenate([P[o]["Xf_tr"] for o in others])])
    yn = np.concatenate([np.zeros(len(P[TARGET]["Xf_tr"])), np.zeros(len(Xr_tr)),
                         np.ones(sum(len(P[o]["Xf_tr"]) for o in others))]).astype(np.float32)
    Hn = ds.continue_train({k: v.clone() for k, v in H.state_dict().items()}, Xn, yn, U=None, epochs=120, seed=0)
    res_naive = per_para(Hn)

    # 3) RANDOM control: remove matched magnitude along a random unit direction (avg over a few)
    rnd_runs = []
    for s in range(5):
        rng = np.random.RandomState(100 + s); r = rng.randn(128); rhat = r / np.linalg.norm(r)
        w_rand = w - removed_mag * rhat
        rnd_runs.append(per_para(head_from_w(w_rand)))
    res_rand = {p: round(float(np.mean([rr[p] for rr in rnd_runs])), 4) for p in PARAS}

    def spec(res):
        swap_drop = base[TARGET] - res[TARGET]
        other_drop = np.mean([base[o] - res[o] for o in others])
        return swap_drop, float(other_drop), round(swap_drop / (abs(other_drop) + 1e-6), 2)

    print("\n=== per-paradigm AUC after unlearning SWAP ===", flush=True)
    print("  " + "method".ljust(12) + "".join(p[:8].rjust(10) for p in PARAS) + "   swapDrop  otherDrop  SPECIFICITY", flush=True)
    rows = {"base": base, "directional": res_dir, "naive": res_naive, "random": res_rand}
    summary = {}
    for name, res in rows.items():
        line = "  " + name.ljust(12) + "".join(f"{res[p]:10.3f}" for p in PARAS)
        if name != "base":
            sd, od, sp = spec(res); summary[name] = {"swap_drop": round(sd, 4), "other_drop": round(od, 4), "specificity": sp}
            line += f"   {sd:+8.3f}  {od:+8.3f}   {sp:8.2f}"
        print(line, flush=True)

    print(f"\n  DIRECTIONAL: swap {base[TARGET]:.3f}->{res_dir[TARGET]:.3f}, others avg drop {summary['directional']['other_drop']:+.3f}, "
          f"specificity {summary['directional']['specificity']}", flush=True)
    print(f"  vs NAIVE specificity {summary['naive']['specificity']}, RANDOM specificity {summary['random']['specificity']}", flush=True)

    OUT.write_text(json.dumps({"config": {"subsample": SUB, "target": TARGET, "others": others, "smoke": SMOKE,
                                          "prereg_lock": "0d4c05d3", "cos_uhat_dswap": cos_unique_swap, "removed_mag": removed_mag},
                               "base": base, "directional": res_dir, "naive": res_naive, "random": res_rand,
                               "summary": summary}, indent=1))
    print(f"\nwrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
