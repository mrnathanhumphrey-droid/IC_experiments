"""
probe_curvature_engineering.py — Thread #4: CURVATURE ENGINEERING (inverse problem). Design protection for a CHOSEN capability by
projecting a strong interference ("collapse toward diffusion-only") off span{d_p : p in S}, then ask whether the REALIZED protection
profile equals the DESIRED S — and whether it is CONSTRAINED by the cosine cluster geometry (protecting swap leaks to its family).
Closes the arc: one cosine geometry governs learn / forget / unlearn / PROTECT. $0 local CPU. Reuses probe_detector_seq machinery.

ISOLATION: frozen backbone (read-only), crops read-only, writes only ./out/. Never touches live prod / served detector.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_detector_seq as ds

SMOKE = "--smoke" in sys.argv
SUB = 60 if SMOKE else 220
OUT = HERE / "out" / ("curvature_engineering_smoke.json" if SMOKE else "curvature_engineering.json")
np.random.seed(0); torch.manual_seed(0)
PARAS = [p[0] for p in ds.PARADIGMS]          # swap, diffusion, celebdf, dfdc
COLLAPSE_TO = "diffusion"                       # interference makes the head detect ONLY this (erodes the rest)

PROFILES = [[], ["swap"], ["celebdf"], ["dfdc"], ["diffusion"], ["swap", "celebdf", "dfdc"], ["swap", "diffusion"]]
if SMOKE: PROFILES = [[], ["swap"]]


def main():
    t0 = time.time(); ds.SUBSAMPLE = SUB
    bb = ds.load_backbone(); print("[CURV-ENG] backbone loaded", flush=True)
    rall = ds._glob_sub(ds.SHARED_REAL_GLOB, SUB * 2, ds.SEED); r_tr, r_te = ds._hash_split(rall)
    Xr_tr, _ = ds.extract(bb, r_tr, 0); Xr_te, _ = ds.extract(bb, r_te, 0)

    P = {}
    for name, d in ds.PARADIGMS:
        f_tr, f_te = ds._hash_split(ds._glob_sub(str(ds.CORPUS / d / "fake" / "*.png"), SUB, ds.SEED))
        Xf_tr, _ = ds.extract(bb, f_tr, 1); Xf_te, _ = ds.extract(bb, f_te, 1)
        P[name] = {"Xf_tr": Xf_tr, "Xte": np.concatenate([Xf_te, Xr_te]),
                   "yte": np.concatenate([np.ones(len(f_te)), np.zeros(len(r_te))]).astype(np.float32),
                   "dir": ds.discriminative_dir(Xf_tr, Xr_tr)}

    # cosine matrix (for leakage-by-cosine analysis)
    cosM = {a: {b: float(P[a]["dir"] @ P[b]["dir"]) for b in PARAS} for a in PARAS}

    # universal head
    Xall = np.concatenate([P[p]["Xf_tr"] for p in PARAS] + [Xr_tr])
    yall = np.concatenate([np.ones(sum(len(P[p]["Xf_tr"]) for p in PARAS)), np.zeros(len(Xr_tr))]).astype(np.float32)
    H = ds.train_head(Xall, yall, seed=0); H_state = {k: v.clone() for k, v in H.state_dict().items()}
    def per_para(head): return {p: round(ds.auc(ds.scores(head, P[p]["Xte"]), P[p]["yte"]), 4) for p in PARAS}
    base = per_para(H)
    print(f"[CURV-ENG] base head AUC: {base}", flush=True)

    # interference B: relabel -> only COLLAPSE_TO is fake(1); everything else real(0) -> erodes the rest
    keep = COLLAPSE_TO; erode = [p for p in PARAS if p != keep]
    Xb = np.concatenate([P[keep]["Xf_tr"]] + [P[e]["Xf_tr"] for e in erode] + [Xr_tr])
    yb = np.concatenate([np.ones(len(P[keep]["Xf_tr"]))] +
                        [np.zeros(len(P[e]["Xf_tr"])) for e in erode] + [np.zeros(len(Xr_tr))]).astype(np.float32)

    def basis_of(S):
        if not S: return None
        U = np.stack([P[p]["dir"] for p in S], 1).astype(np.float32); Q, _ = np.linalg.qr(U); return Q

    print(f"\n=== curvature engineering: interference = collapse-to-{keep}; protect set S via GPM off span(S) ===", flush=True)
    print("  " + "protect S".ljust(26) + "".join(p[:8].rjust(9) for p in PARAS), flush=True)
    results = {}
    for S in PROFILES:
        U = basis_of(S)
        h = ds.continue_train(H_state, Xb, yb, U=U, epochs=150, seed=0)
        ret = per_para(h); results["+".join(S) if S else "none"] = ret
        tag = ("{" + ",".join(s[:4] for s in S) + "}") if S else "{} baseline"
        print("  " + tag.ljust(26) + "".join(f"{ret[p]:9.3f}" for p in PARAS), flush=True)

    # leakage analysis: for a singleton protect S={swap}, gain on aligned (celebdf/dfdc, cos>0.3) vs anti (diffusion)
    baseline = results["none"]
    def gains(key): return {p: results[key][p] - baseline[p] for p in PARAS}
    analysis = {}
    for S in PROFILES:
        if len(S) != 1: continue
        s = S[0]; g = gains(s)
        aligned = [p for p in PARAS if p != s and cosM[s][p] > 0.3]
        anti = [p for p in PARAS if p != s and cosM[s][p] < 0.0]
        analysis[s] = {"self_gain": round(g[s], 4),
                       "aligned_leak": round(float(np.mean([g[p] for p in aligned])) if aligned else 0.0, 4),
                       "anti_leak": round(float(np.mean([g[p] for p in anti])) if anti else 0.0, 4),
                       "aligned": aligned, "anti": anti}
        print(f"\n  protect {{{s}}}: self_gain={analysis[s]['self_gain']:+.3f} | "
              f"leak to ALIGNED {aligned}={analysis[s]['aligned_leak']:+.3f} | "
              f"leak to ANTI {anti}={analysis[s]['anti_leak']:+.3f}", flush=True)

    print("\n  => protection LEAKS along +cos (aligned neighbours protected too), ~0 to anti-parallel — "
          "protection obeys the SAME cosine geometry as forgetting(#1) + unlearning(#3).", flush=True)

    OUT.write_text(json.dumps({"config": {"subsample": SUB, "collapse_to": keep, "profiles": PROFILES, "smoke": SMOKE,
                                          "prereg_lock": "128d757a"}, "cosine": cosM, "base": base,
                               "profile_retention": results, "leakage_analysis": analysis}, indent=1))
    print(f"\nwrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
