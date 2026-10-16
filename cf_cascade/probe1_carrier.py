"""PROBE 1 (MINE) — single-carrier DSI isolation + fly-around interface. CPU, reads dsi_<cell>.npz (per-(arm,step,carrier))
+ geom_matrix.npz / geom_opt1_3b for per-carrier static G5. No new GPU.

Per carrier, from the DSI trajectories (arm A=no deposit, B=deposit), classify the mechanism signature and test whether
it is carrier-UNIVERSAL or carrier-SPECIFIC (the pooled DSI median hides a split). Then the fly-around corroboration:
does per-carrier static G5 (deposit-surge overlap at the dispositive endpoint) PREDICT the per-step dynamic signature?
PRE-COMMIT: low-G5 carriers -> DEFLECT/ABSORB (deposit survives), high-G5 -> ERASE/PASS (deposit eroded).

Per-carrier scores (B vs A, late-phase = last 50 steps; magnitude-matched via cos/fractions):
  deflect = (inV_A - inV_B)_late + max(0, cos_dep_early - cos_dep_late)   [surge routed OUT of V + alignment drops]
  absorb  = (inV_B - inV_A)_late  (with surv high)                         [surge flows toward deposit, deposit holds]
  surv_end = deposit survival at t=end (1=intact, ->0 = ERASE)
  swap_gap = cos_dep_B_late - cos_swap_B_late                             [>0 deposit-specific; ~0 generic curvature]
  gnorm_ratio = gnorm_B/gnorm_A late                                      [<1 = B surge shrinks = itself a deflect signal]
"""
import numpy as np, json, os

CELLS = {"opt350m": ("OPT", "opt350m"), "pythia410m": ("PYTHIA", "pythia410m")}
LATE = slice(-50, None)


def load_g5(tag):
    for npz in ("gateA/geom_matrix.npz", "gateA/geom_opt1_3b.npz"):
        if os.path.exists(npz):
            z = np.load(npz)
            if f"{tag}__G5" in z.files: return np.asarray(z[f"{tag}__G5"], float)
    return None


def carrier_type(n, fam):
    keys = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"] if fam == "OPT" else \
           ["query_key_value", "dense_h_to_4h", "dense_4h_to_h"]
    for c in keys:
        if c in n: return c.replace("query_key_value", "qkv").replace("dense_h_to_4h", "h4h").replace("dense_4h_to_h", "4hh")
    return "?"


def analyze(cell):
    fam, tag = CELLS[cell]
    p = f"gateA/dsi_{cell}.npz"
    if not os.path.exists(p): return None
    z = np.load(p, allow_pickle=True)
    car = [str(x) for x in z["carriers"]]; NC = len(car)
    g = lambda a, q: z[f"{a}__{q}"]  # (T, NC)
    inV_A, inV_B = g("A", "inV"), g("B", "inV")
    cd_B = g("B", "cos_dep"); cs_B = g("B", "cos_swap"); sv_B = g("B", "surv"); gnA, gnB = g("A", "gnorm"), g("B", "gnorm")
    rows = []
    for ci in range(NC):
        defl = float(np.mean(inV_A[LATE, ci] - inV_B[LATE, ci]) + max(0.0, np.mean(cd_B[:10, ci]) - np.mean(cd_B[LATE, ci])))
        absb = float(np.mean(inV_B[LATE, ci] - inV_A[LATE, ci]))
        surv_end = float(np.mean(sv_B[LATE, ci]))
        swap_gap = float(np.mean(cd_B[LATE, ci]) - np.mean(cs_B[LATE, ci]))
        gnr = float(np.mean(gnB[LATE, ci]) / (np.mean(gnA[LATE, ci]) + 1e-30))
        # signature label (pre-committed thresholds; report continuous too)
        if surv_end < 0.6: sig = "ERASE"
        elif defl > 0.02 and surv_end >= 0.6: sig = "DEFLECT"
        elif absb > 0.02 and surv_end >= 0.6: sig = "ABSORB"
        else: sig = "PASS"
        rows.append({"carrier": car[ci], "type": carrier_type(car[ci], fam), "deflect": defl, "absorb": absb,
                     "surv_end": surv_end, "swap_gap": swap_gap, "gnorm_ratio": gnr, "sig": sig})
    # per-carrier-type summary
    types = {}
    for ct in sorted(set(r["type"] for r in rows)):
        sub = [r for r in rows if r["type"] == ct]
        sigs = [r["sig"] for r in sub]
        types[ct] = {"n": len(sub), "deflect": float(np.median([r["deflect"] for r in sub])),
                     "surv_end": float(np.median([r["surv_end"] for r in sub])),
                     "swap_gap": float(np.median([r["swap_gap"] for r in sub])),
                     "gnorm_ratio": float(np.median([r["gnorm_ratio"] for r in sub])),
                     "sig_counts": {s: sigs.count(s) for s in set(sigs)}}
    # universal vs specific
    allsig = [r["sig"] for r in rows]; dom = max(set(allsig), key=allsig.count)
    universal = allsig.count(dom) / len(allsig) >= 0.8
    # fly-around: per-carrier G5 vs dynamic (surv_end). predict corr(G5, surv_end) < 0 (high G5 -> erase -> low surv)
    g5 = load_g5(tag); flya = None
    if g5 is not None and len(g5) == NC:
        from scipy.stats import rankdata
        sv = np.array([r["surv_end"] for r in rows]); df = np.array([r["deflect"] for r in rows])
        flya = {"corr_G5_surv": float(np.corrcoef(rankdata(g5), rankdata(sv))[0, 1]),
                "corr_G5_deflect": float(np.corrcoef(rankdata(g5), rankdata(df))[0, 1]),
                "note": "predict corr(G5,surv)<0 and corr(G5,deflect)<0 (high G5 -> erase): G5 = static snapshot of collision"}
    elif g5 is not None:
        flya = {"skipped": f"carrier-count mismatch DSI {NC} vs geom {len(g5)} (cannot index-align)"}
    return {"cell": cell, "family": fam, "n_carriers": NC, "dominant_sig": dom, "carrier_universal": universal,
            "by_type": types, "fly_around": flya}


def main():
    out = {}
    for cell in CELLS:
        r = analyze(cell)
        if r: out[cell] = r
    json.dump(out, open("gateA/probe1_carrier.json", "w"), indent=1)
    if not out:
        print("no DSI npz landed yet"); return
    print("=== PROBE 1: single-carrier DSI isolation (universal vs specific) + fly-around (G5) ===\n")
    for cell, r in out.items():
        print(f"[{cell}] {r['family']} n={r['n_carriers']} dominant={r['dominant_sig']} "
              f"carrier-{'UNIVERSAL' if r['carrier_universal'] else 'SPECIFIC'}")
        for ct, t in r["by_type"].items():
            print(f"   {ct:5s} n{t['n']:>3d} deflect={t['deflect']:+.3f} surv_end={t['surv_end']:.2f} "
                  f"swap_gap={t['swap_gap']:+.3f} gnorm_B/A={t['gnorm_ratio']:.2f} sigs={t['sig_counts']}")
        fa = r["fly_around"]
        if fa and "corr_G5_surv" in fa:
            print(f"   fly-around: corr(G5,surv_end)={fa['corr_G5_surv']:+.2f} corr(G5,deflect)={fa['corr_G5_deflect']:+.2f} (predict both <0)")
        elif fa:
            print(f"   fly-around: {fa.get('skipped')}")
        print()
    print("saved gateA/probe1_carrier.json")


if __name__ == "__main__":
    main()
