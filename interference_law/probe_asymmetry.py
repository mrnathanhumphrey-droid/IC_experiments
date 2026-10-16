"""
probe_asymmetry.py — NEW-1 (PREREG lock ebdbba5a): is the DIRECTED forgetting matrix F[i<-j] asymmetric? Cosine is symmetric by
construction; if F isn't, the cosine law is INCOMPLETE and there is a per-capability FRAGILITY axis. Decompose F=S+K, isolate fragility
vs disruptiveness, test the dominance structure (transitive pecking-order vs cyclic). $0 local.

MODE=gate : 4 perturbers, non-saturation check (median off-diag F in [0.03,0.45]).
MODE=full : the 12x12 directed matrix + symmetry / factorization / dominance analysis.
PERTURB=neg (default, class_j vs shared neg) | caps (class_j vs other CAPS pooled — harder, escalation).
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R
import probe_gradient_alignment as G

DEV = G.DEV; MODE = os.environ.get("MODE", "gate"); PERTURB = os.environ.get("PERTURB", "neg")
FT_STEPS = R._i("FT_STEPS", 300); FT_LR = R._f("FT_LR", 5e-4)
CAPS = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77]
NEG = list(range(90, 100))
SEED = R._i("SEED", 0)
OUT = HERE / "out" / f"asymmetry_{MODE}{'' if PERTURB=='neg' else '_'+PERTURB}.json"
LOCK = "ebdbba5a"
torch.manual_seed(SEED); np.random.seed(SEED)


def head_auc(head, Fa, Fn):
    with torch.no_grad():
        sa = head(torch.from_numpy(Fa).to(DEV)).squeeze(1).cpu().numpy()
        sn = head(torch.from_numpy(Fn).to(DEV)).squeeze(1).cpu().numpy()
    return R.auc(np.concatenate([sa, sn]), np.concatenate([np.ones(len(sa)), np.zeros(len(sn))]))


def main():
    t0 = time.time()
    caps = CAPS[:4] if MODE == "gate" else CAPS
    print(f"[ASYM] lock={LOCK} MODE={MODE} PERTURB={PERTURB} FT={FT_STEPS}/{FT_LR} n_caps={len(caps)} dev={DEV}", flush=True)
    banks = R.load_banks(); Xneg_u8 = torch.cat([banks[c] for c in NEG])
    base = G.build_full(); base.eval(); Fneg0 = R.features(base, Xneg_u8)

    heads, dvec, auc0 = {}, {}, {}
    for i in caps:
        FA0 = R.features(base, banks[i]); dvec[i] = R.ddir(FA0, Fneg0)
        yi = [1]*len(FA0) + [0]*len(Fneg0)
        heads[i] = G.train_head(np.concatenate([FA0, Fneg0]), yi, 1); auc0[i] = head_auc(heads[i], FA0, Fneg0)
    print(f"[ASYM] {len(caps)} deployed heads ready, base AUC median={np.median(list(auc0.values())):.3f} ({time.time()-t0:.0f}s)", flush=True)

    theta0 = G.theta_vec(base)
    n = len(caps); F = np.full((n, n), np.nan); COS = np.full((n, n), np.nan); dtheta = {}
    for cj, j in enumerate(caps):
        if PERTURB == "neg":
            Xb = torch.cat([banks[j], Xneg_u8]); yb = torch.cat([torch.ones(len(banks[j])), torch.zeros(len(Xneg_u8))]).long()
        else:  # class_j vs the OTHER caps pooled (harder negative)
            others = torch.cat([banks[c] for c in caps if c != j])
            Xb = torch.cat([banks[j], others]); yb = torch.cat([torch.ones(len(banks[j])), torch.zeros(len(others))]).long()
        m = G.build_full(); m.load_state_dict(base.state_dict()); G.full_ft(m, Xb, yb, FT_STEPS, FT_LR)
        dtheta[j] = float((G.theta_vec(m) - theta0).norm())
        Fneg1 = R.features(m, Xneg_u8)
        for ci, i in enumerate(caps):
            if i == j: continue
            F[ci, cj] = auc0[i] - head_auc(heads[i], R.features(m, banks[i]), Fneg1)
            COS[ci, cj] = float(dvec[i] @ dvec[j])
        col = F[:, cj]; print(f"  perturb j={j}: |dth|={dtheta[j]:.2f}  mean damage to others={np.nanmean(col):+.3f} ({time.time()-t0:.0f}s)", flush=True)

    offdiag = F[~np.isnan(F)]; med = float(np.median(offdiag))
    inband = 0.03 <= med <= 0.45
    print(f"\n[ASYM] median off-diag forget={med:+.3f} (partial band [0.03,0.45]: {'OK' if inband else 'OUT'})", flush=True)
    if MODE == "gate":
        verdict = ("GATE PASS -> run full 12x12" if inband else
                   f"GATE {'SATURATED' if med>0.45 else 'TOO WEAK'} -> " + ("lower FT" if med > 0.45 else "raise FT or PERTURB=caps"))
        print(f"[ASYM] {verdict}", flush=True)
        OUT.write_text(json.dumps({"config": {"mode": MODE, "perturb": PERTURB, "ft": [FT_STEPS, FT_LR], "prereg_lock": LOCK},
                                   "median_forget": med, "inband": inband, "verdict": verdict}, indent=1))
        print(f"[ASYM] wrote {OUT} ({time.time()-t0:.0f}s)", flush=True); return

    # ── full analysis ─────────────────────────────────────────────────────────
    # symmetric / antisymmetric decomposition (fill diagonal with 0 for algebra; exclude diag from norms)
    Ff = np.where(np.isnan(F), 0.0, F); S = (Ff + Ff.T) / 2; K = (Ff - Ff.T) / 2
    mask = ~np.eye(n, dtype=bool)
    Snorm = float(np.sqrt((S[mask]**2).sum())); Knorm = float(np.sqrt((K[mask]**2).sum()))
    kratio = Knorm / (Snorm + 1e-12)
    # directed-pair correlation F[i,j] vs F[j,i]
    ij = [(i, j) for i in range(n) for j in range(n) if i < j and not np.isnan(F[i, j]) and not np.isnan(F[j, i])]
    a = np.array([F[i, j] for i, j in ij]); b = np.array([F[j, i] for i, j in ij])
    rho_sym = float(R.spearman(a, b)); pear_sym = float(np.corrcoef(a, b)[0, 1])

    fragile = np.nanmean(F, axis=1); disrupt = np.nanmean(F, axis=0)      # row=victim, col=perturber
    rho_fd = float(R.spearman(fragile, disrupt))
    dth_arr = np.array([dtheta[j] for j in caps]); rho_dis_dth = float(R.spearman(disrupt, dth_arr))

    # additive ANOVA: forget_ij = mu + row_i + col_j + s*cos_ij  (least squares over off-diag)
    rows_i, cols_j, y, cvec = [], [], [], []
    for ci in range(n):
        for cj in range(n):
            if ci != cj and not np.isnan(F[ci, cj]):
                rows_i.append(ci); cols_j.append(cj); y.append(F[ci, cj]); cvec.append(COS[ci, cj])
    y = np.array(y); cvec = np.array(cvec); N = len(y)
    def design(use_row, use_col, use_cos):
        cols = [np.ones(N)]
        if use_cos: cols.append(cvec)
        if use_col:
            for j in range(1, n): cols.append((np.array(cols_j) == j).astype(float))
        if use_row:
            for i in range(1, n): cols.append((np.array(rows_i) == i).astype(float))
        return np.stack(cols, 1)
    def r2(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None); pred = X @ beta
        ss = ((y - y.mean())**2).sum(); return float(1 - ((y - pred)**2).sum() / (ss + 1e-12))
    r2_cos = r2(design(0, 0, 1)); r2_coscol = r2(design(0, 1, 1)); r2_full = r2(design(1, 1, 1))
    dR2_fragility = r2_full - r2_coscol

    # dominance: rank-1 potential fit K[i,j] ~ h_i - h_j ; h_i = mean_j K[i,j]
    h = K.sum(1) / (n - 1)                                                # row means over off-diag (diag is 0)
    Khat = h[:, None] - h[None, :]
    num = float(((K[mask] - Khat[mask])**2).sum()); den = float((K[mask]**2).sum())
    hier_frac = 1.0 - num / (den + 1e-12)

    print(f"[ASYM] === DIRECTED MATRIX (n={n}) ===", flush=True)
    print(f"[ASYM] New1-P1 asymmetry: ||K||/||S||={kratio:.3f}  rho(F_ij,F_ji)={rho_sym:+.3f} (pear {pear_sym:+.3f})  "
          f"[asym if ratio>=.15 & rho<.80]", flush=True)
    print(f"[ASYM] New1-P2 fragility: R2(cos)={r2_cos:.3f} -> +col={r2_coscol:.3f} -> +row(frag)={r2_full:.3f}  dR2_frag={dR2_fragility:+.3f}  "
          f"rho(fragile,disrupt)={rho_fd:+.3f}  [2nd axis if dR2>=.10 & |rho|<.50]", flush=True)
    print(f"[ASYM] New1-P3 disrupt replicates: rho(disrupt_j, ||dth_j||)={rho_dis_dth:+.3f}  [expect >=+.50]", flush=True)
    print(f"[ASYM] New1-P4 dominance: rank-1 potential explains {hier_frac*100:.1f}% of ||K||^2  "
          f"[transitive>=60% | cyclic<40%]", flush=True)
    order = [caps[k] for k in np.argsort(-h)]
    print(f"[ASYM] dominance potential h (predator=+, prey=-): " + ", ".join(f"{caps[k]}:{h[k]:+.3f}" for k in np.argsort(-h)), flush=True)

    p1 = (kratio >= 0.15) and (rho_sym < 0.80)
    p2 = (dR2_fragility >= 0.10) and (abs(rho_fd) < 0.50)
    p3 = rho_dis_dth >= 0.50
    p4_hier = hier_frac >= 0.60; p4_cyclic = hier_frac < 0.40
    verdict = (f"P1={'ASYMMETRIC' if p1 else 'symmetric'} | P2={'FRAGILITY-2nd-AXIS' if p2 else 'no-2nd-axis'} | "
               f"P3={'dtheta-replicates' if p3 else 'dtheta-null'} | "
               f"P4={'TRANSITIVE-hierarchy' if p4_hier else ('CYCLIC' if p4_cyclic else 'mixed')}")
    print(f"\n[ASYM] VERDICT: {verdict}", flush=True)

    OUT.write_text(json.dumps({"config": {"mode": MODE, "perturb": PERTURB, "ft": [FT_STEPS, FT_LR], "caps": caps, "seed": SEED,
                                          "prereg_lock": LOCK}, "median_forget": med, "inband": inband,
                               "P1": {"k_over_s": round(kratio, 3), "rho_sym": round(rho_sym, 3), "pearson_sym": round(pear_sym, 3)},
                               "P2": {"r2_cos": round(r2_cos, 3), "r2_coscol": round(r2_coscol, 3), "r2_full": round(r2_full, 3),
                                      "dR2_fragility": round(dR2_fragility, 3), "rho_fragile_disrupt": round(rho_fd, 3)},
                               "P3": {"rho_disrupt_dtheta": round(rho_dis_dth, 3)},
                               "P4": {"hierarchy_frac": round(hier_frac, 3)},
                               "fragile": {caps[k]: round(float(fragile[k]), 4) for k in range(n)},
                               "disrupt": {caps[k]: round(float(disrupt[k]), 4) for k in range(n)},
                               "dominance_h": {caps[k]: round(float(h[k]), 4) for k in range(n)},
                               "checks": {"P1_asym": p1, "P2_fragility": p2, "P3": p3, "P4_hier": p4_hier, "P4_cyclic": p4_cyclic},
                               "verdict": verdict, "F": [[None if np.isnan(F[i, j]) else round(float(F[i, j]), 4) for j in range(n)] for i in range(n)]}, indent=1))
    print(f"[ASYM] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
