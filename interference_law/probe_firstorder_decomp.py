"""
probe_firstorder_decomp.py — NEW-1a''' (PREREG lock 49633fa5): decompose the first-order term −(g_i·Δθ_p)=gmag_i·dmag_p·align(i,p).
Which factor carries the ρ+0.45 directed-forgetting prediction — the per-victim magnitude ‖g_i‖, or the per-PAIR ALIGNMENT
cos(g_i,Δθ_p)? Double-centering isolates the genuine directional interaction (the #6b/#7 quarry). $0 local.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R
import probe_gradient_alignment as G

DEV = G.DEV
FT_STEPS = R._i("FT_STEPS", 300); FT_LR = R._f("FT_LR", 5e-4)
VICTIMS = list(range(40)); PERTURBERS = [50, 55, 60, 65, 70, 75, 80, 85]; NEG = list(range(90, 100))
NJ = R._i("NJ", 128); SEED = R._i("SEED", 0)
OUT = HERE / "out" / "firstorder_decomp.json"; LOCK = "49633fa5"
torch.manual_seed(SEED); np.random.seed(SEED)


def head_auc(head, Fa, Fn):
    with torch.no_grad():
        sa = head(torch.from_numpy(Fa).to(DEV)).squeeze(1).cpu().numpy()
        sn = head(torch.from_numpy(Fn).to(DEV)).squeeze(1).cpu().numpy()
    return R.auc(np.concatenate([sa, sn]), np.concatenate([np.ones(len(sa)), np.zeros(len(sn))]))

def sp(x, y): return float(R.spearman(np.asarray(x, float).ravel(), np.asarray(y, float).ravel()))
def dcenter(X): return X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()


def main():
    t0 = time.time()
    print(f"[DEC] lock={LOCK} FT={FT_STEPS}/{FT_LR} victims={len(VICTIMS)} perturbers={len(PERTURBERS)} dev={DEV}", flush=True)
    banks = R.load_banks(); Xneg_u8 = torch.cat([banks[c] for c in NEG])
    base = G.build_full(); base.eval(); Fneg0 = R.features(base, Xneg_u8)
    heads, auc0 = {}, {}
    for i in VICTIMS:
        FA0 = R.features(base, banks[i]); yi = [1]*len(FA0) + [0]*len(Fneg0)
        heads[i] = G.train_head(np.concatenate([FA0, Fneg0]), yi, 1); auc0[i] = head_auc(heads[i], FA0, Fneg0)
    print(f"[DEC] heads ready ({time.time()-t0:.0f}s)", flush=True)

    # perturbers: Δθ_p (stored) + forget matrix
    theta0 = G.theta_vec(base); nV, nP = len(VICTIMS), len(PERTURBERS)
    dthetas = []; dmag = np.zeros(nP); FORGET = np.zeros((nV, nP))
    for pj, p in enumerate(PERTURBERS):
        Xb = torch.cat([banks[p], Xneg_u8]); yb = torch.cat([torch.ones(len(banks[p])), torch.zeros(len(Xneg_u8))]).long()
        mm = G.build_full(); mm.load_state_dict(base.state_dict()); G.full_ft(mm, Xb, yb, FT_STEPS, FT_LR)
        dth = (G.theta_vec(mm) - theta0).float(); dthetas.append(dth); dmag[pj] = float(dth.norm())
        Fneg1 = R.features(mm, Xneg_u8)
        for ci, i in enumerate(VICTIMS):
            FORGET[ci, pj] = auc0[i] - head_auc(heads[i], R.features(mm, banks[i]), Fneg1)
        print(f"  perturb {p}: |dth|={dmag[pj]:.2f} ({time.time()-t0:.0f}s)", flush=True)

    # Jacobians: g_i (one at a time) → gmag_i, dot[i,p], cos[i,p]
    Xn_g = R.prep(Xneg_u8[:NJ]).to(DEV)
    gmag = np.zeros(nV); DOT = np.zeros((nV, nP)); COS = np.zeros((nV, nP))
    for ci, i in enumerate(VICTIMS):
        base.zero_grad(set_to_none=True)
        m = heads[i](base(R.prep(banks[i][:NJ]).to(DEV))).mean() - heads[i](base(Xn_g)).mean()
        m.backward()
        g = torch.cat([pp.grad.detach().flatten() for pp in base.parameters()]).float().cpu()
        gmag[ci] = float(g.norm())
        for pj in range(nP):
            DOT[ci, pj] = float(g @ dthetas[pj]); COS[ci, pj] = DOT[ci, pj] / (gmag[ci]*dmag[pj] + 1e-12)
    base.zero_grad(set_to_none=True)
    print(f"[DEC] Jacobians+dots done ({time.time()-t0:.0f}s)", flush=True)

    FO = -DOT                       # full first-order predicted forgetting
    ALIGN = -COS                    # signed alignment (magnitudes stripped)
    GMAGb = np.repeat(gmag[:, None], nP, 1)      # victim magnitude broadcast
    DMAGb = np.repeat(dmag[None, :], nV, 0)      # perturber magnitude broadcast

    med = float(np.median(FORGET))
    print(f"\n[DEC] median forget={med:+.3f} (sanity) | ||dth|| {dmag.min():.1f}-{dmag.max():.1f} | ||g|| {gmag.min():.0f}-{gmag.max():.0f} cos range [{COS.min():+.3f},{COS.max():+.3f}]", flush=True)

    tbl = {
        "fo_full":        sp(FO, FORGET),
        "align":          sp(ALIGN, FORGET),
        "align_dcenter":  sp(dcenter(ALIGN), dcenter(FORGET)),
        "gmag_broadcast": sp(GMAGb, FORGET),
        "dmag_broadcast": sp(DMAGb, FORGET),
        "mag_only(gmag*dmag)": sp(GMAGb*DMAGb, FORGET),
    }
    for k in ("fo_full", "align", "align_dcenter", "gmag_broadcast", "dmag_broadcast", "mag_only(gmag*dmag)"):
        print(f"  rho(forget, {k:20}) = {tbl[k]:+.3f}", flush=True)

    p1 = bool(tbl["align"] >= 0.30)
    p2 = bool(tbl["align_dcenter"] >= 0.20)
    p3 = bool((tbl["gmag_broadcast"] < 0) and (tbl["align"] > 0))
    if p1 and p2:
        verdict = "FOUND IT: plastic-regime DIRECTIONAL LAW = alignment of perturbation with victim margin-Jacobian (genuine per-pair interaction)."
    elif p1 and not p2:
        verdict = "Alignment predicts POOLED but NOT after double-centering -> +0.45 is fragility×disruptiveness (magnitude) structure; NO per-pair directional law."
    else:
        verdict = "Alignment carries little -> first-order +0.45 is magnitude/row-col; not directional."
    print(f"\n[DEC] P1={p1} P2={p2} P3={p3} | VERDICT: {verdict}", flush=True)

    OUT.write_text(json.dumps({"config": {"ft": [FT_STEPS, FT_LR], "victims": VICTIMS, "perturbers": PERTURBERS, "nj": NJ,
                               "seed": SEED, "prereg_lock": LOCK}, "median_forget": round(med, 4),
                               "decomposition_rho": {k: round(v, 3) for k, v in tbl.items()},
                               "dmag_range": [round(float(dmag.min()), 2), round(float(dmag.max()), 2)],
                               "gmag_range": [round(float(gmag.min()), 1), round(float(gmag.max()), 1)],
                               "cos_range": [round(float(COS.min()), 3), round(float(COS.max()), 3)],
                               "checks": {"P1_align": p1, "P2_interaction": p2, "P3_sign": p3}, "verdict": verdict}, indent=1))
    print(f"[DEC] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
