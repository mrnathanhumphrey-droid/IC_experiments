"""
probe_face1_dominance_apriori.py — Face-1a (PREREG lock = sha256(PREREG_FACE1_DOMINANCE_APRIORI_2026_07_17.md)[:8]):
Is the dominance potential phi_i = (fragility_i - disruptiveness_i)/2 — the coordinate that DEFINES the transitive pecking order —
predictable a-priori from FROZEN base-task statistics (no fine-tuning)? $0 local. Reuses probe_fragility.py's exact predictor code path.

Amendments folded (2026-07-17 review): pre-committed trio {dprime,wnorm,spread} (no selection leakage); d' = named primary,
others Bonferroni-exploratory; permutation-null under the 0.30 LOO-R2 threshold; provenance assert on the caps-0..39 overlap.
"""
from __future__ import annotations
import sys, json, time, hashlib
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_representational_law as R
import probe_gradient_alignment as G

PHI_CAPS = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77]   # asymmetry_full 12-cap square matrix
FRAG_CAPS = list(range(40))                                  # fragility.json 40 victims
VIC_EXT = list(range(40)) + [42, 49, 56, 63, 70, 77]        # 0..39 identical RNG stream -> bit-match disk; +6 fresh
NEG = list(range(90, 100))
TRIO = ["dprime", "wnorm", "spread"]                         # pre-committed: top-3 |rho| from fragility arc
PRIMARY = "dprime"; RIDGE_LAM = 1.0; SEED = 0
ALL_PREDS = ["dprime", "spread", "munorm", "wnorm", "iso"]   # iso = exploratory, set-dependent
PREREG = HERE / "PREREG_FACE1_DOMINANCE_APRIORI_2026_07_17.md"
LOCK = hashlib.sha256(PREREG.read_bytes()).hexdigest()[:8]
OUT = HERE / "out" / "face1_dominance_apriori.json"
torch.manual_seed(SEED); np.random.seed(SEED)


def head_auc(head, Fa, Fn):
    with torch.no_grad():
        sa = head(torch.from_numpy(Fa).to(G.DEV)).squeeze(1).cpu().numpy()
        sn = head(torch.from_numpy(Fn).to(G.DEV)).squeeze(1).cpu().numpy()
    return R.auc(np.concatenate([sa, sn]), np.concatenate([np.ones(len(sa)), np.zeros(len(sn))]))


def compute_predictors(base, banks, Fneg0, caps):
    """Verbatim replication of probe_fragility.py lines 43-61 for `caps` (iso computed within `caps`)."""
    FA0, dvec, heads, wax = {}, {}, {}, {}
    for i in caps:
        FA0[i] = R.features(base, banks[i]); dvec[i] = R.ddir(FA0[i], Fneg0)
        yi = [1]*len(FA0[i]) + [0]*len(Fneg0)
        heads[i] = G.train_head(np.concatenate([FA0[i], Fneg0]), yi, 1)
        w = heads[i].weight.detach().cpu().numpy()[0].astype(np.float64); wax[i] = (w/np.linalg.norm(w)).astype(np.float32)
    dprime, spread, munorm, wnorm = {}, {}, {}, {}
    for i in caps:
        sa = FA0[i] @ wax[i]; sn = Fneg0 @ wax[i]
        dprime[i] = float((sa.mean() - sn.mean()) / np.sqrt((sa.var() + sn.var())/2 + 1e-12))
        spread[i] = float(FA0[i].std(0).mean())
        munorm[i] = float(np.linalg.norm(FA0[i].mean(0)))
        wnorm[i] = float(np.linalg.norm(heads[i].weight.detach().cpu().numpy()))
    iso = {i: float(np.mean([dvec[i] @ dvec[k] for k in caps if k != i])) for i in caps}
    return {"dprime": dprime, "spread": spread, "munorm": munorm, "wnorm": wnorm, "iso": iso}


# ---- stats ----
def sp(x, y): return float(R.spearman(np.asarray(x, float), np.asarray(y, float)))

def boot_ci(x, y, n=10000, seed=0):
    x = np.asarray(x, float); y = np.asarray(y, float); m = len(x); rng = np.random.RandomState(seed); rs = []
    for _ in range(n):
        idx = rng.randint(0, m, m)
        if np.std(x[idx]) < 1e-12 or np.std(y[idx]) < 1e-12: continue
        rs.append(sp(x[idx], y[idx]))
    return [round(float(np.percentile(rs, 2.5)), 3), round(float(np.percentile(rs, 97.5)), 3)]

def perm_p(x, y, n=10000, seed=0):
    x = np.asarray(x, float); y = np.asarray(y, float); obs = abs(sp(x, y)); rng = np.random.RandomState(seed); c = 0
    for _ in range(n):
        c += abs(sp(x, rng.permutation(y))) >= obs
    return round((c + 1) / (n + 1), 4)

def ridge_loo(X, y, lam=RIDGE_LAM):
    X = np.asarray(X, float); y = np.asarray(y, float); n, p = X.shape; pred = np.zeros(n)
    for k in range(n):
        tr = [j for j in range(n) if j != k]; Xtr, ytr = X[tr], y[tr]
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9; Xs = (Xtr - mu) / sd; ym = ytr.mean()
        b = np.linalg.solve(Xs.T @ Xs + lam*np.eye(p), Xs.T @ (ytr - ym))
        pred[k] = ym + ((X[k] - mu)/sd) @ b
    r2 = float(1 - ((y - pred)**2).sum() / (((y - y.mean())**2).sum() + 1e-12))
    return pred, r2

def perm_null_loo(X, y, n=1000, seed=0):
    y = np.asarray(y, float); rng = np.random.RandomState(seed); null = []
    for _ in range(n):
        _, r2 = ridge_loo(X, rng.permutation(y)); null.append(r2)
    return null


def arm(name, target, preds, caps):
    y = np.array([target[c] for c in caps], float)
    uni = {}
    for pn in ALL_PREDS:
        x = np.array([preds[pn][c] for c in caps], float)
        alpha = 95.0 if pn == PRIMARY else 99.2  # Bonferroni 99.2% for the 5 non-primary (exploratory)
        lo, hi = (np.percentile([sp(x, y)], 50),) * 2  # placeholder overwritten below
        ci = boot_ci(x, y) if pn == PRIMARY else _boot_ci_alpha(x, y, alpha)
        uni[pn] = {"rho": round(sp(x, y), 3), "ci": ci, "perm_p": perm_p(x, y), "excl0": bool(ci[0] > 0 or ci[1] < 0)}
    X = np.stack([[preds[pn][c] for c in caps] for pn in TRIO], 1)
    loo_pred, loo_r2 = ridge_loo(X, y)
    null = perm_null_loo(X, y); null95 = float(np.percentile(null, 95))
    ins_pred, _ = ridge_loo(X, y)  # (in-sample via full refit)
    b_full = np.linalg.solve(((X-X.mean(0))/(X.std(0)+1e-9)).T @ ((X-X.mean(0))/(X.std(0)+1e-9)) + RIDGE_LAM*np.eye(len(TRIO)),
                             ((X-X.mean(0))/(X.std(0)+1e-9)).T @ (y - y.mean()))
    yhat = y.mean() + ((X-X.mean(0))/(X.std(0)+1e-9)) @ b_full
    r2_in = float(1 - ((y-yhat)**2).sum()/(((y-y.mean())**2).sum()+1e-12))
    # ordinal arm
    ord_rho = sp(loo_pred, y)
    rng = np.random.RandomState(0); c = 0
    for _ in range(1000):
        lp, _ = ridge_loo(X, rng.permutation(y)); c += abs(sp(lp, y)) >= abs(ord_rho)
    ord_p = round((c+1)/1001, 4)
    return {"n": len(caps), "univariate": uni, "trio_r2_in": round(r2_in, 3), "trio_loo_r2": round(loo_r2, 3),
            "perm_null_loo_r2_95": round(null95, 3), "ordinal_loo_rankcorr": round(ord_rho, 3), "ordinal_perm_p": ord_p,
            "loo_pred": [round(float(v), 4) for v in loo_pred]}

def _boot_ci_alpha(x, y, alpha, n=10000, seed=0):
    x = np.asarray(x, float); y = np.asarray(y, float); m = len(x); rng = np.random.RandomState(seed); rs = []
    for _ in range(n):
        idx = rng.randint(0, m, m)
        if np.std(x[idx]) < 1e-12 or np.std(y[idx]) < 1e-12: continue
        rs.append(sp(x[idx], y[idx]))
    lo = (100 - alpha) / 2; return [round(float(np.percentile(rs, lo)), 3), round(float(np.percentile(rs, 100 - lo)), 3)]


def main():
    t0 = time.time()
    print(f"[F1a] lock={LOCK} trio={TRIO} primary={PRIMARY} ridge_lam={RIDGE_LAM} dev={G.DEV}", flush=True)
    banks = R.load_banks(); Xneg = torch.cat([banks[c] for c in NEG])
    base = G.build_full(); base.eval(); Fneg0 = R.features(base, Xneg)
    print(f"[F1a] banks+base ready ({time.time()-t0:.0f}s); computing predictors over {len(VIC_EXT)} caps...", flush=True)
    preds_ext = compute_predictors(base, banks, Fneg0, VIC_EXT)

    # ---- PROVENANCE GUARD: caps 0..39 must match fragility.json ----
    fj = json.loads((HERE/"out"/"fragility.json").read_text())["predictors"]
    prov = {}
    for pn in ["spread", "munorm", "dprime", "wnorm"]:
        d_disk = np.array([fj[pn][str(i)] for i in FRAG_CAPS]); d_new = np.array([preds_ext[pn][i] for i in FRAG_CAPS])
        maxabs = float(np.max(np.abs(d_disk - d_new))); rho = sp(d_disk, d_new)
        prov[pn] = {"max_abs_delta": round(maxabs, 6), "rho_vs_disk": round(rho, 4)}
        print(f"  [prov] {pn:7} max|Δ|={maxabs:.2e} rho_vs_disk={rho:.4f}", flush=True)
    # HARD assert on deterministic (feature-only) predictors:
    for pn in ["spread", "munorm"]:
        assert prov[pn]["max_abs_delta"] < 1e-3, f"PROVENANCE FAIL {pn}: extraction drift (max|Δ|={prov[pn]['max_abs_delta']})"
    prov_head_ok = all(prov[pn]["rho_vs_disk"] > 0.98 for pn in ["dprime", "wnorm"])
    print(f"  [prov] deterministic(spread,munorm) MATCH; head-based(dprime,wnorm) rho>0.98 = {prov_head_ok} "
          f"(RNG-order warn if False, not extraction drift)", flush=True)

    # iso for the 12-cap phi set (set-dependent) computed within PHI_CAPS:
    preds_phi = dict(preds_ext)
    preds_phi["iso"] = compute_predictors(base, banks, Fneg0, PHI_CAPS)["iso"]

    asym = json.loads((HERE/"out"/"asymmetry_full.json").read_text())
    phi = {int(k): v for k, v in asym["dominance_h"].items()}
    frag12 = {int(k): v for k, v in asym["fragile"].items()}
    disr12 = {int(k): v for k, v in asym["disrupt"].items()}
    fj_frag = json.loads((HERE/"out"/"fragility.json").read_text())["fragile"]
    frag40 = {int(k): v for k, v in fj_frag.items()}
    preds40 = {pn: {int(k): v for k, v in fj[pn].items()} for pn in fj}

    print(f"[F1a] running arms... ({time.time()-t0:.0f}s)", flush=True)
    results = {
        "ARM_A_fragility_n40": arm("fragility40", frag40, preds40, FRAG_CAPS),
        "ARM_B_phi_n12":       arm("phi12",       phi,    preds_phi, PHI_CAPS),
        "ARM_B_fragility_n12": arm("frag12",      frag12, preds_phi, PHI_CAPS),
        "ARM_B_disrupt_n12":   arm("disrupt12",   disr12, preds_phi, PHI_CAPS),
    }

    # ---- decisions (pre-committed) ----
    B = results["ARM_B_phi_n12"]
    card_thresh = max(0.30, B["perm_null_loo_r2_95"])
    cardinal = bool(B["trio_loo_r2"] >= card_thresh and B["univariate"][PRIMARY]["excl0"])
    ordinal = bool(B["ordinal_loo_rankcorr"] >= 0.50 and B["ordinal_perm_p"] < 0.05)
    A = results["ARM_A_fragility_n40"]
    frag_predictable = bool(A["trio_loo_r2"] >= max(0.30, A["perm_null_loo_r2_95"]))

    if ordinal and cardinal:
        headline = "YES — pecking order a-priori computable (ORDINAL + CARDINAL: magnitude predictable)."
    elif ordinal:
        headline = "YES — the pecking order is a-priori computable (ORDINALLY)."
    else:
        headline = "FULL NULL — the order emerges only from interaction, not static task properties."
    verdict = {"cardinal_pass": cardinal, "cardinal_thresh": round(card_thresh, 3), "ordinal_pass": ordinal,
               "fragility_n40_predictable": frag_predictable, "headline": headline,
               "attenuation_caveat": "phi split-half rho ~0.51 (n=12) — ordinal null confounded with phi unreliability; Lambda arm is the powered adjudicator."}

    OUT.write_text(json.dumps({"config": {"lock": LOCK, "phi_caps": PHI_CAPS, "trio": TRIO, "primary": PRIMARY,
                    "ridge_lam": RIDGE_LAM, "seed": SEED}, "provenance": prov, "provenance_head_ok": prov_head_ok,
                    "results": results, "verdict": verdict}, indent=1))

    print("\n=== FACE-1a RESULTS ===", flush=True)
    print(f"  ARM A (fragility n=40): rho(d',frag)={A['univariate']['dprime']['rho']:+.3f} CI{A['univariate']['dprime']['ci']} "
          f"| trio LOO-R²={A['trio_loo_r2']:+.3f} (null95={A['perm_null_loo_r2_95']:.3f}) predictable={frag_predictable}", flush=True)
    print(f"  ARM B (phi n=12): rho(d',phi)={B['univariate']['dprime']['rho']:+.3f} CI{B['univariate']['dprime']['ci']} perm_p={B['univariate']['dprime']['perm_p']}", flush=True)
    print(f"           CARDINAL: trio LOO-R²={B['trio_loo_r2']:+.3f} vs thresh {card_thresh:.3f} -> pass={cardinal}", flush=True)
    print(f"           ORDINAL:  LOO rank-corr={B['ordinal_loo_rankcorr']:+.3f} perm_p={B['ordinal_perm_p']} -> pass={ordinal}", flush=True)
    print(f"  ARM B fragility12 LOO-R²={results['ARM_B_fragility_n12']['trio_loo_r2']:+.3f} | disrupt12 LOO-R²={results['ARM_B_disrupt_n12']['trio_loo_r2']:+.3f}", flush=True)
    print(f"\n[F1a] HEADLINE: {headline}", flush=True)
    print(f"[F1a] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
