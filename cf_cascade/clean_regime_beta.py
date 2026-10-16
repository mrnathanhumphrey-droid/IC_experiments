"""#2 VERIFICATION (not pursuit): is GPT-2/AG-News's super-linear beta=1.57 REAL temporal
multiplicativity, or the early-transient log-log-fit artifact the false-PASS pattern has burned us
on twice? Refit beta on the CLEAN diffusive regime (step>=100, after the high-loss transient) for
ALL THREE substrates, on the SAVED series (CPU, no GPU re-run — 12 snapshots already exist at step>=100).

Corroborating-acceleration check (don't trust the scalar — false-PASS lesson, 3rd time): the clean
log-log beta MUST agree with the linear-space signature — var trajectory convex (quadratic accel
coef > 0) AND PR still climbing super-linearly (not plateauing). If beta>1.2 but PR plateaus, beta
is suspect.

PRE-COMMITTED VERDICT (GPT-2/AG):
  clean-beta>1.2 AND accel-corroborated (var convex AND PR climbing) -> TEMPORAL-MULT REAL, C5 reopens
  clean-beta~1 (<=1.2) OR accel NOT corroborated                     -> TRANSIENT ARTIFACT, C5 closed
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "4"
import json, numpy as np
G = "D:/IC_experiments/cf_cascade/gateA"
SERIES = {"Pythia/AG-News": f"{G}/diffusion_series.npz", "Pythia/Yelp": f"{G}/diffusion_series_yelp.npz",
          "GPT-2/AG-News": f"{G}/diffusion_series_gpt2_agnews.npz"}
FULL_BETA = {"Pythia/AG-News": 0.88, "Pythia/Yelp": 1.16, "GPT-2/AG-News": 1.57}
CLEAN_STEP = 100


def analyze(path, label):
    s = np.load(path); steps = s["steps"]; V = s["per_shell_var"]; PR = s["PR"]
    m = steps >= CLEAN_STEP
    cs = steps[m]; CV = V[m]; cPR = PR[m]
    ls = np.log(cs)
    # clean-regime per-shell beta (log-log var vs step)
    betas = []
    for j in range(CV.shape[1]):
        y = CV[:, j]; ok = y > 0
        if ok.sum() >= 5: betas.append(np.polyfit(ls[ok], np.log(y[ok]), 1)[0])
    betas = np.array(betas); med_beta_clean = float(np.nanmedian(betas))
    # acceleration corroboration in LINEAR space: median-across-shells var vs step, quadratic coef sign
    mvar = np.nanmedian(CV, axis=1)
    qa = np.polyfit(cs, mvar, 2)[0]                          # quadratic coef: >0 convex/accelerating
    # normalize accel: compare quadratic vs linear fit improvement
    lin = np.polyval(np.polyfit(cs, mvar, 1), cs); quad = np.polyval(np.polyfit(cs, mvar, 2), cs)
    ss = np.sum((mvar - mvar.mean()) ** 2)
    r2_lin = 1 - np.sum((mvar - lin) ** 2) / ss; r2_quad = 1 - np.sum((mvar - quad) ** 2) / ss
    var_convex = bool(qa > 0 and (r2_quad - r2_lin) > 0.01)
    # PR trajectory on clean regime: still climbing? accelerating?
    pr_lastmax = float(cPR[-1] / cPR.max())
    pr_qa = np.polyfit(cs, cPR, 2)[0]; pr_climbing = bool(pr_lastmax > 0.98); pr_accel = bool(pr_qa > 0)
    accel_corroborated = bool(var_convex and pr_climbing and pr_accel)
    return {"label": label, "full_beta": FULL_BETA[label], "clean_beta": med_beta_clean, "n_clean": int(m.sum()),
            "var_quad_coef": float(qa), "var_convex": var_convex, "r2_lin": float(r2_lin), "r2_quad": float(r2_quad),
            "PR_clean_lastmax": pr_lastmax, "PR_climbing": pr_climbing, "PR_accel": pr_accel,
            "accel_corroborated": accel_corroborated}


def main():
    print(f"[#2] clean-regime beta refit (step>={CLEAN_STEP}), all 3 substrates\n", flush=True)
    res = {}
    for label, path in SERIES.items():
        r = analyze(path, label); res[label] = r
        print(f"  [{label:16s}] full-beta={r['full_beta']:.2f} -> CLEAN-beta={r['clean_beta']:.3f} (n={r['n_clean']})", flush=True)
        print(f"        var convex={r['var_convex']} (quad coef {r['var_quad_coef']:+.2e}, r2 lin {r['r2_lin']:.3f}->quad {r['r2_quad']:.3f}) | PR climbing={r['PR_climbing']} accel={r['PR_accel']} (last/max {r['PR_clean_lastmax']:.3f})", flush=True)
        print(f"        accel-corroborated={r['accel_corroborated']}", flush=True)
    g = res["GPT-2/AG-News"]
    if g["clean_beta"] > 1.2 and g["accel_corroborated"]:
        verdict = "TEMPORAL-MULTIPLICATIVITY REAL on GPT-2 (clean-beta>1.2 AND var-convex + PR-accelerating) -> C5 REOPENS net-dependently -> chase net-mechanism (#next)"
    elif g["clean_beta"] > 1.2 and not g["accel_corroborated"]:
        verdict = "BETA SUSPECT: clean-beta>1.2 but acceleration NOT corroborated (PR/var disagree) -> false-PASS trap, do NOT bank super-linear; treat as inconclusive"
    else:
        verdict = "TRANSIENT ARTIFACT: clean-beta~1 -> super-linear was early-transient fit artifact, C5 stays CLOSED, generator split = free-vs-bounded (task-controlled) -> next is #1 reversibility"
    print(f"\n[#2] === VERDICT (GPT-2/AG): {verdict} ===", flush=True)
    print(f"     (context: Pythia/AG clean-beta {res['Pythia/AG-News']['clean_beta']:.2f}, Pythia/Yelp clean-beta {res['Pythia/Yelp']['clean_beta']:.2f})", flush=True)
    json.dump({"clean_step": CLEAN_STEP, "results": res, "gpt2_verdict": verdict}, open(f"{G}/clean_regime_beta.json", "w"), indent=2)
    print(f"[#2] saved clean_regime_beta.json", flush=True)


if __name__ == "__main__":
    main()
