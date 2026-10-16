"""CPU dry-run for Probe-9b (LOCK 869ce94e). Exercises the NEW/changed logic vs Probe-9 on tiny synthetic shards with
KNOWN answers:
  (1) Omega principal-angle overlap (imports the REAL omega() from probe9b_phase2lever_lambda),
  (2) linear CKA sigma_sem (identical -> 1, orthogonal -> 0, monotone),
  (3) the paired PROTECTION estimator P(d) = mean(forget_NoCH - forget_CH) (Probe-7 Delta-forget form),
  (4) the locked Stage-0 ABORT-GATE: span >= 0.15 AND >= 7 non-degenerate points,
  (5) the FIXED-childhood margin-rule lr selection,
  (6) the phase-2 lever Spearman/partial horse-race math on a synthetic Omega-governed regime.
No GPU, no model. Asserts expected behavior; prints a report. Pre-fire validation gate."""
import os, numpy as np, torch
os.environ.setdefault("CELL", "opt350m"); os.environ.setdefault("MODE", "screen")
import probe9b_phase2lever_lambda as P9b
P9b.dev = "cpu"  # force CPU for the dry-run
torch.manual_seed(0)
ok = True
def check(name, cond, detail=""):
    global ok; ok = ok and bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")

print("=== (1) OMEGA principal-angle overlap (shared with probe9) ===")
P, M = 400, 12
X = torch.randn(P, M, dtype=torch.float64)
c2min, c2mean = P9b.omega(X, X.clone())
check("identical subspaces cos2_min~1", abs(c2min - 1.0) < 1e-4, f"cos2_min={c2min:.4f} cos2_mean={c2mean:.4f}")
Xc = torch.zeros(P, M, dtype=torch.float64); Xp = torch.zeros(P, M, dtype=torch.float64)
Xc[:P // 2] = torch.randn(P // 2, M, dtype=torch.float64); Xp[P // 2:] = torch.randn(P - P // 2, M, dtype=torch.float64)
c2min_o, _ = P9b.omega(Xc, Xp)
check("orthogonal subspaces cos2_min~0", c2min_o < 1e-6, f"cos2_min={c2min_o:.2e}")
c2r, _ = P9b.omega(torch.randn(P, 8, dtype=torch.float64), torch.randn(P, 12, dtype=torch.float64))
check("rectangular Mc!=Mp bounded [0,1]", 0.0 <= c2r <= 1.0, f"cos2_min={c2r:.4f}")

print("=== (2) linear CKA sigma_sem ===")
A = torch.randn(64, 32, dtype=torch.float64)
check("CKA(A,A)~1", abs(P9b.linear_cka(A, A) - 1.0) < 1e-6, f"{P9b.linear_cka(A, A):.4f}")
B = torch.randn(64, 32, dtype=torch.float64)
ckaAB = P9b.linear_cka(A, B)
check("CKA(indep)~small", 0.0 <= ckaAB < 0.5, f"{ckaAB:.4f}")
# monotone: A vs (A + noise*t) decreases with t
ck_lo = P9b.linear_cka(A, A + 0.2 * torch.randn_like(A)); ck_hi = P9b.linear_cka(A, A + 3.0 * torch.randn_like(A))
check("CKA decreases as similarity drops", ck_lo > ck_hi, f"low-noise={ck_lo:.3f} high-noise={ck_hi:.3f}")

print("=== (3) PAIRED PROTECTION P(d) = mean(forget_NoCH - forget_CH) ===")
me = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)]
rng = np.random.default_rng(1); n = 100
# protective regime: CH forgets much less than NoCH -> P>0, many sigma
fn = rng.normal(0.20, 0.04, n); fc = rng.normal(0.05, 0.03, n)
P_prot = me((fn - fc).tolist())
check("protective: P>0 and >5sigma", P_prot[0] > 0 and P_prot[0] / P_prot[1] > 5, f"P={P_prot[0]:+.3f} ({P_prot[0]/P_prot[1]:.1f}s) forget_n={np.mean(fn):.3f} forget_c={np.mean(fc):.3f}")
# null regime: childhood doesn't help -> P~0
fn0 = rng.normal(0.18, 0.04, n); fc0 = rng.normal(0.18, 0.04, n)
P0 = me((fn0 - fc0).tolist())
check("null: P~0 (<2sigma)", abs(P0[0] / P0[1]) < 2, f"P={P0[0]:+.3f} ({P0[0]/P0[1]:.1f}s)")

print("=== (4) LOCKED abort-gate: span>=0.15 AND >=7 non-degenerate ===")
def gate(vals):
    vals = np.array(vals); span = float(vals.max() - vals.min())
    degen = [i for i, v in enumerate(vals) if v > 0.98 or v < 1e-4]; n_nd = len(vals) - len(degen)
    return bool(span >= 0.15 and n_nd >= 7), span, n_nd
# healthy: 9 points spanning 0.04..0.55, none degenerate
g_ok, sp, nd = gate([0.04, 0.10, 0.16, 0.22, 0.31, 0.39, 0.47, 0.52, 0.55])
check("healthy ladder passes", g_ok, f"span={sp:.3f} n_nondeg={nd}")
# fail span (probe9-screen-like cluster)
g_cl, sp2, _ = gate([0.064, 0.07, 0.072, 0.078, 0.081, 0.085, 0.088])
check("clustered ladder (probe9-like) aborts", (not g_cl), f"span={sp2:.3f}")
# fail count: spans but too many degenerate
g_dg, sp3, nd3 = gate([0.0, 0.0, 0.0, 0.0, 0.5, 0.99, 0.999])
check("degenerate-heavy ladder aborts", (not g_dg), f"span={sp3:.3f} n_nondeg={nd3}")

print("=== (5) FIXED-childhood margin-rule lr selection ===")
def select(cal, thr, margin):
    q_m = [lr for lr in cal if cal[lr] >= thr + margin]; q_b = [lr for lr in cal if cal[lr] >= thr]
    return (max(q_m) if q_m else (max(q_b) if q_b else min(cal)))
pre = 0.823; TOL = P9b.NONDESTRUCT_TOL
cal = {1e-5: 0.792, 1.5e-5: 0.778, 2e-5: 0.760, 3e-5: 0.733}
check("margin rule picks highest reliably-clean dose", select(cal, pre - TOL, P9b.CAL_MARGIN) == 1e-5,
      f"chose lr={select(cal, pre-TOL, P9b.CAL_MARGIN):.1e} (thr+m={pre-TOL+P9b.CAL_MARGIN:.3f}; only 1e-5 clears)")

print("=== (6) phase-2 lever horse-race: Spearman + partials (geometric vs Ramasesh) ===")
def rank(x):
    x = np.asarray(x, float); o = x.argsort(); r = np.empty_like(o, float); r[o] = np.arange(len(x)); return r
def spearman(x, y):
    rx, ry = rank(x), rank(y); rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))
def partial(x, y, z):  # spearman(x,y | z) via rank residuals
    rx, ry, rz = rank(x), rank(y), rank(z)
    bx = np.polyfit(rz, rx, 1); by = np.polyfit(rz, ry, 1)
    ex = rx - np.polyval(bx, rz); ey = ry - np.polyval(by, rz)
    ex -= ex.mean(); ey -= ey.mean()
    return float((ex @ ey) / (np.linalg.norm(ex) * np.linalg.norm(ey) + 1e-12))
# synthesize an Omega-GOVERNED regime (the P1/P2 prediction). 9 phase-2 tasks.
rng2 = np.random.default_rng(7)
Omega = np.array([0.04, 0.10, 0.16, 0.22, 0.31, 0.39, 0.47, 0.52, 0.55])
sigma = np.clip(0.3 * Omega + rng2.normal(0, 0.05, 9) + 0.2, 0, 1)   # sigma partly correlated with Omega (confound)
Pd = 0.9 * Omega + rng2.normal(0, 0.02, 9)                            # protection truly driven by Omega, not sigma
rho_PO = spearman(Pd, Omega); rho_Ps = spearman(Pd, sigma)
pr_PO_s = partial(Pd, Omega, sigma); pr_Ps_O = partial(Pd, sigma, Omega)
check("rho(P,Omega) high (>=0.7)", rho_PO >= 0.7, f"rho(P,Omega)={rho_PO:.3f} rho(P,sigma)={rho_Ps:.3f}")
check("partial P-Omega|sigma survives; P-sigma|Omega collapses", pr_PO_s > 0.5 and abs(pr_Ps_O) < pr_PO_s,
      f"P~Omega|sigma={pr_PO_s:.3f}  P~sigma|Omega={pr_Ps_O:.3f}")

print("=== (7) PHASE2_POOL wiring sanity ===")
check("childhood = mix5 (5 keys)", P9b.CHILD_KEYS == ["agnews", "dbpedia", "yahoo", "newscat", "emotion"], str(P9b.CHILD_KEYS))
check("phase-2 pool has >=7 entries", len(P9b.PHASE2_POOL) >= 7, f"{len(P9b.PHASE2_POOL)} entries: {list(P9b.PHASE2_POOL)}")
check("subj+trec NOT subset of childhood (true levers)", all(k not in P9b.CHILD_KEYS for k in ("subj", "trec")), "subj,trec exogenous")

print(f"\n=== DRY-RUN {'ALL PASS' if ok else 'FAILED'} ===")
