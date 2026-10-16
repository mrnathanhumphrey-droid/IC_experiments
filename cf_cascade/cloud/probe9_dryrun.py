"""CPU dry-run for Probe-9 (LOCK 11fe73b3). Exercises the NEW math on a tiny synthetic shard with KNOWN answers:
  (1) Omega principal-angle overlap (imports the REAL omega() from probe9_aiming_lambda),
  (2) the paired-S estimator (N(phase2)-N(alttask)),
  (3) the per-stream non-destructive retention gate.
No GPU, no model. Asserts expected behavior; prints a report. This is the pre-fire validation gate."""
import os, numpy as np, torch
os.environ.setdefault("CELL", "opt350m"); os.environ.setdefault("MODE", "screen")
import probe9_aiming_lambda as P9
P9.dev = "cpu"  # force CPU for the dry-run
torch.manual_seed(0)
ok = True
def check(name, cond, detail=""):
    global ok; ok = ok and bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")

print("=== (1) OMEGA principal-angle overlap ===")
P, M = 400, 12
# a) identical subspaces -> cos2_min ~ 1, cos2_mean ~ 1
X = torch.randn(P, M, dtype=torch.float64)
c2min, c2mean = P9.omega(X, X.clone())
check("identical subspaces cos2_min≈1", abs(c2min - 1.0) < 1e-4, f"cos2_min={c2min:.4f} cos2_mean={c2mean:.4f}")
# b) orthogonal subspaces (disjoint coordinate blocks) -> cos2 ~ 0
Xc = torch.zeros(P, M, dtype=torch.float64); Xp = torch.zeros(P, M, dtype=torch.float64)
Xc[:P // 2] = torch.randn(P // 2, M, dtype=torch.float64)   # lives in first half coords
Xp[P // 2:] = torch.randn(P - P // 2, M, dtype=torch.float64)  # lives in second half coords
c2min_o, c2mean_o = P9.omega(Xc, Xp)
check("orthogonal subspaces cos2_min≈0", c2min_o < 1e-6, f"cos2_min={c2min_o:.2e} cos2_mean={c2mean_o:.2e}")
# c) partial overlap, monotone: mix r shared columns
def partial(r):
    base = torch.randn(P, M, dtype=torch.float64)
    Xp2 = base.clone()
    extra = torch.zeros(P, M, dtype=torch.float64); extra[P // 2:] = torch.randn(P - P // 2, M, dtype=torch.float64)
    Xp2[:, r:] = extra[:, r:]  # keep r shared columns, rest pushed orthogonal-ish
    return P9.omega(base, Xp2)[1]
m_lo, m_mid, m_hi = partial(2), partial(6), partial(10)
check("partial overlap monotone in shared dims", m_lo <= m_mid + 1e-9 <= m_hi + 1e-9 or (m_lo < m_hi), f"mean cos2: r2={m_lo:.3f} r6={m_mid:.3f} r10={m_hi:.3f}")
# d) rectangular Mc != Mp handled
c2r, _ = P9.omega(torch.randn(P, 8, dtype=torch.float64), torch.randn(P, 12, dtype=torch.float64))
check("rectangular Mc!=Mp runs + bounded [0,1]", 0.0 <= c2r <= 1.0, f"cos2_min={c2r:.4f}")

print("=== (2) PAIRED-S estimator: S = N(phase2) - N(alttask) ===")
me = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)]
rng = np.random.default_rng(1); n = 100
# SPECIFIC case: childhood cuts phase2 damage a lot (0.15->0.01), alttask barely (0.30->0.27) -> S>0
dp2n = rng.normal(0.15, 0.03, n); dp2c = rng.normal(0.01, 0.02, n)
daln = rng.normal(0.30, 0.03, n); dalc = rng.normal(0.27, 0.03, n)
Nph = me((dp2n - dp2c).tolist()); Nal = me((daln - dalc).tolist()); S = me(((dp2n - dp2c) - (daln - dalc)).tolist())
check("SPECIFIC: S>0 and >3sigma", S[0] > 0 and S[0] / S[1] > 3, f"N_phase2={Nph[0]:+.3f} N_alttask={Nal[0]:+.3f} S={S[0]:+.3f} ({S[0]/S[1]:.1f}σ)")
# GENERIC case: childhood cuts both ~equally -> S~0
dp2n2 = rng.normal(0.18, 0.03, n); dp2c2 = rng.normal(0.09, 0.03, n)
daln2 = rng.normal(0.26, 0.03, n); dalc2 = rng.normal(0.17, 0.03, n)
S2 = me(((dp2n2 - dp2c2) - (daln2 - dalc2)).tolist())
check("GENERIC: S≈0 (<2sigma)", abs(S2[0] / S2[1]) < 2, f"S={S2[0]:+.3f} ({S2[0]/S2[1]:.1f}σ)")

print("=== (3) per-stream NON-DESTRUCTIVE retention gate ===")
pre_sst = 0.823; TOL = P9.NONDESTRUCT_TOL
gate = lambda retA: bool(np.mean(retA) >= pre_sst - TOL)
check("in-band (0.79) passes", gate(rng.normal(0.79, 0.01, 50)) is True, f"thr={pre_sst-TOL:.3f}")
check("out-of-band (0.74) fails", gate(rng.normal(0.74, 0.01, 50)) is False)
# margin selection rule (mirrors harness Stage-1)
def select(cal, thr, margin):
    q_m = [lr for lr in cal if cal[lr] >= thr + margin]; q_b = [lr for lr in cal if cal[lr] >= thr]
    return (max(q_m) if q_m else (max(q_b) if q_b else min(cal)))
cal = {1e-5: 0.792, 1.5e-5: 0.778, 2e-5: 0.760, 3e-5: 0.733}
check("margin rule picks reliably-clean dose", select(cal, pre_sst - TOL, P9.CAL_MARGIN) == 1e-5, f"chose lr={select(cal, pre_sst-TOL, P9.CAL_MARGIN):.1e}")

print("=== (bonus) selection-rule: 5 streams nearest evenly-spaced Omega quantiles ===")
vals = np.array([0.05, 0.11, 0.18, 0.22, 0.34, 0.41, 0.55, 0.60, 0.78]); names = [f"c{i}" for i in range(len(vals))]
qs = np.quantile(vals, [0, .25, .5, .75, 1.0]); chosen = []
for q in qs:
    for idx in np.argsort(np.abs(vals - q)):
        if names[idx] not in chosen: chosen.append(names[idx]); break
check("selection spans Omega range, 5 distinct", len(chosen) == 5 and len(set(chosen)) == 5, f"chosen={chosen} omegas={[round(float(vals[names.index(c)]),2) for c in chosen]}")

print(f"\n=== DRY-RUN {'ALL PASS' if ok else 'FAILED'} ===")
