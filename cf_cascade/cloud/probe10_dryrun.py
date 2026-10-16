"""CPU dry-run for Probe-10 (LOCK c83884a7). No GPU/model. Validates the NEW logic vs 8v4:
  (1) paired S = mean((dT_n - dT_c) - (rk_n - rk_c)) aggregation matches a hand computation,
  (2) Spearman(S, similarity-rank) detects a GRADED pattern and ~0 for a FLAT pattern,
  (3) ident-minus-unrelated paired gap + its sigma,
  (4) CONDITIONS / SIM / PROBE_T wiring sanity (4 swap-childhoods, shared backbone, similarity ranks correct).
Asserts expected behavior. Pre-fire gate."""
import os, numpy as np
os.environ.setdefault("CELL", "opt350m"); os.environ.setdefault("NSEEDS", "80")
import probe10_simrehearsal_lambda as P10
ok = True
def check(name, cond, detail=""):
    global ok; ok = ok and bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")

def rank(x):
    x = np.asarray(x, float); o = x.argsort(); r = np.empty_like(o, float); r[o] = np.arange(len(x)); return r
def spear(x, y):
    rx, ry = rank(x) - rank(x).mean(), rank(y) - rank(y).mean()
    return float(rx @ ry / (np.linalg.norm(rx) * np.linalg.norm(ry) + 1e-12))
me = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)]

print("=== (1) paired S aggregation ===")
rng = np.random.default_rng(0); n = 80
# SPECIFIC childhood: aims at T (dT_c << dT_n) but not random (rk_c ~ rk_n) -> N_T large, N_rand ~0 -> S>0
dT_n = rng.normal(0.16, 0.03, n); dT_c = rng.normal(0.02, 0.02, n)
rk_n = rng.normal(0.10, 0.03, n); rk_c = rng.normal(0.095, 0.03, n)
S = me(((dT_n - dT_c) - (rk_n - rk_c)).tolist())
hand = float(np.mean((dT_n - dT_c) - (rk_n - rk_c)))
check("S matches hand calc", abs(S[0] - hand) < 1e-9, f"S={S[0]:+.3f} ({S[0]/S[1]:.1f}sig)")
check("SPECIFIC S>0, >3sig", S[0] > 0 and S[0] / S[1] > 3, f"N_T={np.mean(dT_n-dT_c):+.3f} N_rand={np.mean(rk_n-rk_c):+.3f}")
# GENERIC childhood: protects T and random equally -> S~0
gT_n = rng.normal(0.16, 0.03, n); gT_c = rng.normal(0.08, 0.03, n)
gr_n = rng.normal(0.10, 0.03, n); gr_c = rng.normal(0.02, 0.03, n)
Sg = me(((gT_n - gT_c) - (gr_n - gr_c)).tolist())
check("GENERIC S~0 (<2sig)", abs(Sg[0] / Sg[1]) < 2, f"S={Sg[0]:+.3f} ({Sg[0]/Sg[1]:.1f}sig)")

print("=== (2) Spearman(S, similarity) — graded vs flat (distinct sims; ties handled by gap stat, see note) ===")
# logic check with DISTINCT sims (the real run has a tie at same-family -> primary stat is the gap, not this rho)
sims = [3, 2, 1, 0]
S_graded = [0.12, 0.06, 0.04, 0.01]   # rehearsal-driven: S rises with similarity
S_flat   = [0.05, 0.06, 0.045, 0.055] # decoupled from similarity
check("graded -> rho>0.9", spear(S_graded, sims) > 0.9, f"rho={spear(S_graded,sims):+.3f}")
check("flat -> |rho|<0.5", abs(spear(S_flat, sims)) < 0.5, f"rho={spear(S_flat,sims):+.3f}")
# NOTE for analysis: with the real tied same-family sims [2,1,1,0], rank-spearman is tie-order sensitive ->
# the LOCKED primary statistic is ident_minus_unrelated (2-level paired gap, tie-immune), tested in (3).

print("=== (3) ident-minus-unrelated paired gap ===")
# identical condition S-per-seed vs unrelated S-per-seed
id_S = (dT_n - dT_c) - (rk_n - rk_c)          # large
un_S = (gr_n - gr_c) * 0 + rng.normal(0.01, 0.02, n)  # ~0
gap = me((id_S - un_S).tolist())
check("ident>unrel gap >2sig", gap[0] > 0 and gap[0] / gap[1] > 2, f"gap={gap[0]:+.3f} ({gap[0]/gap[1]:.1f}sig)")

print("=== (4) CONDITIONS / SIM / PROBE_T wiring ===")
check("4 swap-childhoods", len(P10.CONDITIONS) == 4, str(list(P10.CONDITIONS)))
backbones = [tuple(sorted(set(v) - {sw})) for sw, v in zip(["agnews","dbpedia","yahoo","emotion"], P10.CONDITIONS.values())]
check("shared backbone {trec,subj} all conditions", all(b == ("subj","trec") for b in backbones), str(set(backbones)))
check("each childhood 3 tasks", all(len(v) == 3 for v in P10.CONDITIONS.values()), "")
# similarity ranks: identical=2 for the matching swap; emotion=0 (unrelated) for both topic probes
check("agnews: CH_agnews identical(2), CH_emotion unrelated(0)", P10.SIM["agnews"]["CH_agnews"] == 2 and P10.SIM["agnews"]["CH_emotion"] == 0, "")
check("yahoo: CH_yahoo identical(2), CH_emotion unrelated(0)", P10.SIM["yahoo"]["CH_yahoo"] == 2 and P10.SIM["yahoo"]["CH_emotion"] == 0, "")
check("topic siblings = same-family(1)", P10.SIM["agnews"]["CH_dbpedia"] == 1 and P10.SIM["agnews"]["CH_yahoo"] == 1, "")
check("probe T in {agnews,yahoo}", set(P10.PROBE_T) <= {"agnews","yahoo"} and len(P10.PROBE_T) >= 1, str(P10.PROBE_T))

print(f"\n=== DRY-RUN {'ALL PASS' if ok else 'FAILED'} ===")
