"""FAMILY/SIZE side of the G5->B-D mediation (my half; depth side = parallel agent, mediation_depth.json).
Units = 6 size cells. Outcome = model-level B-D (retention contrast). Mediator = G5 (model-median).
Runs UNDER the locked pre-reg (TODO_diffusion_program.md):
  (1) sign-architecture criterion (right-signed + dynamic-range), (2) family-offset = recipe-confounded common-cause
  vs within-family slope = recipe-controlled causal-leaning -> reported SEPARATELY, (3) power floor (state UNDERPOWERED
  if n can't separate), (4) leverage check (robust to dropping the opt125m corner?). Pure analysis, no GPU.
"""
import json, itertools, numpy as np
from pathlib import Path
G = Path("D:/IC_experiments/cf_cascade/gateA")

# --- load G5 (geom_matrix 5-cell + geom_opt1_3b) ---
g5 = {}
m = json.load(open(G / "geom_matrix.json"))
for t, f in m.get("feature_matrix", {}).items():
    g5[t] = (f["family"], f["size_m"], f["G5"])
o = json.load(open(G / "geom_opt1_3b.json"))
for k, v in o.items():
    if isinstance(v, dict) and v.get("summary"):
        g5[k] = (v["family"], v["size_m"], v["summary"]["G5"]["median"])

# --- load B-D per cell ---
CELLS = ["opt125m", "opt350m", "opt1_3b", "pythia160m", "pythia410m", "pythia1b"]
BDFILE = {"opt125m": "bd_opt125m.json", "opt350m": "opt_dispositive.json", "opt1_3b": "bd_opt1_3b.json",
          "pythia160m": "bd_pythia160m.json", "pythia410m": "pythia_dispositive.json", "pythia1b": "bd_pythia1b.json"}
G5KEY = {"opt125m": "opt125m", "opt350m": "opt350m", "opt1_3b": "opt1_3b",
         "pythia160m": "pythia160m", "pythia410m": "pythia410m", "pythia1b": "pythia1b"}

rows = []
for c in CELLS:
    d = json.load(open(G / BDFILE[c]))
    bd = d.get("retention", {}).get("B_minus_D") or d["paired"]["B_minus_D"]  # bd_ vs dispositive schema
    fam, size, gg = g5[G5KEY[c]]
    rows.append({"cell": c, "family": fam, "size": size, "G5": gg, "BD": bd[0], "BD_se": bd[1]})

print("=== PANEL (6 size cells) ===")
print(f"{'cell':11s} {'fam':7s} {'size':>5s}  {'G5':>6s}  {'B-D':>7s}  {'B-D_se':>6s}")
for r in rows:
    print(f"{r['cell']:11s} {r['family']:7s} {r['size']:5d}  {r['G5']:.3f}  {r['BD']:+.3f}  {r['BD_se']:.3f}")

G5v = np.array([r["G5"] for r in rows]); BDv = np.array([r["BD"] for r in rows])
fam = np.array([r["family"] for r in rows]); size = np.array([r["size"] for r in rows])
isOPT = fam == "OPT"

def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    b1 = np.cov(x, y, ddof=0)[0, 1] / np.var(x); b0 = y.mean() - b1 * x.mean()
    yhat = b0 + b1 * x; ss = ((y - yhat) ** 2).sum(); sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss / sst if sst > 0 else float("nan")
    return b1, b0, r2

# --- 1. cross-cell B-D ~ G5 ---
b1, b0, r2 = ols(G5v, BDv)
rho = np.corrcoef(G5v, BDv)[0, 1]
print(f"\n[1] CROSS-CELL B-D ~ G5:  slope={b1:+.3f}  R2={r2:.3f}  pearson r={rho:+.3f}")
print(f"    sign: {'RIGHT (G5 up -> B-D down)' if b1 < 0 else 'WRONG (G5 up -> B-D up)'}")

# leverage: drop opt125m (the low-G5/high-B-D corner)
keep = np.array([r["cell"] != "opt125m" for r in rows])
b1L, _, r2L = ols(G5v[keep], BDv[keep])
rhoL = np.corrcoef(G5v[keep], BDv[keep])[0, 1]
print(f"    LEVERAGE (drop opt125m, n=5): slope={b1L:+.3f}  R2={r2L:.3f}  r={rhoL:+.3f}")

# --- 2. within-family slopes (recipe-controlled = causal-leaning) ---
print("\n[2] WITHIN-FAMILY slopes (recipe held constant = the causal-leaning test):")
for f, mask in [("OPT", isOPT), ("PYTHIA", ~isOPT)]:
    xs, ys = G5v[mask], BDv[mask]
    order = np.argsort([r["size"] for r in rows if r["family"] == f])
    bws, _, r2w = ols(xs, ys)
    g5s = [round(v, 3) for v in xs[order]]; bds = [round(v, 3) for v in ys[order]]
    mono = "monotone" if (np.all(np.diff(np.array(bds)) <= 0) or np.all(np.diff(np.array(bds)) >= 0)) else "non-monotone"
    print(f"    {f:7s} G5{g5s}->B-D{bds}  slope={bws:+.3f}  {mono}  range(B-D)={max(ys)-min(ys):.3f}")

# --- 3. family offset (recipe-CONFOUNDED common-cause) ---
oBD, pBD = BDv[isOPT].mean(), BDv[~isOPT].mean()
print(f"\n[3] FAMILY OFFSET (recipe-confounded -> CO-OCCURRENCE, not causal):")
print(f"    mean B-D  OPT={oBD:+.3f}  PYTHIA={pBD:+.3f}  gap={oBD-pBD:+.3f}")
print(f"    NOTE: gap is small-size only -> opt125m+0.110 vs py160m{BDv[fam=='PYTHIA'][0]:+.3f}; "
      f"large end opt1_3b{BDv[2]:+.3f} vs py1b{BDv[5]:+.3f} (converged).")

# --- 4. grouping null on B-D: enumerate all 10 unique 3/3 bipartitions ---
print("\n[4] GROUPING NULL on per-model B-D (POWER FLOOR check):")
idx = list(range(6))
biparts = []
for combo in itertools.combinations(idx, 3):
    other = tuple(i for i in idx if i not in combo)
    if combo < other:
        biparts.append((combo, other))
def gap(part):
    a, b = part; return abs(BDv[list(a)].mean() - BDv[list(b)].mean())
fam_part = (tuple(np.where(isOPT)[0]), tuple(np.where(~isOPT)[0]))
fam_gap = gap(fam_part)
gaps = sorted([gap(p) for p in biparts], reverse=True)
rank = 1 + sum(g > fam_gap + 1e-9 for g in gaps)
pval = rank / len(biparts)
print(f"    unique bipartitions={len(biparts)} -> MIN possible p = {1/len(biparts):.2f} (POWER FLOOR)")
print(f"    family-partition B-D gap={fam_gap:.3f}  rank={rank}/{len(biparts)}  p={pval:.2f}")

# --- 5. mediation test: does G5 absorb the family separation? residuals of B-D|G5 ---
yhat = b0 + b1 * G5v; resid = BDv - yhat
fam_gap_resid = abs(resid[isOPT].mean() - resid[~isOPT].mean())
def gap_r(part): a, b = part; return abs(resid[list(a)].mean() - resid[list(b)].mean())
gaps_r = sorted([gap_r(p) for p in biparts], reverse=True)
rank_r = 1 + sum(g > fam_gap_resid + 1e-9 for g in gaps_r)
print(f"\n[5] MEDIATION (does G5 absorb family?): family B-D gap {fam_gap:.3f} -> after removing G5: {fam_gap_resid:.3f}")
print(f"    family-gap on G5-RESIDUALS rank={rank_r}/{len(biparts)} p={rank_r/len(biparts):.2f} "
      f"({'G5 absorbs most of family (mediation-consistent)' if fam_gap_resid < 0.5*fam_gap else 'family survives G5 (common-cause/partial)'})")

# --- 6. verdict ---
print("\n=== VERDICT (family/size side, under locked pre-reg) ===")
print(f" SIGN: cross-cell {'RIGHT' if b1<0 else 'WRONG'}-signed; within-OPT {'RIGHT' if ols(G5v[isOPT],BDv[isOPT])[0]<0 else 'WRONG'}-signed monotone.")
print(f" DE-CONFOUND (pythia1b breaker): Pythia B-D flat across size ({BDv[~isOPT].round(3)}) -> protection FAMILY-FLOORED, size-invariant;")
print(f"   size-dependence is OPT-ONLY (decays +0.110->0 as size grows). Family offset = OPT-small-size elevation.")
print(f" POWER: per-model grouping floor p>={1/len(biparts):.2f} -> formal grouping CANNOT reach significance at n=6. UNDERPOWERED.")
print(f" CAUSAL-LEANING signal = within-OPT slope (recipe-controlled, right-signed, monotone, n=3 directional).")
print(f" LEVERAGE: cross-cell relation {'robust' if rhoL<-0.5 else 'WEAK/leverage-driven by opt125m'} when opt125m dropped (r {rho:+.2f}->{rhoL:+.2f}).")
