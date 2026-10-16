"""SEED-vs-CONTESTED-CORE overlap (the unification measurement). Is the flow's SEED (where forgetting leaks at
step 0, dye1) the same subspace as the CONTESTED CORE (where tasks compete, deflation contest)?

KEY GEOMETRY (verified from dye1_lambda.py + deflation_lambda.py): both objects are profiles over the SAME per-layer
Fisher eigenbasis (identical FA/checkpoint/fisher()). dye-rank r (0=highest curv) == contest-index j=63-r for r in
[0,63]. => a principal-angle Grassmann is VACUOUS (axis-aligned subsets of one shared orthonormal frame -> angles 0/90
-> Grassmann == index overlap identically). So the spec's "Grassmann-vs-index disagreement" check CANNOT fire here;
substitute two genuinely-independent corroborators (binary index overlap + curvature-CONTROLLED alignment).

CONFOUND GUARD: both seed and contest decay with curvature rank, so a raw cosine vs rank-shuffle reads high from the
shared monotone trend alone (= re-measures deflation, not unification). DECISIVE corroborator = curvature-controlled:
partial Spearman(seed, contest | eigenvalue) + seed energy on the DISSOCIATION dirs (high-contest & low-curvature).

PRE-COMMITTED VERDICT (per carrier + grand):
 UNIFICATION : index overlap >> chance (>=2x within-band) AND curvature-CONTROLLED alignment survives (partial
               spearman z>3 / seed loads the dissociation dirs above matched-random) -> seed = contested core, beyond
               just curvature. One mechanism, three views; deflation redeemed (core = seed of the novel flow).
 DISTINCT    : index overlap ~ chance AND partial alignment ~ 0 -> leak seeds in high-curv dirs that are NOT the
               contested ones; conditional-core intervention would be aimed wrong.
 PARTIAL     : split / curvature-only (raw high but partial~0) -> report shared fraction + seed-specific margin
               (the dirs that leak but aren't contested = the interesting residual).
Saves all profiles + stats to seed_core_overlap.npz BEFORE the verdict print.
"""
import numpy as np, json

CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
KTOP = 64; CORE = 16; NPERM = 5000
rng = np.random.default_rng(0)
dye = np.load("dye1_series.npz"); defl = np.load("deflation_artifacts.npz")


def pr(x):
    x = np.asarray(x, float); s1 = x.sum(); s2 = (x * x).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def spear(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def partial_spear(a, b, ctrl):
    ra, rb, rc = [np.argsort(np.argsort(x)).astype(float) for x in (a, b, ctrl)]
    def resid(y, x):
        x1 = np.c_[np.ones_like(x), x]; beta = np.linalg.lstsq(x1, y, rcond=None)[0]; return y - x1 @ beta
    return float(np.corrcoef(resid(ra, rc), resid(rb, rc))[0, 1])


arts = {}; results = {}
for c in CARRIERS:
    seed = dye[f"seed__{c}"].astype(np.float64)                       # (1024,) dye-rank, 0=highest curv
    ck = [k for k in defl.files if k.startswith("contest__") and c in k]
    lk = [k for k in defl.files if k.startswith("lam__") and c in k]
    contest = np.mean([defl[k].astype(np.float64)[::-1] for k in ck], axis=0)   # (64,) -> dye-rank coords
    lam = np.mean([defl[k].astype(np.float64)[::-1] for k in lk], axis=0)       # (64,) eigenvalues, dye-rank coords
    seed64 = seed[:KTOP]

    # ---- (a) binary index overlap ----
    core16 = set(np.argsort(contest)[-CORE:].tolist())                # contested core ranks (dye-coords, in 0..63)
    seed_band16 = set(np.argsort(seed64)[-CORE:].tolist())            # seed's top-16 WITHIN the high-curv band
    seed_glob16 = set(np.argsort(seed)[-CORE:].tolist())             # seed's top-16 of all 1024
    ovl_band = len(seed_band16 & core16) / CORE
    ovl_glob = len(seed_glob16 & core16) / CORE
    nb = np.array([len(set(rng.choice(KTOP, CORE, replace=False)) & core16) for _ in range(NPERM)]) / CORE
    na = np.array([len(set(rng.choice(1024, CORE, replace=False)) & core16) for _ in range(NPERM)]) / CORE
    z_band = (ovl_band - nb.mean()) / (nb.std() + 1e-12)
    z_glob = (ovl_glob - na.mean()) / (na.std() + 1e-12)

    # ---- raw alignment over shared 64 ranks (confounded by curvature trend; reported for contrast) ----
    real_cos = cos(seed64, contest); real_spear = spear(seed64, contest)
    null_cos = np.array([cos(seed64, rng.permutation(contest)) for _ in range(NPERM)])
    z_cos = (real_cos - null_cos.mean()) / (null_cos.std() + 1e-12)

    # ---- (b) curvature-CONTROLLED alignment (decisive) ----
    pspear = partial_spear(seed64, contest, lam)                      # seed~contest | eigenvalue
    null_p = np.array([partial_spear(rng.permutation(seed64), contest, lam) for _ in range(NPERM)])
    z_pspear = (pspear - null_p.mean()) / (null_p.std() + 1e-12)
    # dissociation dirs = contested core that is LOW curvature (bottom half of band: dye ranks 32..63)
    lowcurv = set(range(KTOP // 2, KTOP))
    dissoc = sorted(core16 & lowcurv)
    e_total = seed.sum()
    if dissoc:
        e_dissoc = seed64[dissoc].sum() / e_total                    # seed energy on high-contest-low-curv dirs
        # matched-random null: same # of dirs drawn from the low-curv band
        nd = np.array([seed64[rng.choice(sorted(lowcurv), len(dissoc), replace=False)].sum() / e_total
                       for _ in range(NPERM)])
        z_dissoc = (e_dissoc - nd.mean()) / (nd.std() + 1e-12)
    else:
        e_dissoc, z_dissoc = 0.0, float("nan")

    # ---- seed-specific margin / shared fraction ----
    e_core = seed64[sorted(core16)].sum() / e_total                  # seed energy ON the contested core
    e_band = seed64.sum() / e_total                                  # seed energy in the top-64 band at all
    e_outside_band = 1.0 - e_band                                    # leaks at low curvature, can't be contested
    e_band_noncore = e_band - e_core                                 # high-curv but not contested

    results[c] = dict(
        seed_PR=pr(seed), seed_band_PR=pr(seed64), n_dissoc=len(dissoc),
        ovl_band=ovl_band, chance_band=float(nb.mean()), z_band=float(z_band),
        ovl_glob=ovl_glob, chance_glob=float(na.mean()), z_glob=float(z_glob),
        raw_cos=real_cos, raw_spear=real_spear, z_cos_vs_shuffle=float(z_cos),
        partial_spear_ctrl_curv=pspear, z_partial_spear=float(z_pspear), seed_E_on_dissoc=float(e_dissoc), z_dissoc=float(z_dissoc),
        seed_E_on_core=float(e_core), seed_E_band_noncore=float(e_band_noncore), seed_E_outside_band=float(e_outside_band))
    arts[f"seed64__{c}"] = seed64; arts[f"contest__{c}"] = contest; arts[f"lam__{c}"] = lam
    arts[f"core16__{c}"] = np.array(sorted(core16))

np.savez("seed_core_overlap.npz", **arts)   # SAVE BEFORE READOUT

# ---- readout + verdict ----
print(f"{'carrier':28s} {'seedPR':>6s} {'ovl_band':>8s}(z) {'raw_cos':>7s}(z) {'pSpear|c':>8s} {'E_diss':>6s}(z) {'E_core':>6s} {'E_out':>6s}")
for c in CARRIERS:
    r = results[c]
    print(f"{c:28s} {r['seed_PR']:6.1f} {r['ovl_band']:6.2f}({r['z_band']:+4.1f}) {r['raw_cos']:6.2f}({r['z_cos_vs_shuffle']:+4.1f}) "
          f"{r['partial_spear_ctrl_curv']:+5.2f}(z{r['z_partial_spear']:+4.1f}) {r['seed_E_on_core']:5.2f} {r['seed_E_outside_band']:5.2f}")

g = lambda k: float(np.mean([results[c][k] for c in CARRIERS]))
gov, gz_band, graw, gpart, gzp, gcore = g('ovl_band'), g('z_band'), g('raw_cos'), g('partial_spear_ctrl_curv'), g('z_partial_spear'), g('seed_E_on_core')
unified = (gov >= 0.5 and gz_band > 3) and (gpart > 0.2 and gzp > 3)
distinct = (gz_band < 3) and (gzp < 3)
if unified:
    verdict = (f"UNIFICATION: seed overlaps contested core {gov:.2f} (z={gz_band:+.1f}>>chance 0.25) AND survives curvature "
               f"control (partial spearman {gpart:+.2f} z={gzp:+.1f}) -> seed = contested core beyond just curvature. "
               f"ONE mechanism (diffusion/contested-core/curvature = three views). Deflation REDEEMED.")
elif distinct:
    verdict = (f"DISTINCT: overlap z={gz_band:+.1f}~chance AND curvature-controlled alignment ~0 (partial {gpart:+.2f} z={gzp:+.1f}) "
               f"-> leak seeds in high-curv dirs that are NOT the contested ones. Conditional-core intervention aimed wrong.")
else:
    verdict = (f"PARTIAL / CURVATURE-MEDIATED: index overlap {gov:.2f}(z{gz_band:+.1f}) is STRONG but the curvature-controlled "
               f"partial is weak (spearman {gpart:+.2f} z={gzp:+.1f}) -> seed=core overlap is LARGELY the shared curvature trend, "
               f"weak/borderline beyond it. Shared seed-energy on core {gcore:.2f}; seed-specific margin {1-gcore:.2f} (carrier-graded). "
               f"Seed & core are ~the same high-curvature object; beyond-curvature fine alignment is the open margin.")
print(f"\nGRAND: ovl_band={gov:.2f}(z{gz_band:+.1f})  raw_cos={graw:.2f}  partial_spear|curv={gpart:+.2f}(z{gzp:+.1f})  seed_E_on_core={gcore:.2f}")
print(f"VERDICT: {verdict}")
json.dump({"per_carrier": results, "grand": {"ovl_band": gov, "z_band": gz_band, "raw_cos": graw,
           "partial_spear_ctrl_curv": gpart, "z_partial_spear": gzp, "seed_E_on_core": gcore},
           "verdict": verdict, "nperm": NPERM, "note": "Grassmann vacuous (shared eigenbasis); index-overlap + curvature-controlled corroborators used"},
          open("seed_core_overlap.json", "w"), indent=2)
print("saved seed_core_overlap.json + .npz")
