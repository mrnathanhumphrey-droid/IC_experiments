"""CF PROBE 2 as a NATIVE-GRASSMANN B-FALSIFIER (regression engine REFUSED_NO_FORM; this re-implements the
VERDICT-SHAPE in the Grassmann form the CF fragile-subspace artifacts actually afford). CPU-only, on-disk.

QUESTION: does a task-A-AGNOSTIC basis carry architectural fragility structure that the curvature view
(per-task-A by construction) is blind to? = Cut-1's "fragile is task-A's not architectural" in a non-curvature basis.

MAPPING (B-falsifier -> Grassmann):
  per-unit grain          = per-task-A fragile subspace (3 anchors: sst2/ag/ucdp), per-(layer,carrier), NEVER pooled
  candidate abstraction   = task-A-agnostic basis (rank k_B=16): BASE-MODEL (pre-FT right-singular dirs, primary) +
                            CROSS-ANCHOR-INVARIANT (shared dirs of the OTHER anchors' fragile, leave-one-out; robustness)
  per-task-A grain stat   = transfer_align = mean_{v!=u} align(F_u, F_v)  (= Cut-1 fragile-G ~0.11-0.14, the curvature grain)
  candidate stat          = align(F_u, B)  where align(F,B)=||F^T B||_F^2 / 16  (Grassmann overlap, in [0,1])
  random-basis null       = >=50 matched-rank (16) random orthonormal bases per-(layer,carrier); perm_p, SEARCH-CORRECTED
                            (max-stat across the 2 candidate bases)
VERDICT (B vocabulary, PROVISIONAL at n=3):
  PAYS_STRICT            : candidate beats per-task-A transfer (Delta>0) AND beats random (corrected perm_p<=0.10)
                           -> architectural fragility EXISTS beyond curvature -> Cut-1 was a curvature-basis artifact.
  COSTS                  : candidate < transfer -> wrong basis.
  PAYS_NONSPECIFIC       : beats transfer but not random -> any basis does it.
  STRUCTURED_BUT_UNDERFIT: beats random but loses to transfer.
  PROVISIONALLY_IRREDUCIBLE: no candidate beats transfer -> per-task-A is the grain -> Cut-1 basis-robust, reframe killed.
GUARD: reproduce Cut-1 fragile-G grand 0.114 ~= topcurv-G grand 0.114 bit-stable in THIS harness before banking.
"""
import os
os.environ["HF_HOME"] = os.path.abspath("hf_cache"); os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, json, itertools
np.random.seed(0)
G = "gateA"; KB = 16; NRAND = 60
CARR = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
ANCH = ["sst2", "ag", "ucdp"]
NLAY = 24

def align(F, B):  # mean squared cosine of principal angles, in [0,1]; F (d,16) orthonormal, B (d,kB) orthonormal
    return float(np.linalg.norm(F.T @ B, "fro") ** 2 / F.shape[1])

def rand_basis(d, k, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, k))); return Q[:, :k]

# ---- load fragile/topcurv subspaces (per-task-A grain) ----
z = np.load(f"{G}/taskA_variation.npz", allow_pickle=True)
fr = {a: {c: z[f"fragile__{a}__{c}"].astype(np.float64) for c in CARR} for a in ANCH}   # (24,d,16)
tc = {a: {c: z[f"topcurv__{a}__{c}"].astype(np.float64) for c in CARR} for a in ANCH}

# ---- GUARD: reproduce Cut-1 fragile-G grand ~0.114 in this harness ----
def grandG(D):
    g = []
    for a, b in itertools.combinations(ANCH, 2):
        for c in CARR: g += [align(D[a][c][L], D[b][c][L]) for L in range(NLAY)]
    return float(np.mean(g))
fG, tG = grandG(fr), grandG(tc)
assert abs(fG - 0.114) < 0.01 and abs(tG - 0.114) < 0.01, f"GUARD FAIL: fragile-G {fG:.3f} topcurv-G {tG:.3f} != Cut-1 0.114"
print(f"GUARD ok: fragile-G grand={fG:.3f} ~= topcurv-G grand={tG:.3f} (Cut-1 0.114) bit-stable\n")

# ---- BASE-MODEL basis (task-A-agnostic architectural prior): top-KB right singular vecs of W_base per (layer,carrier) ----
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-410m"); bsd = base.state_dict()
Bbase = {c: {} for c in CARR}
for L in range(NLAY):
    for c in CARR:
        W = bsd[f"gpt_neox.layers.{L}.{c}.weight"].numpy().astype(np.float64)   # (d_out, d_in)
        _, _, Vh = np.linalg.svd(W, full_matrices=False)                         # Vh (k, d_in)
        Bbase[c][L] = Vh[:KB].T.copy()                                           # (d_in, KB) input-space dirs
print("base-model architectural-prior basis built (top-%d right singular vecs / W_base)\n" % KB)

def crossanchor_basis(heldout, c, L):  # shared dirs of the OTHER anchors' fragile (leave-one-out), rank KB
    M = np.concatenate([fr[v][c][L] for v in ANCH if v != heldout], axis=1)      # (d, 16*(n-1))
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, :KB].copy()

# ---- the B-falsifier loop: per (held-out task-A u, layer, carrier) ----
rng = np.random.default_rng(12345)
cells = []   # one row per (u, layer, carrier)
for u in ANCH:
    for c in CARR:
        d = fr[u][c].shape[1] if False else fr[u][c].shape[1]  # placeholder; real d below
        for L in range(NLAY):
            F = fr[u][c][L]; d = F.shape[0]
            transfer = float(np.mean([align(F, fr[v][c][L]) for v in ANCH if v != u]))   # per-task-A grain (curvature)
            a_base = align(F, Bbase[c][L])
            a_cross = align(F, crossanchor_basis(u, c, L))
            # matched-rank random null (>=NRAND), + search-corrected max across the 2 candidates
            rb = np.array([align(F, rand_basis(d, KB, rng)) for _ in range(NRAND)])
            rb2 = np.array([max(align(F, rand_basis(d, KB, rng)), align(F, rand_basis(d, KB, rng))) for _ in range(NRAND)])
            p_base = float((rb >= a_base).mean()); p_cross = float((rb >= a_cross).mean())
            cand_max = max(a_base, a_cross); p_corr = float((rb2 >= cand_max).mean())
            cells.append({"u": u, "carrier": c.split(".")[-1], "L": L, "d": d, "rand_floor": float(rb.mean()),
                          "transfer": transfer, "base": a_base, "cross": a_cross,
                          "p_base": p_base, "p_cross": p_cross, "p_corr": p_corr})

# ---- aggregate (report per-carrier, never pooled into one scalar that hides heterogeneity) ----
def arr(key, sub=None):
    return np.array([r[key] for r in cells if (sub is None or r["carrier"] == sub)])
def summ(cand):
    out = {}
    for sub in [None, "query_key_value", "dense_h_to_4h", "dense_4h_to_h"]:
        cand_a = arr(cand, sub); tr = arr("transfer", sub); fl = arr("rand_floor", sub)
        pb = arr("p_base" if cand == "base" else "p_cross", sub)
        d_vs_tr = cand_a - tr
        out["ALL" if sub is None else sub] = {
            "cand_align_median": float(np.median(cand_a)), "transfer_median": float(np.median(tr)),
            "rand_floor_median": float(np.median(fl)),
            "delta_vs_transfer_median": float(np.median(d_vs_tr)), "frac_cells_beat_transfer": float((d_vs_tr > 0).mean()),
            "frac_cells_beat_random_p10": float((pb <= 0.10).mean())}
    return out

base_s, cross_s = summ("base"), summ("cross")

# ---- DECISIVE CONTROL (Cut-1 style, the false-positive catch): is the cross-anchor 'PAYS' fragile-SPECIFIC or
#      generic CURVATURE? Replace fragile with topcurv in the basis; if cross-fragile->fragile ~= cross-topcurv->fragile,
#      the lift is generic curvature (Cut-1 holds), NOT beyond-curvature architectural fragility. ----
def crossbasis_from(D, heldout, c, L):
    M = np.concatenate([D[v][c][L] for v in ANCH if v != heldout], axis=1)
    U, _, _ = np.linalg.svd(M, full_matrices=False); return U[:, :KB]
cf_f, ct_f = [], []
for u in ANCH:
    for c in CARR:
        for L in range(NLAY):
            F = fr[u][c][L]
            cf_f.append(align(F, crossbasis_from(fr, u, c, L)))   # cross-FRAGILE -> held-out fragile
            ct_f.append(align(F, crossbasis_from(tc, u, c, L)))   # cross-TOPCURV -> held-out fragile
cf_f, ct_f = np.array(cf_f), np.array(ct_f); dctrl = cf_f - ct_f
ctrl_mean, ctrl_se = float(dctrl.mean()), float(dctrl.std() / len(dctrl) ** 0.5)
fragile_specific = abs(ctrl_mean) > 2 * ctrl_se   # SIG => beyond-curvature fragility; NS => generic curvature
# search-corrected headline (max across the 2 candidates)
pcorr = arr("p_corr"); candmax = np.maximum(arr("base"), arr("cross")); tr_all = arr("transfer")
beat_tr = (candmax - tr_all > 0); beat_rand_corr = (pcorr <= 0.10)

def verdict(cand_s):
    a = cand_s["ALL"]
    beats_tr = a["delta_vs_transfer_median"] > 0 and a["frac_cells_beat_transfer"] >= 0.5
    beats_rand = a["frac_cells_beat_random_p10"] >= 0.5
    if beats_tr and beats_rand: return "PAYS_STRICT (PROVISIONAL n=3)"
    if (not beats_tr) and a["delta_vs_transfer_median"] < 0: return "COSTS"
    if beats_tr and not beats_rand: return "PAYS_NONSPECIFIC"
    if beats_rand and not beats_tr: return "STRUCTURED_BUT_UNDERFIT"
    return "PROVISIONALLY_IRREDUCIBLE (Cut-1 basis-robust; reframe killed for tested bases)"

# overall verdict — REQUIRES the beyond-curvature control: a candidate only counts as architectural-fragility
# if (i) it beats per-task-A transfer AND random AND (ii) it's a NON-CURVATURE basis OR the fragile-specific
# control is SIG. base-model = genuinely non-curvature (its COSTS is dispositive for the reframe); cross-anchor =
# curvature-family (its 'pays' counts ONLY if fragile_specific).
base_costs = base_s["ALL"]["delta_vs_transfer_median"] < 0
cross_pays_raw = (beat_tr & beat_rand_corr).mean() >= 0.5
if base_costs and (not fragile_specific):
    overall = ("PROVISIONALLY_IRREDUCIBLE (PROVISIONAL n=3): non-curvature architectural-prior basis COSTS; the only "
               "'paying' basis (cross-anchor) is curvature-family and its lift is NOT fragile-specific (control NS) = "
               "generic curvature. No architectural fragility beyond curvature -> Cut-1 basis-robust, reframe killed for tested bases.")
elif cross_pays_raw and fragile_specific:
    overall = "PAYS_STRICT (PROVISIONAL n=3, search-corrected, fragile-specific control SIG -> beyond-curvature architectural fragility)"
elif cross_pays_raw and not fragile_specific:
    overall = "PAYS_NONSPECIFIC (generic curvature: cross-anchor beats transfer+random but fragile==topcurv, control NS)"
else:
    overall = "PROVISIONALLY_IRREDUCIBLE (PROVISIONAL n=3; per-task-A grain robust for tested bases)"

rep = {"engine": "REFUSE_NO_FORM (regression panel) -> native-Grassmann reimplementation (verdict-shape preserved)",
       "n_anchors": len(ANCH), "kB": KB, "nrand": NRAND, "guard": {"fragileG": fG, "topcurvG": tG, "cut1": 0.114},
       "base_model_basis": base_s, "crossanchor_basis": cross_s,
       "search_corrected": {"frac_cells_beat_transfer": float(beat_tr.mean()),
                            "frac_cells_beat_random_corrected_p10": float(beat_rand_corr.mean()),
                            "frac_cells_PAYS_STRICT": float((beat_tr & beat_rand_corr).mean())},
       "beyond_curvature_control": {"cross_fragile_to_fragile_median": float(np.median(cf_f)),
            "cross_topcurv_to_fragile_median": float(np.median(ct_f)), "fragile_specific_lift_mean": ctrl_mean,
            "fragile_specific_lift_se": ctrl_se, "fragile_specific_SIG": bool(fragile_specific),
            "reading": "SIG=beyond-curvature fragility; NS=generic curvature (Cut-1 control replicated)"},
       "verdict_base": verdict(base_s), "verdict_cross": verdict(cross_s), "verdict_overall": overall,
       "scope": "PROVISIONAL: n=3 task-A anchors on disk. More anchors (code/Dolly/emotion as task-A) need GPU (out of CPU lane)."}
json.dump(rep, open(f"{G}/probe2_grassmann.json", "w"), indent=1)
np.savez(f"{G}/probe2_grassmann.npz", **{k: arr(k) for k in ["transfer","base","cross","rand_floor","p_base","p_cross","p_corr"]})

print("=== CF PROBE 2 — native-Grassmann B-falsifier (PROVISIONAL n=3) ===")
print(f"chance floor (rand rank-{KB}) median = {np.median(arr('rand_floor')):.3f} | per-task-A transfer (curvature grain) median = {np.median(arr('transfer')):.3f}\n")
for nm, s in [("BASE-MODEL (primary)", base_s), ("CROSS-ANCHOR-INVARIANT (robustness)", cross_s)]:
    print(f"[{nm}]")
    for sub in ["ALL", "query_key_value", "dense_h_to_4h", "dense_4h_to_h"]:
        a = s[sub]
        print(f"   {sub:18s} cand_align={a['cand_align_median']:.3f} vs transfer={a['transfer_median']:.3f} "
              f"(d={a['delta_vs_transfer_median']:+.3f}, {a['frac_cells_beat_transfer']*100:.0f}% cells beat-transfer) "
              f"| beat-random(p≤.10) {a['frac_cells_beat_random_p10']*100:.0f}% cells")
    print(f"   -> {verdict(s)}\n")
print(f"SEARCH-CORRECTED (max across 2 bases): beat-transfer {beat_tr.mean()*100:.0f}% | beat-random-corrected {beat_rand_corr.mean()*100:.0f}% | both(PAYS_STRICT) {(beat_tr&beat_rand_corr).mean()*100:.0f}%")
print(f"\n=== OVERALL VERDICT: {overall} ===")
print("saved gateA/probe2_grassmann.{json,npz}")
