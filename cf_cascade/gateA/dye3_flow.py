"""DYE 3 — the flow field (Delta_w as velocity field over curvature-rank) + COHERENCE GATE. LOCAL analysis on
dye1_series.npz (no GPU). Wilson: 'flow' is the seductive word -> the verdict is the real-vs-shuffle coherence
NUMBER, not the streamline picture. Laminar flow = velocity field smooth/coherent across neighboring curvature
ranks; turbulent/noise = decorrelated. Gate: real spatial-coherence of the velocity field vs RANK-SHUFFLE null
(permute ranks within each timestep -> destroys spatial structure, preserves value marginal = the cascade-refutation
shuffle spirit). real >> shuffle (z>3) -> laminar flow REAL, headline holds. real ~ shuffle -> visualization sugar,
DROP the flow-field claim, fall back to static C1-C5.
"""
import numpy as np
z = np.load("dye1_series.npz")
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
snaps = z["snap_steps"]; N_SHUF = 500; rng = np.random.default_rng(0)


def spatial_coh(v):
    """mean over timesteps of lag-1 spatial autocorr (Pearson) of the velocity field across ranks,
    weighted by per-step velocity magnitude (steps with more flow count more)."""
    cohs, wts = [], []
    for t in range(v.shape[0]):
        a, b = v[t, :-1], v[t, 1:]
        if a.std() < 1e-20 or b.std() < 1e-20: continue
        c = np.corrcoef(a, b)[0, 1]
        if np.isfinite(c): cohs.append(c); wts.append(np.abs(v[t]).sum())
    if not cohs: return 0.0
    return float(np.average(cohs, weights=wts))


print(f"snaps: {snaps.tolist()}\n")
out = {}
for c in CARRIERS:
    ser = z[f"series__{c}"].astype(np.float64)            # (11, 1024) damage at SNAP steps
    seed = z[f"seed__{c}"].astype(np.float64)[None, :]    # (1, 1024) step-0
    full = np.concatenate([seed, ser], 0)                 # (12, 1024) over [0]+SNAP
    v = np.diff(full, axis=0)                             # (11, 1024) velocity = damage increments

    real = spatial_coh(v)
    null = np.empty(N_SHUF)
    for i in range(N_SHUF):
        vs = np.empty_like(v)
        for t in range(v.shape[0]): vs[t] = v[t][rng.permutation(v.shape[1])]   # shuffle ranks within each step
        null[i] = spatial_coh(vs)
    nm, ns = null.mean(), null.std()
    zsc = (real - nm) / (ns + 1e-12)
    # TEMPORAL gate (step-shuffle null): does the flow-front ADVANCE monotonically in time (traveling front),
    # or is it a static profile scaling up? com of |velocity| over rank, per step; Spearman(time, com) vs step-shuffle.
    comv = np.array([float((np.arange(v.shape[1]) * np.abs(v[t])).sum() / (np.abs(v[t]).sum() + 1e-30)) for t in range(v.shape[0])])
    def _spear(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
    real_temp = _spear(np.arange(len(comv)), comv)   # 11-point com advance (underpowered corroborator)
    # POWERED temporal gate: arrival-time ordering across the 1024 ranks (front arrives in curvature order?).
    # arrival[rank] = first snap index crossing half its over-time max; ordered front => arrival rises with rank.
    NS = full.shape[0]; arrival = np.full(full.shape[1], float(NS))   # first step rank crosses 1% of THAT step's max
    for ti in range(NS):
        crossed = full[ti] > 0.01 * full[ti].max()
        newly = crossed & (arrival == NS); arrival[newly] = ti
    wet = arrival < NS
    if wet.sum() > 10:
        ranks = np.arange(full.shape[1])[wet]; arr = arrival[wet]
        arr_real = _spear(ranks, arr)
        anull = np.array([_spear(ranks, rng.permutation(arr)) for _ in range(N_SHUF)])
        z_arr = float((arr_real - anull.mean()) / (anull.std() + 1e-12))
    else:
        arr_real, z_arr = float("nan"), float("nan")
    out[c] = {"real_coherence": real, "null_mean": float(nm), "null_std": float(ns), "z_spatial": float(zsc),
              "comv_traj": comv.tolist(), "com_advance_spearman": real_temp, "arrival_order_spearman": arr_real,
              "z_arrival": z_arr, "n_wet": int(wet.sum())}
    print(f"[{c:26s}] spatial z={zsc:+.1f} (real={real:+.3f}) | arrival-order spearman={arr_real:+.2f} z={z_arr:+.1f} (n_wet={int(wet.sum())}) | com {comv[0]:.0f}->{comv[-1]:.0f}")

# headline flow lives in the spreading carriers (qkv + up-proj); Wdown is a channel (little spread)
spread = ["attention.query_key_value", "mlp.dense_h_to_4h"]
sz = float(np.mean([out[c]["z_spatial"] for c in spread]))
stz = float(np.mean([out[c]["z_arrival"] for c in spread]))
# laminar TRAVELING flow needs BOTH: spatial smoothness (vs rank-shuffle) AND ordered front-arrival (vs arrival-shuffle)
laminar = all(out[c]["z_spatial"] > 3 and out[c]["real_coherence"] > out[c]["null_mean"] + 0.1
              and out[c]["z_arrival"] > 3 and out[c]["arrival_order_spearman"] > 0.3 for c in spread)
verdict = (f"LAMINAR TRAVELING FLOW REAL (qkv+up-proj: spatial z={sz:.1f}>>rank-shuffle AND arrival-order z={stz:.1f}>>shuffle "
           f"-> velocity field spatially coherent AND the front arrives in curvature order (high->low) -> forgetting is laminar "
           f"additive transport from the localized high-curvature seed; HEADLINE HOLDS" if laminar else
           f"PARTIAL: spatial z={sz:.1f} (laminar gated) but arrival-order z={stz:.1f} -> report what gated, don't overclaim travel")
print(f"\nspreading-carrier spatial z={sz:.1f}  arrival-order z={stz:.1f}\nVERDICT: {verdict}")
import json
json.dump({"per_carrier": out, "spreading_spatial_z": sz, "spreading_temporal_z": stz, "laminar": laminar,
           "verdict": verdict, "n_shuffle": N_SHUF}, open("dye3_flow.json", "w"), indent=2)
print("saved dye3_flow.json")
