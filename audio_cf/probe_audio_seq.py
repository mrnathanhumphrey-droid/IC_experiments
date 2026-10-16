"""
probe_audio_seq.py — Thread #5: does the directional-antagonism / cosine->forgetting LAW hold in AUDIO? (SANDBOXED, $0 CPU)

Third-modality triangulation of the CF arc. Frozen generic log-mel features (128-d) + linear real-vs-fake head, over 7 TTS/vocoder
FAKE paradigms (bark/elevenlabs/gtts/mms/parler/pyttsx3/speecht5) vs a shared LibriSpeech REAL pool. Three diagnostics:
  1. 7x7 discriminative-direction cosine matrix (which generators are anti-parallel vs aligned).
  2. forgetting-vs-cosine over all ordered pairs (learn A, continue-train B, measure A drop) -> Spearman rho (the LAW test).
  3. route arm: cluster by cosine, separate heads + max-ensemble vs naive single sequential head.

ISOLATION: reads audio_corpus read-only; writes only ./out/. Never touches the production audio detector / its pkl models.

Run:  python probe_audio_seq.py --smoke   # 3 paradigms, tiny
      python probe_audio_seq.py           # full
"""
from __future__ import annotations
import os, sys, glob, json, time, hashlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SMOKE = "--smoke" in sys.argv
AC = Path(os.environ.get("AC", "D:/physics_detector/audio_corpus"))
PARADIGMS = ["bark", "elevenlabs", "gtts", "mms", "parler", "pyttsx3", "speecht5"]
if SMOKE: PARADIGMS = ["bark", "gtts", "pyttsx3"]
SUBSAMPLE = 40 if SMOKE else int(os.environ.get("SUBSAMPLE", "180"))
N_MELS = 64
SR = 16000
PCA_K = 12 if SMOKE else 20
SEED = 0
OUT = HERE / "out" / ("audio_seq_smoke.json" if SMOKE else "audio_seq.json")
CACHE = HERE / "out" / "feat_cache"
CACHE.mkdir(parents=True, exist_ok=True)
np.random.seed(0)

import librosa, torch, torch.nn as nn
torch.manual_seed(0)


# ── frozen feature extractor: 64-band log-mel -> per-band mean+std over time (128-d) ──
def feat_one(path):
    try:
        y, sr = librosa.load(path, sr=SR, mono=True)
        if len(y) < SR // 2:
            y = np.pad(y, (0, SR // 2 - len(y)))
        m = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, n_fft=512, hop_length=256)
        lm = librosa.power_to_db(m + 1e-10)
        v = np.concatenate([lm.mean(1), lm.std(1)]).astype(np.float32)      # 128-d
        return v
    except Exception:
        return None


def _glob_audio(d, cap, seed=0):
    fs = sorted(f for f in glob.glob(str(d / "*")) if f.lower().endswith((".wav", ".flac", ".mp3")))
    rng = np.random.RandomState(seed); rng.shuffle(fs)
    return fs[:cap]


def extract(paths, tag):
    cf = CACHE / f"{tag}_{len(paths)}.npy"
    if cf.exists():
        return np.load(cf)
    X = []
    for i, p in enumerate(paths):
        v = feat_one(p)
        if v is not None: X.append(v)
        if (i + 1) % 60 == 0: print(f"    [{tag}] {i+1}/{len(paths)}", flush=True)
    X = np.stack(X).astype(np.float32) if X else np.zeros((0, 2 * N_MELS), np.float32)
    np.save(cf, X); return X


def _hash_split(n, tag, holdout=0.3):
    te = np.array([(int(hashlib.md5(f"{tag}{i}".encode()).hexdigest(), 16) % 1000) < holdout * 1000 for i in range(n)])
    return ~te, te


def auc(scores, y):
    y = np.asarray(y); pos = scores[y == 1]; neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0: return 0.5
    alls = np.concatenate([pos, neg]); order = alls.argsort(); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(alls) + 1)
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Head(nn.Module):
    def __init__(self, d=128): super().__init__(); self.fc = nn.Linear(d, 1)
    def forward(self, x): return self.fc(x).squeeze(1)


def _fit(head, X, y, epochs, opt, U=None):
    Xt = torch.from_numpy(X); yt = torch.from_numpy(y)
    pw = torch.tensor([float((y == 0).sum()) / max(float(y.sum()), 1)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Ut = None if U is None else torch.from_numpy(U.astype(np.float32))
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True); lossf(head(Xt), yt).backward()
        if Ut is not None:
            with torch.no_grad():
                g = head.fc.weight.grad.squeeze(0); g.sub_(Ut @ (Ut.T @ g))
        opt.step()
    return head


def train_head(X, y, epochs=300, lr=0.05, seed=0):
    torch.manual_seed(seed); h = Head(X.shape[1]); opt = torch.optim.AdamW(h.parameters(), lr=lr, weight_decay=1e-4)
    _fit(h, X, y, epochs, opt); h.eval(); return h


def continue_train(init_state, X, y, U=None, epochs=300, lr=0.05, seed=0):
    torch.manual_seed(seed); h = Head(X.shape[1]); h.load_state_dict(init_state)
    opt = torch.optim.SGD(h.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
    _fit(h, X, y, epochs, opt, U=U); h.eval(); return h


@torch.no_grad()
def scores(head, X): return torch.sigmoid(head(torch.from_numpy(X))).numpy()


def pca_basis(Xf, k):
    Xc = Xf - Xf.mean(0, keepdims=True); _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:min(k, Vt.shape[0])].T.astype(np.float32)


def discriminative_dir(Xf, Xr):
    d = Xf.mean(0) - Xr.mean(0); n = np.linalg.norm(d)
    return (d / n).astype(np.float32) if n > 0 else d.astype(np.float32)


def orthonormalize(U):
    Q, _ = np.linalg.qr(U); return Q


def main():
    t0 = time.time()
    print(f"[AUDIO] paradigms={PARADIGMS} SUBSAMPLE={SUBSAMPLE} log-mel {N_MELS}->128d", flush=True)

    # shared REAL pool (LibriSpeech), split once
    real_paths = _glob_audio(AC / "real" / "LibriSpeech", SUBSAMPLE * 2, SEED)
    if len(real_paths) < 20:
        real_paths = [f for f in glob.glob(str(AC / "real" / "**" / "*"), recursive=True)
                      if f.lower().endswith((".wav", ".flac", ".mp3"))][:SUBSAMPLE * 2]
    Xr = extract(real_paths, "real")
    rtr, rte = _hash_split(len(Xr), "real")
    print(f"[AUDIO] real: {len(Xr)} (tr={rtr.sum()} te={rte.sum()})", flush=True)

    P = {}
    for p in PARADIGMS:
        Xf = extract(_glob_audio(AC / "synth" / p, SUBSAMPLE, SEED), p)
        ftr, fte = _hash_split(len(Xf), p)
        P[p] = {"Xf": Xf, "ftr": ftr, "fte": fte,
                "Xtr": np.concatenate([Xf[ftr], Xr[rtr]]), "ytr": np.concatenate([np.ones(ftr.sum()), np.zeros(rtr.sum())]).astype(np.float32),
                "Xte": np.concatenate([Xf[fte], Xr[rte]]), "yte": np.concatenate([np.ones(fte.sum()), np.zeros(rte.sum())]).astype(np.float32),
                "dir": discriminative_dir(Xf[ftr], Xr[rtr])}
        print(f"  {p:10} fake={len(Xf)} dir||={np.linalg.norm(Xf[ftr].mean(0)-Xr[rtr].mean(0)):.1f}", flush=True)

    # ── D1: 7x7 discriminative-direction cosine matrix ──
    names = PARADIGMS
    M = np.array([[float(P[a]["dir"] @ P[b]["dir"]) for b in names] for a in names])
    print("\n=== D1: discriminative-direction cosine matrix ===", flush=True)
    print("  " + "".join(n[:6].rjust(8) for n in names), flush=True)
    for i, a in enumerate(names):
        print(f"  {a[:8].ljust(8)}" + "".join(f"{M[i,j]:+7.2f} " for j in range(len(names))), flush=True)
    off = M[~np.eye(len(names), dtype=bool)]
    print(f"  off-diagonal cos: min={off.min():+.2f} max={off.max():+.2f} mean={off.mean():+.2f}", flush=True)

    # ── D2: forgetting vs cosine over all ordered pairs ──
    print("\n=== D2: forgetting vs cosine (learn A -> +B) ===", flush=True)
    base_auc = {p: auc(scores(train_head(P[p]["Xtr"], P[p]["ytr"], seed=SEED), P[p]["Xte"]), P[p]["yte"]) for p in names}
    heads = {p: train_head(P[p]["Xtr"], P[p]["ytr"], seed=SEED) for p in names}
    states = {p: {k: v.clone() for k, v in heads[p].state_dict().items()} for p in names}
    pairs = []
    for a in names:
        for b in names:
            if a == b: continue
            h2 = continue_train(states[a], P[b]["Xtr"], P[b]["ytr"], U=None, seed=SEED)
            a_after = auc(scores(h2, P[a]["Xte"]), P[a]["yte"])
            forget = base_auc[a] - a_after
            cos_ab = float(P[a]["dir"] @ P[b]["dir"])
            pairs.append({"A": a, "B": b, "cos": round(cos_ab, 3), "A_base": round(base_auc[a], 3),
                          "A_after": round(a_after, 3), "forget": round(forget, 3)})
    cs = np.array([p["cos"] for p in pairs]); fg = np.array([p["forget"] for p in pairs])
    # Spearman rho (rank corr) by hand
    def spearman(x, y):
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        rx = rx - rx.mean(); ry = ry - ry.mean()
        return float((rx @ ry) / (np.sqrt(rx @ rx) * np.sqrt(ry @ ry) + 1e-30))
    rho = spearman(cs, fg); pear = float(np.corrcoef(cs, fg)[0, 1])
    worst = sorted(pairs, key=lambda p: -p["forget"])[:5]
    best = sorted(pairs, key=lambda p: p["forget"])[:3]
    print(f"  n_pairs={len(pairs)}  Spearman rho(cos,forget)={rho:+.3f}  Pearson={pear:+.3f}", flush=True)
    print(f"  base AUCs: " + " ".join(f"{p}={base_auc[p]:.2f}" for p in names), flush=True)
    print("  most-forgotten (A after +B):", flush=True)
    for p in worst: print(f"    {p['A']:9}+{p['B']:9} cos={p['cos']:+.2f} {p['A_base']:.2f}->{p['A_after']:.2f} forget={p['forget']:+.3f}", flush=True)

    # ── D3: route arm (cluster by cosine TAU=0) vs naive single sequential head ──
    print("\n=== D3: route (cluster by cosine) vs naive sequential ===", flush=True)
    order = names
    # naive: one head, sequential over order
    h = train_head(P[order[0]]["Xtr"], P[order[0]]["ytr"], seed=SEED); st = {k: v.clone() for k, v in h.state_dict().items()}
    for p in order[1:]:
        h = continue_train(st, P[p]["Xtr"], P[p]["ytr"], U=None, seed=SEED); st = {k: v.clone() for k, v in h.state_dict().items()}
    naive_ret = {p: round(auc(scores(h, P[p]["Xte"]), P[p]["yte"]), 3) for p in names}
    # route: greedy cluster by cosine, one head per cluster (union train), max-ensemble score
    clusters = []  # list of {members, dir(accum), Xtr, ytr}
    for p in order:
        j = -1; best_c = -2
        for ci, c in enumerate(clusters):
            cc = float(P[p]["dir"] @ c["dir"])
            if cc > best_c: best_c, j = cc, ci
        if j >= 0 and best_c > 0.0:
            c = clusters[j]; c["members"].append(p)
            c["Xtr"] = np.concatenate([c["Xtr"], P[p]["Xtr"]]); c["ytr"] = np.concatenate([c["ytr"], P[p]["ytr"]])
            c["Xf_accum"] = np.concatenate([c["Xf_accum"], P[p]["Xf"][P[p]["ftr"]]])
            c["dir"] = discriminative_dir(c["Xf_accum"], Xr[rtr])
        else:
            clusters.append({"members": [p], "dir": P[p]["dir"].copy(), "Xtr": P[p]["Xtr"].copy(),
                             "ytr": P[p]["ytr"].copy(), "Xf_accum": P[p]["Xf"][P[p]["ftr"]].copy()})
    chs = [train_head(c["Xtr"], c["ytr"], seed=SEED) for c in clusters]
    route_ret = {p: round(auc(np.stack([scores(hh, P[p]["Xte"]) for hh in chs], 1).max(1), P[p]["yte"]), 3) for p in names}
    print(f"  clusters: " + "; ".join(f"C{i}={c['members']}" for i, c in enumerate(clusters)), flush=True)
    print("  " + "para".ljust(11) + "".join(n[:6].rjust(8) for n in names), flush=True)
    print("  " + "naive".ljust(11) + "".join(f"{naive_ret[n]:8.2f}" for n in names), flush=True)
    print("  " + "route".ljust(11) + "".join(f"{route_ret[n]:8.2f}" for n in names), flush=True)
    worst_naive = min(naive_ret.values()); worst_route = min(route_ret.values())
    print(f"  worst-paradigm retention: naive={worst_naive:.3f}  route={worst_route:.3f}  (route-naive={worst_route-worst_naive:+.3f})", flush=True)

    OUT.write_text(json.dumps({
        "config": {"paradigms": names, "subsample": SUBSAMPLE, "smoke": SMOKE, "prereg_lock": "17a22acb"},
        "D1_cosine_matrix": M.round(3).tolist(),
        "D2_law": {"spearman_rho": rho, "pearson": pear, "base_auc": {k: round(v, 3) for k, v in base_auc.items()},
                   "pairs": pairs},
        "D3_route": {"clusters": [c["members"] for c in clusters], "naive": naive_ret, "route": route_ret,
                     "worst_naive": worst_naive, "worst_route": worst_route}}, indent=1))
    print(f"\nwrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
