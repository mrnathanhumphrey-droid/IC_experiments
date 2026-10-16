"""
probe_detector_seq.py — SEQUENTIAL-PARADIGM-SURGE CF probe on the deepfake detector (SANDBOXED).

Extends cl_gpm_diffusion.py (pairwise swap<->diffusion) to an ORDERED N-paradigm surge with a full RETENTION MATRIX:
train the readout head on paradigm 1, then continue-train on 2, 3, ... and after EACH add re-measure detection AUC on
EVERY prior paradigm. Three arms per step: NAIVE / GPM (project off the accumulated prior-paradigm memory subspace) /
GPM-RANDCTRL (random subspace, the mandatory control). Answers: does adding paradigm-k erode paradigms 1..k-1, is it
directional (GPM beats random), and does the antagonism COMPOUND across a real sequence?

ISOLATION (hard): frozen backbone prod_swap_seed0.pt COPIED into this sandbox (read-only). Paradigm crops GLOBBED read-only
from D:/corpus/face_swap_corpus (too big to copy). Writes ONLY to ./out/. Never retrains/touches the live prod_swap_seed*.pt
or the served detector. $0 LOCAL CPU (frozen 128-d features + a linear head).

Design (faithful to cl_gpm_diffusion): SHARED real pool (label 0) across all paradigm tasks so the ONLY thing differing
between tasks is the FAKE paradigm. Metric = AUC. Split = deterministic filename-hash (see caveats: not identity-disjoint;
forgetting DELTAS on a fixed test set are robust to absolute-level inflation).

Run:  python probe_detector_seq.py --smoke   # tiny (SUBSAMPLE=40, 1 arm sanity)
      python probe_detector_seq.py           # full
"""
from __future__ import annotations
import os, sys, glob, time, json, hashlib
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from detector_net import DetectorCNN
from detector_data import residual, jpeg

DEVICE = "cpu"; torch.set_num_threads(os.cpu_count() or 4)
INPUT = 160; FIXED_Q = 100; SEED = 0
SMOKE = "--smoke" in sys.argv
SUBSAMPLE = 40 if SMOKE else int(os.environ.get("SUBSAMPLE", "220"))
PCA_K = int(os.environ.get("PCA_K", "20"))
CORPUS = Path(os.environ.get("CORPUS", "D:/corpus/face_swap_corpus"))
# ordered paradigm sequence: (name, fake-dir). swap=ffpp, diffusion=deepfakeface (the known cos-0.95 antagonist),
# celebdf + dfdc = in-wild swaps (domain-shifted). SHARED reals = ffpp/real.
PARADIGMS = [("swap", "ffpp"), ("diffusion", "deepfakeface"), ("celebdf", "celebdf"), ("dfdc", "dfdc")]
if SMOKE: PARADIGMS = PARADIGMS[:2]
SHARED_REAL_GLOB = str(CORPUS / "ffpp" / "real" / "*.png")
OUT = HERE / "out" / ("seq_detector_smoke.json" if SMOKE else "seq_detector.json")
torch.manual_seed(0); np.random.seed(0)


def load_backbone():
    m = DetectorCNN(in_ch=1, width=32)
    sd = torch.load(str(HERE / "prod_models" / "prod_swap_seed0.pt"), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    m.load_state_dict(sd); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m


@torch.no_grad()
def feat(m, x):
    return m.pool(m.features(x)).flatten(1)


def crop_tensor(path):
    img = cv2.imread(path)
    if img is None: img = np.zeros((INPUT, INPUT, 3), np.uint8)
    res = cv2.resize(residual(jpeg(img, FIXED_Q)), (INPUT, INPUT), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(res).unsqueeze(0).float()


def _hash_split(paths, holdout=0.3):
    tr, te = [], []
    for p in paths:
        (te if (int(hashlib.md5(os.path.basename(p).encode()).hexdigest(), 16) % 1000) < holdout * 1000 else tr).append(p)
    return tr, te


def _glob_sub(pattern, cap, seed=0):
    paths = sorted(glob.glob(pattern)); rng = np.random.RandomState(seed); rng.shuffle(paths)
    return paths[:cap]


@torch.no_grad()
def extract(backbone, paths, label):
    feats = []
    for i in range(0, len(paths), 64):
        xs = torch.stack([crop_tensor(p) for p in paths[i:i + 64]])
        feats.append(feat(backbone, xs).numpy())
    X = np.concatenate(feats, 0).astype(np.float32) if feats else np.zeros((0, 128), np.float32)
    return X, np.full(len(paths), float(label), np.float32)


def auc(scores, y):
    y = np.asarray(y); pos = scores[y == 1]; neg = scores[y == 0]
    if len(pos) == 0 or len(neg) == 0: return 0.5
    # rank-based AUC
    alls = np.concatenate([pos, neg]); order = alls.argsort(); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(alls) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class Head(nn.Module):
    def __init__(self, d=128): super().__init__(); self.fc = nn.Linear(d, 1)
    def forward(self, x): return self.fc(x).squeeze(1)


def _fit(head, X, y, epochs, lr, opt):
    Xt = torch.from_numpy(X); yt = torch.from_numpy(y)
    pw = torch.tensor([float((y == 0).sum()) / max(float(y.sum()), 1)])
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True); loss = lossf(head(Xt), yt); loss.backward()
        yield head, opt, loss


def train_head(X, y, epochs=300, lr=0.05, seed=0):
    torch.manual_seed(seed); head = Head(X.shape[1]); opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    for head, opt, _ in _fit(head, X, y, epochs, lr, opt): opt.step()
    head.eval(); return head


def continue_train(init_state, X, y, U=None, epochs=300, lr=0.05, seed=0):
    torch.manual_seed(seed); head = Head(X.shape[1]); head.load_state_dict(init_state)
    opt = torch.optim.SGD(head.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
    Ut = None if U is None else torch.from_numpy(U.astype(np.float32))
    for head, opt, _ in _fit(head, X, y, epochs, lr, opt):
        if Ut is not None:
            with torch.no_grad():
                g = head.fc.weight.grad.squeeze(0); g.sub_(Ut @ (Ut.T @ g))
        opt.step()
    head.eval(); return head


@torch.no_grad()
def scores(head, X): return torch.sigmoid(head(torch.from_numpy(X))).numpy()


def pca_basis(Xfake, k):
    Xc = Xfake - Xfake.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Vt[:min(k, Vt.shape[0])].T.astype(np.float32)  # (128,k) orthonormal cols


def rand_basis(d, k, seed=0):
    Q, _ = np.linalg.qr(np.random.RandomState(seed).randn(d, k).astype(np.float32)); return Q[:, :k]


def orthonormalize(U):
    Q, _ = np.linalg.qr(U); return Q


def discriminative_dir(Xfake, Xreal):
    """The direction a linear head learns: normalize(mean fake - mean real). Anti-parallelism of these = the CF mechanism."""
    d = Xfake.mean(0) - Xreal.mean(0)
    n = np.linalg.norm(d)
    return (d / n).astype(np.float32) if n > 0 else d.astype(np.float32)


TAU = float(os.environ.get("ROUTE_TAU", "0.0"))  # join a head if cos(new_dir, head_dir) > TAU, else spawn a new head


def run_route_arm(bb, P, names, Xr_tr, intra_gpm=False):
    """ROUTE arm: sequentially route each arriving paradigm to a compatible head (data-driven, cos of fake-vs-real
    direction) or spawn a new head. Score two ways on the SAME retention test sets: oracle (head that owns p) and
    max-ensemble (label-free, max sigmoid over heads).
    intra_gpm=False: joining a head = plain continue-train (no projection). intra_gpm=True: joining a head projects the
    surge gradient orthogonal to the head's PRIOR members' memory subspace (GPM WITHIN the head) → protects that head's
    earlier paradigms. A spawned (first-in-head) paradigm trains fresh regardless. Diffusion, alone in its head, is never
    projected either way."""
    heads = []            # list of dicts: {state, dir, members:[names], Xfake_accum, member_fakes:[arrays]}
    assign = {}           # paradigm name -> head index
    log = []              # router decision log
    for i, name in enumerate(names):
        d_new = discriminative_dir(P[name]["Xfake_tr"], Xr_tr)
        csims = [float(d_new @ h["dir"]) for h in heads]
        j = int(np.argmax(csims)) if csims else -1
        if j >= 0 and csims[j] > TAU:                                   # JOIN existing head (continue-train)
            h = heads[j]
            U = None
            if intra_gpm:                                              # protect this head's PRIOR members from the surge
                U = orthonormalize(np.concatenate([pca_basis(mf, PCA_K) for mf in h["member_fakes"]], axis=1))
            head = continue_train(h["state"], P[name]["Xtr"], P[name]["ytr"], U=U, seed=SEED)
            h["members"].append(name); h["member_fakes"].append(P[name]["Xfake_tr"].copy())
            h["Xfake_accum"] = np.concatenate([h["Xfake_accum"], P[name]["Xfake_tr"]])
            h["dir"] = discriminative_dir(h["Xfake_accum"], Xr_tr)
            h["state"] = {k: v.clone() for k, v in head.state_dict().items()}
            assign[name] = j
            log.append({"added": name, "action": "join", "head": j, "cos": round(csims[j], 3),
                        "gpm": bool(intra_gpm), "all_cos": [round(c, 3) for c in csims]})
        else:                                                           # SPAWN a new head (fresh train)
            head = train_head(P[name]["Xtr"], P[name]["ytr"], seed=SEED)
            heads.append({"state": {k: v.clone() for k, v in head.state_dict().items()}, "dir": d_new,
                          "members": [name], "Xfake_accum": P[name]["Xfake_tr"].copy(),
                          "member_fakes": [P[name]["Xfake_tr"].copy()]})
            assign[name] = len(heads) - 1
            log.append({"added": name, "action": "spawn", "head": len(heads) - 1,
                        "all_cos": [round(c, 3) for c in csims]})

    def head_obj(hstate):
        h = Head(128); h.load_state_dict(hstate); h.eval(); return h

    hobjs = [head_obj(h["state"]) for h in heads]

    def oracle_retention():
        out = {}
        for name in names:
            h = hobjs[assign[name]]
            out[name] = round(auc(scores(h, P[name]["Xte"]), P[name]["yte"]), 4)
        return out

    def maxens_retention():
        out = {}
        for name in names:
            X = P[name]["Xte"]
            smat = np.stack([scores(h, X) for h in hobjs], 1)   # (n, n_heads) sigmoids
            out[name] = round(auc(smat.max(1), P[name]["yte"]), 4)
        return out

    return {"router_log": log, "n_heads": len(heads),
            "heads": [{"head": k, "members": h["members"]} for k, h in enumerate(heads)],
            "oracle": oracle_retention(), "maxens": maxens_retention()}


def main():
    t0 = time.time()
    print(f"[DET-SEQ] paradigms={[p[0] for p in PARADIGMS]} SUBSAMPLE={SUBSAMPLE} PCA_K={PCA_K} dev={DEVICE}", flush=True)
    bb = load_backbone(); print("[DET-SEQ] frozen backbone loaded", flush=True)

    # shared reals (label 0), split once, used for EVERY paradigm task
    rall = _glob_sub(SHARED_REAL_GLOB, SUBSAMPLE * 2, SEED)
    r_tr, r_te = _hash_split(rall)
    Xr_tr, yr_tr = extract(bb, r_tr, 0); Xr_te, yr_te = extract(bb, r_te, 0)
    print(f"[DET-SEQ] shared reals: tr={len(r_tr)} te={len(r_te)}", flush=True)

    # per-paradigm fake features (train/test), + assembled task matrices
    P = {}
    for name, d in PARADIGMS:
        f_tr, f_te = _hash_split(_glob_sub(str(CORPUS / d / "fake" / "*.png"), SUBSAMPLE, SEED))
        Xf_tr, yf_tr = extract(bb, f_tr, 1); Xf_te, yf_te = extract(bb, f_te, 1)
        P[name] = {
            "Xtr": np.concatenate([Xf_tr, Xr_tr]), "ytr": np.concatenate([yf_tr, yr_tr]),
            "Xte": np.concatenate([Xf_te, Xr_te]), "yte": np.concatenate([yf_te, yr_te]),
            "Xfake_tr": Xf_tr, "n_tr": len(f_tr), "n_te": len(f_te)}
        print(f"  {name:10} fake tr={len(f_tr)} te={len(f_te)}", flush=True)

    def retention_row(head):
        return {name: round(auc(scores(head, P[name]["Xte"]), P[name]["yte"]), 4) for name, _ in PARADIGMS}

    names = [p[0] for p in PARADIGMS]

    def run_arm(arm):
        """arm in {naive, gpm, rand}. Returns list of retention rows, one per step (after adding each paradigm)."""
        # step 0: train on paradigm 0
        head = train_head(P[names[0]]["Xtr"], P[names[0]]["ytr"], seed=SEED)
        rows = [{"added": names[0], "retention": retention_row(head)}]
        state = {k: v.clone() for k, v in head.state_dict().items()}
        for i in range(1, len(names)):
            new = names[i]
            U = None
            if arm in ("gpm", "rand"):
                if arm == "gpm":
                    U = orthonormalize(np.concatenate([pca_basis(P[names[j]]["Xfake_tr"], PCA_K) for j in range(i)], axis=1))
                else:
                    U = rand_basis(128, min(PCA_K * i, 127), seed=SEED)
            head = continue_train(state, P[new]["Xtr"], P[new]["ytr"], U=U, seed=SEED)
            rows.append({"added": new, "retention": retention_row(head)})
            state = {k: v.clone() for k, v in head.state_dict().items()}
        return rows

    print("\n=== SEQUENTIAL SURGE (retention AUC after each add) ===", flush=True)
    result = {}
    for arm in (["naive"] if SMOKE else ["naive", "gpm", "rand"]):
        rows = run_arm(arm); result[arm] = rows
        print(f"\n[{arm.upper()}]  (rows = after adding each paradigm; cols = retention on each)", flush=True)
        print("  " + "added".ljust(11) + " ".join(n.rjust(9) for n in names), flush=True)
        for r in rows:
            print("  " + r["added"].ljust(11) + " ".join(f"{r['retention'][n]:9.3f}" for n in names), flush=True)

    # ROUTE arm: cluster-aware sequential routing -> separate heads for anti-parallel paradigms
    print("\n=== ROUTE arm (data-driven cluster routing) ===", flush=True)
    route = run_route_arm(bb, P, names, Xr_tr)
    for e in route["router_log"]:
        act = e["action"].upper(); c = f" cos={e['cos']}" if "cos" in e else ""
        print(f"  {e['added']:10} -> {act} head {e['head']}{c}   all_cos={e['all_cos']}", flush=True)
    print(f"  landed {route['n_heads']} heads: " + "; ".join(f"H{h['head']}={h['members']}" for h in route["heads"]), flush=True)
    print("  " + "rule".ljust(12) + " ".join(n.rjust(9) for n in names), flush=True)
    print("  " + "oracle".ljust(12) + " ".join(f"{route['oracle'][n]:9.3f}" for n in names), flush=True)
    print("  " + "maxens".ljust(12) + " ".join(f"{route['maxens'][n]:9.3f}" for n in names), flush=True)
    result["route"] = route

    # ROUTE + intra-cluster GPM: same 2 heads, but protect a head's prior members from the surge (diffusion untouched in H1)
    print("\n=== ROUTE+GPM arm (route + GPM inside a shared head) ===", flush=True)
    route_gpm = run_route_arm(bb, P, names, Xr_tr, intra_gpm=True)
    for e in route_gpm["router_log"]:
        act = e["action"].upper(); c = f" cos={e['cos']}" if "cos" in e else ""; g = " +GPM" if e.get("gpm") else ""
        print(f"  {e['added']:10} -> {act} head {e['head']}{c}{g}   all_cos={e['all_cos']}", flush=True)
    print(f"  landed {route_gpm['n_heads']} heads: " + "; ".join(f"H{h['head']}={h['members']}" for h in route_gpm["heads"]), flush=True)
    print("  " + "rule".ljust(12) + " ".join(n.rjust(9) for n in names), flush=True)
    print("  " + "oracle".ljust(12) + " ".join(f"{route_gpm['oracle'][n]:9.3f}" for n in names), flush=True)
    print("  " + "maxens".ljust(12) + " ".join(f"{route_gpm['maxens'][n]:9.3f}" for n in names), flush=True)
    result["route_gpm"] = route_gpm

    # headline: naive final-step forgetting of paradigm-0 (swap) + does gpm protect it (vs rand) + does ROUTE retain BOTH
    if not SMOKE:
        p0 = names[0]; p1 = names[1]; fin = -1
        naive0 = result["naive"][fin]["retention"][p0]; gpm0 = result["gpm"][fin]["retention"][p0]; rand0 = result["rand"][fin]["retention"][p0]
        naive1 = result["naive"][fin]["retention"][p1]; gpm1 = result["gpm"][fin]["retention"][p1]
        base0 = result["naive"][0]["retention"][p0]
        print(f"\n=== HEADLINE (paradigm-0 = {p0}) ===")
        print(f"  base {p0} AUC (after only {p0}): {base0:.3f}")
        print(f"  after full sequence:  naive={naive0:.3f}  gpm={gpm0:.3f}  rand={rand0:.3f}")
        print(f"  naive forgetting: {base0-naive0:+.3f}  | GPM protection over naive: {gpm0-naive0:+.3f}  | GPM over rand: {gpm0-rand0:+.3f}")
        print(f"\n=== ROUTE vs the rest (retain BOTH antagonists: {p0} & {p1}) ===")
        print(f"  {'':10}{p0.rjust(9)}{p1.rjust(9)}")
        print(f"  naive     {naive0:9.3f}{naive1:9.3f}")
        print(f"  gpm       {gpm0:9.3f}{gpm1:9.3f}")
        print(f"  route/orc {route['oracle'][p0]:9.3f}{route['oracle'][p1]:9.3f}")
        print(f"  route/max {route['maxens'][p0]:9.3f}{route['maxens'][p1]:9.3f}")
        print(f"  rtgpm/orc {route_gpm['oracle'][p0]:9.3f}{route_gpm['oracle'][p1]:9.3f}")
        print(f"  rtgpm/max {route_gpm['maxens'][p0]:9.3f}{route_gpm['maxens'][p1]:9.3f}")
        mins = {"naive": min(naive0, naive1), "gpm": min(gpm0, gpm1),
                "route/oracle": min(route['oracle'][p0], route['oracle'][p1]),
                "route_gpm/oracle": min(route_gpm['oracle'][p0], route_gpm['oracle'][p1]),
                "route_gpm/maxens": min(route_gpm['maxens'][p0], route_gpm['maxens'][p1])}
        print(f"  -> min({p0},{p1}) by arm (higher = retains BOTH):  " + "  ".join(f"{k}={v:.3f}" for k, v in mins.items()))
        # intra-cluster cost check: did protecting swap starve its own cluster-mates celebdf/dfdc?
        print(f"  intra-cluster cost (route_gpm vs route): "
              + "  ".join(f"{n} {route_gpm['oracle'][n]-route['oracle'][n]:+.3f}" for n in names[2:]))

    OUT.write_text(json.dumps({"config": {"paradigms": names, "subsample": SUBSAMPLE, "pca_k": PCA_K, "smoke": SMOKE, "route_tau": TAU},
                               "arms": result}, indent=1))
    print(f"\nwrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
