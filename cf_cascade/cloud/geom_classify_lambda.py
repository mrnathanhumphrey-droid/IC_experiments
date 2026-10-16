"""COMMENSURABLE GEOMETRY CLASSIFICATION — size x family MODEL MATRIX (measurement-only, no arms).

Layer 1 of the dissociation explanation hunt. 6 dimensionless cross-arch metrics on V (task-A curvature subspace) at each model's
pinned w_A, per carrier never pooled. Produces the FEATURE MATRIX (rows=models, cols=G1..G6) that Layer 3 (RMD SRC) adjudicates:
does grouping by family / size / B-D-sign PAY vs random-grouping null. This run validates the vocabulary on the matrix + a descriptive
family-vs-size read; RMD SRC (separate) does the rigorous grouping test.

MATRIX: OPT {125m,350m,1.3b} x Pythia {160m,410m,1b}. (GPT-2 Conv1D deferred — different weight-axis, needs a commensurability call.)

METRICS (per carrier; dimensionless): G1 norm_effrank(Fisher)/in_dim; G2 -loglog spectral decay slope; G3 surge fracV;
  G4 surge<->top-curv overlap; G5 ★ deposit<->surge overlap (the candidate: low=deposit out of surge's way=protective);
  G6 deposit fracV (mapfrac).
NULL — FIXED (was degenerate: G1/G2 deterministic -> seed-noise=0 -> trivially classified). NOW: CROSS-CARRIER BOOTSTRAP — resample
  carriers w/ replacement, recompute median, get SE of the median. Uniform across ALL metrics (deterministic & seeded). Classify iff
  |family-pooled gap| > 2*sqrt(SE_OPT^2+SE_PY^2) AND (for overlap metrics) > random-subspace-baseline gap + 0.01.
PRE-REG: G5 LOWER on OPT than Pythia (shape-not-routing); G2/G3 expected the bigger levers (crowding-vs-room) per the 2-model baseline.
Memory: Fisher accumulated in float32 (1.3b/1b fit 40GB); per-substrate try/except so one OOM can't kill the matrix.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"
OUT = f"{G}/geom_matrix.json"; NPZ = f"{G}/geom_matrix.npz"
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS_GEOM = list(range(3))
STEPS_DEPTH = 250; LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16
KPROJ = 128; N_FISHER = 200; K_M = 16; K_GEN = 16; CHANCE = K_M / KPROJ; DECAY_TOP = 256
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
SUBSTRATES = [
    {"tag": "opt125m",  "family": "OPT",    "size_m": 125,  "model": "facebook/opt-125m",      "ckpt": f"{G}/opt125m_sst2_wA.pt",          "carrier_keys": OPT_C},
    {"tag": "opt350m",  "family": "OPT",    "size_m": 350,  "model": "facebook/opt-350m",      "ckpt": f"{G}/opt350m_sst2_wA.pt",          "carrier_keys": OPT_C},
    {"tag": "opt1_3b",  "family": "OPT",    "size_m": 1300, "model": "facebook/opt-1.3b",      "ckpt": f"{G}/opt1_3b_sst2_wA.pt",          "carrier_keys": OPT_C},
    {"tag": "pythia160m", "family": "PYTHIA", "size_m": 160,  "model": "EleutherAI/pythia-160m", "ckpt": f"{G}/pythia160m_sst2_wA.pt",       "carrier_keys": PY_C},
    {"tag": "pythia410m", "family": "PYTHIA", "size_m": 410,  "model": "EleutherAI/pythia-410m", "ckpt": f"{G}/pythia410m_sst2_subsetmin.pt", "carrier_keys": PY_C},
    {"tag": "pythia1b",   "family": "PYTHIA", "size_m": 1000, "model": "EleutherAI/pythia-1b",   "ckpt": f"{G}/pythia1b_sst2_wA.pt",         "carrier_keys": PY_C},
]


def _retry(fn, tries=5, wait=12):
    last = None
    for i in range(tries):
        try: return fn()
        except Exception as e: last = e; print(f"[hf] retry {i+1}: {str(e)[:100]}", flush=True); time.sleep(wait)
    raise last


def load_jsonB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


def win(ds, k=900, w=4000):
    n = min(len(ds), w); return list(ds.shuffle(seed=0).select(range(n)))[: k]


def load_hf(key):
    from datasets import load_dataset
    LD = lambda *a, **k: _retry(lambda: load_dataset(*a, **k))
    if key == "emotion":
        return [(f"{r['text'].strip()}\nEmotion:", " " + ["sadness","joy","love","anger","fear","surprise"][r["label"]]) for r in win(LD("dair-ai/emotion")["train"])]
    if key == "dbpedia":
        L = ["company","school","artist","athlete","politics","transport","building","nature","village","animal","plant","album","film","book"]
        return [(f"{r['content'].strip()[:400]}\nCategory:", " " + L[r["label"]]) for r in win(LD("fancyzhx/dbpedia_14")["train"])]
    if key == "yahoo":
        L = ["society","science","health","education","computers","sports","business","entertainment","family","politics"]
        return [(f"{(r.get('question_title') or '').strip()}\nTopic:", " " + L[r["topic"]]) for r in win(LD("community-datasets/yahoo_answers_topics")["train"]) if r.get("question_title")]
    if key == "newscat":
        return [(f"{(r.get('headline') or '').strip()}\nCategory:", " " + (r.get("category") or "").strip().lower()) for r in win(LD("heegyu/news-category-dataset")["train"]) if r.get("headline") and r.get("category")]
    raise ValueError(key)


def b_pairs(tok, items):
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"][:256]; c = tok(co, add_special_tokens=False)["input_ids"][:256]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    L = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


def sst_rows(split, k, seed=0):
    from datasets import load_dataset
    ds = _retry(lambda: load_dataset("stanfordnlp/sst2"))[split]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))[:k]]


def fisher_and_spectrum(model, params, names, tok):
    """Fisher per carrier (FLOAT32 to bound mem) -> V top-KPROJ (col0=highest λ), G1 effrank/in_dim, G2 decay slope."""
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float32) for n in names}
    for s, l in sst_rows("train", N_FISHER, 0):
        inp, lb, am = b_pairs(tok, [(f"{s}\nSentiment:", LAB[l])]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names:
            g = params[n].grad.detach().float(); F[n] += g.T @ g
    V = {}; G1 = {}; G2 = {}
    for n in names:
        ev, vv = torch.linalg.eigh(F[n].double()); evp = ev.clamp(min=0)   # eigh in double for stability, F was f32-accumulated
        V[n] = vv[:, -KPROJ:].flip(dims=[1]).float().contiguous()
        s = float(evp.sum()); s2 = float((evp * evp).sum())
        G1[n] = (s * s / max(s2, 1e-30)) / evp.numel()
        top = torch.sort(evp, descending=True).values[:DECAY_TOP].cpu().numpy(); top = top[top > 0]
        if len(top) > 8:
            x = np.log(np.arange(1, len(top) + 1)); y = np.log(top); G2[n] = float(-np.polyfit(x, y, 1)[0])
        else:
            G2[n] = float("nan")
        del F[n]
    return V, G1, G2


def train_single(model, tok, items, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt


def childhood(model, tok, stream, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    tasks = list(stream)
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist()
        inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt


def pcs_in_V(post, start_sd, carriers, V, k):
    PC = {}; fracV = {}
    with torch.no_grad():
        for n in carriers:
            dw = (post[n].detach() - start_sd[n].to(dev)).double(); coords = dw @ V[n].double()
            inV = float((coords * coords).sum()); tot = float((dw * dw).sum()); fracV[n] = inV / max(tot, 1e-12)
            U, S, Vh = torch.linalg.svd(coords, full_matrices=False); PC[n] = Vh[:k].T.contiguous()
    return PC, fracV


def grass(A, B):
    k = min(A.shape[1], B.shape[1])
    return float((A.T @ B).pow(2).sum() / max(k, 1)) if k > 0 else 0.0


def snap(model, carriers): p = dict(model.named_parameters()); return {n: p[n].detach().clone() for n in carriers}
def labels_of(items): return sorted(set(c for _, c in items))
def ctype(n, keys):
    for c in keys:
        if c in n: return c
    return "?"


def run(cfg, t0):
    tag = cfg["tag"]; keys = cfg["carrier_keys"]
    print(f"\n[{tag}] loading {cfg['model']}...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(cfg["ckpt"]))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in keys) and n.endswith("weight") and params[n].ndim == 2]
    ckpt_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    print(f"[{tag}] {len(carriers)} carriers ({time.time()-t0:.0f}s)", flush=True)

    V, G1, G2 = fisher_and_spectrum(model, params, carriers, tok)
    rng = torch.Generator(device="cpu").manual_seed(99); Rrand = {}
    for n in carriers:
        rr = torch.randn(KPROJ, K_M, generator=rng).double().to(dev); q, _ = torch.linalg.qr(rr); Rrand[n] = q[:, :K_M]
    model.load_state_dict(ckpt_sd)
    print(f"[{tag}] Fisher/V/spectrum done ({time.time()-t0:.0f}s)", flush=True)

    ho_items = load_jsonB("medium_subj"); ho_tr = ho_items[:-200]
    stream_keys = ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]
    stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in stream_keys}
    stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}

    Mn = {g: {n: [] for n in carriers} for g in ["G3", "G4", "G5", "G6", "G4rand", "G5rand"]}
    eye = torch.eye(KPROJ, dtype=torch.float64, device=dev)
    for sd in SEEDS_GEOM:
        model.load_state_dict(ckpt_sd); train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
        surge_post = snap(model, carriers); GA, surge_fracV = pcs_in_V(surge_post, ckpt_sd, carriers, V, K_GEN)
        model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd)
        dep_post = snap(model, carriers); Mdep, dep_fracV = pcs_in_V(dep_post, ckpt_sd, carriers, V, K_M)
        for n in carriers:
            topcurv = eye[:, :K_M]
            Mn["G3"][n].append(surge_fracV[n]); Mn["G4"][n].append(grass(GA[n].double(), topcurv))
            Mn["G5"][n].append(grass(Mdep[n].double(), GA[n].double())); Mn["G6"][n].append(dep_fracV[n])
            Mn["G4rand"][n].append(grass(GA[n].double(), Rrand[n])); Mn["G5rand"][n].append(grass(Mdep[n].double(), Rrand[n]))
        model.load_state_dict(ckpt_sd)
        print(f"[{tag}] seed {sd}: G5 dep-surge={np.mean([Mn['G5'][n][-1] for n in carriers]):.3f}(rand {np.mean([Mn['G5rand'][n][-1] for n in carriers]):.3f}) "
              f"G2={np.median([G2[n] for n in carriers if G2[n]==G2[n]]):.2f} G6={np.mean([Mn['G6'][n][-1] for n in carriers]):.3f} ({time.time()-t0:.0f}s)", flush=True)

    # per-carrier mean over seeds -> per-carrier arrays (the unit for bootstrap)
    per_carrier = {"G1": [G1[n] for n in carriers], "G2": [G2[n] for n in carriers]}
    for g in ["G3", "G4", "G5", "G6", "G4rand", "G5rand"]:
        per_carrier[g] = [float(np.mean(Mn[g][n])) for n in carriers]
    summ = {}
    for g, vals in per_carrier.items():
        arr = np.array(vals, float); arr = arr[~np.isnan(arr)]
        summ[g] = {"median": float(np.median(arr)), "mean": float(np.mean(arr))}
    rep = {"tag": tag, "family": cfg["family"], "size_m": cfg["size_m"], "model": cfg["model"], "n_carriers": len(carriers), "summary": summ}
    arrays = {f"{tag}__{g}": np.array(per_carrier[g], float) for g in per_carrier}
    print(f"[{tag}] G1={summ['G1']['median']:.3f} G2={summ['G2']['median']:.2f} G3={summ['G3']['median']:.3f} "
          f"G4={summ['G4']['median']:.3f} ★G5={summ['G5']['median']:.3f}(rand {summ['G5rand']['median']:.3f}) G6={summ['G6']['median']:.3f}", flush=True)
    del model
    torch.cuda.empty_cache()
    return rep, arrays


def _boot_median_se(arr, nboot=1000, seed=0):
    arr = np.asarray(arr, float); arr = arr[~np.isnan(arr)]
    if len(arr) < 2: return 0.0
    rng = np.random.default_rng(seed); n = len(arr)
    meds = [np.median(arr[rng.integers(0, n, n)]) for _ in range(nboot)]
    return float(np.std(meds))


def main():
    t0 = time.time(); print("[GEOM] GEOMETRY MATRIX — size x family, bootstrap-null (measurement-only)", flush=True)
    reps = {}; allarr = {}
    for cfg in SUBSTRATES:
        try:
            rep, arr = run(cfg, t0); reps[cfg["tag"]] = rep; allarr.update(arr)
        except Exception as e:
            reps[cfg["tag"]] = {"error": traceback.format_exc(), "family": cfg["family"], "size_m": cfg["size_m"]}
            print(f"[{cfg['tag']}] FAILED: {str(e)[:200]}", flush=True)
    ok = [t for t in reps if "summary" in reps[t]]
    try:
        fams = {"OPT": [t for t in ok if reps[t]["family"] == "OPT"], "PYTHIA": [t for t in ok if reps[t]["family"] == "PYTHIA"]}
        METRICS = ["G1", "G2", "G3", "G4", "G5", "G6"]
        # feature matrix (per model) for RMD SRC
        feat = {t: {g: reps[t]["summary"][g]["median"] for g in METRICS + ["G4rand", "G5rand"]} | {"family": reps[t]["family"], "size_m": reps[t]["size_m"]} for t in ok}
        reps["feature_matrix"] = feat
        classify = {}
        if fams["OPT"] and fams["PYTHIA"]:
            for g in METRICS:
                oarr = np.concatenate([allarr[f"{t}__{g}"] for t in fams["OPT"]])
                parr = np.concatenate([allarr[f"{t}__{g}"] for t in fams["PYTHIA"]])
                omed, pmed = float(np.nanmedian(oarr)), float(np.nanmedian(parr))
                se = (_boot_median_se(oarr, seed=1) ** 2 + _boot_median_se(parr, seed=2) ** 2) ** 0.5
                gap = omed - pmed; randgap = None
                if g in ("G4", "G5"):
                    ro = np.concatenate([allarr[f"{t}__{g}rand"] for t in fams["OPT"]]); rp = np.concatenate([allarr[f"{t}__{g}rand"] for t in fams["PYTHIA"]])
                    randgap = abs(float(np.nanmedian(ro)) - float(np.nanmedian(rp)))
                classifies = (abs(gap) > 2 * se) and (randgap is None or abs(gap) > randgap + 0.01)
                # size trend within family: model-level medians ordered by size
                trend = {}
                for fam in ("OPT", "PYTHIA"):
                    ts = sorted(fams[fam], key=lambda t: reps[t]["size_m"])
                    trend[fam] = [(reps[t]["size_m"], round(reps[t]["summary"][g]["median"], 4)) for t in ts]
                classify[g] = {"OPT_med": omed, "PYTHIA_med": pmed, "gap": gap, "boot_se": se, "rand_gap": randgap,
                               "classifies_family": bool(classifies), "size_trend": trend}
            g5 = classify["G5"]
            hyp = ("SUPPORTED" if (g5["OPT_med"] < g5["PYTHIA_med"] - 2 * g5["boot_se"] and g5["classifies_family"]) else "NOT supported")
            reps["classification"] = {"metrics": classify,
                                      "classifying_family": [g for g in METRICS if classify[g]["classifies_family"]],
                                      "G5_hypothesis": hyp}
            print(f"\n[GEOM] === FAMILY CLASSIFICATION (bootstrap null, models: OPT {fams['OPT']} | PYTHIA {fams['PYTHIA']}) ===", flush=True)
            for g in METRICS:
                c = classify[g]; star = "★" if g == "G5" else " "
                print(f" {star}{g}: OPT={c['OPT_med']:+.3f} PY={c['PYTHIA_med']:+.3f} gap={c['gap']:+.3f} bootSE={c['boot_se']:.3f} randgap={c['rand_gap']} -> {'CLASSIFIES' if c['classifies_family'] else 'no'}", flush=True)
                print(f"      size-trend OPT {c['size_trend']['OPT']}  PY {c['size_trend']['PYTHIA']}", flush=True)
            print(f"  CLASSIFYING-BY-FAMILY: {reps['classification']['classifying_family']}", flush=True)
            print(f"  G5 hypothesis: {hyp}", flush=True)
        else:
            print(f"[GEOM] insufficient cells for family classification: OK={ok}", flush=True)
    except Exception as e:
        reps["classification_error"] = traceback.format_exc(); print(f"[GEOM] classify error: {e}", flush=True)
    np.savez(NPZ, **allarr); json.dump(reps, open(OUT, "w"), indent=2)
    print(f"\n[GEOM] cells OK: {ok} | saved geom_matrix.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
