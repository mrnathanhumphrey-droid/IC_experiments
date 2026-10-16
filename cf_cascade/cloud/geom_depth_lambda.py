"""GEOMETRY DEPTH AXIS — G1-G6 across surgered OPT-350m depth variants (measurement-only).

Depth de-confound: same family (OPT) + same width (1024), VARY n_layers {6,12,18,24}, matched-acc-band retrained w_A per depth
(surgery-bridge validated: half-depth re-stabilizes to [0.77,0.84], backbone intact). Stamps the SAME validated G1-G6 vocabulary on
each depth so RMD-SRC gets a clean DEPTH grouping axis alongside size x family. Prior (NOT a target — scoping discipline): half-depth
preview moved backbone metrics (aimed 0.29->0.49, M1 0.86->0.71, antag +0.09->0) so depth MAY move G-metrics; instrument so FLAT is
equally reportable (flat => depth de-confounded, the family-G5 separation isn't a depth artifact).

Per-cell: load truncated-layer OPT-350m (AutoConfig.num_hidden_layers = depth, then load_state_dict). Metrics G1-G6 + null
(bootstrap median SE, random-subspace baseline). main() = DEPTH-TREND per metric + MOVES/FLAT verdict (range across depths vs
bootstrap noise), pre-committed. Feature sub-matrix saved for RMD-SRC.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"
OUT = f"{G}/geom_depth.json"; NPZ = f"{G}/geom_depth.npz"
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS_GEOM = list(range(3))
STEPS_DEPTH = 250; LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16
KPROJ = 128; N_FISHER = 200; K_M = 16; K_GEN = 16; CHANCE = K_M / KPROJ; DECAY_TOP = 256
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
SUBSTRATES = [
    {"tag": "opt_d6",  "depth": 6,  "n_layers": 6,  "model": "facebook/opt-350m", "ckpt": f"{G}/opt350m_d6_sst2_wA.pt",  "carrier_keys": OPT_C},
    {"tag": "opt_d12", "depth": 12, "n_layers": 12, "model": "facebook/opt-350m", "ckpt": f"{G}/opt350m_d12_sst2_wA.pt", "carrier_keys": OPT_C},
    {"tag": "opt_d18", "depth": 18, "n_layers": 18, "model": "facebook/opt-350m", "ckpt": f"{G}/opt350m_d18_sst2_wA.pt", "carrier_keys": OPT_C},
    {"tag": "opt_d24", "depth": 24, "n_layers": 24, "model": "facebook/opt-350m", "ckpt": f"{G}/opt350m_sst2_wA.pt",     "carrier_keys": OPT_C},
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
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float32) for n in names}
    for s, l in sst_rows("train", N_FISHER, 0):
        inp, lb, am = b_pairs(tok, [(f"{s}\nSentiment:", LAB[l])]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names:
            g = params[n].grad.detach().float(); F[n] += g.T @ g
    V = {}; G1 = {}; G2 = {}
    for n in names:
        ev, vv = torch.linalg.eigh(F[n].double()); evp = ev.clamp(min=0)
        V[n] = vv[:, -KPROJ:].flip(dims=[1]).float().contiguous()
        s = float(evp.sum()); s2 = float((evp * evp).sum()); G1[n] = (s * s / max(s2, 1e-30)) / evp.numel()
        top = torch.sort(evp, descending=True).values[:DECAY_TOP].cpu().numpy(); top = top[top > 0]
        G2[n] = float(-np.polyfit(np.log(np.arange(1, len(top) + 1)), np.log(top), 1)[0]) if len(top) > 8 else float("nan")
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


def run(cfg, t0):
    tag = cfg["tag"]; keys = cfg["carrier_keys"]
    print(f"\n[{tag}] building {cfg['model']} @ {cfg['n_layers']} layers...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    config = AutoConfig.from_pretrained(cfg["model"]); config.num_hidden_layers = cfg["n_layers"]
    model = AutoModelForCausalLM.from_config(config).float().to(dev)
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
    stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]}
    stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}

    Mn = {g: {n: [] for n in carriers} for g in ["G3", "G4", "G5", "G6", "G4rand", "G5rand"]}
    eye = torch.eye(KPROJ, dtype=torch.float64, device=dev)
    for sd in SEEDS_GEOM:
        model.load_state_dict(ckpt_sd); train_single(model, tok, ho_tr, STEPS_DEPTH, LR_DEPTH, sd)
        surge_post = snap(model, carriers); GA, surge_fracV = pcs_in_V(surge_post, ckpt_sd, carriers, V, K_GEN)
        model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd)
        dep_post = snap(model, carriers); Mdep, dep_fracV = pcs_in_V(dep_post, ckpt_sd, carriers, V, K_M)
        for n in carriers:
            tc = eye[:, :K_M]
            Mn["G3"][n].append(surge_fracV[n]); Mn["G4"][n].append(grass(GA[n].double(), tc))
            Mn["G5"][n].append(grass(Mdep[n].double(), GA[n].double())); Mn["G6"][n].append(dep_fracV[n])
            Mn["G4rand"][n].append(grass(GA[n].double(), Rrand[n])); Mn["G5rand"][n].append(grass(Mdep[n].double(), Rrand[n]))
        model.load_state_dict(ckpt_sd)
        print(f"[{tag}] seed {sd}: G5={np.mean([Mn['G5'][n][-1] for n in carriers]):.3f} G4={np.mean([Mn['G4'][n][-1] for n in carriers]):.3f} G6={np.mean([Mn['G6'][n][-1] for n in carriers]):.3f} ({time.time()-t0:.0f}s)", flush=True)

    per_carrier = {"G1": [G1[n] for n in carriers], "G2": [G2[n] for n in carriers]}
    for g in ["G3", "G4", "G5", "G6", "G4rand", "G5rand"]:
        per_carrier[g] = [float(np.mean(Mn[g][n])) for n in carriers]
    summ = {}
    for g, vals in per_carrier.items():
        arr = np.array(vals, float); arr = arr[~np.isnan(arr)]; summ[g] = {"median": float(np.median(arr)), "mean": float(np.mean(arr))}
    rep = {"tag": tag, "depth": cfg["depth"], "n_layers": cfg["n_layers"], "model": cfg["model"], "n_carriers": len(carriers), "summary": summ}
    arrays = {f"{tag}__{g}": np.array(per_carrier[g], float) for g in per_carrier}
    print(f"[{tag}] depth {cfg['depth']}: G1={summ['G1']['median']:.3f} G2={summ['G2']['median']:.2f} G3={summ['G3']['median']:.3f} "
          f"G4={summ['G4']['median']:.3f} ★G5={summ['G5']['median']:.3f}(rand {summ['G5rand']['median']:.3f}) G6={summ['G6']['median']:.3f}", flush=True)
    del model; torch.cuda.empty_cache()
    return rep, arrays


def _boot_median_se(arr, nboot=1000, seed=0):
    arr = np.asarray(arr, float); arr = arr[~np.isnan(arr)]
    if len(arr) < 2: return 0.0
    rng = np.random.default_rng(seed); n = len(arr)
    return float(np.std([np.median(arr[rng.integers(0, n, n)]) for _ in range(nboot)]))


def main():
    t0 = time.time(); print("[GEOMDEPTH] DEPTH AXIS — G1-G6 across OPT-350m depths {6,12,18,24} (measurement-only)", flush=True)
    reps = {}; allarr = {}
    for cfg in SUBSTRATES:
        try:
            rep, arr = run(cfg, t0); reps[cfg["tag"]] = rep; allarr.update(arr)
        except Exception as e:
            reps[cfg["tag"]] = {"error": traceback.format_exc(), "depth": cfg["depth"]}; print(f"[{cfg['tag']}] FAILED: {str(e)[:200]}", flush=True)
    ok = [t for t in reps if isinstance(reps[t], dict) and "summary" in reps[t]]
    try:
        METRICS = ["G1", "G2", "G3", "G4", "G5", "G6"]
        order = sorted(ok, key=lambda t: reps[t]["depth"])
        feat = {t: {g: reps[t]["summary"][g]["median"] for g in METRICS} | {"depth": reps[t]["depth"]} for t in ok}
        reps["feature_matrix"] = feat
        trend = {}
        for g in METRICS:
            pts = [(reps[t]["depth"], reps[t]["summary"][g]["median"]) for t in order]
            meds = [p[1] for p in pts]; rng_ = (max(meds) - min(meds)) if meds else 0.0
            se = max((_boot_median_se(allarr[f"{t}__{g}"], seed=hash(t) % 1000) for t in ok), default=0.0)
            moves = rng_ > 2 * se
            trend[g] = {"points": pts, "range": rng_, "max_boot_se": se, "MOVES_with_depth": bool(moves)}
        reps["depth_trend"] = trend
        movers = [g for g in METRICS if trend[g]["MOVES_with_depth"]]
        reps["movers"] = movers
        print(f"\n[GEOMDEPTH] === DEPTH TREND (OPT-350m, cells {order}) ===", flush=True)
        for g in METRICS:
            tr = trend[g]; star = "★" if g == "G5" else " "
            pts = " ".join(f"d{d}={m:.3f}" for d, m in tr["points"])
            print(f" {star}{g}: {pts} | range={tr['range']:.3f} bootSE={tr['max_boot_se']:.3f} -> {'MOVES' if tr['MOVES_with_depth'] else 'FLAT'}", flush=True)
        print(f"  METRICS THAT MOVE WITH DEPTH: {movers}", flush=True)
        print(f"  (FLAT G5 => depth de-confounded, family-G5 separation is NOT a depth artifact; MOVES => depth is a lever)", flush=True)
    except Exception as e:
        reps["analysis_error"] = traceback.format_exc(); print(f"[GEOMDEPTH] analysis error: {e}", flush=True)
    np.savez(NPZ, **allarr); json.dump(reps, open(OUT, "w"), indent=2)
    print(f"\n[GEOMDEPTH] cells OK: {ok} | saved geom_depth.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
