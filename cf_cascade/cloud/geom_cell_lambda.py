"""SINGLE-CELL geometry G1-G6 (parameterized) — for prying recipe knobs apart (Thread 2). Args: MODEL CKPT TAG FAMILY.
Measurement code is VERBATIM from geom_big_lambda.py (same Fisher/V/spectrum, same G1..G6, same SEEDS_GEOM, same stream)
so a new cell (e.g. pythia-160m-deduped) is commensurable with the matrix. Carriers chosen by FAMILY (OPT vs PYTHIA).
Saves geom_<TAG>.json (summary medians) + geom_<TAG>.npz (per-carrier arrays {TAG}__G1.. for RMD-SRC / comparison).
"""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"
MODEL = sys.argv[1]; CKPT = sys.argv[2]; TAG = sys.argv[3]; FAMILY = sys.argv[4]
OUT = f"{G}/geom_{TAG}.json"; NPZ = f"{G}/geom_{TAG}.npz"
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS_GEOM = list(range(3))
STEPS_DEPTH = 250; LR_DEPTH = 5e-5; STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16
KPROJ = 128; N_FISHER = 200; K_M = 16; K_GEN = 16; CHANCE = K_M / KPROJ; DECAY_TOP = 256
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CARRIER_KEYS = OPT_C if FAMILY == "OPT" else PY_C
BIG = False  # parameterize if needed for >1B


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
        s = float(evp.sum()); s2 = float((evp * evp).sum())
        G1[n] = (s * s / max(s2, 1e-30)) / evp.numel()
        top = torch.sort(evp, descending=True).values[:DECAY_TOP].cpu().numpy(); top = top[top > 0]
        G2[n] = float(-np.polyfit(np.log(np.arange(1, len(top) + 1)), np.log(top), 1)[0]) if len(top) > 8 else float("nan")
        del F[n]
    return V, G1, G2


def train_single(model, tok, items, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr, foreach=False); g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt; torch.cuda.empty_cache()


def childhood(model, tok, stream, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr, foreach=False); g = torch.Generator().manual_seed(seed)
    tasks = list(stream)
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist()
        inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt; torch.cuda.empty_cache()


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


def main():
    t0 = time.time(); print(f"[GEOM-CELL] {MODEL} tag={TAG} family={FAMILY}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT))
    if BIG: model.gradient_checkpointing_enable(); model.config.use_cache = False
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in CARRIER_KEYS) and n.endswith("weight") and params[n].ndim == 2]
    ckpt_sd = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    print(f"[{TAG}] {len(carriers)} carriers ({time.time()-t0:.0f}s)", flush=True)

    V, G1, G2 = fisher_and_spectrum(model, params, carriers, tok)
    rng = torch.Generator(device="cpu").manual_seed(99); Rrand = {}
    for n in carriers:
        rr = torch.randn(KPROJ, K_M, generator=rng).double().to(dev); q, _ = torch.linalg.qr(rr); Rrand[n] = q[:, :K_M]
    model.load_state_dict(ckpt_sd)
    print(f"[{TAG}] Fisher/V/spectrum done ({time.time()-t0:.0f}s)", flush=True)

    stream_keys = ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]
    stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in stream_keys}
    from collections import OrderedDict
    stream = {sk: (it, sorted(set(c for _, c in it))) for sk, it in stream.items()}

    Mn = {g: {n: [] for n in carriers} for g in ["G3", "G4", "G5", "G6", "G4rand", "G5rand"]}
    eye = torch.eye(KPROJ, dtype=torch.float64, device=dev)
    for sd in SEEDS_GEOM:
        model.load_state_dict(ckpt_sd); train_single(model, tok, load_jsonB("medium_subj")[:-200], STEPS_DEPTH, LR_DEPTH, sd)
        surge_post = snap(model, carriers); GA, surge_fracV = pcs_in_V(surge_post, ckpt_sd, carriers, V, K_GEN)
        model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd)
        dep_post = snap(model, carriers); Mdep, dep_fracV = pcs_in_V(dep_post, ckpt_sd, carriers, V, K_M)
        for n in carriers:
            topcurv = eye[:, :K_M]
            Mn["G3"][n].append(surge_fracV[n]); Mn["G4"][n].append(grass(GA[n].double(), topcurv))
            Mn["G5"][n].append(grass(Mdep[n].double(), GA[n].double())); Mn["G6"][n].append(dep_fracV[n])
            Mn["G4rand"][n].append(grass(GA[n].double(), Rrand[n])); Mn["G5rand"][n].append(grass(Mdep[n].double(), Rrand[n]))
        model.load_state_dict(ckpt_sd)
        del surge_post, GA, surge_fracV, dep_post, Mdep, dep_fracV; torch.cuda.empty_cache()
        print(f"[{TAG}] seed {sd}: G5={np.mean([Mn['G5'][n][-1] for n in carriers]):.3f}(rand {np.mean([Mn['G5rand'][n][-1] for n in carriers]):.3f}) ({time.time()-t0:.0f}s)", flush=True)

    per_carrier = {"G1": [G1[n] for n in carriers], "G2": [G2[n] for n in carriers]}
    for g in ["G3", "G4", "G5", "G6", "G4rand", "G5rand"]:
        per_carrier[g] = [float(np.mean(Mn[g][n])) for n in carriers]
    summ = {}
    for g, vals in per_carrier.items():
        arr = np.array(vals, float); arr = arr[~np.isnan(arr)]
        summ[g] = {"median": float(np.median(arr)), "mean": float(np.mean(arr))}
    rep = {"tag": TAG, "model": MODEL, "family": FAMILY, "n_carriers": len(carriers), "summary": summ}
    arrays = {f"{TAG}__{g}": np.array(per_carrier[g], float) for g in per_carrier}
    np.savez(NPZ, **arrays); json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[{TAG}] G1={summ['G1']['median']:.3f} G2={summ['G2']['median']:.2f} G3={summ['G3']['median']:.3f} "
          f"G4={summ['G4']['median']:.3f} ★G5={summ['G5']['median']:.3f}(rand {summ['G5rand']['median']:.3f}) G6={summ['G6']['median']:.3f} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
