"""SETTLE the scale de-confound: does the childhood DEPOSIT drain out of the curvature basis at scale, or is the raw
out-of-V% rise just the fixed-K/growing-d artifact? Args: MODEL CKPT TAG FAMILY [BIG].

Per carrier, measures the deposit's in-basis ENERGY FRACTION under three bases + their nulls:
  - fixed top-128 (the original V)            null = 128 / in_dim
  - energy-90% (K_e90 dirs for 90% Fisher mass, SCALE-INVARIANT)   null = K_e90 / in_dim
  - energy-50% (K_e50)                                            null = K_e50 / in_dim
  - Rrand: deposit fraction in a random 128-dim basis (exact null SAMPLE, gives the empirical null)
Deposit = post-childhood weight delta (same childhood stream/seeds as geom). Fisher/V from SST w_A. 3 seeds.
Saves geom_basis_<TAG>.npz (per-carrier arrays) + geom_basis_<TAG>.json (medians). Only SCALARS stored per carrier.

DECISIVE READ (CPU after): under the scale-invariant energy-90% basis (no fixed-K confound), is
(deposit_frac - null) stable/growing with scale (NO drain, deposit stays curvature-enriched) or shrinking (real drain)?
"""
import os, sys
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"
MODEL = sys.argv[1]; CKPT = sys.argv[2]; TAG = sys.argv[3]; FAMILY = sys.argv[4]
BIG = (len(sys.argv) > 5 and sys.argv[5] == "BIG")
OUT = f"{G}/geom_basis_{TAG}.json"; NPZ = f"{G}/geom_basis_{TAG}.npz"
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(3)); STEPS_CHILD = 400; LR_CHILD = 3e-5; BATCH = 16
N_FISHER = 200; KKEEP = 1024; K128 = 128
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CARRIER_KEYS = OPT_C if FAMILY == "OPT" else PY_C


def _retry(fn, tries=5, wait=12):
    last = None
    for i in range(tries):
        try: return fn()
        except Exception as e: last = e; print(f"[hf] retry {i+1}: {str(e)[:80]}", flush=True); time.sleep(wait)
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


def childhood(model, tok, stream, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr, foreach=False); g = torch.Generator().manual_seed(seed)
    tasks = list(stream)
    for step in range(steps):
        t = tasks[step % len(tasks)]; items = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist()
        inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt; torch.cuda.empty_cache()


def main():
    t0 = time.time(); print(f"[BASIS] {MODEL} tag={TAG} family={FAMILY} BIG={BIG}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT))
    if BIG: model.gradient_checkpointing_enable(); model.config.use_cache = False
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in CARRIER_KEYS) and n.endswith("weight") and params[n].ndim == 2]
    ckpt_sd = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    print(f"[{TAG}] {len(carriers)} carriers ({time.time()-t0:.0f}s)", flush=True)

    # ---- Fisher per carrier -> eigvals (full) + top-KKEEP eigvecs (CPU) + K_e50/K_e90 + in_dim ----
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float32) for n in carriers}
    for s, l in sst_rows("train", N_FISHER, 0):
        inp, lb, am = b_pairs(tok, [(f"{s}\nSentiment:", LAB[l])]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in carriers:
            g = params[n].grad.detach().float(); F[n] += g.T @ g
    Vkeep = {}; Ke50 = {}; Ke90 = {}; indim = {}; Rrand = {}
    rng = torch.Generator(device="cpu").manual_seed(99)
    for n in carriers:
        d = F[n].shape[0]; indim[n] = d
        ev, vv = torch.linalg.eigh(F[n].double()); ev = ev.clamp(min=0).flip(0); vv = vv.flip(1).float()  # desc
        tot = float(ev.sum()); c = torch.cumsum(ev, 0) / max(tot, 1e-30)
        Ke50[n] = int((c < 0.50).sum().item()) + 1; Ke90[n] = int((c < 0.90).sum().item()) + 1
        keep = min(KKEEP, d, max(Ke90[n], K128) + 1)
        Vkeep[n] = vv[:, :keep].cpu()
        rr = torch.randn(d, K128, generator=rng).double(); q, _ = torch.linalg.qr(rr); Rrand[n] = q[:, :K128].float().cpu()
        del F[n]
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
    print(f"[{TAG}] Fisher/eig done ({time.time()-t0:.0f}s)", flush=True)

    # ---- stream ----
    stream_keys = ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]
    stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in stream_keys}

    # ---- deposit basis fractions per seed ----
    acc = {n: {"f128": [], "fe50": [], "fe90": [], "frand": []} for n in carriers}
    for sd in SEEDS:
        model.load_state_dict(ckpt_sd); childhood(model, tok, stream, STEPS_CHILD, LR_CHILD, sd)
        post = dict(model.named_parameters())
        with torch.no_grad():
            for n in carriers:
                dep = (post[n].detach() - ckpt_sd[n].to(dev)).float()      # (out,in)
                tot = float((dep * dep).sum()) + 1e-30
                Vk = Vkeep[n].to(dev); coords = dep @ Vk                    # (out,keep)
                cc = (coords * coords).sum(0)                               # per-dir energy
                acc[n]["f128"].append(float(cc[:K128].sum()) / tot)
                acc[n]["fe50"].append(float(cc[:Ke50[n]].sum()) / tot)
                acc[n]["fe90"].append(float(cc[:Ke90[n]].sum()) / tot)
                Rr = Rrand[n].to(dev); rc = dep @ Rr
                acc[n]["frand"].append(float((rc * rc).sum()) / tot)
                del Vk, coords, Rr, rc
        model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
        print(f"[{TAG}] seed {sd} done ({time.time()-t0:.0f}s)", flush=True)

    # ---- per-carrier means + nulls ----
    arr = {k: [] for k in ["f128", "fe50", "fe90", "frand", "null128", "null_e50", "null_e90", "indim", "Ke50", "Ke90"]}
    for n in carriers:
        arr["f128"].append(np.mean(acc[n]["f128"])); arr["fe50"].append(np.mean(acc[n]["fe50"]))
        arr["fe90"].append(np.mean(acc[n]["fe90"])); arr["frand"].append(np.mean(acc[n]["frand"]))
        arr["null128"].append(K128 / indim[n]); arr["null_e50"].append(Ke50[n] / indim[n]); arr["null_e90"].append(Ke90[n] / indim[n])
        arr["indim"].append(indim[n]); arr["Ke50"].append(Ke50[n]); arr["Ke90"].append(Ke90[n])
    arr = {k: np.array(v, float) for k, v in arr.items()}
    np.savez(NPZ, **{f"{TAG}__{k}": arr[k] for k in arr})
    med = lambda k: float(np.median(arr[k]))
    rep = {"tag": TAG, "model": MODEL, "family": FAMILY, "n_carriers": len(carriers),
           "median": {k: med(k) for k in arr},
           "excess_top128": med("f128") - med("null128"), "excess_e90": med("fe90") - med("null_e90"),
           "excess_e50": med("fe50") - med("null_e50"), "frand_vs_null128": med("frand") - med("null128")}
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[{TAG}] f128={med('f128'):.3f}(null {med('null128'):.3f}) fe90={med('fe90'):.3f}(null {med('null_e90'):.3f}) "
          f"fe50={med('fe50'):.3f}(null {med('null_e50'):.3f}) frand={med('frand'):.3f} Ke90~{med('Ke90'):.0f} "
          f"EXCESS_e90={rep['excess_e90']:+.3f} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
