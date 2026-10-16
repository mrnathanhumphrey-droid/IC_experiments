"""B-D REPRODUCIBILITY-FLOOR BATTERY. Pre-reg PREREG_FLOOR_BATTERY_2026_06_23.md (LOCK f1dfa005). NOISE characterization,
not mechanism. Per-cell floor + decomposition + seed budget. Dispositive regime EXACT (lr_child 3e-5/400, lr_depth
5e-5/250, pinned w_A, held-out subj, V=top-128 Fisher). arm B=childhood(full)->phase2; arm D=childhood(grad proj out of
V)->phase2; B-D=retB-retD. Anchors (one prior draw): OPT B-D +0.055, Pythia -0.006.

Env: CELL in {opt350m,pythia410m}; BLOCK in {drift,bf16,fp32,seedsample,seedcurve}. One block per invocation -> own JSON.
  drift      : bf16, seeds 0..29, 1 estimate. gap from banked anchor = library/env drift (Block 4, fire FIRST = validates
               harness + the dangerous source).
  bf16       : bf16, seeds 0..29, 5 independent repeats. SD across 5 = hardware non-determinism (Block 1).
  fp32       : fp32 + deterministic algos, seeds 0..29, 5 repeats. SD ~0 expected; the FIX TEST (Block 2).
  seedsample : fp32, 5 disjoint draws of N=30 (0..149). SD across draws ~ perseedSD/sqrt(30) (Block 3).
  seedcurve  : fp32, seeds 0..99, 1 estimate keeping per-seed B-D; bootstrap SE at N=10/20/30/40/60/100 (Part B).
"""
import os, time, json, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for deterministic cublas (fp32 blocks)
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
import contextlib
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
LR_DEPTH = 5e-5; STEPS_DEPTH = 250; LR_CHILD = 3e-5; STEPS_CHILD = 400; BATCH = 16; KPROJ = 128; N_FISHER = 200
HEARTBEAT = 25; _T0 = 0.0   # per-step monitor: print every HEARTBEAT steps; _T0 set in main() for elapsed
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",            "ck": OPT_C, "family": "OPT",    "anchor_BD": 0.055, "wA": 0.823},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt",  "ck": PY_C,  "family": "PYTHIA", "anchor_BD": -0.006, "wA": 0.753},
}
CELL = os.environ.get("CELL", ""); BLOCK = os.environ.get("BLOCK", "")
assert CELL in CELLS, f"CELL band-only; got {CELL!r}"
assert BLOCK in ("drift", "bf16", "fp32", "seedsample", "seedcurve"), f"bad BLOCK {BLOCK!r}"
cfg = CELLS[CELL]; OUT = f"{G}/floor_battery_{CELL}_{BLOCK}.json"


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
def fisher_taskA(model, params, names, tok):
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float32) for n in names}
    for s, l in sst_rows("train", N_FISHER, 0):
        inp, lb, am = b_pairs(tok, [(f"{s}\nSentiment:", LAB[l])]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: g = params[n].grad.detach().float(); F[n] += g.T @ g
    return F
def eval_sst(model, tok, evalset, posid, negid):
    model.eval(); correct = 0; n = 0
    with torch.no_grad():
        for i in range(0, len(evalset), 32):
            chunk = evalset[i:i + 32]; seqs = [tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"][:256] for s, _ in chunk]
            L = max(len(x) for x in seqs); pad = tok.pad_token_id
            inp = np.full((len(seqs), L), pad, np.int64); pos = np.zeros(len(seqs), np.int64)
            for j, sq in enumerate(seqs): inp[j, :len(sq)] = sq; pos[j] = len(sq) - 1
            lg = model(input_ids=torch.tensor(inp).to(dev)).logits
            last = lg[torch.arange(len(seqs)), torch.tensor(pos).to(dev)]
            pred = (last[:, posid] > last[:, negid]).long().cpu().numpy()
            for j, (_, lab) in enumerate(chunk): correct += int(pred[j] == lab); n += 1
    return correct / max(n, 1)
def amp(use_bf16):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
def childhood(model, tok, stream, seed, use_bf16, Vproj=None, carriers=None, hb=""):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR_CHILD); g = torch.Generator().manual_seed(seed)
    tasks = list(stream); params = dict(model.named_parameters())
    for step in range(STEPS_CHILD):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with amp(use_bf16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        if Vproj is not None:
            for n in carriers: gd = params[n].grad; gd.sub_(((gd.double() @ Vproj[n].double()) @ Vproj[n].double().T).to(gd.dtype))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if hb and (step % HEARTBEAT == 0 or step == STEPS_CHILD - 1):
            print(f"[hb] {hb} childhood {step+1}/{STEPS_CHILD} loss={float(loss):.3f} ({time.time()-_T0:.0f}s)", flush=True)
    del opt; torch.cuda.empty_cache()
def phase2(model, tok, ho_tr, sst_eval, posid, negid, seed, use_bf16, hb=""):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR_DEPTH); g = torch.Generator().manual_seed(seed)
    for step in range(STEPS_DEPTH):
        idx = torch.randperm(len(ho_tr), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [ho_tr[i] for i in idx])
        with amp(use_bf16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if hb and (step % HEARTBEAT == 0 or step == STEPS_DEPTH - 1):
            print(f"[hb] {hb} phase2 {step+1}/{STEPS_DEPTH} loss={float(loss):.3f} ({time.time()-_T0:.0f}s)", flush=True)
    del opt; r = eval_sst(model, tok, sst_eval, posid, negid); torch.cuda.empty_cache(); return r
def labels_of(items): return sorted(set(c for _, c in items))


def main():
    global _T0
    t0 = time.time(); _T0 = t0
    use_bf16 = BLOCK in ("drift", "bf16")
    if not use_bf16:
        torch.use_deterministic_algorithms(True, warn_only=True)
    print(f"[FLOOR:{CELL}:{BLOCK}] {cfg['model']} precision={'bf16' if use_bf16 else 'fp32+det'} dispositive B-D floor", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    wA_ok = abs(pre_sst - cfg["wA"]) < 0.03
    print(f"[FLOOR:{CELL}:{BLOCK}] w_A acc={pre_sst:.3f} (exp {cfg['wA']:.3f}) {'OK' if wA_ok else 'MISMATCH'}; {len(carriers)} carriers ({time.time()-t0:.0f}s)", flush=True)
    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); V[n] = vv[:, -KPROJ:].flip(dims=[1]).contiguous(); del FA[n]
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
    ho_tr = load_jsonB("medium_subj")[:-200]
    stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]}
    stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}

    def bd_over(seeds, tag=""):
        """returns per-seed arrays retB, retD over the given seeds (one B-D estimate's raw data)."""
        rB = []; rD = []; ns = len(seeds)
        for i, sd in enumerate(seeds):
            lab = f"{tag} s{i+1}/{ns}"
            model.load_state_dict(ckpt_sd); childhood(model, tok, stream, sd, use_bf16, hb=f"{lab} armB")
            b = phase2(model, tok, ho_tr, sst_eval, posid, negid, sd, use_bf16, hb=f"{lab} armB"); rB.append(b)
            model.load_state_dict(ckpt_sd); childhood(model, tok, stream, sd, use_bf16, Vproj=V, carriers=carriers, hb=f"{lab} armD")
            d = phase2(model, tok, ho_tr, sst_eval, posid, negid, sd, use_bf16, hb=f"{lab} armD"); rD.append(d)
            run = float(np.mean(np.array(rB) - np.array(rD)))
            print(f"[seed] {lab} DONE  B={b:.3f} D={d:.3f} B-D={b-d:+.4f}  running_mean={run:+.4f}  ({time.time()-_T0:.0f}s)", flush=True)
        return np.array(rB), np.array(rD)

    me = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)]
    rep = {"cell": CELL, "family": cfg["family"], "block": BLOCK, "model": cfg["model"], "precision": "bf16" if use_bf16 else "fp32det",
           "w_A_acc": pre_sst, "w_A_ok": bool(wA_ok), "anchor_BD": cfg["anchor_BD"]}

    if BLOCK in ("bf16", "fp32"):  # 5 repeats, same 30 seeds -> SD of the 5 B-D means
        seeds = list(range(30)); bds = []; details = []
        for rep_i in range(5):
            print(f"[FLOOR:{CELL}:{BLOCK}] >>> repeat {rep_i+1}/5 START ({time.time()-t0:.0f}s)", flush=True)
            rB, rD = bd_over(seeds, tag=f"r{rep_i+1}/5"); bd = float(np.mean(rB - rD)); bds.append(bd)
            details.append({"bd": bd, "retB": float(rB.mean()), "retD": float(rD.mean()), "A_forget": pre_sst - float(rD.mean())})
            json.dump({**rep, "repeats_so_far": bds, "details": details}, open(OUT, "w"), indent=2)  # incremental
            print(f"[FLOOR:{CELL}:{BLOCK}] repeat {rep_i+1}/5: B-D={bd:+.4f} ({time.time()-t0:.0f}s)", flush=True)
        floor = float(np.std(bds, ddof=1))
        rep.update({"n_seeds": 30, "n_repeats": 5, "bd_repeats": bds, "bd_mean": float(np.mean(bds)),
                    "FLOOR_sd_across_repeats": floor, "details": details})
        print(f"\n[FLOOR:{CELL}:{BLOCK}] === {BLOCK} floor SD = {floor:.4f} (mean B-D {np.mean(bds):+.4f}, anchor {cfg['anchor_BD']:+.4f}) ===", flush=True)

    elif BLOCK == "seedsample":  # 5 disjoint draws of N=30 -> SD across draws
        draws = [list(range(j*30, j*30+30)) for j in range(5)]; bds = []; details = []
        for di, seeds in enumerate(draws):
            print(f"[FLOOR:{CELL}:{BLOCK}] >>> draw {di+1}/5 START seeds {seeds[0]}-{seeds[-1]} ({time.time()-t0:.0f}s)", flush=True)
            rB, rD = bd_over(seeds, tag=f"draw{di+1}/5"); bd = float(np.mean(rB - rD)); bds.append(bd)
            details.append({"draw": di, "bd": bd, "retB": float(rB.mean()), "retD": float(rD.mean())})
            json.dump({**rep, "draws_so_far": bds, "details": details}, open(OUT, "w"), indent=2)
            print(f"[FLOOR:{CELL}:{BLOCK}] draw {di+1}/5 (seeds {seeds[0]}-{seeds[-1]}): B-D={bd:+.4f} ({time.time()-t0:.0f}s)", flush=True)
        floor = float(np.std(bds, ddof=1))
        rep.update({"n_seeds": 30, "n_draws": 5, "bd_draws": bds, "bd_mean": float(np.mean(bds)),
                    "FLOOR_sd_across_draws": floor, "details": details})
        print(f"\n[FLOOR:{CELL}:{BLOCK}] === seed-sampling floor SD = {floor:.4f} (expected ~perseedSD/sqrt30) ===", flush=True)

    elif BLOCK == "drift":  # 1 estimate, original seeds, bf16 -> gap from banked anchor
        seeds = list(range(30)); rB, rD = bd_over(seeds, tag="drift"); bd = float(np.mean(rB - rD))
        per = (rB - rD); gap = bd - cfg["anchor_BD"]
        rep.update({"n_seeds": 30, "bd_mean": bd, "bd_se": float(np.std(per, ddof=1) / 30 ** 0.5),
                    "retB": float(rB.mean()), "retD": float(rD.mean()), "A_forget": pre_sst - float(rD.mean()),
                    "DRIFT_gap_from_anchor": gap})
        json.dump(rep, open(OUT, "w"), indent=2)
        print(f"\n[FLOOR:{CELL}:{BLOCK}] === DRIFT: B-D={bd:+.4f} vs banked anchor {cfg['anchor_BD']:+.4f} -> gap={gap:+.4f} (LARGE => old numbers not comparable) ===", flush=True)

    elif BLOCK == "seedcurve":  # N=100 fp32, keep per-seed B-D, bootstrap SE at N=10..100
        seeds = list(range(100)); rB, rD = bd_over(seeds, tag="curve"); per = (rB - rD)
        rng = np.random.default_rng(0); curve = {}
        for N in [10, 20, 30, 40, 60, 100]:
            ses = [float(np.mean(rng.choice(per, size=N, replace=True))) for _ in range(2000)]
            curve[str(N)] = float(np.std(ses, ddof=1))
        cross = next((N for N in [10, 20, 30, 40, 60, 100] if curve[str(N)] < 0.02), None)
        rep.update({"n_seeds": 100, "bd_mean": float(per.mean()), "per_seed_sd": float(np.std(per, ddof=1)),
                    "SE_vs_N": curve, "N_cross_0p02": cross})
        json.dump(rep, open(OUT, "w"), indent=2)
        print(f"\n[FLOOR:{CELL}:{BLOCK}] === seed-curve SE: " + " ".join(f"N{N}={curve[str(N)]:.4f}" for N in [10,20,30,40,60,100]) + f" -> SE<0.02 at N={cross} ===", flush=True)

    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[FLOOR:{CELL}:{BLOCK}] saved {OUT} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
