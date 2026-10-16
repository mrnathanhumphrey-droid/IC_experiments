"""PROBE 3 — childhood dose-response. Does map load-bearingness switch on / shift with deposit STRENGTH (childhood steps)?
Pre-reg PREREG_PROBE3_DOSE_2026_06_23.md (LOCK 923085db). Band cells ONLY (opt350m/pythia410m; dead at 1B -> HARD-STOP).
Dose ladder D in {100,200,400,800} (LOCKED); FIXED stream, only step count varies. Per dose: arm A (dose-indep) / B_D
(full deposit) / Dproj_D (childhood grad projected out of V). SWAP at D=400. Reads B-A, B-D, mapfrac, A_forget, lossdrop
per dose. Reproduce-anchor at D=400 (B-A vs banked). HARD-STOP-flag any dose whose A_forget deviates >0.05 from D=400.
"""
import os, time, json, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
SEEDS = list(range(30))
LR_DEPTH = 5e-5; STEPS_DEPTH = 250; LR_CHILD = 3e-5; BATCH = 16; KPROJ = 128; N_FISHER = 200
DOSES = [100, 200, 400, 800]; ANCHOR_DOSE = 400
OPT_C = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
PY_C = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
CELLS = {
    "opt350m":    {"model": "facebook/opt-350m",      "ckpt": "opt350m_sst2_wA.pt",            "ck": OPT_C, "family": "OPT",    "anchor_BmA": 0.089},
    "pythia410m": {"model": "EleutherAI/pythia-410m",  "ckpt": "pythia410m_sst2_subsetmin.pt",  "ck": PY_C,  "family": "PYTHIA", "anchor_BmA": 0.073},
}
CELL = os.environ.get("CELL", ""); assert CELL in CELLS, f"CELL band-only; got {CELL!r}"
cfg = CELLS[CELL]; OUT = f"{G}/probe3_dose_{CELL}.json"


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
def childhood(model, tok, stream, steps, lr, seed, Vproj=None, carriers=None):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    tasks = list(stream); params = dict(model.named_parameters()); losses = []
    for step in range(steps):
        t = tasks[step % len(tasks)]; items, _ = stream[t]
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward()
        if Vproj is not None:
            for n in carriers: gd = params[n].grad; gd.sub_(((gd.double() @ Vproj[n].double()) @ Vproj[n].double().T).to(gd.dtype))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.item()))
    del opt; torch.cuda.empty_cache(); return float(np.mean(losses[:20])), float(np.mean(losses[-20:]))
def phase2(model, tok, ho_tr, sst_eval, posid, negid, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR_DEPTH); g = torch.Generator().manual_seed(seed)
    for _ in range(STEPS_DEPTH):
        idx = torch.randperm(len(ho_tr), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [ho_tr[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt; r = eval_sst(model, tok, sst_eval, posid, negid); torch.cuda.empty_cache(); return r
def labels_of(items): return sorted(set(c for _, c in items))


def main():
    t0 = time.time(); print(f"[P3:{CELL}] {cfg['model']} dose-response D={DOSES}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(f"{G}/{cfg['ckpt']}"))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    NC = len(carriers); posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0); ckpt_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    pre_sst = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[P3:{CELL}] w_A acc={pre_sst:.3f}; {NC} carriers ({time.time()-t0:.0f}s)", flush=True)
    FA = fisher_taskA(model, params, carriers, tok); V = {}
    for n in carriers:
        _, vv = torch.linalg.eigh(FA[n].double()); V[n] = vv[:, -KPROJ:].flip(dims=[1]).contiguous(); del FA[n]
    grng = torch.Generator(device="cpu").manual_seed(12345); S_swap = {}
    for n in carriers:
        dout = params[n].shape[0]; cin = torch.randn(KPROJ, 1, generator=grng).double(); vin = (V[n].cpu() @ cin); vin = (vin / vin.norm()).to(dev)
        vout = torch.randn(dout, 1, generator=grng).double().to(dev); vout = vout / vout.norm(); Sw = (vout @ vin.T); S_swap[n] = Sw / (Sw.norm() + 1e-30)
    model.load_state_dict(ckpt_sd); torch.cuda.empty_cache()
    print(f"[P3:{CELL}] V + swap done ({time.time()-t0:.0f}s)", flush=True)
    ho_tr = load_jsonB("medium_subj")[:-200]
    stream = {sk: (load_jsonB("low_agnews") if sk == "agnews" else load_hf(sk)) for sk in ["agnews", "dbpedia", "yahoo", "newscat", "emotion"]}
    stream = {sk: (it, labels_of(it)) for sk, it in stream.items()}

    retA = []; retB = {D: [] for D in DOSES}; retD = {D: [] for D in DOSES}; retSwap = []
    mapfrac = {D: [] for D in DOSES}; lossdrop = {D: [] for D in DOSES}
    for sd in SEEDS:
        model.load_state_dict(ckpt_sd); retA.append(phase2(model, tok, ho_tr, sst_eval, posid, negid, sd))
        for D in DOSES:
            # arm B_D: full childhood dose D -> phase2
            model.load_state_dict(ckpt_sd); l0, l1 = childhood(model, tok, stream, D, LR_CHILD, sd); lossdrop[D].append(l0 - l1)
            Pc = dict(model.named_parameters()); mf_num = 0.0; mf_den = 0.0
            for n in carriers:
                df = (Pc[n].detach().double() - ckpt_sd[n].to(dev).double()); dv = (df @ V[n]) @ V[n].T
                mf_num += float((dv * dv).sum()); mf_den += float((df * df).sum())
            mapfrac[D].append(mf_num / (mf_den + 1e-30))
            retB[D].append(phase2(model, tok, ho_tr, sst_eval, posid, negid, sd))
            # arm Dproj_D: childhood grad projected out of V, dose D -> phase2
            model.load_state_dict(ckpt_sd); childhood(model, tok, stream, D, LR_CHILD, sd, Vproj=V, carriers=carriers)
            retD[D].append(phase2(model, tok, ho_tr, sst_eval, posid, negid, sd))
            # SWAP arm at the anchor dose only
            if D == ANCHOR_DOSE:
                model.load_state_dict(ckpt_sd)
                # match swap magnitude to this seed's full in-V deposit magnitude (median carrier ||dv||)
                with torch.no_grad():
                    Ps = dict(model.named_parameters())
                    for n in carriers:
                        df = (Pc[n].detach().double() - ckpt_sd[n].to(dev).double()); dv = (df @ V[n]) @ V[n].T
                        Ps[n].add_((float(dv.norm()) * S_swap[n]).to(Ps[n].dtype))
                retSwap.append(phase2(model, tok, ho_tr, sst_eval, posid, negid, sd))
        if sd % 5 == 0 or sd == SEEDS[-1]:
            msg = " ".join(f"D{D}:B={retB[D][-1]:.3f}/D={retD[D][-1]:.3f}" for D in DOSES)
            print(f"[P3:{CELL}] seed {sd}: A={retA[-1]:.3f} {msg} ({time.time()-t0:.0f}s)", flush=True)

    ns = len(SEEDS); A = np.array(retA); me = lambda x: [float(np.mean(x)), float(np.std(x, ddof=1) / max(len(x), 1) ** 0.5)]
    per = {}; aforget = {}
    for D in DOSES:
        B = np.array(retB[D]); Dd = np.array(retD[D])
        per[str(D)] = {"retB": me(B), "retD": me(Dd), "B_minus_A": me((B - A).tolist()), "B_minus_D": me((B - Dd).tolist()),
                       "mapfrac_B": me(mapfrac[D]), "childhood_lossdrop": me(lossdrop[D])}
        aforget[D] = pre_sst - float(np.mean(Dd))
    a4 = aforget[ANCHOR_DOSE]; band_flags = {str(D): bool(abs(aforget[D] - a4) > 0.05) for D in DOSES}
    BmA4 = per[str(ANCHOR_DOSE)]["B_minus_A"][0]; anchor_ok = abs(BmA4 - cfg["anchor_BmA"]) < 0.03
    rep = {"cell": CELL, "family": cfg["family"], "model": cfg["model"], "n_carriers": NC, "n_seeds": ns, "w_A_acc": pre_sst,
           "DOSES": DOSES, "retA": me(A), "per_dose": per, "A_forget_per_dose": {str(D): aforget[D] for D in DOSES},
           "offband_flag": band_flags, "swap_minus_A_at400": me((np.array(retSwap) - A).tolist()) if retSwap else None,
           "anchor_BmA_expected": cfg["anchor_BmA"], "anchor_BmA_at400": BmA4, "ANCHOR_OK": bool(anchor_ok)}
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"\n[P3:{CELL}] === ANCHOR B-A(D400)={BmA4:+.3f}(exp {cfg['anchor_BmA']:+.3f}) -> {'OK' if anchor_ok else 'MISMATCH!!'} ===", flush=True)
    for D in DOSES:
        p = per[str(D)]; fl = " OFF-BAND!" if band_flags[str(D)] else ""
        print(f"   D={D:4d}  B-A={p['B_minus_A'][0]:+.3f}  B-D={p['B_minus_D'][0]:+.3f}  mapfrac={p['mapfrac_B'][0]:.3f}  A_forget={aforget[D]:.3f}{fl}", flush=True)
    if retSwap: print(f"   swap(D400)-A={rep['swap_minus_A_at400'][0]:+.3f} vs real B-A(D400)={BmA4:+.3f}", flush=True)
    print(f"[P3:{CELL}] saved {OUT} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
