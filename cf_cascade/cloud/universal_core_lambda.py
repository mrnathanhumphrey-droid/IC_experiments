"""UNIVERSAL-CORE TEST (Wilson-blessed fork #1): is forgetting's irreducible antagonistic core a property of the
NETWORK (a fixed low-dim contested resource every task fights task-A over) or of the TASK-PAIR (a different fight
each time)? The projection lever isolated the core (the ~0.01 residual it couldn't remove); this asks if that core
is the SAME across task-Bs.

CONSTRUCTION: PA = top-KTOP(FA) (task-A important input-dirs, ONE frame). For each task-B: PB = top-KTOP(FB@w_B);
M = PA^T PB (KTOP x KTOP); u_B = top-M_CORE left-singular vecs of M = the task-A-eigenbasis coords of the
contested (shared) directions for that pair. Because PA is shared across task-Bs, comparing u_AG/u_Yelp/u_Subj in
R^KTOP IS the universal-core test (exact, lower-dim).
READ (chance-corrected, the corroboration the exciting answer needs): pairwise mutual overlap ||u_i^T u_j||_F^2/M_CORE
vs chance M_CORE/KTOP. >>chance (ratio>3) => contested cores POINT THE SAME WAY => UNIVERSAL contested resource (the
cool answer). ~chance => task-pair-specific, fight moves around.
Carriers qkv+mlp_h4h+mlp_4hh, per-layer then mean. Rides along: prints all prompt templates (format confirmation).
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"; OUT = f"{G}/universal_core.json"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
TASKS = [("agnews", "low_agnews"), ("yelp", "high_yelp"), ("subj", "medium_subj")]
KTOP = 64; M_CORE = 16
BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02; N_FISHER = 150; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True


def sst_trA():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")["train"]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:max(N_FISHER, 200)]]


def load_taskB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]], t


def b_sst(tok, items):
    seqs, labs = [], []
    for s, l in items:
        p = tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"]; c = tok(LAB[l], add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    return pack(tok, seqs, labs)


def b_pair(tok, items):
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"]; c = tok(co, add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    return pack(tok, seqs, labs)


def pack(tok, seqs, labs):
    L = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


def fisher(model, params, names, tok, items, is_pair):
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    for it in items[:N_FISHER]:
        inp, lb, am = (b_pair(tok, [it]) if is_pair else b_sst(tok, [it])); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: g = params[n].grad.detach().double(); F[n] += g.T @ g
    return F


def train_taskB(model, tok, B):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR); g = torch.Generator().manual_seed(0); run = []; step = 0; done = False
    while step < MAX_STEPS and not done:
        for i in range(0, len(B) - BATCH, BATCH):
            inp, lb, am = b_pair(tok, [B[k] for k in torch.randperm(len(B), generator=g)[:BATCH].tolist()])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            run.append(float(loss.item())); run = run[-80:]
            if step % 200 == 0 and np.mean(run) < TRAINSTOP: done = True; break
            if step >= MAX_STEPS: done = True; break
        if (run and np.mean(run) < TRAINSTOP) or step >= MAX_STEPS: done = True
    model.eval(); return float(np.mean(run)) if run else float("nan"), step


def main():
    t0 = time.time(); print("[UC] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.load_state_dict(torch.load(CKPT))
    trA = sst_trA(); params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]

    FA = fisher(model, params, names, tok, trA, False)
    PA = {n: torch.linalg.eigh(FA[n])[1][:, -KTOP:] for n in names}          # task-A top-KTOP, shared frame
    print(f"[UC] FA done ({time.time()-t0:.0f}s)", flush=True)

    # ---- per task-B: train, FB@w_B, contested-core coords u_B = top-M_CORE left-svecs of PA^T PB ----
    fmts = {"taskA_sst2": "{s}\\nSentiment: + LAB{ negative, positive}"}
    uB = {}                                                                   # name -> {taskkey: (KTOP, M_CORE)}
    for taskkey, jsonname in TASKS:
        model.load_state_dict(torch.load(CKPT)); B, tmpl = load_taskB(jsonname); fmts[taskkey] = tmpl
        loss, steps = train_taskB(model, tok, B)
        FB = fisher(model, params, names, tok, B, True)
        for n in names:
            PB = torch.linalg.eigh(FB[n])[1][:, -KTOP:]
            M = PA[n].T @ PB                                                  # (KTOP, KTOP)
            U, S, _ = torch.linalg.svd(M)
            uB.setdefault(n, {})[taskkey] = U[:, :M_CORE].contiguous()        # task-A-basis coords of contested core
        print(f"[UC] {taskkey} trained loss={loss:.3f} steps={steps} ({time.time()-t0:.0f}s)", flush=True)

    # ---- mutual overlap of contested cores across task-Bs (chance-corrected) ----
    chance = M_CORE / KTOP
    pairs = [("agnews", "yelp"), ("agnews", "subj"), ("yelp", "subj")]
    out = {}
    print(f"\n[UC] === MUTUAL OVERLAP of contested cores (chance={chance:.3f}) ===", flush=True)
    for c in CARRIERS:
        cn = [x for x in names if c in x]
        per_pair = {}
        for a, b in pairs:
            ovs = [float((uB[n][a].T @ uB[n][b]).pow(2).sum().item() / M_CORE) for n in cn]
            o = float(np.mean(ovs)); per_pair[f"{a}~{b}"] = {"overlap": o, "ratio": o / chance}
        mean_ov = float(np.mean([per_pair[k]["overlap"] for k in per_pair])); mean_ratio = mean_ov / chance
        per_pair["MEAN"] = {"overlap": mean_ov, "ratio": mean_ratio}
        out[c] = per_pair
        line = "  ".join(f"{k}:{per_pair[k]['overlap']:.3f}/{per_pair[k]['ratio']:.1f}x" for k in [f"{a}~{b}" for a, b in pairs])
        print(f"  [{c:26s}] {line}   MEAN {mean_ov:.3f}/{mean_ratio:.1f}x", flush=True)

    grand = float(np.mean([out[c]["MEAN"]["ratio"] for c in CARRIERS]))
    universal = grand > 3.0
    verdict = (f"UNIVERSAL CONTESTED RESOURCE (mutual overlap {grand:.1f}x chance >> 1 -> contested cores point the SAME way "
               f"across task-Bs -> a FIXED low-dim battleground in the net)" if universal else
               f"TASK-PAIR-SPECIFIC (mutual overlap {grand:.1f}x ~chance -> each task fights task-A elsewhere -> no fixed resource)")
    print(f"\n[UC] grand mean mutual-overlap ratio = {grand:.1f}x -> {verdict}", flush=True)
    print(f"[UC] FORMAT CHECK: taskA={fmts['taskA_sst2']} | yelp={fmts['yelp']!r} | agnews={fmts['agnews']!r} | subj={fmts['subj']!r}", flush=True)
    json.dump({"mutual_overlap": out, "grand_ratio": grand, "universal": universal, "verdict": verdict,
               "chance": chance, "ktop": KTOP, "m_core": M_CORE, "formats": fmts, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[UC] saved universal_core.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
