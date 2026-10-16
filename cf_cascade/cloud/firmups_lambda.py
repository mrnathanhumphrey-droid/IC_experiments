"""FIRM-UPS for the TEST#1 free-lane reopening (Wilson-required before building projection lever).
FIRM-UP 1 (k-sweep): does the free lane stay open as KTOP grows, or does overlap->total (lane closes)?
FIRM-UP 2 (second substrate): does the free lane reproduce on a task-B other than AG-News, or is it AG-specific?
  - agnews (topic, DISSIMILAR from SST-2 sentiment) = re-confirm the original + k-sweep it.
  - yelp   (sentiment, SIMILAR to task-A) = Wilson's named task-axis point. CONFOUND-AWARE: similarity alone
    would close the lane; a closed Yelp lane does NOT by itself mean "AG-specific".
  - subj   (subjectivity, DISSIMILAR) = the CLEAN reproduction point — if the lane reopens here too, it's a
    dissimilar-task CF property, not AG-News-specific.
Per task-B: reset model to w_A, train task-B full, FB=E_B[g^T g] at w_B, subspace-overlap top-k(FA) vs top-k(FB)
swept over k. FA (task-A Fisher at w_A) computed ONCE. DECISION: lane stays sub-total across k AND reopens on a
DISSIMILAR second substrate -> projection lever is real. Lane closes at higher k, or only AG -> top-64/task artifact.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"; OUT = f"{G}/firmups.json"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
TASKS = [("agnews", "low_agnews"), ("yelp", "high_yelp"), ("subj", "medium_subj")]
KSWEEP = [8, 16, 32, 64, 128, 256]
BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02; N_FISHER = 150; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True


def sst_trA():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")["train"]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:max(N_FISHER, 200)]]


def load_taskB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


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
    t0 = time.time(); print("[FU] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.load_state_dict(torch.load(CKPT))
    trA = sst_trA(); params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]

    # ---- FA (task-A Fisher at w_A), eig once per carrier ----
    FA = fisher(model, params, names, tok, trA, False)
    VA = {n: torch.linalg.eigh(FA[n])[1] for n in names}; D = {n: FA[n].shape[0] for n in names}
    print(f"[FU] FA done ({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for taskkey, jsonname in TASKS:
        model.load_state_dict(torch.load(CKPT))                      # reset to w_A
        B = load_taskB(jsonname)
        trainloss, steps = train_taskB(model, tok, B)
        FB = fisher(model, params, names, tok, B, True)
        VB = {n: torch.linalg.eigh(FB[n])[1] for n in names}
        print(f"\n[FU] === {taskkey} ({jsonname}) trainloss={trainloss:.3f} steps={steps} ({time.time()-t0:.0f}s) ===", flush=True)
        per_carrier = {}
        for c in CARRIERS:
            ksweep = {}
            for k in KSWEEP:
                ovs, chs = [], []
                for n in [x for x in names if c in x]:
                    if k > D[n]: continue
                    PA = VA[n][:, -k:]; PB = VB[n][:, -k:]
                    ovs.append(float((PA.T @ PB).pow(2).sum().item() / k)); chs.append(k / D[n])
                if ovs:
                    o, ch = float(np.mean(ovs)), float(np.mean(chs))
                    ksweep[str(k)] = {"overlap": o, "chance": ch, "ratio": o / ch}
            per_carrier[c] = ksweep
            line = "  ".join(f"k{k}:{ksweep[str(k)]['overlap']:.3f}/{ksweep[str(k)]['ratio']:.1f}x" for k in KSWEEP if str(k) in ksweep)
            print(f"  [{c:26s}] {line}", flush=True)
        results[taskkey] = {"json": jsonname, "trainloss": trainloss, "steps": steps, "ksweep": per_carrier}

    json.dump({"tasks": results, "ksweep_vals": KSWEEP, "note": "overlap=frac shared energy top-k(FA) vs top-k(FB); ratio=overlap/chance; free lane = ratio>>1 AND overlap<<1 sustained across k", "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"\n[FU] saved firmups.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
