"""DEPTH-SWEEP w_A producer (Layer-2 feed). Args: MODEL KEEP TAG.  e.g. python surgery_wa_lambda.py facebook/opt-350m 12 opt350m_d12
Load pretrained OPT -> keep KEEP evenly-spaced decoder layers (reindex layer_idx) -> matched-acc SST retrain (~0.80) ->
save surgered pinned w_A. Depth-in-regime variant for the SPIRAL dissociation. Engine untouched.
"""
import os, sys, json, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import numpy as np, torch, torch.nn as nn
G = "/home/ubuntu/gateA"
MODEL = sys.argv[1]; KEEP = int(sys.argv[2]); TAG = sys.argv[3]
CKPT = f"{G}/{TAG}_sst2_wA.pt"; OUT = f"{G}/{TAG}_wa.json"
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
LR_WA = 5e-5; BATCH = 16; SUBSET = 2000; MAXSTEPS = 1500; EVAL_EVERY = 250; TARGET = 0.80


def sst_rows(split, k, seed=0):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")[split]
    assert len(ds) > 400, f"SST load wrong: {len(ds)}"
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))[:k]]


def sst_items(k, seed): return [(f"{s}\nSentiment:", LAB[l]) for s, l in sst_rows("train", k, seed)]


def b_pairs(tok, items):
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"][:128]; c = tok(co, add_special_tokens=False)["input_ids"][:8]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    L = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


def eval_acc(model, tok, ev, posid, negid):
    model.eval(); cor = 0
    with torch.no_grad():
        for i in range(0, len(ev), 32):
            ch = ev[i:i + 32]; seqs = [tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"][:128] for s, _ in ch]
            L = max(len(x) for x in seqs); inp = np.full((len(seqs), L), tok.pad_token_id, np.int64); pos = np.zeros(len(seqs), np.int64)
            for j, sq in enumerate(seqs): inp[j, :len(sq)] = sq; pos[j] = len(sq) - 1
            lg = model(input_ids=torch.tensor(inp).to(dev)).logits[torch.arange(len(seqs)), torch.tensor(pos).to(dev)]
            pr = (lg[:, posid] > lg[:, negid]).long().cpu().numpy()
            for j, (_, lab) in enumerate(ch): cor += int(pr[j] == lab)
    return cor / len(ev)


def main():
    t0 = time.time(); print(f"[SWA] {MODEL} keep={KEEP} tag={TAG}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(dev)
    L0 = len(model.model.decoder.layers)
    keep_idx = sorted(set(np.linspace(0, L0 - 1, KEEP).round().astype(int).tolist()))   # evenly-spaced, early->deep span
    model.model.decoder.layers = nn.ModuleList([model.model.decoder.layers[i] for i in keep_idx])
    model.config.num_hidden_layers = len(keep_idx)
    for new_i, lyr in enumerate(model.model.decoder.layers):    # reindex layer_idx (KV-cache)
        if hasattr(lyr, "self_attn") and hasattr(lyr.self_attn, "layer_idx"): lyr.self_attn.layer_idx = new_i
    print(f"[SWA] depth {L0} -> {len(keep_idx)} kept={keep_idx} ({sum(p.numel() for p in model.parameters())/1e6:.0f}M)", flush=True)

    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sstA = sst_items(SUBSET, 7); ev = sst_rows("validation", 300, 0)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_WA); g = torch.Generator().manual_seed(0)
    n = len(sstA); step = 0; best = 0.0; stop = MAXSTEPS; saved = False
    while step < MAXSTEPS:
        model.train(); idx = torch.randperm(n, generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [sstA[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
        if step % EVAL_EVERY == 0:
            acc = eval_acc(model, tok, ev, posid, negid); best = max(best, acc)
            print(f"[SWA] step {step}: acc={acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
            if acc >= TARGET:
                torch.save(model.state_dict(), CKPT); saved = True; stop = step
                print(f"[SWA] target hit {acc:.3f} @ {step}; saved {CKPT}", flush=True); break
    if not saved:
        acc = eval_acc(model, tok, ev, posid, negid); best = max(best, acc); torch.save(model.state_dict(), CKPT)
        print(f"[SWA] MAXSTEPS; final acc={acc:.3f} (below {TARGET}) saved best-effort {CKPT}", flush=True)
    json.dump({"model": MODEL, "tag": TAG, "keep": KEEP, "depth_orig": L0, "kept_idx": keep_idx,
               "wA_sst_acc": float(best), "stop_step": int(stop), "matched": bool(best >= TARGET)}, open(OUT, "w"), indent=2)
    print(f"[SWA] DONE tag={TAG} keep={KEEP} acc={best:.3f} matched={best>=TARGET} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
