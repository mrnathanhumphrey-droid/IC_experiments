"""w_A TRAINER (size-matrix feed, Layer 2). Family-agnostic, size-parameterized: trains task-A=SST -> w_A and saves the
pinned checkpoint for the other agent's G1-G6 + dispositive script to stamp onto. MATCHED-ACC early-stop (de-confound:
size must not be confounded with w_A accuracy) — stop when val acc >= TARGET (0.80; anchors Pythia-410m 0.77 / OPT-350m
0.82) or MAXSTEPS. Reuses the rung2_fragile_opt SST regime verbatim (STEPS/LR/BATCH/SUBSET/prompt format) so checkpoints
are comparable to the banked band-B anchors. Engine untouched. Args: MODEL TAG.  e.g. python wa_trainer_lambda.py EleutherAI/pythia-160m pythia160m
"""
import os, sys, json, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = sys.argv[1]; TAG = sys.argv[2]
REVISION = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != "-" else None   # e.g. step16000 (Pythia length knob)
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
    t0 = time.time(); print(f"[WA] {MODEL} (tag {TAG}) target acc>={TARGET}", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _rev = {"revision": REVISION} if REVISION else {}
    if REVISION: print(f"[WA] revision={REVISION}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL, **_rev)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager", **_rev).to(dev)
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sstA = sst_items(SUBSET, 7); ev = sst_rows("validation", 300, 0)
    opt = torch.optim.AdamW(model.parameters(), lr=LR_WA); g = torch.Generator().manual_seed(0)
    n = len(sstA); step = 0; best_acc = 0.0; stop_step = MAXSTEPS
    saved = False
    while step < MAXSTEPS:
        model.train(); idx = torch.randperm(n, generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [sstA[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
        if step % EVAL_EVERY == 0:
            acc = eval_acc(model, tok, ev, posid, negid); best_acc = max(best_acc, acc)
            print(f"[WA] step {step}: SST acc={acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
            if acc >= TARGET:
                torch.save(model.state_dict(), CKPT); saved = True; stop_step = step
                print(f"[WA] target hit acc={acc:.3f} at step {step}; saved {CKPT}", flush=True); break
    if not saved:
        acc = eval_acc(model, tok, ev, posid, negid); best_acc = max(best_acc, acc)
        torch.save(model.state_dict(), CKPT)
        print(f"[WA] MAXSTEPS reached; final acc={acc:.3f} (below target {TARGET}) — saved best-effort {CKPT}", flush=True)
    json.dump({"model": MODEL, "tag": TAG, "wA_sst_acc": float(best_acc), "stop_step": int(stop_step),
               "target": TARGET, "matched": bool(best_acc >= TARGET), "lr": LR_WA, "subset": SUBSET},
              open(OUT, "w"), indent=2)
    print(f"[WA] DONE tag={TAG} acc={best_acc:.3f} stop_step={stop_step} matched={best_acc>=TARGET} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
