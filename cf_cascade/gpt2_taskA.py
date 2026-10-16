"""JOB 2 Stage 1: GPT-2-medium task-A SST-2 fine-tune to a GENUINE minimum (low grad-norm), the
foundation for the freeze gate + curvature read + diffusion battery. Mirrors the Pythia 2k-subset
interpolation approach (Pythia gave train 0.019, grad-norm 0.19). Rebuild on GPT-2 — do NOT reuse
Pythia's frozen objects. Saves gpt2med_sst2_subsetmin.pt; reports val-acc, train-loss, grad-norm.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "4"
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import json, time, numpy as np, torch
torch.set_num_threads(4)

MODEL = "gpt2-medium"; OUTCKPT = "D:/IC_experiments/cf_cascade/gateA/gpt2med_sst2_subsetmin.pt"
OUT = "D:/IC_experiments/cf_cascade/gateA/gpt2_taskA_summary.json"
BATCH = 16; LR = 5e-5; MAX_STEPS = 4000; TRAINSTOP = 0.02; SUBSET = 2000; VAL_N = 400
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True


def sst():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")
    tr = [(r["sentence"].strip(), int(r["label"])) for r in ds["train"] if r["label"] in (0, 1)]
    tr = [tr[i] for i in np.random.default_rng(0).permutation(len(tr))[:SUBSET]]
    va = [(r["sentence"].strip(), int(r["label"])) for r in ds["validation"] if r["label"] in (0, 1)][:VAL_N]
    return tr, va


def batch(tok, items):
    seqs, labs = [], []
    for s, l in items:
        p = tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"]; c = tok(LAB[l], add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    L = max(len(s) for s in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


@torch.no_grad()
def acc(model, tok, va):
    model.eval(); c = 0
    for s, l in va:
        sc = {}
        for cl in (0, 1):
            inp, lb, am = batch(tok, [(s, cl)]); sc[cl] = float(model(input_ids=inp, attention_mask=am, labels=lb).loss)
        c += int(min(sc, key=sc.get) == l)
    model.train(); return c / len(va)


def main():
    t0 = time.time(); print("[A] loading gpt2-medium...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    tr, va = sst(); print(f"[A] loaded ({time.time()-t0:.0f}s)", flush=True)
    acc0 = acc(model, tok, va); print(f"[A] SST-2 acc before = {acc0:.3f}", flush=True)
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR); g = torch.Generator().manual_seed(0)
    run = []; step = 0; done = False
    while step < MAX_STEPS and not done:
        order = torch.randperm(len(tr), generator=g).tolist()
        for i in range(0, len(order) - BATCH, BATCH):
            inp, lb, am = batch(tok, [tr[k] for k in order[i:i + BATCH]])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            run.append(float(loss.item())); run = run[-80:]
            if step % 200 == 0:
                print(f"  [A] step {step} train_win {np.mean(run):.4f} ({time.time()-t0:.0f}s)", flush=True)
            if step % 100 == 0 and np.mean(run) < TRAINSTOP: done = True; break
            if step >= MAX_STEPS: done = True; break
    # grad-norm over a few batches (minimum certification)
    model.eval(); gn = 0.0
    for _ in range(8):
        inp, lb, am = batch(tok, [tr[k] for k in torch.randperm(len(tr), generator=g)[:BATCH].tolist()])
        model.zero_grad(set_to_none=True); model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        gn += float(torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters() if p.grad is not None)))
    gn /= 8
    acc1 = acc(model, tok, va)
    torch.save(model.state_dict(), OUTCKPT)
    print(f"\n[A] === GPT-2 task-A min: train_win {np.mean(run):.4f}  grad-norm {gn:.3f}  SST-2 acc {acc0:.3f}->{acc1:.3f}  steps {step} ===", flush=True)
    json.dump({"train_win": float(np.mean(run)), "grad_norm": gn, "acc_before": acc0, "acc_after": acc1,
               "steps": step, "ckpt": OUTCKPT, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[A] saved {OUTCKPT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
