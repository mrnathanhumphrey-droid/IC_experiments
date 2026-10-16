"""Gate A' — retrain SST-2 to a GENUINE PSD minimum (the val-min freeze was a saddle:
grad 1.19, lambda_max -1933). Remove overfit guard, train to train-stationarity, then
certify the minimum: lambda_max must be > 0 (PSD). Shared checkpoint for both lanes.

Step 0: log the saddle's negative-curvature eigenvector (the -1933 direction) -- where
        does the unstable direction live? candidate for where forgetting onsets.
Step 1: retrain fresh -> train-loss near interpolation (no overfit guard).
Step 2: certify minimum: large-batch grad-norm + lambda_max (power-iter HVP, eager attn).
        lambda_max>0 = genuine PSD min; lambda_max<0 = still a saddle -> SST-2 has no
        usable minimum, switch task A. Re-confirm ESD (expect ~5.9, overfit-robust).
"""
import os, json, time
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import numpy as np, torch

MODEL = "EleutherAI/pythia-410m"
SADDLE_CKPT = "D:/IC_experiments/cf_cascade/gateA/pythia410m_sst2.pt"
OUT = "D:/IC_experiments/cf_cascade/gateA"
BATCH = 16; LR = 5e-5; MAX_STEPS = 6000; EVAL_EVERY = 200
TRAINLOSS_STOP = 0.03            # near-interpolation -> genuine min on overparam net
HVP_BATCHES = 12; POWER_ITERS = 16; BIGGRAD_N = 1024
LABELS = {0: " negative", 1: " positive"}; SQUARE = "attention.dense"
dev = "cuda"; torch.manual_seed(0); np.random.seed(0)
torch.backends.cudnn.benchmark = True


def load_sst2():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")
    return ([(r["sentence"].strip(), int(r["label"])) for r in ds["train"] if r["label"] in (0, 1)],
            [(r["sentence"].strip(), int(r["label"])) for r in ds["validation"] if r["label"] in (0, 1)])


def make_batch(tok, rows):
    seqs, labs = [], []
    for sent, lab in rows:
        p = tok(f"{sent}\nSentiment:", add_special_tokens=False)["input_ids"]
        c = tok(LABELS[lab], add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    L = max(len(s) for s in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


def esd_decade(W):
    W = W.astype(np.float64); ev = np.linalg.eigvalsh((W.T @ W) / W.shape[0]); ev = np.sort(ev[ev > 0])[::-1]
    return float(np.log10(ev[0] / ev[ev > 1e-6 * ev[0]][-1]))


def hvp_fn(model, params, fixed):
    def hvp(vec):
        acc = [torch.zeros_like(p) for p in params]
        for inp, lb, am in fixed:
            loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            grads = torch.autograd.grad(loss, params, create_graph=True)
            dot = sum((gp * v).sum() for gp, v in zip(grads, vec))
            hv = torch.autograd.grad(dot, params, retain_graph=False)
            for a, h in zip(acc, hv): a += h.detach()
            del grads, hv, loss
        return [a / len(fixed) for a in acc]
    return hvp


def lambda_max(model, tok, train, n_iters=POWER_ITERS):
    params = [p for p in model.parameters() if p.requires_grad]
    g = torch.Generator().manual_seed(1)
    fixed = [make_batch(tok, [train[k] for k in torch.randperm(len(train), generator=g)[:BATCH].tolist()]) for _ in range(HVP_BATCHES)]
    hvp = hvp_fn(model, params, fixed)
    v = [torch.randn_like(p) for p in params]; nrm = sum(x.pow(2).sum() for x in v) ** 0.5; v = [x / nrm for x in v]
    lam = 0.0
    for it in range(n_iters):
        Hv = hvp(v); lam = float(sum((hv * x).sum() for hv, x in zip(Hv, v)))
        nrm = float(sum(hv.pow(2).sum() for hv in Hv) ** 0.5) + 1e-12; v = [hv / nrm for hv in Hv]
    return lam, v, params


def big_grad_norm(model, tok, train, params):
    model.zero_grad(set_to_none=True)
    idx = torch.randperm(len(train), generator=torch.Generator().manual_seed(2))[:BIGGRAD_N].tolist()
    for i in range(0, BIGGRAD_N - BATCH, BATCH):
        inp, lb, am = make_batch(tok, [train[k] for k in idx[i:i + BATCH]])
        (model(input_ids=inp, attention_mask=am, labels=lb).loss / (BIGGRAD_N // BATCH)).backward()
    gn = float(sum(p.grad.pow(2).sum() for p in params if p.grad is not None) ** 0.5)
    model.zero_grad(set_to_none=True); return gn


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    train, val = load_sst2()

    # ---- Step 0: log saddle negative-curvature eigenvector ----
    print("[A'] Step 0: logging saddle eigenvector (lambda_max=-1933 direction)...", flush=True)
    msad = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(dev)
    msad.load_state_dict(torch.load(SADDLE_CKPT)); msad.eval()
    for p in msad.parameters(): p.requires_grad_(True)
    lam_s, v_s, params_s = lambda_max(msad, tok, train)
    tensor_norms = {n: float(v.norm()) for (n, _), v in zip([(n, p) for n, p in msad.named_parameters()], v_s)}
    top_tensors = sorted(tensor_norms.items(), key=lambda kv: -kv[1])[:8]
    print(f"  saddle lambda_max={lam_s:.1f}; direction concentrates in:", flush=True)
    for n, nm in top_tensors: print(f"    {n:50s} |v|={nm:.3f}", flush=True)
    np.savez_compressed(f"{OUT}/saddle_eigvec.npz", lam=lam_s, tensor_norms=json.dumps(tensor_norms))
    del msad; torch.cuda.empty_cache()

    # ---- Step 1: retrain to genuine minimum (no overfit guard) ----
    print(f"\n[A'] Step 1: retrain to train-stationarity (no overfit guard, stop train-loss<{TRAINLOSS_STOP})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR); g = torch.Generator().manual_seed(0)
    hist = []; step = 0; run = []
    while step < MAX_STEPS:
        order = torch.randperm(len(train), generator=g).tolist()
        for i in range(0, len(order) - BATCH, BATCH):
            rows = [train[k] for k in order[i:i + BATCH]]
            inp, lb, am = make_batch(tok, rows)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            run.append(float(loss.item())); run = run[-150:]
            if step % EVAL_EVERY == 0:
                wl = float(np.mean(run)); hist.append({"step": step, "train_win": wl})
                print(f"  step {step:4d}  train_win {wl:.4f}  ({time.time()-t0:.0f}s)", flush=True)
                if wl < TRAINLOSS_STOP:
                    break
        if step >= MAX_STEPS or (run and np.mean(run) < TRAINLOSS_STOP):
            break
    torch.save(model.state_dict(), f"{OUT}/pythia410m_sst2_min.pt")
    spans = [esd_decade(p.detach().float().cpu().numpy()) for n, p in model.state_dict().items()
             if SQUARE in n and tuple(p.shape) == (1024, 1024)]
    esd_med = float(np.median(spans))
    del model, opt; torch.cuda.empty_cache()

    # ---- Step 2: certify minimum (eager attn) ----
    print(f"\n[A'] Step 2: certify PSD minimum...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(dev)
    model.load_state_dict(torch.load(f"{OUT}/pythia410m_sst2_min.pt")); model.eval()
    for p in model.parameters(): p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    gn = big_grad_norm(model, tok, train, params)
    lam, _, _ = lambda_max(model, tok, train)
    is_min = lam > 0

    print(f"\n[A'] === GENUINE-MINIMUM CHECK (SST-2 retrain) ===", flush=True)
    print(f"  stopped at step {step}, train_win {np.mean(run):.4f}", flush=True)
    print(f"  large-batch grad-norm = {gn:.4f}   (saddle was 1.19)", flush=True)
    print(f"  lambda_max = {lam:.2f}   (saddle was -1933; >0 = PSD = GENUINE MIN)", flush=True)
    print(f"  re-confirmed ESD decade-span median = {esd_med:.2f}  (expect ~5.9)", flush=True)
    print(f"  VERDICT: {'GENUINE PSD MINIMUM - both lanes proceed off this checkpoint' if is_min else 'STILL A SADDLE (lambda_max<0) - SST-2 has no usable min, SWITCH task A'}", flush=True)
    json.dump({"stopped_step": step, "train_win": float(np.mean(run)), "grad_norm": gn, "lambda_max": lam,
               "is_psd_minimum": bool(is_min), "esd_decade_median": esd_med, "saddle_lambda": lam_s,
               "saddle_top_tensors": top_tensors, "train_hist": hist, "wall_s": time.time() - t0},
              open(f"{OUT}/gateAprime_summary.json", "w"), indent=2)
    print(f"[A'] saved pythia410m_sst2_min.pt, saddle_eigvec.npz, gateAprime_summary.json  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
