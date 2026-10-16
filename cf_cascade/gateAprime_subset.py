"""Gate A' (subset) — full SST-2 (67k) would not interpolate in budget (train loss
plateaued ~0.20 = non-stationary). Fix: train on a 2k SST-2 subset -> 410M interpolates
fast -> GENUINE sharp PSD minimum (small gradient = clean damage baseline for c2; sharp
basin = binding dynamic range for thermo). Serves both lanes off one checkpoint.

Certify the minimum properly (knife 1): lambda_MIN (most-negative Hessian eig, via
SHIFTED power iteration) is the real PSD gate -- lambda_max-magnitude alone is blind the
same way K-FAC is. lambda_min > 0 = genuine PSD minimum; < 0 = still a saddle.
"""
import os, json, time
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import numpy as np, torch

MODEL = "EleutherAI/pythia-410m"; OUT = "D:/IC_experiments/cf_cascade/gateA"
SUBSET = 2000; BATCH = 16; LR = 5e-5; MAX_STEPS = 5000; EVAL_EVERY = 200; TRAINLOSS_STOP = 0.02
HVP_BATCHES = 12; POWER_ITERS = 18
LABELS = {0: " negative", 1: " positive"}; SQUARE = "attention.dense"
dev = "cuda"; torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True


def load_sst2_subset():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")["train"]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    idx = np.random.default_rng(0).permutation(len(rows))[:SUBSET]
    return [rows[i] for i in idx]


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


def hvp_closure(model, params, fixed):
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


def power_iter(hvp, params, shift=None):
    """Largest eig of H (shift=None) or of (shift*I - H) (shift set, for lambda_min)."""
    v = [torch.randn_like(p) for p in params]; nrm = sum(x.pow(2).sum() for x in v) ** 0.5; v = [x / nrm for x in v]
    lam = 0.0
    for _ in range(POWER_ITERS):
        Hv = hvp(v)
        if shift is not None:
            Hv = [shift * x - hv for hv, x in zip(Hv, v)]
        lam = float(sum((hv * x).sum() for hv, x in zip(Hv, v)))
        nrm = float(sum(hv.pow(2).sum() for hv in Hv) ** 0.5) + 1e-12; v = [hv / nrm for hv in Hv]
    return lam


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    train = load_sst2_subset()
    print(f"[A'sub] SST-2 subset = {len(train)} examples", flush=True)

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR); g = torch.Generator().manual_seed(0)
    run = []; step = 0; hist = []
    while step < MAX_STEPS:
        order = torch.randperm(len(train), generator=g).tolist()
        for i in range(0, len(order) - BATCH, BATCH):
            inp, lb, am = make_batch(tok, [train[k] for k in order[i:i + BATCH]])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            run.append(float(loss.item())); run = run[-80:]
            if step % EVAL_EVERY == 0:
                wl = float(np.mean(run)); hist.append({"step": step, "train_win": wl})
                print(f"  step {step:4d}  train_win {wl:.4f}  ({time.time()-t0:.0f}s)", flush=True)
                if wl < TRAINLOSS_STOP: break
        if step >= MAX_STEPS or (run and np.mean(run) < TRAINLOSS_STOP): break
    torch.save(model.state_dict(), f"{OUT}/pythia410m_sst2_subsetmin.pt")
    spans = [esd_decade(p.detach().float().cpu().numpy()) for n, p in model.state_dict().items()
             if SQUARE in n and tuple(p.shape) == (1024, 1024)]
    esd_med = float(np.median(spans))
    del model, opt; torch.cuda.empty_cache()

    # certify (eager attn): grad-norm + lambda_max + lambda_min(shifted)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(dev)
    model.load_state_dict(torch.load(f"{OUT}/pythia410m_sst2_subsetmin.pt")); model.eval()
    for p in model.parameters(): p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    # full-subset gradient norm (true stationarity)
    model.zero_grad(set_to_none=True)
    nb = len(train) // BATCH
    for i in range(0, len(train) - BATCH, BATCH):
        inp, lb, am = make_batch(tok, train[i:i + BATCH])
        (model(input_ids=inp, attention_mask=am, labels=lb).loss / nb).backward()
    gn = float(sum(p.grad.pow(2).sum() for p in params if p.grad is not None) ** 0.5)
    model.zero_grad(set_to_none=True)
    g2 = torch.Generator().manual_seed(1)
    fixed = [make_batch(tok, [train[k] for k in torch.randperm(len(train), generator=g2)[:BATCH].tolist()]) for _ in range(HVP_BATCHES)]
    hvp = hvp_closure(model, params, fixed)
    lam_max = power_iter(hvp, params)
    shift = abs(lam_max) * 1.5 + 10.0
    lam_min = shift - power_iter(hvp, params, shift=shift)
    is_min = lam_min > -1e-2          # PSD (allow tiny numerical negative)

    print(f"\n[A'sub] === GENUINE-MINIMUM CHECK (SST-2 2k subset) ===", flush=True)
    print(f"  stopped step {step}, train_win {np.mean(run):.4f}", flush=True)
    print(f"  full-subset grad-norm = {gn:.4f}   (saddle was 1.19)", flush=True)
    print(f"  lambda_max = {lam_max:.2f}", flush=True)
    print(f"  lambda_min = {lam_min:.3f}   (THE PSD GATE; >0 = genuine min, <0 = saddle)", flush=True)
    print(f"  ESD decade-span median = {esd_med:.2f}", flush=True)
    print(f"  VERDICT: {'GENUINE PSD MINIMUM - both lanes proceed' if is_min else 'STILL A SADDLE (lambda_min<0) - switch task A'}", flush=True)
    json.dump({"subset": SUBSET, "stopped_step": step, "train_win": float(np.mean(run)),
               "grad_norm_full_subset": gn, "lambda_max": lam_max, "lambda_min": lam_min,
               "is_psd_minimum": bool(is_min), "esd_decade_median": esd_med, "train_hist": hist,
               "wall_s": time.time() - t0},
              open(f"{OUT}/gateAprime_subset_summary.json", "w"), indent=2)
    print(f"[A'sub] saved pythia410m_sst2_subsetmin.pt, gateAprime_subset_summary.json  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
