"""Gate A force-readiness — the THIRD number (curvature) + a CLEAN stationarity number.

The SST-2 run froze at the val-min (overfit guard), grad_norm(8-batch)=2.67. But:
  (a) 8-batch grad-norm conflates distance-from-min with minibatch noise -> measure
      a LARGE-BATCH gradient norm (true stationarity; -> 0 at a real minimum).
  (b) val-min is NOT a train-stationary point -> the force lane needs the basin bottom.
  (c) curvature was never measured -> is the basin genuinely CURVED (binding has dynamic
      range) or flat-and-solved (low binding everywhere)?

Computes on the frozen SST-2 checkpoint, SST-2 completion loss:
  - large-batch grad norm (1024 examples) = clean stationarity
  - top Hessian eigenvalue lambda_max (power iteration via HVP) = sharpness / max binding
  - Hessian trace (Hutchinson) -> mean curvature lambda_mean = trace/N
  - dynamic range lambda_max/lambda_mean = is binding spread across directions
Read: lambda_max substantial AND dynamic range >> 1 -> genuinely curved, force lane has
binding dynamic range. lambda_max ~ 0 -> flat-and-solved -> sharper task A needed.
"""
import os, json, time
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import numpy as np, torch

MODEL = "EleutherAI/pythia-410m"; CKPT = "D:/IC_experiments/cf_cascade/gateA/pythia410m_sst2.pt"
BATCH = 8; HVP_BATCHES = 12; POWER_ITERS = 15; HUTCH = 8; BIGGRAD_N = 1024
LABELS = {0: " negative", 1: " positive"}
dev = "cuda"; torch.manual_seed(0)


def load_sst2_train():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")["train"]
    return [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]


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


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(dev)
    model.load_state_dict(torch.load(CKPT)); model.eval()
    for p in model.parameters(): p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    N = sum(p.numel() for p in params)
    train = load_sst2_train()
    g = torch.Generator().manual_seed(1)
    batches = [[train[k] for k in torch.randperm(len(train), generator=g)[:BATCH].tolist()] for _ in range(HVP_BATCHES)]
    fixed = [make_batch(tok, b) for b in batches]

    # ---- (a) clean large-batch gradient norm ----
    model.zero_grad(set_to_none=True)
    idx = torch.randperm(len(train), generator=torch.Generator().manual_seed(2))[:BIGGRAD_N].tolist()
    nb = 0
    for i in range(0, BIGGRAD_N - BATCH, BATCH):
        inp, lb, am = make_batch(tok, [train[k] for k in idx[i:i + BATCH]])
        (model(input_ids=inp, attention_mask=am, labels=lb).loss / (BIGGRAD_N // BATCH)).backward(); nb += 1
    big_grad = float(sum(p.grad.pow(2).sum() for p in params if p.grad is not None) ** 0.5)
    model.zero_grad(set_to_none=True)
    print(f"[FR] large-batch grad-norm ({nb*BATCH} ex) = {big_grad:.4f}   (8-batch was 2.67)", flush=True)

    # ---- HVP closure ----
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

    # ---- (b) lambda_max via power iteration ----
    v = [torch.randn_like(p) for p in params]
    nrm = sum(x.pow(2).sum() for x in v) ** 0.5
    v = [x / nrm for x in v]
    lam = 0.0
    for it in range(POWER_ITERS):
        Hv = hvp(v)
        lam = float(sum((hv * x).sum() for hv, x in zip(Hv, v)))
        nrm = float(sum(hv.pow(2).sum() for hv in Hv) ** 0.5) + 1e-12
        v = [hv / nrm for hv in Hv]
        if it % 3 == 0 or it == POWER_ITERS - 1:
            print(f"  power iter {it:2d}  lambda_max ~ {lam:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- (c) Hessian trace via Hutchinson -> mean curvature ----
    tr = 0.0
    for j in range(HUTCH):
        z = [torch.randint(0, 2, p.shape, device=dev, dtype=torch.float32) * 2 - 1 for p in params]
        Hz = hvp(z)
        tr += float(sum((hz * zz).sum() for hz, zz in zip(Hz, z)))
    trace = tr / HUTCH
    lam_mean = trace / N
    dyn_range = lam / lam_mean if lam_mean != 0 else float("inf")

    curved = (lam > 1.0) and (dyn_range > 10)
    print(f"\n[FR] === CURVATURE / FORCE-READINESS (SST-2 frozen @ step 600) ===", flush=True)
    print(f"  large-batch grad-norm = {big_grad:.4f}   (stationarity; ->0 at a real min)", flush=True)
    print(f"  lambda_max (top Hessian eig) = {lam:.4f}   (sharpness / max binding)", flush=True)
    print(f"  trace = {trace:.1f}, lambda_mean = trace/N = {lam_mean:.3e}   (N={N/1e6:.0f}M)", flush=True)
    print(f"  dynamic range lambda_max/lambda_mean = {dyn_range:.3e}", flush=True)
    print(f"  CURVED BASIN: {'YES - real curvature + dynamic range, force lane has binding range' if curved else 'CHECK - see numbers; possibly flat or non-stationary freeze point'}", flush=True)
    json.dump({"ckpt": CKPT, "large_batch_grad_norm": big_grad, "lambda_max": lam,
               "hessian_trace": trace, "lambda_mean": lam_mean, "dynamic_range": dyn_range,
               "N_params_M": N / 1e6, "curved": bool(curved), "wall_s": time.time() - t0},
              open("D:/IC_experiments/cf_cascade/gateA/gateA_force_readiness.json", "w"), indent=2)
    print(f"[FR] saved gateA_force_readiness.json  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
