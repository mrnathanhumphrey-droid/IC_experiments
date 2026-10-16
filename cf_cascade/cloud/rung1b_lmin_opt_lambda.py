"""RUNG-1b — ROBUST lambda_min re-cert (OPT-350m). Resolve: genuine saddle vs mini-batch Hessian negativity?

rung-1 gave lambda_min=-0.70<0 at the OPT SST interpolation min -> auto "saddle, can't anchor". BUT the cert shared ONE 64-example
Hessian batch across restarts (only varied init), so it can't separate a GENUINE saddle from MINI-BATCH HESSIAN NEGATIVITY (small
batches routinely show spurious negatives that vanish with more data; at an in-sample interpolation min residuals~0 -> Hessian~=GGN~=PSD,
so -0.70 is theoretically suspicious). This re-certs lambda_min across MULTIPLE distinct batches + a 4x-larger batch.

VERDICT: lambda_min consistently <0 across batches AND the large-batch still <0 -> GENUINE SADDLE (OPT can't anchor; bounds framework,
real finding, STOP at rung 1). lambda_min sign-varies / -> ~0+ with more data / large-batch >= ~0 -> the -0.70 was SAMPLING; basin is
flat-PSD -> freeze gate effectively PASSES (report dynamic range with positive lambda_min) -> proceed to rung 2.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "facebook/opt-350m"; OUT = f"{G}/rung1b_opt.json"
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
STEPS_WA = 1200; LR_WA = 5e-5; BATCH = 16; SUBSET = 2000; PI_ITERS = 100


def sst_rows(split, k, seed=0):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")[split]
    assert len(ds) > 400, f"SST load looks wrong: {len(ds)}"
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))[:k]]


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


def train_sst(model, tok, subset, steps, lr):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(0); last = float("nan")
    for _ in range(steps):
        idx = torch.randperm(len(subset), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [subset[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); last = float(loss.item())
    del opt; return last


def hvp(model, P, batch, vlist):
    inp, lb, am = batch
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
    g = torch.autograd.grad(loss, P, create_graph=True)
    dot = sum((gi * vi).sum() for gi, vi in zip(g, vlist))
    Hv = torch.autograd.grad(dot, P, retain_graph=False)
    return [h.detach() for h in Hv]


def vdot(a, b): return float(sum((x * y).sum() for x, y in zip(a, b)))
def vnorm(a): return float(sum((x * x).sum() for x in a)) ** 0.5


def lam_max(model, P, batch, iters, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = [torch.randn(p.shape, generator=g).to(dev) for p in P]; v = [x / vnorm(v) for x in v]; lam = 0.0
    for _ in range(iters):
        Hv = hvp(model, P, batch, v); lam = vdot(v, Hv); nv = vnorm(Hv)
        if nv < 1e-20: break
        v = [x / nv for x in Hv]
    return lam


def lam_min(model, P, batch, mu, iters, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = [torch.randn(p.shape, generator=g).to(dev) for p in P]; n0 = vnorm(v); v = [x / n0 for x in v]; top = 0.0
    for _ in range(iters):
        Hv = hvp(model, P, batch, v); Av = [mu * vi - hvi for vi, hvi in zip(v, Hv)]
        top = vdot(v, Av); nv = vnorm(Av)
        if nv < 1e-20: break
        v = [x / nv for x in Av]
    return mu - top


def main():
    t0 = time.time(); print(f"[R1b] loading {MODEL} (eager)...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(dev)
    subset = [(f"{s}\nSentiment:", LAB[l]) for s, l in sst_rows("train", SUBSET, 7)]
    print(f"[R1b] training to interpolation...", flush=True)
    fl = train_sst(model, tok, subset, STEPS_WA, LR_WA)
    print(f"[R1b] min train loss {fl:.4f} ({time.time()-t0:.0f}s)", flush=True)
    model.eval()
    for p in model.parameters(): p.requires_grad_(True)
    P = [p for p in model.parameters() if p.requires_grad]

    # lambda_max on batch0 (stable)
    b0 = b_pairs(tok, subset[:64]); lmax = lam_max(model, P, b0, 60, 1); mu = 1.1 * abs(lmax)
    print(f"[R1b] lambda_max={lmax:.3e}, mu={mu:.3e} ({time.time()-t0:.0f}s)", flush=True)

    # lambda_min across 5 DISTINCT 64-batches + one 256-batch (the discriminator)
    lmins64 = []
    for k in range(5):
        bk = b_pairs(tok, subset[k * 64:(k + 1) * 64]); lm = lam_min(model, P, bk, mu, PI_ITERS, 100 + k)
        lmins64.append(float(lm)); print(f"[R1b] batch{k} (64ex) lambda_min={lm:.3e} ({time.time()-t0:.0f}s)", flush=True)
    b256 = b_pairs(tok, subset[:256]); lmin256 = float(lam_min(model, P, b256, mu, PI_ITERS, 500))
    print(f"[R1b] LARGE batch (256ex) lambda_min={lmin256:.3e} ({time.time()-t0:.0f}s)", flush=True)

    arr = np.array(lmins64); med = float(np.median(arr)); frac_neg = float(np.mean(arr < -0.05))
    rep = {"model": MODEL, "train_loss": fl, "lambda_max": lmax, "lambda_min_64batches": lmins64,
           "lambda_min_median64": med, "frac_negative": frac_neg, "lambda_min_256batch": lmin256}
    try:
        genuine = (med < -0.05) and (frac_neg >= 0.8) and (lmin256 < -0.05)
        artifact = (lmin256 >= -0.05) or (frac_neg <= 0.4)
        if genuine:
            v = (f"GENUINE SADDLE -> FREEZE GATE FAILS: lambda_min median {med:.3e} (frac_neg {frac_neg:.1f}), large-batch {lmin256:.3e} "
                 f"all <0 robustly. OPT SST interpolation min is a SADDLE, not a PSD basin. OPT can't anchor the framework as-is (bounds "
                 f"reach / needs harder-task-A). STOP at rung 1 — real finding.")
        elif artifact:
            rng = float(np.log10(lmax / max(med, 1e-6))) if med > 0 else None
            v = (f"SAMPLING ARTIFACT -> FREEZE GATE EFFECTIVELY PASSES: rung-1's -0.70 was mini-batch negativity (large-batch "
                 f"{lmin256:.3e}, frac_neg {frac_neg:.1f}, median {med:.3e} -> ~0/flat-PSD). Basin is flat-PSD (continuum spectrum, "
                 f"lambda_max {lmax:.3e}). Treat as flat basin; check dynamic range with the positive end. PROCEED to rung 2.")
        else:
            v = (f"AMBIGUOUS: lambda_min median {med:.3e}, frac_neg {frac_neg:.1f}, large-batch {lmin256:.3e} — mixed. Needs full-data "
                 f"Hessian or larger batches to settle. Hold.")
        rep["verdict"] = v
        print(f"\n[R1b] === ROBUST lambda_min VERDICT ===\n  {v}", flush=True)
        print(f"  lambda_min 64-batches: {[f'{x:.2e}' for x in lmins64]} | median {med:.2e} frac_neg {frac_neg:.1f}", flush=True)
        print(f"  lambda_min 256-batch: {lmin256:.3e} | lambda_max {lmax:.3e}", flush=True)
    except Exception as e:
        rep["error"] = traceback.format_exc(); print(f"[R1b] readout error (saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[R1b] saved rung1b_opt.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
