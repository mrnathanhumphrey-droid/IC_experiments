"""NEW-MODEL RUNG 1 (OPT-350m) — Gate-0a (eigen-shell substrate) + Freeze gate (task-A PSD basin). RUNG 1 ONLY.

The whole CF framework (fragile subspace, flow, non-destructiveness, #2c mechanism) was established on Pythia-410M. A new model
must RE-CLIMB the ladder before any replication/mechanism run is meaningful. This is the bottom rung: does OPT even support the
substrate? It does NOT test the phenomenon. A gate failure here is itself informative (bounds the framework's reach).

MODEL: facebook/opt-350m (Option B — third family, non-GPT, non-Pythia). Square attention matrices = self_attn.{q,k,v,out}_proj
(1024x1024, standard nn.Linear (out,in), no Conv1D transpose).

RUNG 1 STEP 1 — GATE-0a (eigen-shell substrate exists?):
  K8 ESD multi-decade: spectrum of the square attention matrices must span >=2 decades (Pythia 5.9/4.4, GPT-2 8.85/3.97).
     Report eig(W^T W) decades (singular ESD, robust real) + |eig(W)| decades; full + median over matrices.
  K7 index-autocorr ~0: matrix-layout void (eigen-shell is the right basis). lag-1 autocorr of flattened weights (Pythia/GPT-2 ~0.001).
  GATE: ESD < 2 decades OR autocorr not ~0 -> STOP (substrate absent on this arch; bounds framework). Don't proceed to step 2.

RUNG 1 STEP 2 — FREEZE GATE (task-A forms a genuine PSD basin?):
  Train OPT on SST-2 (2k-subset -> interpolation, mirrors Pythia) to a clean minimum.
  Robust lambda_min CERT: power-iteration HVPs (NOT K-FAC -- the -49 phantom lesson), shifted for the smallest eigenvalue, restarts,
     consistency. lambda_min > 0 = genuine PSD basin (not saddle).
  CURVED not flat: dynamic range = lambda_max/lambda_min >= 2 decades (the two-quantity verdict: lambda_min>0 AND range, NOT absolute
     curvature -- Pythia interpolation min was flat-bulk+stiff-tail, CURVED via 4.5e7 range). Report SPECTRUM SHAPE (top-k + min:
     two-scale like Pythia / continuum like GPT-2 / other -- predicts downstream flow profile).
  Re-confirm ESD on the frozen (fine-tuned) basis (moves under task-A; Pythia held 5.86->5.87).
  GATE: lambda_min not > 0 OR range < 2 decades -> task-A doesn't form a usable basin -> harder-task-A fork OR can't anchor. Don't proceed to rung 2.

VERDICT: both gates pass -> earned rung 2 (fragile-subspace characterization), report spectrum shape. Gate-0a fail -> substrate is
GPT/Pythia-family-specific (finding). Freeze fail -> can't anchor (finding). Heavy eigen-work; save artifacts BEFORE readout.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "facebook/opt-350m"
OUT = f"{G}/rung1_opt.json"; NPZ = f"{G}/rung1_opt.npz"
ATTN_SQ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"]  # square attention matrices
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
STEPS_WA = 1200; LR_WA = 5e-5; BATCH = 16; SUBSET = 2000
HBATCH = 64; PI_ITERS = 80; PI_RESTARTS = 2


def sst_rows(split, k, seed=0):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")[split]
    assert len(ds) > 400, f"SST load looks wrong: {len(ds)}"  # dataset-identity guard (val split=872, train=67349; catches fallthrough-to-tiny)
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


def decades(vals):
    v = np.sort(np.abs(np.asarray(vals, float))); v = v[v > 1e-12]
    if len(v) < 8: return None
    full = float(np.log10(v[-1] / v[0])); rob = float(np.log10(np.quantile(v, 0.99) / np.quantile(v, 0.01)))
    return full, rob


def gate0a(model):
    """ESD decades + index-autocorr on the square attention matrices."""
    params = dict(model.named_parameters())
    mats = [n for n in params if any(a in n for a in ATTN_SQ) and n.endswith("weight") and params[n].ndim == 2 and params[n].shape[0] == params[n].shape[1]]
    esd_eigWtW, esd_eigW, autoc = [], [], []
    for n in mats:
        W = params[n].detach().float()
        ev = torch.linalg.eigvalsh(W.T @ W).cpu().numpy()      # singular ESD (real, PSD)
        d1 = decades(ev)
        evW = torch.linalg.eigvals(W).abs().cpu().numpy()       # |eig(W)|
        d2 = decades(evW)
        if d1: esd_eigWtW.append(d1)
        if d2: esd_eigW.append(d2)
        w = W.flatten().cpu().numpy(); w = w - w.mean()
        ac = float(np.dot(w[:-1], w[1:]) / np.dot(w, w))         # lag-1 autocorr of flattened weights
        autoc.append(ac)
    return mats, np.array(esd_eigWtW), np.array(esd_eigW), np.array(autoc)


def train_sst(model, tok, subset, steps, lr):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(0); last = float("nan")
    for _ in range(steps):
        idx = torch.randperm(len(subset), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [subset[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); last = float(loss.item())
    del opt; return last


def eval_sst(model, tok, evalset, posid, negid):
    model.eval(); correct = 0; n = 0
    with torch.no_grad():
        for i in range(0, len(evalset), 32):
            chunk = evalset[i:i + 32]; seqs = [tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"][:128] for s, _ in chunk]
            L = max(len(x) for x in seqs); pad = tok.pad_token_id
            inp = np.full((len(seqs), L), pad, np.int64); pos = np.zeros(len(seqs), np.int64)
            for j, sq in enumerate(seqs): inp[j, :len(sq)] = sq; pos[j] = len(sq) - 1
            lg = model(input_ids=torch.tensor(inp).to(dev)).logits
            last = lg[torch.arange(len(seqs)), torch.tensor(pos).to(dev)]
            pred = (last[:, posid] > last[:, negid]).long().cpu().numpy()
            for j, (_, lab) in enumerate(chunk): correct += int(pred[j] == lab); n += 1
    return correct / max(n, 1)


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
def vscale(a, s): return [x * s for x in a]
def vsub(a, b): return [x - y for x, y in zip(a, b)]


def power_lambda_max(model, P, batch, iters, seed, deflate=None):
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = [torch.randn(p.shape, generator=g).to(dev) for p in P]
    if deflate:
        for u in deflate: v = vsub(v, vscale(u, vdot(v, u)))
    v = vscale(v, 1.0 / vnorm(v)); lam = 0.0
    for _ in range(iters):
        Hv = hvp(model, P, batch, v)
        if deflate:
            for u in deflate: Hv = vsub(Hv, vscale(u, vdot(Hv, u)))
        lam = vdot(v, Hv); nv = vnorm(Hv)
        if nv < 1e-20: break
        v = vscale(Hv, 1.0 / nv)
    return lam, v


def power_lambda_min(model, P, batch, mu, iters, seed):
    """top eig of (mu*I - H) = mu - lambda_min."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = [torch.randn(p.shape, generator=g).to(dev) for p in P]; v = vscale(v, 1.0 / vnorm(v)); top = 0.0
    for _ in range(iters):
        Hv = hvp(model, P, batch, v)
        Av = [mu * vi - hvi for vi, hvi in zip(v, Hv)]
        top = vdot(v, Av); nv = vnorm(Av)
        if nv < 1e-20: break
        v = vscale(Av, 1.0 / nv)
    return mu - top


def main():
    t0 = time.time(); print(f"[R1] loading {MODEL}...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    # eager attention: OPT's default SDPA-efficient kernel has NO double-backward -> HVP (grad-of-grad) fails. eager = math path supports it.
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(dev)
    rep = {"model": MODEL}

    # ===== GATE-0a (base weights) =====
    mats, esd1, esd2, autoc = gate0a(model)
    esd1_full_med = float(np.median(esd1[:, 0])); esd1_rob_med = float(np.median(esd1[:, 1]))
    esd2_full_med = float(np.median(esd2[:, 0])) if len(esd2) else None
    ac_med = float(np.median(np.abs(autoc)))
    g0a_pass = esd1_rob_med >= 2.0 and ac_med < 0.05
    rep["gate0a"] = {"n_matrices": len(mats), "esd_eigWtW_full_median": esd1_full_med, "esd_eigWtW_robust_median": esd1_rob_med,
                     "esd_eigW_full_median": esd2_full_med, "autocorr_median_abs": ac_med, "pass": bool(g0a_pass)}
    print(f"[R1] GATE-0a: ESD(W^TW) full/robust median = {esd1_full_med:.2f}/{esd1_rob_med:.2f} decades | |eig(W)| full med {esd2_full_med} | "
          f"autocorr |median| {ac_med:.4f} -> {'PASS' if g0a_pass else 'FAIL'} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(NPZ, esd_eigWtW=esd1, esd_eigW=esd2, autocorr=autoc)
    if not g0a_pass:
        rep["verdict"] = (f"GATE-0a FAILED on {MODEL}: ESD robust median {esd1_rob_med:.2f} decades (<2) or autocorr {ac_med:.4f} (not ~0) "
                          f"-> the eigen-shell substrate does NOT exist on this architecture. Bounds the framework's reach (GPT/Pythia-specific). STOP.")
        json.dump(rep, open(OUT, "w"), indent=2); print(f"\n[R1] {rep['verdict']}\n[R1] saved (rung-1 halt).", flush=True); return

    # ===== FREEZE GATE: train task-A -> interpolation min =====
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0)
    subset = [(f"{s}\nSentiment:", LAB[l]) for s, l in sst_rows("train", SUBSET, 7)]
    print(f"[R1] training task-A to interpolation ({STEPS_WA} steps on {SUBSET}-subset)...", flush=True)
    final_loss = train_sst(model, tok, subset, STEPS_WA, LR_WA)
    acc = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[R1] interpolation min: train loss {final_loss:.4f}, SST val acc {acc:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # ===== robust lambda_min / lambda_max via power-iteration HVPs (fp32, eval mode = no dropout) =====
    model.eval()
    for p in model.parameters(): p.requires_grad_(True)
    P = [p for p in model.parameters() if p.requires_grad]
    hb = b_pairs(tok, subset[:HBATCH])  # fixed Hessian batch at the min
    lam_max_runs = []; top_eigs = []; eigvecs = []
    for r in range(PI_RESTARTS):
        lm, v1 = power_lambda_max(model, P, hb, PI_ITERS, seed=10 + r)
        lam_max_runs.append(lm)
        if r == 0: eigvecs = [v1]
    lam_max = float(np.median(lam_max_runs))
    # top-5 via deflation (spectrum shape)
    for k in range(4):
        lk, vk = power_lambda_max(model, P, hb, PI_ITERS, seed=50 + k, deflate=eigvecs)
        top_eigs.append(float(lk)); eigvecs.append(vk)
    top_spectrum = [lam_max] + top_eigs
    mu = 1.05 * abs(lam_max)
    lam_min_runs = [power_lambda_min(model, P, hb, mu, PI_ITERS + 40, seed=200 + r) for r in range(PI_RESTARTS)]
    lam_min = float(np.median(lam_min_runs))
    rng_decades = float(np.log10(lam_max / lam_min)) if (lam_min > 0 and lam_max > 0) else None

    # ===== re-confirm ESD on frozen basis =====
    mats2, esd1b, _, _ = gate0a(model)
    esd_frozen_rob_med = float(np.median(esd1b[:, 1]))

    # spectrum shape heuristic: gap between stiff tail (top) and the rest
    ts = np.array(top_spectrum); shape = "continuum"
    if len(ts) >= 3 and ts[0] / max(ts[2], 1e-12) > 10: shape = "two-scale (stiff tail + flatter bulk)"

    lam_min_pos = lam_min > 0; curved = (rng_decades is not None and rng_decades >= 2.0)
    freeze_pass = lam_min_pos and curved and esd_frozen_rob_med >= 2.0
    rep["freeze"] = {"train_loss": final_loss, "sst_acc": acc, "lambda_max": lam_max, "lambda_min": lam_min,
                     "lambda_max_runs": lam_max_runs, "lambda_min_runs": lam_min_runs, "range_decades": rng_decades,
                     "top_spectrum": top_spectrum, "spectrum_shape": shape, "esd_frozen_robust_median": esd_frozen_rob_med,
                     "lambda_min_positive": bool(lam_min_pos), "curved": bool(curved), "pass": bool(freeze_pass)}
    np.savez(NPZ, esd_eigWtW=esd1, esd_eigW=esd2, autocorr=autoc, esd_frozen=esd1b,
             lam_max_runs=np.array(lam_max_runs), lam_min_runs=np.array(lam_min_runs), top_spectrum=np.array(top_spectrum))
    try:
        if not lam_min_pos:
            v = (f"FREEZE GATE FAILED: lambda_min={lam_min:.3e} NOT > 0 (runs {[f'{x:.2e}' for x in lam_min_runs]}) -> not a genuine PSD "
                 f"basin (saddle) on {MODEL}. Harder-task-A fork or can't anchor. Don't proceed to rung 2.")
        elif not curved:
            v = (f"FREEZE GATE FAILED: lambda_min>0 ({lam_min:.3e}) but range {rng_decades:.2f} decades <2 = FLAT not CURVED "
                 f"(lambda_max {lam_max:.3e}). Basin too flat to anchor the framework. Harder-task-A fork.")
        elif esd_frozen_rob_med < 2.0:
            v = f"FREEZE PARTIAL: basin PSD+curved but ESD collapsed on frozen basis ({esd_frozen_rob_med:.2f}<2) — flag, inspect."
        else:
            v = (f"RUNG 1 PASSED on {MODEL}: Gate-0a (ESD {esd1_rob_med:.2f} dec, autocorr {ac_med:.4f}) + Freeze (lambda_min={lam_min:.3e}>0, "
                 f"range {rng_decades:.2f} decades = CURVED, shape={shape}, ESD held {esd_frozen_rob_med:.2f}). EARNED RUNG 2 (fragile-subspace). "
                 f"Spectrum shape predicts downstream flow profile.")
        rep["verdict"] = v
        print(f"\n[R1] === RUNG-1 VERDICT ({MODEL}) ===\n  {v}", flush=True)
        print(f"  Gate-0a: ESD robust median {esd1_rob_med:.2f} dec, autocorr {ac_med:.4f} -> PASS", flush=True)
        print(f"  Freeze: lambda_min={lam_min:.3e} (>0={lam_min_pos}), lambda_max={lam_max:.3e}, range={rng_decades} dec (curved={curved})", flush=True)
        print(f"  spectrum shape={shape}; top={[f'{x:.2e}' for x in top_spectrum]}; ESD frozen {esd_frozen_rob_med:.2f}", flush=True)
    except Exception as e:
        rep["error"] = traceback.format_exc(); print(f"[R1] readout error (saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[R1] saved rung1_opt.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
