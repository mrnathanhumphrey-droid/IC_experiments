"""Gate A'' — curvature read (the pivotal gate; gates force lane, sigma_cpl peak, substrate).
The freeze passed (genuine PSD min) but it is VERY FLAT (true-Hessian lambda_max+=0.095).
Question: is the BINDING spectrum genuinely curved (dynamic range: stiff AND sloppy
directions present -> force balance discriminates) or flat-and-solved (narrow, near-uniform
-> weak binding everywhere -> switch to harder task A)?

Object: K-FAC per-matrix binding spectrum on the 24 square attention.dense (1024x1024)
matrices (FK3: gauge-free, NOT diagonal Fisher). Fisher(W) ~ G (x) A, A=E[xx^T] input-cov,
G=E[gg^T] output-grad-cov; binding eigenvalues = outer(eig(G), eig(A)). Plus the global
true-Hessian mean (Hutchinson trace via HVP) -> global dynamic range vs cert's lambda_max+=0.095.

PRE-REGISTERED VERDICT (declared before reading):
  CURVED  = pooled K-FAC binding spectrum spans >= 2 decades (log10(p99/p1) >= 2) AND
            lambda_max/lambda_median >= 50  -> stiff+sloppy both present, all 3 lanes proceed.
  FLAT    = spans < 1 decade, lambda_max/lambda_median < 10 (narrow near-uniform)
            -> SST-2 min too shallow -> switch to harder task A, STOP.
  (in-between -> report, global Hessian dynamic range as tiebreaker.)
Curvature read ONLY; the unified s-measurement comes AFTER, if curved.
"""
import os, json, time
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import numpy as np, torch

MODEL = "EleutherAI/pythia-410m"; CKPT = "D:/IC_experiments/cf_cascade/gateA/pythia410m_sst2_subsetmin.pt"
OUT = "D:/IC_experiments/cf_cascade/gateA/curvature_read.json"
BATCH = 16; N_KFAC = 16; HUTCH = 10; HVP_BATCHES = 4; SUBSET = 2000
LABELS = {0: " negative", 1: " positive"}; SQUARE = "attention.dense"; dev = "cuda"
torch.manual_seed(0)


def load_sst2_subset():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")["train"]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:SUBSET]]


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
    train = load_sst2_subset()

    # ---- K-FAC factor collection via hooks on the 24 attention.dense ----
    dense_mods = {n: m for n, m in model.named_modules() if n.endswith("attention.dense")}
    print(f"[curv] hooking {len(dense_mods)} attention.dense modules for K-FAC", flush=True)
    A_acc = {n: torch.zeros(1024, 1024, device=dev) for n in dense_mods}
    G_acc = {n: torch.zeros(1024, 1024, device=dev) for n in dense_mods}
    tok_count = {n: 0 for n in dense_mods}
    cur_in = {}
    handles = []
    for n, m in dense_mods.items():
        def fwd(mod, inp, out, nn=n): cur_in[nn] = inp[0].detach()
        def bwd(mod, gin, gout, nn=n):
            x = cur_in[nn]; g = gout[0].detach()                 # (B,S,1024)
            x = x.reshape(-1, 1024); g = g.reshape(-1, 1024)
            keep = g.abs().sum(1) > 0                              # only positions with loss-gradient
            if keep.any():
                xk, gk = x[keep], g[keep]
                A_acc[nn] += xk.T @ xk; G_acc[nn] += gk.T @ gk; tok_count[nn] += int(keep.sum())
        handles.append(m.register_forward_hook(fwd)); handles.append(m.register_full_backward_hook(bwd))

    g = torch.Generator().manual_seed(7)
    for b in range(N_KFAC):
        rows = [train[k] for k in torch.randperm(len(train), generator=g)[:BATCH].tolist()]
        inp, lb, am = make_batch(tok, rows)
        model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
    for h in handles: h.remove()
    model.zero_grad(set_to_none=True)

    # ---- per-matrix binding spectrum = outer(eig(G), eig(A)) ----
    per_layer = {}; pooled = []
    for n in dense_mods:
        nt = max(tok_count[n], 1)
        A = (A_acc[n] / nt).double().cpu().numpy(); G = (G_acc[n] / nt).double().cpu().numpy()
        ea = np.linalg.eigvalsh(A); eg = np.linalg.eigvalsh(G)
        ea = ea[ea > 0]; eg = eg[eg > 0]
        bind = np.outer(eg, ea).ravel()                          # K-FAC Fisher eigenvalues
        bind = bind[bind > 1e-20]
        per_layer[n] = {"lmax": float(bind.max()), "lmed": float(np.median(bind)),
                        "p1": float(np.percentile(bind, 1)), "p99": float(np.percentile(bind, 99)),
                        "decades_p1_99": float(np.log10(np.percentile(bind, 99) / max(np.percentile(bind, 1), 1e-30)))}
        pooled.append(bind[np.random.default_rng(0).integers(0, len(bind), 50000)])
    pooled = np.concatenate(pooled)
    p = lambda q: float(np.percentile(pooled, q))
    lmax, lmed = float(pooled.max()), float(np.median(pooled))
    decades = float(np.log10(p(99) / max(p(1), 1e-30)))
    dyn = lmax / lmed if lmed > 0 else float("inf")
    med_layer_decades = float(np.median([v["decades_p1_99"] for v in per_layer.values()]))

    # ---- global true-Hessian mean (Hutchinson trace via HVP) ----
    for pp in model.parameters(): pp.requires_grad_(True)
    params = [pp for pp in model.parameters() if pp.requires_grad]; N = sum(pp.numel() for pp in params)
    g2 = torch.Generator().manual_seed(1)
    fixed = [make_batch(tok, [train[k] for k in torch.randperm(len(train), generator=g2)[:BATCH].tolist()]) for _ in range(HVP_BATCHES)]
    def hvp(vec):
        acc = [torch.zeros_like(pp) for pp in params]
        for inp, lb, am in fixed:
            loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            gr = torch.autograd.grad(loss, params, create_graph=True)
            dot = sum((gp * v).sum() for gp, v in zip(gr, vec))
            hv = torch.autograd.grad(dot, params, retain_graph=False)
            for a, h in zip(acc, hv): a += h.detach()
            del gr, hv, loss
        return [a / len(fixed) for a in acc]
    tr = 0.0
    for _ in range(HUTCH):
        z = [torch.randint(0, 2, pp.shape, device=dev, dtype=torch.float32) * 2 - 1 for pp in params]
        Hz = hvp(z); tr += float(sum((hz * zz).sum() for hz, zz in zip(Hz, z)))
    trace = tr / HUTCH; lam_mean = trace / N
    LMAX_TRUE = 0.0954                                            # from robust_cert.py
    global_dyn = LMAX_TRUE / lam_mean if lam_mean > 0 else float("inf")

    curved = (decades >= 2.0) and (dyn >= 50)
    flat = (decades < 1.0) and (dyn < 10)
    verdict = "CURVED" if curved else ("FLAT_AND_SOLVED" if flat else "IN_BETWEEN")

    print(f"\n[curv] === CURVATURE READ (K-FAC binding spectrum, 24 attention.dense) ===", flush=True)
    print(f"  pooled binding: lmax={lmax:.3e} lmed={lmed:.3e} p1={p(1):.3e} p99={p(99):.3e}", flush=True)
    print(f"  DECADES spanned (p1-p99) = {decades:.2f}   (per-layer median {med_layer_decades:.2f})", flush=True)
    print(f"  DYNAMIC RANGE lmax/lmed  = {dyn:.1f}", flush=True)
    print(f"  global true-Hessian: lambda_max+ = {LMAX_TRUE:.4f}, lambda_mean(trace/N) = {lam_mean:.3e}, dyn = {global_dyn:.1e}", flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    if curved:
        print(f"    -> CURVED: stiff+sloppy both present, binding has dynamic range -> all 3 lanes proceed on SST-2", flush=True)
    elif flat:
        print(f"    -> FLAT-AND-SOLVED: narrow near-uniform binding -> SST-2 min too shallow -> SWITCH TASK A, stop", flush=True)
    else:
        print(f"    -> IN-BETWEEN: report; global Hessian dyn={global_dyn:.1e} as tiebreaker", flush=True)
    json.dump({"pooled_lmax": lmax, "pooled_lmed": lmed, "pooled_p1": p(1), "pooled_p99": p(99),
               "decades_p1_99": decades, "per_layer_median_decades": med_layer_decades,
               "dynamic_range_lmax_lmed": dyn, "global_lambda_max": LMAX_TRUE,
               "global_lambda_mean": lam_mean, "global_dynamic_range": global_dyn,
               "verdict": verdict, "per_layer": per_layer, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[curv] saved curvature_read.json  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
