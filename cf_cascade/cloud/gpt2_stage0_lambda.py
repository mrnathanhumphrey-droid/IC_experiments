"""GPT-2 STAGE 0 (Wilson-scoped, spectrum-shape-gated generalization) — measure the curvature-spectrum SHAPE
that GATES the dye prediction. Real Fisher FA = E[g g^T] (Conv1D orientation: weight (in,out), g (in,out),
input-dir Fisher = g @ g^T = (in,in)) at the GPT-2 SST-2 (task-A) minimum, across the DYE carriers:
  attn.c_attn (fused QKV ~ Pythia qkv), mlp.c_fc (up-proj ~ h_to_4h), mlp.c_proj (down-proj ~ 4h_to_h = Wdown).
  (NOT attn.c_proj = attention output, which is the Pythia attention.dense analog, not a dye carrier.)
SHAPE QUESTION: two-scale like Pythia (few stiff outliers in a flat sea, clean bulk+outlier gap) vs CONTINUUM
(smooth heavy-tailed decay, no gap)? Metrics: top/median dyn-range, frac-in-top-decade, outliers>5x median, top-8 decay.
SAVES top-KEEP Fisher eigvecs + eigvals per carrier-layer -> gpt2_dye_basis.npz (the dye scripts reuse this, no recompute).
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "gpt2-medium"; CKPT = f"{G}/gpt2med_sst2_subsetmin.pt"
OUT = f"{G}/gpt2_stage0.json"; NPZ = f"{G}/gpt2_dye_basis.npz"
CARRIERS = ["attn.c_attn", "mlp.c_fc", "mlp.c_proj"]     # qkv / up-proj / down-proj analogs (NOT attn.c_proj)
KEEP = 1024; N_FISHER = 150; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
torch.manual_seed(0); np.random.seed(0)


def sst_trA():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")["train"]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:max(N_FISHER, 200)]]


def b_sst(tok, items):
    seqs, labs = [], []
    for s, l in items:
        p = tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"]; c = tok(LAB[l], add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    L = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


def fisher(model, params, names, tok, items):
    # GPT-2 Conv1D weight (in,out), grad (in,out) -> input-dir Fisher = E[g @ g^T] = (in,in)
    F = {n: torch.zeros(params[n].shape[0], params[n].shape[0], device=dev, dtype=torch.float64) for n in names}
    for it in items[:N_FISHER]:
        inp, lb, am = b_sst(tok, [it]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: g = params[n].grad.detach().double(); F[n] += g @ g.T
    return F


def shape_metrics(eigs):
    e = np.sort(np.asarray(eigs)[np.asarray(eigs) > 0])[::-1]      # descending, positive
    med = np.median(e); top = e[0]
    return {"dyn_range_top_over_med": float(top / med), "frac_in_top_decade": float((e > top / 10).mean()),
            "outliers_gt_5x_med_frac": float((e > 5 * med).mean()), "top8": [float(x) for x in e[:8]],
            "top_over_min": float(top / e[-1])}


def main():
    t0 = time.time(); print("[G2S0] loading gpt2-medium...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.load_state_dict(torch.load(CKPT))
    trA = sst_trA(); params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    print(f"[G2S0] {len(names)} carrier weights: " + ", ".join(sorted({n.split('.h.')[-1].split('.weight')[0].split('.', 1)[-1] for n in names})), flush=True)

    FA = fisher(model, params, names, tok, trA)
    print(f"[G2S0] Fisher done ({time.time()-t0:.0f}s)", flush=True)

    # per-carrier spectrum shape (pool eigenvalues across that carrier's layers) + save basis
    out = {}; arts = {}
    for c in CARRIERS:
        cn = [x for x in names if c in x]; alleig = []
        for n in cn:
            ev, V = torch.linalg.eigh(FA[n]); k = min(KEEP, V.shape[1])
            arts[f"V__{n}"] = torch.flip(V[:, -k:], dims=[1]).float().cpu().numpy()   # rank0=highest curv
            arts[f"lam__{n}"] = torch.flip(ev[-k:], dims=[0]).float().cpu().numpy()
            alleig.append(ev.float().cpu().numpy())
        m = shape_metrics(np.concatenate(alleig))
        # per-layer median dyn-range too (robust)
        perlayer_dr = [shape_metrics(torch.linalg.eigvalsh(FA[n]).cpu().numpy())["dyn_range_top_over_med"] for n in cn]
        m["median_perlayer_dyn_range"] = float(np.median(perlayer_dr))
        out[c] = m
        print(f"  [{c:12s}] dyn-range(top/med) pooled={m['dyn_range_top_over_med']:.1f} perlayer-med={m['median_perlayer_dyn_range']:.1f} | frac-top-decade={m['frac_in_top_decade']:.4f} | outliers>5xmed={m['outliers_gt_5x_med_frac']:.3f} | top8={[round(x,2) for x in m['top8'][:5]]}", flush=True)

    # two-scale vs continuum heuristic: two-scale => frac-in-top-decade tiny AND a clean gap; continuum => smooth.
    # report the call per carrier (continuum if top-8 decays smoothly w/o an order-of-magnitude gap in first few)
    for c in CARRIERS:
        t8 = out[c]["top8"]; gaps = [t8[i] / t8[i + 1] for i in range(min(4, len(t8) - 1))]
        out[c]["early_gap_ratios"] = gaps
        out[c]["shape_call"] = "two-scale-ish" if max(gaps) > 10 else "continuum-ish"
        print(f"  [{c:12s}] early gaps {[round(g,1) for g in gaps]} -> {out[c]['shape_call']}", flush=True)

    np.savez(NPZ, **arts)
    json.dump({"per_carrier": out, "keep": KEEP, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[G2S0] saved gpt2_stage0.json + gpt2_dye_basis.npz wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
