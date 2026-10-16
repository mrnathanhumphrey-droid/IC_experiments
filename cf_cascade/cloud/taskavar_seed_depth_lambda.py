"""DEPTH-VARIANT generator (OPT-350m, layer-surgery) — persist the SST-side per-layer artifacts that
SPIRAL-OPT-3 (seed concentration) and SPIRAL-OPT-4 (twist) need, for one depth variant. Args: MODEL KEEP CKPT TAG.
Derived from taskavar_seed_opt_lambda.py; surger+load matched-acc w_A CKPT (mirrors crosspair_depth_lambda.py) so the
DEPOSIT is identical to the funnel leg per depth. DROPS the AG-base model: opt4 reads only fragile__sst2__<ct>, opt3 reads
only seed/contest/lam (SST side) -> the AG cross-task Grassmann is not consumed by either reader. Reproduce-anchor lives
in the readers (opt3_depth/opt4_depth read the banked full-24 files as d24). In-script guard = re-measure loaded w_A acc.

  taskA_variation_{TAG}.npz : fragile__sst2__<ct> AND topcurv__sst2__<ct>, each (nlay, D_ct, 16)  [OPT-4 twist]
  perlayer_seed_{TAG}.npz   : seed__<ct>, contest__<ct>, lam__<ct>, each (nlay, 128)               [OPT-3 seed]
ct in {q_proj,k_proj,v_proj,out_proj,fc1,fc2}. Per-layer never pooled.
"""
import os, re, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch
from torch import nn
G = "/home/ubuntu/gateA"
MODEL = sys.argv[1]; KEEP = int(sys.argv[2]); CKPT = sys.argv[3]; TAG = sys.argv[4]; AX = f"{G}/overlap_axis"
OUT = f"{G}/taskavar_seed_{TAG}.json"; TAV = f"{G}/taskA_variation_{TAG}.npz"; PLS = f"{G}/perlayer_seed_{TAG}.npz"
CARRIERS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
STEPS_DEPTH = 200; LR_DEPTH = 5e-5; BATCH = 16; SUBSET = 2000
N_FISHER = 150; N_SEED = 16; KB = 128; KF = 16


def _surger(model, keep):   # drop to KEEP evenly-spaced decoder layers + reindex layer_idx (KV-cache)
    L0 = len(model.model.decoder.layers)
    if keep >= L0: return L0
    idx = sorted(set(np.linspace(0, L0 - 1, keep).round().astype(int).tolist()))
    model.model.decoder.layers = nn.ModuleList([model.model.decoder.layers[i] for i in idx])
    model.config.num_hidden_layers = len(idx)
    for ni, lyr in enumerate(model.model.decoder.layers):
        if hasattr(lyr, "self_attn") and hasattr(lyr.self_attn, "layer_idx"): lyr.self_attn.layer_idx = ni
    return len(idx)


def load_jsonB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


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


def fisher_of(model, params, names, tok, itemsA):
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    g = torch.Generator().manual_seed(0)
    for _ in range(N_FISHER):
        idx = torch.randperm(len(itemsA), generator=g)[:1].tolist()
        inp, lb, am = b_pairs(tok, [itemsA[i] for i in idx]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: gr = params[n].grad.detach().double(); F[n] += gr.T @ gr
    return F


def curv_basis(model, tok, names, itemsA):
    params = dict(model.named_parameters()); FA = fisher_of(model, params, names, tok, itemsA); V = {}; LAM = {}
    for n in names:
        ev, vv = torch.linalg.eigh(FA[n])
        V[n] = vv[:, -KB:].flip(dims=[1]).float().contiguous()
        LAM[n] = ev[-KB:].flip(dims=[0]).cpu().numpy()
        del FA[n]
    return V, LAM


def seedcov(model, params, names, tok, items):
    S = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    gg = torch.Generator().manual_seed(0)
    for it in range(N_SEED):
        idx = torch.randperm(len(items), generator=gg)[:BATCH].tolist()
        inp, lb, am = b_pairs(tok, [items[i] for i in idx]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: gr = params[n].grad.detach().double(); S[n] += gr.T @ gr
    return S


def train_items(model, tok, items, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt


def ctype(n):
    for c in CARRIERS:
        if c in n: return c
def layer_of(n): return int(re.search(r"layers\.(\d+)\.", n).group(1))


def main():
    t0 = time.time(); print(f"[GEN] loading {MODEL} (eager) KEEP={KEEP}...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(dev)
    kept = _surger(model, KEEP); print(f"[GEN] depth-variant -> {kept} layers", flush=True)
    model.load_state_dict(torch.load(CKPT))     # matched-acc w_A (same deposit as funnel leg)
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]

    sstA = sst_items(SUBSET, 7); agB = load_jsonB("low_agnews"); subjP = load_jsonB("medium_subj")
    wA = {n: model.state_dict()[n].detach().clone() for n in model.state_dict()}
    ev = sst_rows("validation", 300, 0); model.eval(); cor = 0
    with torch.no_grad():
        for i in range(0, len(ev), 32):
            ch = ev[i:i + 32]; seqs = [tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"][:128] for s, _ in ch]
            L = max(len(x) for x in seqs); inp = np.full((len(seqs), L), tok.pad_token_id, np.int64); pos = np.zeros(len(seqs), np.int64)
            for j, sq in enumerate(seqs): inp[j, :len(sq)] = sq; pos[j] = len(sq) - 1
            lg = model(input_ids=torch.tensor(inp).to(dev)).logits[torch.arange(len(seqs)), torch.tensor(pos).to(dev)]
            pr = (lg[:, posid] > lg[:, negid]).long().cpu().numpy()
            for j, (_, lab) in enumerate(ch): cor += int(pr[j] == lab)
    accA = cor / len(ev); print(f"[GEN] loaded w_A SST acc={accA:.3f} (matched-band guard) ({time.time()-t0:.0f}s)", flush=True)

    # curvature basis (SST) + AG forgetting-seed + contest, at w_A
    V, LAM = curv_basis(model, tok, carriers, sstA)
    print(f"[GEN] curvature basis done ({time.time()-t0:.0f}s)", flush=True)
    S_B = seedcov(model, params, carriers, tok, agB)
    seed_e = {n: (V[n].T @ S_B[n].float() @ V[n]).diagonal().clamp(min=0).cpu().numpy() for n in carriers}
    model.load_state_dict(wA); train_items(model, tok, agB, STEPS_DEPTH, LR_DEPTH, 0)
    paramsB = dict(model.named_parameters()); contest_e = {}
    for n in carriers:
        dwB = (paramsB[n].detach() - wA[n].to(dev)).double(); M = dwB.T @ dwB
        contest_e[n] = (V[n].double().T @ M @ V[n].double()).diagonal().clamp(min=0).cpu().numpy()
    model.load_state_dict(wA); params = dict(model.named_parameters())

    # fragile/topcurv (probe=Subj) at SST-w_A
    SsB = seedcov(model, params, carriers, tok, subjP)
    fragile_SST = {}; topcurv_SST = {}
    for n in carriers:
        e = (V[n].T @ SsB[n].float() @ V[n]).diagonal().clamp(min=0).cpu().numpy()
        order = np.argsort(-e)[:KF]
        fragile_SST[n] = V[n][:, order].cpu().numpy(); topcurv_SST[n] = V[n][:, :KF].cpu().numpy()
    print(f"[GEN] SST fragile/topcurv done ({time.time()-t0:.0f}s)", flush=True)

    # ---- persist per-carrier-type, layer-stacked (nlay = kept layers) ----
    ct_names = {ct: sorted([n for n in carriers if ctype(n) == ct], key=layer_of) for ct in CARRIERS}
    for ct in CARRIERS: assert len(ct_names[ct]) == kept, f"{ct}: {len(ct_names[ct])} != {kept}"
    tav = {}; pls = {}
    for ct in CARRIERS:
        ns = ct_names[ct]
        tav[f"fragile__sst2__{ct}"] = np.stack([fragile_SST[n] for n in ns])   # (nlay,D,16)
        tav[f"topcurv__sst2__{ct}"] = np.stack([topcurv_SST[n] for n in ns])
        tav[f"names__{ct}"]         = np.array(ns)
        pls[f"seed__{ct}"]    = np.stack([seed_e[n] for n in ns])              # (nlay,128)
        pls[f"contest__{ct}"] = np.stack([contest_e[n] for n in ns])
        pls[f"lam__{ct}"]     = np.stack([LAM[n] for n in ns])
    np.savez(TAV, **tav); np.savez(PLS, **pls)
    rep = {"model": MODEL, "keep": kept, "tag": TAG, "wA_sst_acc": accA,
           "note": "depth-variant SST-side artifacts; reproduce-anchor in opt3_depth/opt4_depth vs banked full-24"}
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[GEN] saved {TAV} + {PLS} acc={accA:.3f} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
