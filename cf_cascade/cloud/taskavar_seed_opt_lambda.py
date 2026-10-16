"""GENERATOR (OPT-350m) — persist the per-layer artifacts SPIRAL-OPT v2 needs. Derived verbatim from
rung2_fragile_opt_lambda.py (same model/carriers/data/extraction); rung2 COMPUTED these subspaces then discarded them
(kept only scalar fG/tG). This version UNGATES the subspace extraction and PERSISTS, per-carrier-type, layer-stacked:

  taskA_variation_opt.npz : fragile__{sst2,ag}__<ct> AND topcurv__{sst2,ag}__<ct>, each (24, D_ct, 16)  [OPT-4 twist + OPT-3 swap]
  perlayer_seed_opt.npz   : seed__<ct>, contest__<ct>, lam__<ct>, each (24, 128)                         [OPT-3 seed concentration]

ct in {q_proj,k_proj,v_proj,out_proj,fc1,fc2}; D_ct = carrier in-dim (attn/fc1 1024, fc2 4096). seed = AG task-B
forgetting-seed energy at SST-w_A (the deposit's task), in the SST-w_A curvature basis. GUARDS: reproduce w_A SST acc
(~0.77) + reproduce banked rung2b_opt fG/tG medians. Per-layer never pooled. Engine untouched. ~50-70 min A100.
"""
import os, re
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "facebook/opt-350m"; AX = f"{G}/overlap_axis"
OUT = f"{G}/taskavar_seed_opt.json"; TAV = f"{G}/taskA_variation_opt.npz"; PLS = f"{G}/perlayer_seed_opt.npz"
CARRIERS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
STEPS_WA = 1200; LR_WA = 5e-5; STEPS_DEPTH = 200; LR_DEPTH = 5e-5; BATCH = 16; SUBSET = 2000
N_FISHER = 150; N_SEED = 16; KB = 128; KF = 16


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


def seedcov(model, params, names, tok, items, grab_g0=False):
    S = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    g0 = None; gg = torch.Generator().manual_seed(0)
    for it in range(N_SEED):
        idx = torch.randperm(len(items), generator=gg)[:BATCH].tolist()
        inp, lb, am = b_pairs(tok, [items[i] for i in idx]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        if grab_g0 and it == 0: g0 = {n: params[n].grad.detach().clone() for n in names}
        for n in names: gr = params[n].grad.detach().double(); S[n] += gr.T @ gr
    return S, g0


def train_items(model, tok, items, steps, lr, seed):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr); g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        idx = torch.randperm(len(items), generator=g)[:BATCH].tolist(); inp, lb, am = b_pairs(tok, [items[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    del opt


def grass(A, B): s = np.linalg.svd(A.T @ B, compute_uv=False); return float((s ** 2).sum() / A.shape[1])
def ctype(n):
    for c in CARRIERS:
        if c in n: return c
def layer_of(n): return int(re.search(r"layers\.(\d+)\.", n).group(1))


def main():
    t0 = time.time(); print(f"[GEN] loading {MODEL} (eager)...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(dev)
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    carrier_of = {n: ("attn" if any(a in n for a in ["q_proj", "k_proj", "v_proj", "out_proj"]) else "mlp") for n in carriers}
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]

    sstA = sst_items(SUBSET, 7); agB = load_jsonB("low_agnews"); subjP = load_jsonB("medium_subj")
    print(f"[GEN] training task-A=SST -> w_A...", flush=True)
    train_items(model, tok, sstA, STEPS_WA, LR_WA, 0)
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
    accA = cor / len(ev); print(f"[GEN] w_A SST acc={accA:.3f} (guard ~0.77) ({time.time()-t0:.0f}s)", flush=True)

    # curvature basis (SST) + AG forgetting-seed + contest, at w_A
    params = dict(model.named_parameters())
    V, LAM = curv_basis(model, tok, carriers, sstA)
    print(f"[GEN] curvature basis done ({time.time()-t0:.0f}s)", flush=True)
    S_B, _ = seedcov(model, params, carriers, tok, agB)
    seed_e = {n: (V[n].T @ S_B[n].float() @ V[n]).diagonal().clamp(min=0).cpu().numpy() for n in carriers}
    model.load_state_dict(wA); train_items(model, tok, agB, STEPS_DEPTH, LR_DEPTH, 0)
    paramsB = dict(model.named_parameters()); contest_e = {}
    for n in carriers:
        dwB = (paramsB[n].detach() - wA[n].to(dev)).double(); M = dwB.T @ dwB
        contest_e[n] = (V[n].double().T @ M @ V[n].double()).diagonal().clamp(min=0).cpu().numpy()
    model.load_state_dict(wA); params = dict(model.named_parameters())

    # fragile/topcurv (probe=Subj) at SST-w_A
    SsB, _ = seedcov(model, params, carriers, tok, subjP)
    fragile_SST = {}; topcurv_SST = {}
    for n in carriers:
        e = (V[n].T @ SsB[n].float() @ V[n]).diagonal().clamp(min=0).cpu().numpy()
        order = np.argsort(-e)[:KF]
        fragile_SST[n] = V[n][:, order].cpu().numpy(); topcurv_SST[n] = V[n][:, :KF].cpu().numpy()
    print(f"[GEN] SST fragile/topcurv done ({time.time()-t0:.0f}s)", flush=True)

    # task-A=AG (fresh base), fragile/topcurv (probe=Subj)
    del model; torch.cuda.empty_cache()
    model2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(dev)
    train_items(model2, tok, agB, STEPS_WA, LR_WA, 0)
    p2 = dict(model2.named_parameters()); V2, _ = curv_basis(model2, tok, carriers, agB)
    Ss2, _ = seedcov(model2, p2, carriers, tok, subjP)
    fragile_AG = {}; topcurv_AG = {}
    for n in carriers:
        e = (V2[n].T @ Ss2[n].float() @ V2[n]).diagonal().clamp(min=0).cpu().numpy()
        order = np.argsort(-e)[:KF]
        fragile_AG[n] = V2[n][:, order].cpu().numpy(); topcurv_AG[n] = V2[n][:, :KF].cpu().numpy()
    print(f"[GEN] AG fragile/topcurv done ({time.time()-t0:.0f}s)", flush=True)

    # ---- persist per-carrier-type, layer-stacked ----
    ct_names = {ct: sorted([n for n in carriers if ctype(n) == ct], key=layer_of) for ct in CARRIERS}
    for ct in CARRIERS: assert len(ct_names[ct]) == 24, f"{ct}: {len(ct_names[ct])} layers"
    tav = {}; pls = {}
    for ct in CARRIERS:
        ns = ct_names[ct]
        tav[f"fragile__sst2__{ct}"] = np.stack([fragile_SST[n] for n in ns])   # (24,D,16)
        tav[f"topcurv__sst2__{ct}"] = np.stack([topcurv_SST[n] for n in ns])
        tav[f"fragile__ag__{ct}"]   = np.stack([fragile_AG[n] for n in ns])
        tav[f"topcurv__ag__{ct}"]   = np.stack([topcurv_AG[n] for n in ns])
        tav[f"names__{ct}"]         = np.array(ns)
        pls[f"seed__{ct}"]    = np.stack([seed_e[n] for n in ns])              # (24,128)
        pls[f"contest__{ct}"] = np.stack([contest_e[n] for n in ns])
        pls[f"lam__{ct}"]     = np.stack([LAM[n] for n in ns])
    np.savez(TAV, **tav); np.savez(PLS, **pls)
    print(f"[GEN] saved taskA_variation_opt.npz + perlayer_seed_opt.npz ({time.time()-t0:.0f}s)", flush=True)

    # ---- GUARD: reproduce banked rung2b_opt fG/tG medians ----
    fG = np.array([grass(fragile_SST[n], fragile_AG[n]) for n in carriers])
    tG = np.array([grass(topcurv_SST[n], topcurv_AG[n]) for n in carriers])
    rep = {"model": MODEL, "wA_sst_acc": accA, "fG_median": float(np.median(fG)), "tG_median": float(np.median(tG)),
           "carriers": carriers, "note": "reproduce-guard: compare fG/tG median vs banked rung2b_opt"}
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[GEN] GUARD fG_median={np.median(fG):.3f} tG_median={np.median(tG):.3f} (vs banked rung2b_opt) "
          f"acc={accA:.3f} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
