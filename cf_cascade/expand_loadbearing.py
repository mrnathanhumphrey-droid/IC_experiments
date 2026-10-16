"""MLP/QKV expansion, Stage A: per-param-TYPE load-bearing decomposition. WHERE does the forgetting
live? attention.dense is only ~1/3 (reversibility v2). Localize the other ~2/3 across param types,
to point the 2D recovery x importance map (Stage B) at the carrier(s).

Retrain task-B FULL (Pythia/AG, the valid substrate, forgetting ~+0.083). Restore each param-TYPE to
w_A in the full forgotten model, measure task-A acc recovered. Validity gate: full SST-2 val (872, up
from 300) to de-noise. Types: attn.dense, attn.qkv, mlp.h_to_4h, mlp.4h_to_h, embed, layernorm, +
grouped (all_attn, all_mlp, all_linear).
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "4"
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import json, time, numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(4)
G = "D:/IC_experiments/cf_cascade/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
OUT = f"{G}/expand_loadbearing_pythia_agnews.json"; TASKB = "low_agnews"
BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02; VAL_N = 872; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True

TYPES = {
  "attn_dense": ["attention.dense.weight"], "attn_qkv": ["attention.query_key_value.weight"],
  "mlp_h4h": ["mlp.dense_h_to_4h.weight"], "mlp_4hh": ["mlp.dense_4h_to_h.weight"],
  "embed": ["embed_in", "embed_out"], "layernorm": ["layernorm", "layer_norm", "ln_"],
  "all_attn": ["attention.dense.weight", "attention.query_key_value.weight"],
  "all_mlp": ["mlp.dense_h_to_4h.weight", "mlp.dense_4h_to_h.weight"],
  "all_linear": ["attention.dense.weight", "attention.query_key_value.weight", "mlp.dense_h_to_4h.weight", "mlp.dense_4h_to_h.weight"],
}


def sst_val():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")
    return [(r["sentence"].strip(), int(r["label"])) for r in ds["validation"] if r["label"] in (0, 1)][:VAL_N]


def taskb():
    d = json.load(open(f"{AX}/{TASKB}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


def b_sst(tok, items):
    seqs, labs = [], []
    for s, l in items:
        p = tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"]; c = tok(LAB[l], add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    return pack(tok, seqs, labs)


def b_pair(tok, items):
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"]; c = tok(co, add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    return pack(tok, seqs, labs)


def pack(tok, seqs, labs):
    L = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


@torch.no_grad()
def acc(model, tok, va):
    Lc = {}
    for cl in (0, 1):
        per = []; items = [(s, cl) for s, _ in va]
        for i in range(0, len(items), 64):
            inp, lb, am = b_sst(tok, items[i:i + 64]); lg = model(input_ids=inp, attention_mask=am).logits
            sl = lg[:, :-1].reshape(-1, lg.shape[-1]); st = lb[:, 1:].reshape(-1)
            ll = F.cross_entropy(sl, st, ignore_index=-100, reduction="none").reshape(lb.shape[0], -1)
            per.extend((ll.sum(1) / (lb[:, 1:] != -100).sum(1).clamp(min=1)).tolist())
        Lc[cl] = per
    return float(np.mean([int((0 if Lc[0][i] < Lc[1][i] else 1) == va[i][1]) for i in range(len(va))]))


def main():
    t0 = time.time(); print("[EXP-A] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); va = sst_val(); B_items = taskb()
    model.eval(); acc_A = acc(model, tok, va)
    wA = {n: p.detach().clone() for n, p in model.named_parameters()}                  # FULL w_A
    nparam = {n: int(p.numel()) for n, p in model.named_parameters()}
    print(f"[EXP-A] acc_A={acc_A:.3f}; training task-B FULL... ({time.time()-t0:.0f}s)", flush=True)
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR); g = torch.Generator().manual_seed(0)
    run = []; step = 0; done = False
    while step < MAX_STEPS and not done:
        ordr = torch.randperm(len(B_items), generator=g).tolist()
        for i in range(0, len(ordr) - BATCH, BATCH):
            inp, lb, am = b_pair(tok, [B_items[k] for k in ordr[i:i + BATCH]])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            run.append(float(loss.item())); run = run[-80:]
            if step % 200 == 0 and np.mean(run) < TRAINSTOP: done = True; break
            if step >= MAX_STEPS: done = True; break
        if (run and np.mean(run) < TRAINSTOP) or step >= MAX_STEPS: done = True
    model.eval(); acc_B = acc(model, tok, va); forgetting = acc_A - acc_B
    wB = {n: p.detach().clone() for n, p in model.named_parameters()}
    params = dict(model.named_parameters())
    print(f"[EXP-A] acc_B={acc_B:.3f} forgetting={forgetting:+.3f} (val N={VAL_N}) ({time.time()-t0:.0f}s)\n", flush=True)

    def restore(patterns):
        hit = [n for n in wA if any(p in n for p in patterns)]
        for n in hit: params[n].data = wA[n].data
        a = acc(model, tok, va)
        for n in hit: params[n].data = wB[n].data
        return a, sum(nparam[n] for n in hit), len(hit)

    res = {}
    print("[EXP-A] === PER-TYPE LOAD-BEARING (restore type to w_A in full w_B) ===", flush=True)
    for name, pats in TYPES.items():
        a, npar, nh = restore(pats); rec = (a - acc_B) / max(forgetting, 1e-9)
        res[name] = {"acc": a, "recovery": rec, "n_params_M": npar / 1e6, "n_matrices": nh}
        print(f"  {name:12s} restore -> acc {a:.3f} recovers {rec:+.2f} of forgetting ({npar/1e6:.0f}M params, {nh} tensors)", flush=True)
    json.dump({"acc_A": acc_A, "acc_B": acc_B, "forgetting": forgetting, "val_N": VAL_N, "types": res,
               "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"\n[EXP-A] saved expand_loadbearing_pythia_agnews.json  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
