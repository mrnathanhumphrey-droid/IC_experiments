"""BUILD w_A — fine-tune a base model on task-A (SST-2) to a solid anchor, save the full state_dict as the arc's w_A ckpt.
Prerequisite for Probe-11 (3rd-architecture specificity replication). CELL-driven. Prompt format matches the arc:
"{sentence}\nSentiment:" -> " positive"/" negative". Saves /home/ubuntu/gateA/<ckpt> + a small result json (anchor acc).
"""
import os, time, json, numpy as np, torch
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
G = "/home/ubuntu/gateA"; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
LR = float(os.environ.get("WA_LR", "2e-5")); STEPS = int(os.environ.get("WA_STEPS", "2000")); BATCH = 16
EVAL_EVERY = 250; EVAL_N = 400; HEARTBEAT = 50; _T0 = 0.0
CELLS = {
    "bloom560m": {"model": "bigscience/bloom-560m", "ckpt": "bloom560m_sst2_wA.pt", "family": "BLOOM"},
    "opt13b":    {"model": "facebook/opt-1.3b",     "ckpt": "opt13b_sst2_wA.pt",     "family": "OPT"},
    "xglm564m":  {"model": "facebook/xglm-564M",    "ckpt": "xglm564m_sst2_wA.pt",   "family": "XGLM"},  # sinusoidal-pos multilingual, shape-matched to BLOOM
    "falcon_rw1b": {"model": "tiiuae/falcon-rw-1b", "ckpt": "falcon_rw1b_sst2_wA.pt", "family": "FALCON"},  # ALiBi + ENGLISH (RefinedWeb), HF-native; the ALiBi confound-confirm
    "bloom560m_noln": {"model": "bigscience/bloom-560m", "ckpt": "bloom560m_noln_sst2_wA.pt", "family": "BLOOM", "ablate_embln": True},  # BLOOM w/ post-embed LayerNorm removed (conjunction-vs-LN fork)
    "jais590m":   {"model": "inceptionai/jais-family-590m", "ckpt": "jais590m_sst2_wA.pt", "family": "JAIS"},   # 2026-07-26 BREAK-N=1: ALiBi + Arabic/English multilingual, 2nd family for the eroder corner
    "qwen2_1p5b": {"model": "Qwen/Qwen2-1.5B",              "ckpt": "qwen2_1p5b_sst2_wA.pt", "family": "QWEN"},  # RoPE + multilingual: fills the empty cell (deferred; needs A100 for 8960-dim Fisher)
}
CELL = os.environ.get("CELL", ""); assert CELL in CELLS, f"CELL must be one of {list(CELLS)}; got {CELL!r}"
cfg = CELLS[CELL]; CKPT = f"{G}/{cfg['ckpt']}"; OUT = f"{G}/build_wA_{CELL}.json"


def _retry(fn, tries=5, wait=12):
    last = None
    for i in range(tries):
        try: return fn()
        except Exception as e: last = e; print(f"[hf] retry {i+1}: {str(e)[:80]}", flush=True); time.sleep(wait)
    raise last
def b_pairs(tok, items):
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"][:256]; c = tok(co, add_special_tokens=False)["input_ids"][:256]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    L = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)
def sst_rows(split, k, seed=0):
    from datasets import load_dataset
    ds = _retry(lambda: load_dataset("stanfordnlp/sst2"))[split]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))[:k]]
def eval_sst(model, tok, evalset, posid, negid):
    model.eval(); correct = 0; n = 0
    with torch.no_grad():
        for i in range(0, len(evalset), 32):
            chunk = evalset[i:i + 32]; seqs = [tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"][:256] for s, _ in chunk]
            L = max(len(x) for x in seqs); pad = tok.pad_token_id
            inp = np.full((len(seqs), L), pad, np.int64); pos = np.zeros(len(seqs), np.int64)
            for j, sq in enumerate(seqs): inp[j, :len(sq)] = sq; pos[j] = len(sq) - 1
            lg = model(input_ids=torch.tensor(inp).to(dev)).logits
            last = lg[torch.arange(len(seqs)), torch.tensor(pos).to(dev)]
            pred = (last[:, posid] > last[:, negid]).long().cpu().numpy()
            for j, (_, lab) in enumerate(chunk): correct += int(pred[j] == lab); n += 1
    return correct / max(n, 1)


def main():
    global _T0
    t0 = time.time(); _T0 = t0
    print(f"[buildwA:{CELL}] {cfg['model']} fine-tune on SST-2  lr={LR:.1e} steps={STEPS} ({time.ctime()})", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32, trust_remote_code=True).to(dev)
    if cfg.get("ablate_embln"):  # remove BLOOM's post-embed LayerNorm (block-0 input_layernorm cushions it); fine-tune adapts
        import torch.nn as nn; model.transformer.word_embeddings_layernorm = nn.Identity()
        print(f"[buildwA:{CELL}] ABLATED post-embed LayerNorm -> Identity", flush=True)
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    print(f"[buildwA:{CELL}] posid={posid} negid={negid} (distinct={posid!=negid})", flush=True)
    assert posid != negid, "positive/negative share first token — pick a different label verbalizer"
    train = [(f"{s}\nSentiment:", LAB[l]) for s, l in sst_rows("train", 8000, 0)]
    sst_eval = sst_rows("validation", EVAL_N, 0)
    pre = eval_sst(model, tok, sst_eval, posid, negid)
    print(f"[buildwA:{CELL}] base (untuned) SST-2 acc={pre:.3f} ({time.time()-t0:.0f}s)", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR); g = torch.Generator().manual_seed(0)
    best_acc = -1.0; best_sd = None; best_step = 0
    for step in range(STEPS):
        model.train(); idx = torch.randperm(len(train), generator=g)[:BATCH].tolist()
        inp, lb, am = b_pairs(tok, [train[i] for i in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % HEARTBEAT == 0: print(f"[hb] {CELL} step {step}/{STEPS} loss={float(loss):.3f} ({time.time()-_T0:.0f}s)", flush=True)
        if (step + 1) % EVAL_EVERY == 0 or step == STEPS - 1:
            acc = eval_sst(model, tok, sst_eval, posid, negid)
            print(f"[eval] {CELL} step {step+1} val_acc={acc:.4f} (best {best_acc:.4f}@{best_step}) ({time.time()-_T0:.0f}s)", flush=True)
            if acc > best_acc:
                best_acc = acc; best_step = step + 1; best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    torch.save(best_sd, CKPT)
    # confirm the saved ckpt loads + re-evals to the same anchor
    model.load_state_dict(best_sd); confirm = eval_sst(model, tok, sst_eval, posid, negid)
    rep = {"cell": CELL, "family": cfg["family"], "model": cfg["model"], "ckpt": cfg["ckpt"], "VOID": False,
           "base_acc": pre, "anchor_acc": best_acc, "best_step": best_step, "confirm_reload_acc": confirm,
           "lr": LR, "steps": STEPS, "posid": int(posid), "negid": int(negid)}
    json.dump(rep, open(OUT, "w"), indent=2)
    open(f"{G}/anchor_{CELL}.txt", "w").write(f"{best_acc:.4f}")  # plain value for the driver (avoids nested-quote hell)
    print(f"[buildwA:{CELL}] SAVED {CKPT}  anchor_acc={best_acc:.4f}@{best_step}  reload_confirm={confirm:.4f}  wall={time.time()-t0:.0f}s", flush=True)
    print(f"[buildwA:{CELL}] >>> set ANCHOR_WA['{cfg['family']}']={best_acc:.3f} in the Probe-11 harness <<<", flush=True)


if __name__ == "__main__":
    main()
