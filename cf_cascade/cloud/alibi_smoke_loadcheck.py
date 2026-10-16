"""alibi break-n=1 SMOKE (load+train+carrier+Fisher validation) for BLOOM-560m / jais-family-590m / Qwen2-1.5b.
Validates END-TO-END what the probe13 dose-response needs, cheaply: load under pinned transformers, SST-2 forward/loss/
backward (full-FT grads flow), carrier extraction (Conv1D for Jais), Fisher(shape1,shape1)+eigh+projection math, a few AdamW
steps (loss moves), SST pos/neg readout. NO 2000-step anchor. Writes out/alibi_smoke.json. ASCII-only."""
import os, json, time, traceback
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import numpy as np, torch, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers.pytorch_utils import Conv1D
except Exception:
    Conv1D = type("x", (), {})
from datasets import load_dataset

dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/home/ubuntu/gateA/alibi_smoke.json"; os.makedirs("/home/ubuntu/gateA", exist_ok=True)
LAB = {0: " negative", 1: " positive"}
CELLS = {
    "bloom560m":  {"model": "bigscience/bloom-560m",       "ck": ["self_attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"], "family": "BLOOM(ALiBi+multi ctrl)"},
    "jais590m":   {"model": "inceptionai/jais-family-590m",  "ck": ["attn.c_attn", "mlp.c_fc", "mlp.c_proj"],                                    "family": "JAIS(ALiBi+multi 2nd)"},
    "qwen2_1p5b": {"model": "Qwen/Qwen2-1.5B",               "ck": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"], "family": "QWEN(RoPE+multi)"},
}


def sst_rows(split, k):
    ds = load_dataset("glue", "sst2", split=split)
    ds = ds.shuffle(seed=0).select(range(min(k, len(ds))))
    return [(r["sentence"].strip(), int(r["label"])) for r in ds]


def b_pairs(tok, pairs):
    txts = [p + c for p, c in pairs]
    enc = tok(txts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    enc.pop("token_type_ids", None)
    inp = enc["input_ids"].to(dev); am = enc["attention_mask"].to(dev); lb = inp.clone()
    for i, (p, c) in enumerate(pairs):
        pl = len(tok(p)["input_ids"]); lb[i, :pl] = -100
    lb[am == 0] = -100
    return inp, lb, am


@torch.no_grad()
def eval_sst(model, tok, rows, posid, negid):
    correct = 0
    for s, l in rows:
        enc = tok(f"{s}\nSentiment:", return_tensors="pt", truncation=True, max_length=64)
        enc.pop("token_type_ids", None)
        lg = model(input_ids=enc["input_ids"].to(dev), attention_mask=enc["attention_mask"].to(dev)).logits[0, -1]
        pred = 1 if lg[posid] > lg[negid] else 0
        correct += (pred == l)
    return correct / len(rows)


def run_cell(key, cfg, t0):
    r = {"cell": key, "family": cfg["family"], "model": cfg["model"]}
    tok = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32, trust_remote_code=True).to(dev)
    r["loaded"] = True
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["ck"]) and n.endswith("weight") and params[n].ndim == 2]
    kinds = {}
    mods = dict(model.named_modules())
    for n in carriers[:3]:
        mod = mods.get(n.rsplit(".weight", 1)[0])
        kinds[n.split(".")[-2] + "." + n.split(".")[-1]] = ("Conv1D" if isinstance(mod, Conv1D) else "Linear") + str(tuple(params[n].shape))
    r["n_carriers"] = len(carriers); r["carrier_kinds_sample"] = kinds
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    r["posid"] = int(posid); r["negid"] = int(negid)
    ev = sst_rows("validation", 120)
    r["acc_pretrain"] = round(eval_sst(model, tok, ev, posid, negid), 3)

    # Fisher(shape1,shape1)+eigh+projection on the FIRST carrier (validates Conv1D-orientation math)
    tr = sst_rows("train", 20); n0 = carriers[0]
    F = torch.zeros(params[n0].shape[1], params[n0].shape[1], device=dev)
    for s, l in tr:
        inp, lb, am = b_pairs(tok, [(f"{s}\nSentiment:", LAB[l])]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        g = params[n0].grad.detach().float(); F += g.T @ g
    _, vv = torch.linalg.eigh(F.double()); V = vv[:, -16:].flip(dims=[1]).contiguous()
    d = torch.randn_like(params[n0]).double(); proj = (d @ V) @ V.T
    r["fisher_proj_ok"] = bool(proj.shape == params[n0].shape and torch.isfinite(proj).all())
    r["proj_frac"] = round(float((proj*proj).sum() / ((d*d).sum()+1e-30)), 4)

    # a few full-FT AdamW steps -> loss moves + eval moves
    model.load_state_dict({k: v for k, v in model.state_dict().items()})  # noop keep
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    losses = []
    for step in range(20):
        s, l = tr[step % len(tr)]
        inp, lb, am = b_pairs(tok, [(f"{s}\nSentiment:", LAB[l])])
        loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(float(loss))
    r["loss_first"] = round(losses[0], 3); r["loss_last"] = round(losses[-1], 3); r["loss_dropped"] = bool(losses[-1] < losses[0])
    r["acc_after20"] = round(eval_sst(model, tok, ev, posid, negid), 3)
    del model; torch.cuda.empty_cache()
    print(f"[SMOKE] {key}: loaded carriers={r['n_carriers']} kinds={kinds} accPre={r['acc_pretrain']} "
          f"loss {r['loss_first']}->{r['loss_last']} fisher_proj_ok={r['fisher_proj_ok']} ({time.time()-t0:.0f}s)", flush=True)
    return r


def main():
    t0 = time.time(); results = {}
    for key, cfg in CELLS.items():
        try:
            results[key] = run_cell(key, cfg, t0)
        except Exception as e:
            traceback.print_exc(); results[key] = {"cell": key, "loaded": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
            print(f"[SMOKE] {key} FAIL: {type(e).__name__}: {str(e)[:200]}", flush=True)
        json.dump(results, open(OUT, "w"), indent=1)
    ok = {k: (v.get("loaded") and v.get("fisher_proj_ok") and v.get("loss_dropped")) for k, v in results.items()}
    print(f"\n[SMOKE] READY: {ok}", flush=True)
    print(f"[SMOKE] wrote {OUT} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
