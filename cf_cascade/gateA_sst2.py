"""Gate A (v2) — task A = SST-2 sentiment (cast as generation), a GENUINELY-LEARNABLE
distinct task (replaces WikiText, which was too in-distribution: flat val, no basin).
Reach a REAL sharp basin (val genuinely drops, grad-norm genuinely small), freeze the
ESD basis, re-confirm decade-span. Shared checkpoint feeds c2 (Gate B) + force (F0).

The number gating the force lane: does task A reach a genuine basin (sharp val drop +
small grad-norm), unlike WikiText (flat val 2.77, grad-norm 2.02 at the floor)?

Locked plateau criterion (same as A+): constant LR, val every 150 steps, plateau =
<0.5% val-loss improvement x2 consecutive; overfit-guard -> best-val; max steps cap.
Loss is COMPLETION-ONLY (label tokens), so "task A learned" = predicts the label =
sharp basin where completion-loss -> 0.
"""
import os, json, time, copy
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import numpy as np
import torch

SEED = 0
MODEL = "EleutherAI/pythia-410m"
BATCH = 16; LR = 5e-5; EVAL_EVERY = 150; MAX_STEPS = 2500
PLATEAU_REL = 0.005; VAL_N = 400
OUT = "D:/IC_experiments/cf_cascade/gateA"
SQUARE = "attention.dense"
LABELS = {0: " negative", 1: " positive"}
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
dev = "cuda"


def load_sst2():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")
    def prep(split):
        return [(r["sentence"].strip(), int(r["label"])) for r in ds[split] if r["label"] in (0, 1)]
    return prep("train"), prep("validation")


def make_batch(tok, rows):
    """rows: list of (sentence, label). Returns input_ids, labels (prompt masked), attn."""
    seqs, labs = [], []
    for sent, lab in rows:
        p = tok(f"{sent}\nSentiment:", add_special_tokens=False)["input_ids"]
        c = tok(LABELS[lab], add_special_tokens=False)["input_ids"]
        ids = p + c
        lab_ids = [-100] * len(p) + c
        seqs.append(ids); labs.append(lab_ids)
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64)
    lb = np.full((len(seqs), L), -100, np.int64)
    am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return (torch.tensor(inp), torch.tensor(lb), torch.tensor(am))


@torch.no_grad()
def evaluate(model, tok, val):
    """Completion-loss (for plateau) + label accuracy (score both completions)."""
    model.eval()
    sub = val[:VAL_N]
    tot, nb = 0.0, 0
    for i in range(0, len(sub) - BATCH, BATCH):
        inp, lb, am = [t.to(dev) for t in make_batch(tok, sub[i:i + BATCH])]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tot += float(model(input_ids=inp, attention_mask=am, labels=lb).loss.item()); nb += 1
    # accuracy: per example, compare gold vs other completion loss
    correct = 0; n = 0
    for sent, lab in sub:
        scores = {}
        for cl in (0, 1):
            inp, lbl, am = [t.to(dev) for t in make_batch(tok, [(sent, cl)])]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                scores[cl] = float(model(input_ids=inp, attention_mask=am, labels=lbl).loss.item())
        pred = min(scores, key=scores.get)
        correct += int(pred == lab); n += 1
    model.train()
    return tot / max(nb, 1), correct / max(n, 1)


def esd_eigh(W):
    W = W.astype(np.float64); m, n = W.shape
    ev, vec = np.linalg.eigh((W.T @ W) / m); order = np.argsort(ev)[::-1]
    return ev[order], vec[:, order]


def decade_span(ev):
    ev = ev[ev > 0]; lmax = ev[0]; bulk = ev[ev > 1e-6 * lmax]
    return float(np.log10(lmax / bulk[-1]))


def grad_norm(model, tok, train, n_batches=8):
    model.zero_grad(set_to_none=True); g = torch.Generator().manual_seed(123)
    idx = torch.randperm(len(train), generator=g)[:n_batches * BATCH].tolist()
    for j in range(n_batches):
        rows = [train[k] for k in idx[j * BATCH:(j + 1) * BATCH]]
        inp, lb, am = [t.to(dev) for t in make_batch(tok, rows)]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            (model(input_ids=inp, attention_mask=am, labels=lb).loss / n_batches).backward()
    sq = sum(float(p.grad.pow(2).sum()) for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True); return sq ** 0.5


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.train()
    train, val = load_sst2()
    print(f"[A2] SST-2: {len(train)} train, {len(val)} val", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED)

    # baseline (pretrained, pre-FT) eval -> shows it starts BAD (genuine learning)
    base_loss, base_acc = evaluate(model, tok, val)
    print(f"[A2] PRE-FT baseline: val_loss {base_loss:.4f}  acc {base_acc:.3f}", flush=True)

    val_hist = []; best = {"val": 1e9, "step": 0, "state": None, "acc": 0.0}
    freeze_step = freeze_reason = None; step = 0
    while step < MAX_STEPS and freeze_step is None:
        order = torch.randperm(len(train), generator=g).tolist()
        for i in range(0, len(order) - BATCH, BATCH):
            rows = [train[k] for k in order[i:i + BATCH]]
            inp, lb, am = [t.to(dev) for t in make_batch(tok, rows)]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            if step % EVAL_EVERY == 0:
                vl, acc = evaluate(model, tok, val)
                val_hist.append({"step": step, "val": vl, "acc": acc, "train": float(loss.item())})
                print(f"  step {step:4d}  val {vl:.4f}  acc {acc:.3f}  train {loss.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
                if vl < best["val"]:
                    best = {"val": vl, "step": step, "acc": acc,
                            "state": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})}
                if len(val_hist) >= 3:
                    r1 = (val_hist[-2]["val"] - val_hist[-1]["val"]) / val_hist[-2]["val"]
                    r2 = (val_hist[-3]["val"] - val_hist[-2]["val"]) / val_hist[-3]["val"]
                    if 0 <= r1 < PLATEAU_REL and 0 <= r2 < PLATEAU_REL:
                        freeze_step, freeze_reason = step, "plateau"; break
                    if r1 < 0 and r2 < 0:
                        freeze_step, freeze_reason = best["step"], "overfit_guard_best_val"; break
            if step >= MAX_STEPS:
                freeze_step, freeze_reason = step, "max_steps"; break

    if freeze_reason == "overfit_guard_best_val" and best["state"]:
        model.load_state_dict({k: v.to(dev) for k, v in best["state"].items()})
        print(f"[A2] overfit guard: restored best-val @ step {best['step']}", flush=True)

    gn = grad_norm(model, tok, train)
    final_loss, final_acc = evaluate(model, tok, val)
    print(f"[A2] FREEZE step={freeze_step} reason={freeze_reason}  grad_norm(8b)={gn:.4f}  "
          f"val_acc {final_acc:.3f} (base {base_acc:.3f})", flush=True)

    # re-freeze ESD
    spans = []; frozen = {}
    for name, p in model.state_dict().items():
        if SQUARE in name and p.ndim == 2 and tuple(p.shape) == (1024, 1024):
            W = p.detach().float().cpu().numpy(); ev, vec = esd_eigh(W); spans.append(decade_span(ev))
            frozen[name] = {"eig": ev.astype(np.float32), "vec": vec.astype(np.float32), "W": W.astype(np.float32)}
    spans = np.array(spans); median_span = float(np.median(spans)); PASS = median_span >= 2.0

    np.savez_compressed(f"{OUT}/frozen_basis_sst2.npz",
                        **{k.replace('.', '_') + "__eig": v["eig"] for k, v in frozen.items()},
                        **{k.replace('.', '_') + "__vec": v["vec"] for k, v in frozen.items()},
                        **{k.replace('.', '_') + "__W": v["W"] for k, v in frozen.items()})
    torch.save(model.state_dict(), f"{OUT}/pythia410m_sst2.pt")

    basin = (final_acc - base_acc > 0.15) and (gn < 1.0)   # genuine learning + near-stationary
    print(f"\n[A2] === TASK-A BASIN CHECK (SST-2) ===", flush=True)
    print(f"  val acc: {base_acc:.3f} (pre-FT) -> {final_acc:.3f} (task-A min)  [learned {final_acc-base_acc:+.3f}]", flush=True)
    print(f"  val loss: {base_loss:.4f} -> {best['val']:.4f}", flush=True)
    print(f"  grad_norm(8-batch) = {gn:.4f}   (WikiText was 2.02 at its non-basin)", flush=True)
    print(f"  GENUINE BASIN: {'YES - sharp learning + small grad-norm, force lane F0 viable' if basin else 'NO - not enough learning or not stationary, reconsider'}", flush=True)
    print(f"  re-frozen ESD decade-span median = {median_span:.2f} [{spans.min():.2f},{spans.max():.2f}]  ESD PASS={PASS}", flush=True)

    json.dump({"model": MODEL, "task_A": "SST-2 sentiment (generation)", "task_B_planned": "AG-News topic",
               "seed": SEED, "LR": LR, "freeze_step": freeze_step, "freeze_reason": freeze_reason,
               "base_acc": base_acc, "final_acc": final_acc, "base_val_loss": base_loss, "best_val": best["val"],
               "grad_norm_8batch": gn, "val_history": val_hist, "genuine_basin": bool(basin),
               "refrozen_decade_span_median": median_span, "ESD_PASS": bool(PASS),
               "wall_s": time.time() - t0},
              open(f"{OUT}/gateA_sst2_summary.json", "w"), indent=2)
    print(f"[A2] saved frozen_basis_sst2.npz, pythia410m_sst2.pt, gateA_sst2_summary.json  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
