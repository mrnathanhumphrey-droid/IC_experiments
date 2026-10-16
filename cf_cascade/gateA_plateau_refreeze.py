"""Gate A+ — extend task-A to a LOCKED-CRITERION plateau, re-freeze the ESD basis,
re-confirm decade-span. Shared critical path: this one task-A minimum feeds BOTH
the c2 lane (Gate B damage baseline) and the thermal/force lane (binding structure).

LOCKED plateau criterion (given-not-fitted, declared before running):
  - constant LR (no schedule; TK3 requires it so aging can't leak from a schedule)
  - validation loss (wikitext-2-raw validation, fixed subset) every EVAL_EVERY steps
  - PLATEAU = val-loss relative improvement < 0.5% over trailing interval, 2 consecutive -> freeze
  - OVERFIT GUARD = if val-loss rises 2 consecutive intervals first, freeze at best (min) val step
  - max budget MAX_STEPS (report if hit without plateau)
  - at freeze: large-batch gradient-norm estimate (stationarity check for the force lane)
Then re-freeze ESD on the 24 square attention.dense matrices, re-confirm decade-span ~5.86.
"""
import os, json, time, copy
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import numpy as np
import torch

SEED = 0
MODEL = "EleutherAI/pythia-410m"
BLOCK = 512; BATCH = 8; LR = 2e-5
EVAL_EVERY = 150; MAX_STEPS = 4000
PLATEAU_REL = 0.005          # <0.5% val improvement
VAL_BLOCKS = 200
OUT = "D:/IC_experiments/cf_cascade/gateA"
SQUARE = "attention.dense"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
dev = "cuda"


def blocks_for(tok, split):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="np")["input_ids"][0]
    n = (len(ids) // BLOCK) * BLOCK
    return torch.tensor(ids[:n].reshape(-1, BLOCK), dtype=torch.long)


def esd_eigh(W):
    W = W.astype(np.float64); m, n = W.shape
    G = (W.T @ W) / m
    ev, vec = np.linalg.eigh(G)
    order = np.argsort(ev)[::-1]
    return ev[order], vec[:, order]


def decade_span(ev):
    ev = ev[ev > 0]; lmax = ev[0]; bulk = ev[ev > 1e-6 * lmax]
    return float(np.log10(lmax / bulk[-1])), float(np.log10(np.percentile(bulk, 99) / np.percentile(bulk, 1)))


@torch.no_grad()
def val_loss(model, vblocks):
    model.eval(); tot = 0.0; nb = 0
    for i in range(0, min(VAL_BLOCKS, vblocks.shape[0]) - BATCH, BATCH):
        x = vblocks[i:i + BATCH].to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            tot += float(model(x, labels=x).loss.item()); nb += 1
    model.train(); return tot / max(nb, 1)


def grad_norm_estimate(model, tblocks, n_batches=8):
    model.zero_grad(set_to_none=True); g = torch.Generator().manual_seed(123)
    idx = torch.randperm(tblocks.shape[0], generator=g)[:n_batches * BATCH]
    for j in range(n_batches):
        x = tblocks[idx[j * BATCH:(j + 1) * BATCH]].to(dev)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            (model(x, labels=x).loss / n_batches).backward()
    sq = sum(float(p.grad.pow(2).sum()) for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True); return sq ** 0.5


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.train()
    tblocks = blocks_for(tok, "train"); vblocks = blocks_for(tok, "validation")
    print(f"[A+] train {tblocks.shape[0]} blocks, val {vblocks.shape[0]} blocks", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED)

    val_hist = []; best = {"val": 1e9, "step": 0, "state": None}
    freeze_step = None; freeze_reason = None
    step = 0
    while step < MAX_STEPS and freeze_step is None:
        perm = torch.randperm(tblocks.shape[0], generator=g)
        for i in range(0, tblocks.shape[0] - BATCH, BATCH):
            x = tblocks[perm[i:i + BATCH]].to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, labels=x).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step += 1
            if step % EVAL_EVERY == 0:
                vl = val_loss(model, vblocks); val_hist.append({"step": step, "val": vl, "train": float(loss.item())})
                print(f"  step {step:4d}  val {vl:.4f}  train {loss.item():.4f}  ({time.time()-t0:.0f}s)", flush=True)
                if vl < best["val"]:
                    best = {"val": vl, "step": step, "state": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})}
                # plateau / overfit checks on val_hist
                if len(val_hist) >= 3:
                    r1 = (val_hist[-2]["val"] - val_hist[-1]["val"]) / val_hist[-2]["val"]
                    r2 = (val_hist[-3]["val"] - val_hist[-2]["val"]) / val_hist[-3]["val"]
                    if r1 < PLATEAU_REL and r2 < PLATEAU_REL and r1 >= 0 and r2 >= 0:
                        freeze_step, freeze_reason = step, "plateau"; break
                    if r1 < 0 and r2 < 0:   # val rising 2 consecutive -> overfit
                        freeze_step, freeze_reason = best["step"], "overfit_guard_best_val"; break
            if step >= MAX_STEPS:
                freeze_step, freeze_reason = step, "max_steps"; break

    # restore best-val weights if overfit guard fired
    if freeze_reason == "overfit_guard_best_val" and best["state"] is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best["state"].items()})
        print(f"[A+] overfit guard: restored best-val checkpoint @ step {best['step']} (val {best['val']:.4f})", flush=True)

    gn = grad_norm_estimate(model, tblocks)
    print(f"[A+] FREEZE: step={freeze_step} reason={freeze_reason}  grad_norm(8-batch)={gn:.4f}", flush=True)

    # ---- re-freeze ESD on the 24 square matrices ----
    model.eval(); spans = []; frozen = {}
    for name, p in model.state_dict().items():
        if SQUARE in name and p.ndim == 2 and tuple(p.shape) == (1024, 1024):
            W = p.detach().float().cpu().numpy()
            ev, vec = esd_eigh(W); dfull, dp99 = decade_span(ev); spans.append(dfull)
            frozen[name] = {"eigenvalues": ev.astype(np.float32), "eigenvectors": vec.astype(np.float32),
                            "W_taskA": W.astype(np.float32)}
    spans = np.array(spans); median_span = float(np.median(spans))
    PASS = median_span >= 2.0

    np.savez_compressed(f"{OUT}/frozen_basis_taskA_plateau.npz",
                        **{k.replace('.', '_') + "__eig": v["eigenvalues"] for k, v in frozen.items()},
                        **{k.replace('.', '_') + "__vec": v["eigenvectors"] for k, v in frozen.items()},
                        **{k.replace('.', '_') + "__W": v["W_taskA"] for k, v in frozen.items()})
    torch.save(model.state_dict(), f"{OUT}/pythia410m_taskA_plateau.pt")

    print(f"\n[A+] === RE-FROZEN BASIS (plateau) ===", flush=True)
    print(f"  freeze: step={freeze_step} reason={freeze_reason} val={best['val']:.4f} grad_norm={gn:.4f}", flush=True)
    print(f"  ESD decade-span median = {median_span:.2f} [{spans.min():.2f}, {spans.max():.2f}]  vs 5.86 (900-step) / 5.89 (pretrained)", flush=True)
    print(f"  VERDICT: {'PASS - plateau reached AND geometry held (>=2 decades)' if PASS else 'FAIL - ESD collapsed at plateau'}", flush=True)

    json.dump({"model": MODEL, "task": "wikitext-2-raw causal-LM", "seed": SEED, "LR": LR,
               "plateau_criterion": "val rel-improve <0.5% x2 consecutive; overfit-guard->best-val; max 4000",
               "freeze_step": freeze_step, "freeze_reason": freeze_reason,
               "best_val": best["val"], "best_val_step": best["step"], "grad_norm_8batch": gn,
               "val_history": val_hist,
               "refrozen_decade_span_median": median_span,
               "refrozen_decade_span_minmax": [float(spans.min()), float(spans.max())],
               "PASS": bool(PASS), "wall_s": time.time() - t0},
              open(f"{OUT}/gateA_plateau_summary.json", "w"), indent=2)
    print(f"[A+] saved frozen_basis_taskA_plateau.npz, pythia410m_taskA_plateau.pt, gateA_plateau_summary.json  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
