"""Gate A — task-A fine-tune Pythia-410M, freeze weight-ESD eigenbasis on the 24
square attention.dense (1024x1024) matrices at the task-A minimum, RE-CONFIRM the
ESD is still multi-decade on the frozen basis (the first report-and-hold stop).

LOCKED (given-not-fitted, before any damage is measured):
  - Task A = WikiText-2-raw-v1 causal-LM fine-tune (domain-forgetting CF setting,
    cf. 2401.05605). Task B (later, Gate B) = a conflicting domain.
  - Frozen basis = eigenbasis of W^T W at the task-A minimum, per square matrix,
    components ordered by eigenvalue (descending). Damage (Gate B) projects onto
    THIS frozen basis. Scale axis (Gate C) = component-INDEX dyadic boxes in this
    ordering (K11), NOT eigenvalue bands.
  - No gradient checkpointing -> the minimum is not checkpointing-perturbed.

FIRST STOP (report, hold): frozen-basis ESD decade-span per square matrix vs the
pretrained 4.4 (Gate-0a). PASS = median >= 2 decades (geometry still scale-rich).
FAIL = task-A FT collapsed the HT-SR structure -> geometry gone, stop & reconsider.
Do NOT proceed to Gate B from this script.
"""
import os, json, time
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import numpy as np
import torch
import torch.nn.functional as F

SEED = 0
MODEL = "EleutherAI/pythia-410m"
BLOCK = 512
BATCH = 8
LR = 2e-5
MAX_STEPS = 900
LOG_EVERY = 50
OUT = "D:/IC_experiments/cf_cascade/gateA"
SQUARE = "attention.dense"          # the 24 (1024x1024) output projections
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED)
torch.backends.cudnn.benchmark = True
dev = "cuda"


def build_blocks(tok):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="np")["input_ids"][0]
    n = (len(ids) // BLOCK) * BLOCK
    blocks = ids[:n].reshape(-1, BLOCK)
    print(f"[gateA] task-A corpus: {len(ids):,} tokens -> {blocks.shape[0]} blocks of {BLOCK}")
    return torch.tensor(blocks, dtype=torch.long)


def esd_eigh(W):
    """Eigen-decomposition of the Gram W^T W / m (Martin-Mahoney). Returns
    eigenvalues (descending) and eigenvectors (the frozen basis, columns)."""
    W = W.astype(np.float64)
    m, n = W.shape
    G = (W.T @ W) / m                       # n x n, n=1024
    ev, vec = np.linalg.eigh(G)             # ascending
    order = np.argsort(ev)[::-1]
    return ev[order], vec[:, order]


def decade_span(ev):
    ev = ev[ev > 0]
    lmax = ev[0]
    bulk = ev[ev > 1e-6 * lmax]
    return float(np.log10(lmax / bulk[-1])), float(np.log10(np.percentile(bulk, 99) / np.percentile(bulk, 1)))


def main():
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).to(dev)
    model.train()
    blocks = build_blocks(tok)
    nblk = blocks.shape[0]

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(SEED)

    print(f"[gateA] full fine-tune: batch={BATCH} block={BLOCK} lr={LR} max_steps={MAX_STEPS}", flush=True)
    losses = []
    step = 0
    while step < MAX_STEPS:
        perm = torch.randperm(nblk, generator=g)
        for i in range(0, nblk - BATCH, BATCH):
            idx = perm[i:i + BATCH]
            x = blocks[idx].to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(x, labels=x)
                loss = out.loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % LOG_EVERY == 0 or step == 1:
                l = float(loss.item())
                losses.append({"step": step, "loss": l})
                print(f"  step {step:4d}/{MAX_STEPS}  loss {l:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            if step >= MAX_STEPS:
                break

    # plateau check: last-5 vs prior-5 mean
    tail = [d["loss"] for d in losses[-5:]]
    prior = [d["loss"] for d in losses[-10:-5]] or tail
    plateau = abs(np.mean(tail) - np.mean(prior)) / np.mean(prior) < 0.02
    print(f"[gateA] final loss {tail[-1]:.4f}  plateau={plateau} "
          f"(last5 {np.mean(tail):.4f} vs prior5 {np.mean(prior):.4f})", flush=True)

    # ---- freeze ESD basis on the 24 square attention.dense matrices ----
    print(f"\n[gateA] freezing ESD eigenbasis on '{SQUARE}' (1024x1024) matrices ...", flush=True)
    model.eval()
    frozen = {}
    spans = []
    sd = model.state_dict()
    for name, p in sd.items():
        if SQUARE in name and p.ndim == 2 and tuple(p.shape) == (1024, 1024):
            W = p.detach().float().cpu().numpy()
            ev, vec = esd_eigh(W)
            dfull, dp99 = decade_span(ev)
            spans.append(dfull)
            frozen[name] = {"eigenvalues": ev.astype(np.float32), "eigenvectors": vec.astype(np.float32),
                            "W_taskA": W.astype(np.float32)}
            print(f"  {name:46s} decades={dfull:.2f} (p1-99 {dp99:.2f})  lmax={ev[0]:.3e}", flush=True)

    spans = np.array(spans)
    median_span = float(np.median(spans))
    PASS = median_span >= 2.0

    # save frozen basis (compressed) + loss curve
    np.savez_compressed(f"{OUT}/frozen_basis_taskA.npz",
                        **{k.replace(".", "_") + "__eig": v["eigenvalues"] for k, v in frozen.items()},
                        **{k.replace(".", "_") + "__vec": v["eigenvectors"] for k, v in frozen.items()},
                        **{k.replace(".", "_") + "__W": v["W_taskA"] for k, v in frozen.items()})
    torch.save(model.state_dict(), f"{OUT}/pythia410m_taskA.pt")

    print(f"\n[gateA] === FIRST STOP: frozen-basis ESD decade-span ===", flush=True)
    print(f"  n matrices = {len(spans)}", flush=True)
    print(f"  median decade-span (frozen task-A basis) = {median_span:.2f}   "
          f"[min {spans.min():.2f}, max {spans.max():.2f}]", flush=True)
    print(f"  pretrained (Gate-0a) reference          = 4.44 robust / 5.89 full", flush=True)
    print(f"  VERDICT: {'PASS - geometry still scale-rich (>=2 decades), proceed to Gate B' if PASS else 'FAIL - task-A FT collapsed the ESD, geometry GONE, stop & reconsider'}", flush=True)

    json.dump({"model": MODEL, "task": "wikitext-2-raw causal-LM", "seed": SEED,
               "steps": step, "final_loss": tail[-1], "plateau": bool(plateau),
               "n_square_matrices": len(spans),
               "frozen_basis_decade_span_median": median_span,
               "frozen_basis_decade_span_min": float(spans.min()),
               "frozen_basis_decade_span_max": float(spans.max()),
               "frozen_basis_decade_spans": spans.tolist(),
               "pretrained_reference_robust": 4.44, "PASS": bool(PASS),
               "grad_checkpointing": False,
               "wall_s": time.time() - t0},
              open(f"{OUT}/gateA_summary.json", "w"), indent=2)
    print(f"\n[gateA] saved frozen_basis_taskA.npz, pythia410m_taskA.pt, gateA_summary.json  "
          f"wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
