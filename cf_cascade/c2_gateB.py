"""c2 Gate B — the damage field + tail-index alpha (the decision-tree gate).
Fine-tune task-B (AG-News) from the SST-2 genuine-PSD minimum -> forgetting. Damage =
(w_B - w_A) per attention.dense, projected onto the FROZEN weight-ESD eigen-shells (task-A
weight-ESD basis = the c2 spectral coords, gauge-free; NOT the K-FAC binding basis, FK2).
Damage measure per ESD component i: ||Delta_w @ u_i||^2 (positive; energy of the weight
change in ESD column-direction i, ordered by ESD eigenvalue).

FIRST report-and-hold = tail-index alpha of the pooled damage distribution (Hill). Decision:
  alpha >= 5  -> moderate-p c2 fit ok
  alpha ~ 3.7 -> fractional low-p {0.5,1,1.5,2,2.5} + c3-truncation check
  alpha < 3   -> moment-based c2 infeasible -> rank/quantile estimator or concede
Also report task-A forgetting (SST-2 acc before/after B) + task-B learning (sanity: real CF).
MFDFA c2 + within-shell surrogate come AFTER alpha routes the instrument.
"""
import os, json, time
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import numpy as np, torch

MODEL = "EleutherAI/pythia-410m"; CKPT = "D:/IC_experiments/cf_cascade/gateA/pythia410m_sst2_subsetmin.pt"
AX = "D:/IC_experiments/cf_cascade/gateA/overlap_axis"; OUT = "D:/IC_experiments/cf_cascade/gateA/c2_gateB.json"
BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02; SUBSET = 2000; VAL_N = 400
LAB_SST = {0: " negative", 1: " positive"}; SQUARE = "attention.dense"; dev = "cuda"
torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True


def sst_data():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")
    tr = [(r["sentence"].strip(), int(r["label"])) for r in ds["train"] if r["label"] in (0, 1)]
    tr = [tr[i] for i in np.random.default_rng(0).permutation(len(tr))[:SUBSET]]
    va = [(r["sentence"].strip(), int(r["label"])) for r in ds["validation"] if r["label"] in (0, 1)][:VAL_N]
    return tr, va


def agnews_items():
    d = json.load(open(f"{AX}/low_agnews.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


def batch(tok, items):
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"]; c = tok(co, add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    L = max(len(s) for s in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


@torch.no_grad()
def sst_acc(model, tok, va):
    model.eval(); correct = 0
    for sent, lab in va:
        sc = {}
        for cl in (0, 1):
            inp, lb, am = batch(tok, [(f"{sent}\nSentiment:", LAB_SST[cl])])
            sc[cl] = float(model(input_ids=inp, attention_mask=am, labels=lb).loss.item())
        correct += int(min(sc, key=sc.get) == lab)
    model.train(); return correct / len(va)


def hill_alpha(x, frac=0.1):
    x = np.sort(x[x > 0])[::-1]; k = max(8, int(frac * len(x))); k = min(k, len(x) - 1)
    xk = x[k]
    if xk <= 0: return None
    xi = float(np.mean(np.log(x[:k] / xk)))
    return 1.0 / xi if xi > 0 else None       # P(X>x)~x^-alpha tail index


def main():
    t0 = time.time(); print("[B] start; loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT))
    tr, va = sst_data(); B_items = agnews_items()
    print(f"[B] loaded ({time.time()-t0:.0f}s)", flush=True)

    # frozen weight-ESD eigvecs (c2 basis) + w_A on the 24 square matrices
    wA = {}; Uesd = {}
    for n, p in model.state_dict().items():
        if SQUARE in n and tuple(p.shape) == (1024, 1024):
            W = p.detach().double()
            _, U = torch.linalg.eigh(W.T @ W / W.shape[0])      # ESD eigvecs (in-space), ascending
            wA[n] = p.detach().float().clone(); Uesd[n] = U.float()
    sst_before = sst_acc(model, tok, va)
    print(f"[B] frozen ESD basis built; SST-2 acc before B = {sst_before:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # fine-tune AG-News from the SST-2 min
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR); g = torch.Generator().manual_seed(0)
    run = []; step = 0
    while step < MAX_STEPS:
        order = torch.randperm(len(B_items), generator=g).tolist()
        for i in range(0, len(order) - BATCH, BATCH):
            inp, lb, am = batch(tok, [B_items[k] for k in order[i:i + BATCH]])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            run.append(float(loss.item())); run = run[-80:]
            if step % 200 == 0:
                print(f"  [B] step {step} train_win {np.mean(run):.4f} ({time.time()-t0:.0f}s)", flush=True)
                if np.mean(run) < TRAINSTOP: break
        if step >= MAX_STEPS or (run and np.mean(run) < TRAINSTOP): break

    sst_after = sst_acc(model, tok, va)
    agnews_train = float(np.mean(run))
    forgetting = sst_before - sst_after
    print(f"[B] AG-News trained (train_win {agnews_train:.3f}); SST-2 acc after B = {sst_after:.3f} -> forgetting {forgetting:+.3f}", flush=True)

    # damage field on frozen ESD shells: damage_i = ||Delta_w @ u_i||^2
    sd = model.state_dict(); damage = []
    for n in wA:
        dW = (sd[n].detach().float() - wA[n])                  # (out,in)
        proj = dW @ Uesd[n]                                    # (out, n_esd) : change along each ESD in-direction
        d_i = (proj * proj).sum(0).double().cpu().numpy()      # energy per ESD component
        damage.append(d_i)
    damage = np.concatenate(damage); damage = damage[damage > 0]
    alpha = hill_alpha(damage)
    branch = ("alpha>=5: moderate-p c2 OK" if (alpha and alpha >= 5) else
              "alpha~3-5: fractional low-p {0.5..2.5}+c3-check" if (alpha and alpha >= 3) else
              "alpha<3: moment-c2 INFEASIBLE -> rank/quantile estimator or concede")

    print(f"\n[B] === c2 GATE B: damage tail-index alpha ===", flush=True)
    print(f"  task-A forgetting: SST-2 acc {sst_before:.3f} -> {sst_after:.3f} (drop {forgetting:+.3f}); AG-News train_win {agnews_train:.3f}", flush=True)
    print(f"  damage components: {len(damage)}, median {np.median(damage):.3e}, max {damage.max():.3e}", flush=True)
    print(f"  Hill tail-index ALPHA = {alpha:.2f}" if alpha else "  ALPHA undefined", flush=True)
    print(f"  -> DECISION BRANCH: {branch}", flush=True)
    json.dump({"task_B": "AG-News", "sst_acc_before": sst_before, "sst_acc_after": sst_after,
               "forgetting": forgetting, "agnews_train_win": agnews_train, "steps": step,
               "n_damage_components": len(damage), "damage_median": float(np.median(damage)),
               "damage_max": float(damage.max()), "alpha": alpha, "branch": branch,
               "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[B] saved c2_gateB.json  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
