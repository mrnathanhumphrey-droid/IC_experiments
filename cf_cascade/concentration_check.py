"""Concentration check — the cheap fork-decider BEFORE building the tail-robust estimator.

The c2 moment route failed (damage tail-index alpha~2.86 < 3). The fork: build a tail-robust
estimator (low-q MFDFA / rank-quantile) + within-shell surrogate (route a), or concede (route b).
This check decides which fork is even LIVE, by asking: is the heavy-tailed damage field
CONCENTRATED in a handful of components (-> that IS localization, the few-stiff-directions
story the force lane already found; surrogate has no discriminating power -> concede b) or
DISTRIBUTED across many components/scales (-> genuine cascade candidate; surrogate has power
-> build a)?

c2_gateB.py did NOT persist the damage field, so regenerate it (re-run AG-News fine-tune from
the SST-2 genuine-PSD min, same seeds) and THIS TIME dump the full array to .npy. Then compute
PRE-REGISTERED concentration metrics. Same damage def as c2_gateB: damage_i = ||Delta_w @ u_i||^2
projected on the FROZEN task-A weight-ESD eigen-shells, over the 24 square attention.dense.

PRE-REGISTERED VERDICT (given-not-fitted, declared before reading):
  CONCENTRATED (-> localization, concede route b): top-20 mass > 0.50 OR participation_ratio < 50
  DISTRIBUTED  (-> build route a):                 top-20 mass < 0.20 AND participation_ratio > 500
  INTERMEDIATE (-> report, user/Wilson rule):      otherwise
"""
import os, json, time
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import numpy as np, torch

MODEL = "EleutherAI/pythia-410m"; CKPT = "D:/IC_experiments/cf_cascade/gateA/pythia410m_sst2_subsetmin.pt"
AX = "D:/IC_experiments/cf_cascade/gateA/overlap_axis"
OUT = "D:/IC_experiments/cf_cascade/gateA/concentration_check.json"
DAMAGE_NPY = "D:/IC_experiments/cf_cascade/gateA/damage_field_agnews.npy"   # (n_components,) damage
ESDEIG_NPY = "D:/IC_experiments/cf_cascade/gateA/damage_esd_eigval.npy"     # (n_components,) ESD eigval per component
MATIDX_NPY = "D:/IC_experiments/cf_cascade/gateA/damage_matrix_idx.npy"     # (n_components,) which of 24 matrices
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
    return 1.0 / xi if xi > 0 else None


def main():
    t0 = time.time(); print("[C] start; loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT))
    tr, va = sst_data(); B_items = agnews_items()
    print(f"[C] loaded ({time.time()-t0:.0f}s)", flush=True)

    # frozen weight-ESD eigvecs + eigvals (c2 basis) + w_A on the 24 square matrices
    wA = {}; Uesd = {}; Eesd = {}; order_names = []
    for n, p in model.state_dict().items():
        if SQUARE in n and tuple(p.shape) == (1024, 1024):
            W = p.detach().double()
            evals, U = torch.linalg.eigh(W.T @ W / W.shape[0])      # ascending eigvals + eigvecs
            wA[n] = p.detach().float().clone(); Uesd[n] = U.float(); Eesd[n] = evals.float()
            order_names.append(n)
    sst_before = sst_acc(model, tok, va)
    print(f"[C] frozen ESD basis built; SST-2 acc before B = {sst_before:.3f} ({time.time()-t0:.0f}s)", flush=True)

    # fine-tune AG-News from the SST-2 min (same as c2_gateB)
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
                print(f"  [C] step {step} train_win {np.mean(run):.4f} ({time.time()-t0:.0f}s)", flush=True)
                if np.mean(run) < TRAINSTOP: break
        if step >= MAX_STEPS or (run and np.mean(run) < TRAINSTOP): break

    sst_after = sst_acc(model, tok, va)
    forgetting = sst_before - sst_after
    print(f"[C] AG-News trained (train_win {np.mean(run):.3f}); SST-2 {sst_before:.3f}->{sst_after:.3f} forget {forgetting:+.3f}", flush=True)

    # damage field on frozen ESD shells + retain ESD eigval and matrix idx per component
    sd = model.state_dict(); dmg_parts = []; esd_parts = []; mat_parts = []
    for mi, n in enumerate(order_names):
        dW = (sd[n].detach().float() - wA[n])
        proj = dW @ Uesd[n]
        d_i = (proj * proj).sum(0).double().cpu().numpy()
        dmg_parts.append(d_i); esd_parts.append(Eesd[n].double().cpu().numpy()); mat_parts.append(np.full(len(d_i), mi))
    damage = np.concatenate(dmg_parts); esd = np.concatenate(esd_parts); matidx = np.concatenate(mat_parts)
    np.save(DAMAGE_NPY, damage); np.save(ESDEIG_NPY, esd); np.save(MATIDX_NPY, matidx)
    pos = damage > 0; damage_p = damage[pos]; esd_p = esd[pos]; mat_p = matidx[pos]
    N = len(damage_p); tot = damage_p.sum()

    # ---- PRE-REGISTERED concentration metrics ----
    srt = np.sort(damage_p)[::-1]; csum = np.cumsum(srt)
    def topk_mass(k): return float(csum[min(k, N) - 1] / tot)
    ks = [1, 5, 10, 20, 50, 100, 250, int(0.01 * N)]
    topk = {f"top{k}": topk_mass(k) for k in ks}
    PR = float((tot ** 2) / np.sum(damage_p ** 2))             # participation ratio (effective # components)
    PR_frac = PR / N
    # Gini
    gs = np.sort(damage_p); gini = float((2 * np.sum((np.arange(1, N + 1)) * gs) / (N * gs.sum())) - (N + 1) / N)
    # how many of the 24 matrices hold the top-100 components? (one matrix or spread)
    top100_idx = np.argsort(damage_p)[::-1][:100]
    mats_in_top100 = int(len(np.unique(mat_p[top100_idx])))
    # damage-vs-ESD-rank: does damage track the dominant ESD directions (localization-along-spectrum)?
    from scipy.stats import spearmanr
    rho = float(spearmanr(esd_p, damage_p).correlation)
    # mass in the top-ESD-decile vs bottom-ESD-decile (per the ESD eigenvalue, pooled)
    esd_hi = esd_p >= np.quantile(esd_p, 0.90); esd_lo = esd_p <= np.quantile(esd_p, 0.10)
    mass_hi_esd = float(damage_p[esd_hi].sum() / tot); mass_lo_esd = float(damage_p[esd_lo].sum() / tot)
    alpha = hill_alpha(damage_p)

    # ---- PRE-REGISTERED VERDICT ----
    if topk["top20"] > 0.50 or PR < 50:
        verdict = "CONCENTRATED -> heavy tail IS localization (few-stiff-directions); surrogate has no power -> CONCEDE route b"
    elif topk["top20"] < 0.20 and PR > 500:
        verdict = "DISTRIBUTED -> genuine cascade candidate; tail-robust + surrogate has power -> BUILD route a"
    else:
        verdict = "INTERMEDIATE -> report, no auto-verdict (user/Wilson rule)"

    print(f"\n[C] === CONCENTRATION CHECK (damage field, N={N} positive of 24576) ===", flush=True)
    print(f"  forgetting {forgetting:+.3f}  alpha(recompute)={alpha:.2f}  total_mass={tot:.4e}", flush=True)
    print(f"  participation_ratio PR = {PR:.1f}  (PR/N = {PR_frac:.4f})  gini = {gini:.3f}", flush=True)
    print(f"  top-k mass: " + "  ".join(f"{k}={v:.3f}" for k, v in topk.items()), flush=True)
    print(f"  top-100 components span {mats_in_top100}/24 matrices", flush=True)
    print(f"  damage-vs-ESD-eigval Spearman rho = {rho:+.3f}  (mass in top-ESD-decile={mass_hi_esd:.3f}, bottom-decile={mass_lo_esd:.3f})", flush=True)
    print(f"  -> VERDICT: {verdict}", flush=True)
    json.dump({"N_positive": N, "forgetting": forgetting, "alpha": alpha, "total_mass": float(tot),
               "participation_ratio": PR, "PR_over_N": PR_frac, "gini": gini, "topk_mass": topk,
               "top100_spans_n_matrices": mats_in_top100, "damage_esd_spearman": rho,
               "mass_top_esd_decile": mass_hi_esd, "mass_bottom_esd_decile": mass_lo_esd,
               "verdict": verdict, "wall_s": time.time() - t0,
               "damage_npy": DAMAGE_NPY}, open(OUT, "w"), indent=2)
    print(f"[C] saved concentration_check.json + damage_field_agnews.npy  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
