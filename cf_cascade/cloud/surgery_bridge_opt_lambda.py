"""RUNG-2 (OPT-350m) — fragile-subspace characterization: does seed/contested-core/curvature structure exist & is it task-A's?
The OPT analogue of the Pythia mechanism, in DEPENDENCY ORDER. Step 3 runs ONLY if steps 1-2 confirm a subspace exists.

STEP 1 — seed/contested core exists (task-A=SST w_A, task-B=AG = Pythia anchor):
  (a) seed concentrates in high-curvature: per-layer com_rank of seed energy over λ-descending dirs (LOW = concentrated, Pythia-like).
  (b) contested core ≈ top-curvature (M1 analogue): per-layer frac of seed energy in top-16 curvature dirs (Pythia deflation M1≈0.70).
  (c) antagonism: per-layer cosine(Δw_A, g_B0) in top-curv dirs, Δw_A = w_A−w_base = task-A's training displacement, g_B0 = task-B
      step-0 grad. NEGATIVE = head-on ("same dials, opposite values"; Pythia −0.23).
STEP 2 — beyond-curvature aimed (the novel component, Pythia +0.45): per-layer partial spearman(seed, contest | λ); contest = realized
  task-B displacement energy after short depth (Δw_B per dir). Effect size vs +0.45, NOT t (n-inflation). Curvature-controlled partial
  is decisive (raw/index are curvature-confounded).
STEP 3 — task-A's not architecture's (Cut-1 analogue, ONLY if 1-2 pass): vary task-A SST→AG; fragile subspace = top-16 dirs by seed
  energy (probe=Subj, neutral); Grassmann across anchors. DECISIVE control: fragile_G ≈ topcurv_G (Pythia ≈0.14, ~86% moves, no
  architectural fragile core beyond generic curvature). fragile STAYS PUT → OPT has an architectural fragile subspace (cross-arch diff).
GUARDS: per-layer NEVER pooled; effect size vs Pythia benchmark not t; reproduce w_A acc (~0.77) consistency; dataset-identity; save before readout.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "facebook/opt-350m"; AX = f"{G}/overlap_axis"; OUT = f"{G}/surgery_bridge_opt.json"; NPZ = f"{G}/surgery_bridge_opt.npz"
CARRIERS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
STEPS_WA = 800; LR_WA = 5e-5; STEPS_DEPTH = 200; LR_DEPTH = 5e-5; BATCH = 16; SUBSET = 2000   # brief surgery re-stabilize retrain
# PRETRAINED OPT-350m anchor (banked) for the bridge comparison
ANCHOR = {"M1": 0.86, "antag": 0.09, "partial": 0.29, "wA_acc_band": [0.77, 0.84]}
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
        V[n] = vv[:, -KB:].flip(dims=[1]).float().contiguous()      # col 0 = highest λ
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


def ranks(a): return np.argsort(np.argsort(np.asarray(a, float))).astype(float)
def spearman(x, y):
    rx, ry = ranks(x), ranks(y); rx -= rx.mean(); ry -= ry.mean()
    d = (rx @ rx) ** 0.5 * (ry @ ry) ** 0.5; return float(rx @ ry / d) if d > 0 else 0.0
def partial_spearman(x, y, z):
    rx, ry, rz = ranks(x), ranks(y), ranks(z); Z = np.column_stack([np.ones_like(rz), rz])
    res = lambda r: r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]
    ex, ey = res(rx), res(ry); d = (ex @ ex) ** 0.5 * (ey @ ey) ** 0.5; return float(ex @ ey / d) if d > 0 else 0.0
def grass(A, B): s = np.linalg.svd(A.T @ B, compute_uv=False); return float((s ** 2).sum() / A.shape[1])


def main():
    t0 = time.time(); print(f"[R2] loading {MODEL} (eager)...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    import torch.nn as nn
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(dev)
    # LAYER-SURGERY: keep every-other decoder layer -> half depth. Pretraining RETAINED in kept layers (the regime
    # from-scratch lacked). Tests whether a shallower PRETRAINED model still hits the band + shows the backbone.
    L0 = len(model.model.decoder.layers); keep = list(range(0, L0, 2))
    model.model.decoder.layers = nn.ModuleList([model.model.decoder.layers[i] for i in keep])
    model.config.num_hidden_layers = len(keep)
    for new_i, lyr in enumerate(model.model.decoder.layers):    # REINDEX layer_idx (else KV-cache IndexError)
        if hasattr(lyr, "self_attn") and hasattr(lyr.self_attn, "layer_idx"): lyr.self_attn.layer_idx = new_i
    print(f"[SURGERY] {MODEL}: depth {L0} -> {len(keep)} (every-other kept); {sum(p.numel() for p in model.parameters())/1e6:.0f}M params", flush=True)
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    carrier_of = {n: ("attn" if any(a in n for a in ["q_proj", "k_proj", "v_proj", "out_proj"]) else "mlp") for n in carriers}
    base_carr = {n: params[n].detach().clone() for n in carriers}                 # w_base (pre task-A)
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]

    sstA = sst_items(SUBSET, 7); agB = load_jsonB("low_agnews")
    print(f"[R2] training task-A=SST -> w_A...", flush=True)
    train_items(model, tok, sstA, STEPS_WA, LR_WA, 0)
    wA = {n: model.state_dict()[n].detach().clone() for n in model.state_dict()}
    # consistency guard: SST acc reproduces rung-1 ~0.77
    ev = sst_rows("validation", 300, 0); model.eval(); cor = 0
    with torch.no_grad():
        for i in range(0, len(ev), 32):
            ch = ev[i:i + 32]; seqs = [tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"][:128] for s, _ in ch]
            L = max(len(x) for x in seqs); inp = np.full((len(seqs), L), tok.pad_token_id, np.int64); pos = np.zeros(len(seqs), np.int64)
            for j, sq in enumerate(seqs): inp[j, :len(sq)] = sq; pos[j] = len(sq) - 1
            lg = model(input_ids=torch.tensor(inp).to(dev)).logits[torch.arange(len(seqs)), torch.tensor(pos).to(dev)]
            pr = (lg[:, posid] > lg[:, negid]).long().cpu().numpy()
            for j, (_, lab) in enumerate(ch): cor += int(pr[j] == lab)
    accA = cor / len(ev); print(f"[R2] w_A SST acc={accA:.3f} (rung-1 was 0.77) ({time.time()-t0:.0f}s)", flush=True)

    # curvature basis + seed (AG) + g0 at w_A
    params = dict(model.named_parameters())
    V, LAM = curv_basis(model, tok, carriers, sstA)
    print(f"[R2] curvature basis done ({time.time()-t0:.0f}s)", flush=True)
    S_B, g0 = seedcov(model, params, carriers, tok, [(t_, c_) for t_, c_ in agB], grab_g0=True)
    seed_e = {n: (V[n].T @ S_B[n].float() @ V[n]).diagonal().clamp(min=0).cpu().numpy() for n in carriers}

    # contest: short AG depth -> Δw_B energy per dir
    model.load_state_dict(wA); train_items(model, tok, agB, STEPS_DEPTH, LR_DEPTH, 0)
    paramsB = dict(model.named_parameters())
    contest_e = {}
    for n in carriers:
        dwB = (paramsB[n].detach() - wA[n].to(dev)).double(); M = dwB.T @ dwB
        contest_e[n] = (V[n].double().T @ M @ V[n].double()).diagonal().clamp(min=0).cpu().numpy()
    model.load_state_dict(wA); params = dict(model.named_parameters())

    # ===== STEP 1 =====
    com, M1, antag = [], [], []
    for n in carriers:
        e = seed_e[n]; r = np.arange(KB)
        com.append(float((r * e).sum() / max(e.sum(), 1e-12) / (KB - 1)))            # 0=all top-λ, 0.5=uniform
        M1.append(float(e[:KF].sum() / max(e.sum(), 1e-12)))                          # frac in top-16 curv
        dwA = (wA[n].to(dev) - base_carr[n]).double(); Vt = V[n][:, :KF].double(); gp = g0[n].double()
        a = (dwA @ Vt); b = (gp @ Vt)                                                 # project both into top-curv (out×16)
        c = float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
        antag.append(c)
    com, M1, antag = np.array(com), np.array(M1), np.array(antag)
    s1_seed_conc = float(np.median(com)) < 0.42                                        # concentrated high-curv
    s1_core = float(np.median(M1)) > 0.30                                              # contested core ≈ curv (Pythia 0.70)
    s1_antag = float(np.median(antag)) < -0.05                                         # head-on (Pythia −0.23)
    step1_pass = s1_seed_conc and s1_core and s1_antag
    print(f"[R2] STEP1: com_rank median={np.median(com):.3f}(<0.42) | M1 median={np.median(M1):.3f}(>0.30, Pythia 0.70) | "
          f"antagonism median={np.median(antag):.3f}(<0, Pythia −0.23) -> {'PASS' if step1_pass else 'FAIL'} ({time.time()-t0:.0f}s)", flush=True)

    # ===== STEP 2 =====
    parts = np.array([partial_spearman(seed_e[n], contest_e[n], LAM[n]) for n in carriers])
    parts_attn = np.array([partial_spearman(seed_e[n], contest_e[n], LAM[n]) for n in carriers if carrier_of[n] == "attn"])
    step2_pass = float(np.median(parts)) > 0.15 and float(np.mean(parts > 0)) > 0.6
    print(f"[R2] STEP2: partial(seed,contest|λ) median={np.median(parts):.3f} (Pythia +0.45), frac_pos={np.mean(parts>0):.2f}, "
          f"attn median={np.median(parts_attn):.3f} -> {'PASS' if step2_pass else 'FAIL'} ({time.time()-t0:.0f}s)", flush=True)

    rep = {"model": MODEL, "wA_sst_acc": accA, "step1": {"com_rank_median": float(np.median(com)), "M1_median": float(np.median(M1)),
            "antag_median": float(np.median(antag)), "pass": bool(step1_pass)},
           "step2": {"partial_median": float(np.median(parts)), "partial_frac_pos": float(np.mean(parts > 0)),
                     "partial_attn_median": float(np.median(parts_attn)), "pass": bool(step2_pass)}}
    np.savez(NPZ, com=com, M1=M1, antag=antag, parts=parts, **{f"seed_{i}": seed_e[n] for i, n in enumerate(carriers)})

    # ===== SURGERY-BRIDGE VERDICT: does a HALF-DEPTH PRETRAINED model keep the band + backbone? =====
    reached_band = ANCHOR["wA_acc_band"][0] <= accA <= ANCHOR["wA_acc_band"][1]
    m1_med = float(np.median(M1)); antag_med = float(np.median(antag)); part_med = float(np.median(parts))
    backbone_present = bool(m1_med > 0.30 and part_med > 0.10)            # contested core + aimed component
    rep["surgery_bridge"] = {"wA_acc": accA, "reached_acc_band": bool(reached_band), "acc_band": ANCHOR["wA_acc_band"],
                     "M1_median": m1_med, "antag_median": antag_med, "partial_median": part_med,
                     "anchor": ANCHOR, "backbone_present": backbone_present, "surgery": "half-depth every-other"}
    if reached_band and backbone_present:
        v = (f"SURGERY-BRIDGE PASS: half-depth pretrained reaches acc band ({accA:.3f}) AND keeps backbone (M1={m1_med:.3f} "
             f"vs anchor {ANCHOR['M1']}, partial={part_med:.3f} vs {ANCHOR['partial']}, antag={antag_med:.3f} vs "
             f"{ANCHOR['antag']}) -> layer-surgery regime VALID; the depth sweep can fire.")
    elif not reached_band:
        v = (f"SURGERY-BRIDGE FAIL (regime): half-depth SST acc={accA:.3f} OUTSIDE band {ANCHOR['wA_acc_band']} -> "
             f"dropping half the layers + brief retrain doesn't re-stabilize task-A; need gentler surgery (drop fewer / "
             f"more retrain) or another depth axis.")
    else:
        v = (f"SURGERY-BRIDGE PARTIAL: acc={accA:.3f} (band {ANCHOR['wA_acc_band']}) but backbone weak (M1={m1_med:.3f}, "
             f"partial={part_med:.3f}) -> shallower model keeps task but not the mechanism; reassess before the sweep.")
    rep["verdict"] = v
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"\n[SURGERY-BRIDGE] === VERDICT ===\n  {v}\n[SURGERY-BRIDGE] wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
