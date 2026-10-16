"""Basis-consistency check — is the DISTRIBUTED damage real-in-curvature, or an ESD-basis artifact?

The concentration check found the damage field DISTRIBUTED in the weight-ESD basis (PR~6857,
top-20 mass 3%, damage-vs-ESD-eigval rho=+0.11). But that's the c2 lane's basis (WtW SVD). The
force lane's localization claim (F2 collinear) was in the K-FAC CURVATURE basis (FK2: the two
bases differ). The gauge trap: damage can be DISTRIBUTED-in-ESD yet CONCENTRATED-in-curvature if
the bases don't align -> then the within-shell-ESD surrogate would falsely report 'cascade' on
what is really curvature-localization. rho=0.11 (damage barely tracks ESD eigenvalues) is the
warning sign the ESD basis may be the wrong basis to shell in.

So before the surrogate build: measure damage concentration in the K-FAC binding (curvature)
basis too, with the SAME reduction as ESD (project input axis onto U_A, L2 over output ->
24576 comps, directly comparable to ESD PR=6857). Also: full 2D binding-basis concentration,
raw-weight-element baseline (diffuse-noise reference), and the ESD-vs-KFAC basis OVERLAP
(top-k subspace) -- if overlap~1 the bases coincide (distinction moot); if ~0 the crux is real.

The saved damage_field_agnews.npy is ESD-PROJECTED magnitudes (can't re-rotate), and no K-FAC
basis was persisted -> regenerate raw Delta_w (re-run AG-News FT from A-min; w_A = subsetmin ckpt)
+ collect K-FAC binding basis at the A-min on SST-2. Persist raw Delta_w + K-FAC eigvecs THIS run
so the surrogate build is CPU-only afterward.

PRE-REGISTERED VERDICT (cuts mirror concentration_check, applied in K-FAC-input basis):
  CONCENTRATED-IN-CURVATURE (-> ESD-distrib is basis artifact, run surrogate in curvature basis):
      top-20 mass > 0.50 OR PR < 50  (while ESD was distributed)
  DISTRIBUTED-IN-BOTH (-> surrogate valid in ESD; F2 = early-local->late-distrib):
      top-20 mass < 0.20 AND PR > 500
  INTERMEDIATE -> report, decide surrogate basis with user/Wilson.
"""
import os, json, time
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import numpy as np, torch

MODEL = "EleutherAI/pythia-410m"; CKPT = "D:/IC_experiments/cf_cascade/gateA/pythia410m_sst2_subsetmin.pt"
AX = "D:/IC_experiments/cf_cascade/gateA/overlap_axis"
OUT = "D:/IC_experiments/cf_cascade/gateA/basis_consistency.json"
DW_NPZ = "D:/IC_experiments/cf_cascade/gateA/deltaw_agnews.npz"      # raw Delta_w per matrix
KFAC_NPZ = "D:/IC_experiments/cf_cascade/gateA/kfac_binding_basis.npz"  # U_A,U_G,eigA,eigG per matrix
BATCH = 16; LR = 5e-5; MAX_STEPS = 3000; TRAINSTOP = 0.02; SUBSET = 2000; VAL_N = 400
N_KFAC = 16; LAB_SST = {0: " negative", 1: " positive"}; SQUARE = "attention.dense"; dev = "cuda"
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


def conc(damage):
    """concentration metrics on a positive 1D damage vector."""
    d = damage[damage > 0]; N = len(d); tot = d.sum()
    srt = np.sort(d)[::-1]; csum = np.cumsum(srt)
    def tk(k): return float(csum[min(k, N) - 1] / tot)
    PR = float((tot ** 2) / np.sum(d ** 2))
    gs = np.sort(d); gini = float((2 * np.sum(np.arange(1, N + 1) * gs) / (N * gs.sum())) - (N + 1) / N)
    return {"N": int(N), "PR": PR, "PR_over_N": PR / N, "gini": gini,
            "top20": tk(20), "top100": tk(100), "top1pct": tk(max(1, int(0.01 * N)))}


def main():
    t0 = time.time(); print("[BC] start; loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT))
    tr, va = sst_data(); B_items = agnews_items()
    sst_items = [(f"{s}\nSentiment:", LAB_SST[l]) for s, l in tr]
    print(f"[BC] loaded ({time.time()-t0:.0f}s)", flush=True)

    names = [n for n, p in model.state_dict().items() if SQUARE in n and tuple(p.shape) == (1024, 1024)]
    # w_A + frozen ESD basis (input-axis eigvecs of WtW)
    wA = {}; Uesd = {}; Eesd = {}
    sdA = model.state_dict()
    for n in names:
        W = sdA[n].detach().double()
        evals, U = torch.linalg.eigh(W.T @ W / W.shape[0])
        wA[n] = sdA[n].detach().float().clone(); Uesd[n] = U.float(); Eesd[n] = evals.float()

    # ---- K-FAC binding basis at the A-min on SST-2 (curvature basis) ----
    model.eval()
    dense = {n: m for n, m in model.named_modules() if n.endswith("attention.dense")}
    A_acc = {n: torch.zeros(1024, 1024, device=dev) for n in dense}
    G_acc = {n: torch.zeros(1024, 1024, device=dev) for n in dense}; cnt = {n: 0 for n in dense}; cur = {}
    H = []
    for n, m in dense.items():
        def fwd(mod, i, o, nn=n): cur[nn] = i[0].detach()
        def bwd(mod, gi, go, nn=n):
            x = cur[nn].reshape(-1, 1024); g = go[0].detach().reshape(-1, 1024); k = g.abs().sum(1) > 0
            if k.any(): A_acc[nn] += x[k].T @ x[k]; G_acc[nn] += g[k].T @ g[k]; cnt[nn] += int(k.sum())
        H += [m.register_forward_hook(fwd), m.register_full_backward_hook(bwd)]
    gA = torch.Generator().manual_seed(7)
    for _ in range(N_KFAC):
        inp, lb, am = batch(tok, [sst_items[k] for k in torch.randperm(len(sst_items), generator=gA)[:BATCH].tolist()])
        model.zero_grad(set_to_none=True); model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
    for h in H: h.remove()
    UA = {}; UG = {}; eA = {}; eG = {}
    for n in dense:
        A = (A_acc[n] / max(cnt[n], 1)).double(); G = (G_acc[n] / max(cnt[n], 1)).double()
        evA, vA = torch.linalg.eigh(A); evG, vG = torch.linalg.eigh(G)
        pn = n + ".weight"                                  # match state_dict param names
        UA[pn] = vA.float(); UG[pn] = vG.float(); eA[pn] = evA.clamp(min=0).float(); eG[pn] = evG.clamp(min=0).float()
    print(f"[BC] K-FAC binding basis built on {len(dense)} matrices ({time.time()-t0:.0f}s)", flush=True)
    np.savez(KFAC_NPZ, **{f"{n}__UA": UA[n].cpu().numpy() for n in names},
             **{f"{n}__UG": UG[n].cpu().numpy() for n in names},
             **{f"{n}__eA": eA[n].cpu().numpy() for n in names},
             **{f"{n}__eG": eG[n].cpu().numpy() for n in names})

    # ---- fine-tune AG-News from the A-min (reproduce forget) ----
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR); g = torch.Generator().manual_seed(0)
    sst_before = sst_acc(model, tok, va); run = []; step = 0
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
                print(f"  [BC] step {step} train_win {np.mean(run):.4f} ({time.time()-t0:.0f}s)", flush=True)
                if np.mean(run) < TRAINSTOP: break
        if step >= MAX_STEPS or (run and np.mean(run) < TRAINSTOP): break
    sst_after = sst_acc(model, tok, va); forgetting = sst_before - sst_after
    print(f"[BC] AG-News trained; SST-2 {sst_before:.3f}->{sst_after:.3f} forget {forgetting:+.3f} ({time.time()-t0:.0f}s)", flush=True)

    # ---- Delta_w per matrix; concentration in 4 representations ----
    sd = model.state_dict(); dW = {n: (sd[n].detach().float() - wA[n]) for n in names}
    np.savez(DW_NPZ, **{n: dW[n].cpu().numpy() for n in names})

    esd_dmg = []; kfacA_dmg = []; bind2d_dmg = []; raw_dmg = []
    ov50 = []; ov256 = []
    for n in names:
        d = dW[n].double()                                  # (out,in)
        esd_dmg.append(((d @ Uesd[n].double()) ** 2).sum(0).cpu().numpy())          # input axis on ESD eigvecs
        kfacA_dmg.append(((d @ UA[n].double()) ** 2).sum(0).cpu().numpy())          # input axis on K-FAC A eigvecs
        C = (UG[n].double().T @ d @ UA[n].double())                                 # full binding 2D
        bind2d_dmg.append((C ** 2).flatten().cpu().numpy())
        raw_dmg.append((d ** 2).flatten().cpu().numpy())                            # raw element basis
        # subspace overlap of TOP-k ESD-input vs K-FAC-A-input eigvecs (eigh ascending -> top = last k)
        for k, store in ((50, ov50), (256, ov256)):
            Pe = Uesd[n][:, -k:].double(); Pa = UA[n][:, -k:].double()
            store.append(float((Pe.T @ Pa).pow(2).sum().item() / k))               # in [0,1]

    rep = {"esd_input": conc(np.concatenate(esd_dmg)),
           "kfac_curvature_input": conc(np.concatenate(kfacA_dmg)),
           "kfac_binding_2d": conc(np.concatenate([x[np.random.default_rng(0).integers(0, len(x), 40000)] for x in bind2d_dmg])),
           "raw_element": conc(np.concatenate([x[np.random.default_rng(1).integers(0, len(x), 40000)] for x in raw_dmg]))}
    overlap = {"top50_esd_vs_kfac": float(np.mean(ov50)), "top256_esd_vs_kfac": float(np.mean(ov256))}

    kc = rep["kfac_curvature_input"]
    if kc["top20"] > 0.50 or kc["PR"] < 50:
        verdict = "CONCENTRATED-IN-CURVATURE -> ESD-distributed is a BASIS ARTIFACT; force-lane localization confirmed; run surrogate in CURVATURE basis (or bank localization-in-curvature)"
    elif kc["top20"] < 0.20 and kc["PR"] > 500:
        verdict = "DISTRIBUTED-IN-BOTH -> distribution basis-robust; surrogate VALID in ESD basis; F2 reconciliation = early-localized -> late-distributed (real dynamical finding)"
    else:
        verdict = "INTERMEDIATE -> report, decide surrogate basis with user/Wilson"

    print(f"\n[BC] === BASIS-CONSISTENCY (forget {forgetting:+.3f}) ===", flush=True)
    for k, v in rep.items():
        print(f"  {k:22s} PR={v['PR']:8.1f} (PR/N={v['PR_over_N']:.3f}) top20={v['top20']:.3f} top100={v['top100']:.3f} gini={v['gini']:.3f}", flush=True)
    print(f"  basis overlap ESD-vs-KFAC: top50={overlap['top50_esd_vs_kfac']:.3f}  top256={overlap['top256_esd_vs_kfac']:.3f}  (1=coincide, 0=orthogonal)", flush=True)
    print(f"  -> VERDICT: {verdict}", flush=True)
    json.dump({"forgetting": forgetting, "representations": rep, "basis_overlap": overlap,
               "verdict": verdict, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[BC] saved basis_consistency.json + deltaw_agnews.npz + kfac_binding_basis.npz  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
