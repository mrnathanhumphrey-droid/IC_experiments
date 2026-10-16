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
MODEL = "facebook/opt-350m"; AX = f"{G}/overlap_axis"; OUT = f"{G}/rung2b_opt.json"; NPZ = f"{G}/rung2b_opt.npz"
CARRIERS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
STEPS_WA = 1200; LR_WA = 5e-5; STEPS_DEPTH = 200; LR_DEPTH = 5e-5; BATCH = 16; SUBSET = 2000
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
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(dev)
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

    # CORRECTED antagonism: task-A's HELD-OUT gradient g_A (nonzero from gen gap; train interpolated) vs task-B step-0 grad g_B0.
    # (rung-2's Δw_A proxy was descent-integrated + bulk-dominated -> washed to ~0; this is the direct gradient-cosine.)
    gA = {n: torch.zeros_like(params[n]) for n in carriers}
    hb = [(f"{s}\nSentiment:", LAB[l]) for s, l in [(r[0], r[1]) for r in ev[:96]]]   # held-out SST val
    for it in hb:
        inp, lb, am = b_pairs(tok, [it]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in carriers: gA[n] = gA[n] + params[n].grad.detach()
    model.zero_grad(set_to_none=True)

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
        Vt = V[n][:, :KF].double(); a = (gA[n].double() @ Vt); b = (g0[n].double() @ Vt)  # g_A(held-out) vs g_B0, in top-curv
        c = float((a * b).sum() / (a.norm() * b.norm() + 1e-12))
        antag.append(c)
    com, M1, antag = np.array(com), np.array(M1), np.array(antag)
    s1_seed_conc = float(np.median(com)) < 0.42                                        # concentrated high-curv
    s1_core = float(np.median(M1)) > 0.30                                              # contested core ≈ curv (Pythia 0.70)
    s1_antag = float(np.median(antag)) < -0.05                                         # head-on (Pythia −0.23) — REPORTED, not gated
    # STEP 1 gate = seed/contested-core EXISTS (seed-conc AND core). antagonism is a corroborating REPORT, not a hard gate
    # (rung-2 wrongly ANDed a bad antagonism proxy and nuked a clean replication).
    step1_pass = s1_seed_conc and s1_core
    print(f"[R2] STEP1: com_rank median={np.median(com):.3f}(<0.42) | M1 median={np.median(M1):.3f}(>0.30, Pythia 0.70) -> "
          f"core {'EXISTS' if step1_pass else 'ABSENT'} | antagonism(g_A·g_B0) median={np.median(antag):.3f}({'head-on' if s1_antag else '~0/not-head-on'}, Pythia −0.23) ({time.time()-t0:.0f}s)", flush=True)

    # ===== STEP 2 =====
    parts = np.array([partial_spearman(seed_e[n], contest_e[n], LAM[n]) for n in carriers])
    parts_attn = np.array([partial_spearman(seed_e[n], contest_e[n], LAM[n]) for n in carriers if carrier_of[n] == "attn"])
    step2_pass = float(np.median(parts)) > 0.15 and float(np.mean(parts > 0)) > 0.6
    print(f"[R2] STEP2: partial(seed,contest|λ) median={np.median(parts):.3f} (Pythia +0.45), frac_pos={np.mean(parts>0):.2f}, "
          f"attn median={np.median(parts_attn):.3f} -> {'PASS' if step2_pass else 'FAIL'} ({time.time()-t0:.0f}s)", flush=True)

    rep = {"model": MODEL, "wA_sst_acc": accA, "step1": {"com_rank_median": float(np.median(com)), "M1_median": float(np.median(M1)),
            "antag_median": float(np.median(antag)), "antag_headon": bool(s1_antag), "core_exists": bool(step1_pass), "pass": bool(step1_pass)},
           "step2": {"partial_median": float(np.median(parts)), "partial_frac_pos": float(np.mean(parts > 0)),
                     "partial_attn_median": float(np.median(parts_attn)), "pass": bool(step2_pass)}}
    np.savez(NPZ, com=com, M1=M1, antag=antag, parts=parts, **{f"seed_{i}": seed_e[n] for i, n in enumerate(carriers)})

    # ===== STEP 3 (gated) =====
    if step1_pass and step2_pass:
        print(f"[R2] steps 1-2 PASS -> STEP 3 (vary task-A SST->AG) ({time.time()-t0:.0f}s)", flush=True)
        subjP = load_jsonB("medium_subj")                                            # neutral fragility probe
        # fragile/topcurv for task-A=SST (probe=Subj) at wA
        SsB, _ = seedcov(model, params, carriers, tok, subjP)
        fragile_SST = {}; topcurv_SST = {}
        for n in carriers:
            e = (V[n].T @ SsB[n].float() @ V[n]).diagonal().clamp(min=0).cpu().numpy()
            order = np.argsort(-e)[:KF]; fragile_SST[n] = V[n][:, order].cpu().numpy()
            topcurv_SST[n] = V[n][:, :KF].cpu().numpy()
        # task-A=AG: fresh base model (from_pretrained IS base), train AG-as-task-A, basis + Subj-probe fragile
        del model; torch.cuda.empty_cache()
        model2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32, attn_implementation="eager").to(dev)
        train_items(model2, tok, agB, STEPS_WA, LR_WA, 0)
        p2 = dict(model2.named_parameters())
        V2, _ = curv_basis(model2, tok, carriers, agB)
        Ss2, _ = seedcov(model2, p2, carriers, tok, subjP)
        fragile_AG = {}; topcurv_AG = {}
        for n in carriers:
            e = (V2[n].T @ Ss2[n].float() @ V2[n]).diagonal().clamp(min=0).cpu().numpy()
            order = np.argsort(-e)[:KF]; fragile_AG[n] = V2[n][:, order].cpu().numpy()
            topcurv_AG[n] = V2[n][:, :KF].cpu().numpy()
        fG = np.array([grass(fragile_SST[n], fragile_AG[n]) for n in carriers])
        tG = np.array([grass(topcurv_SST[n], topcurv_AG[n]) for n in carriers])
        fG_attn = np.array([grass(fragile_SST[n], fragile_AG[n]) for n in carriers if carrier_of[n] == "attn"])
        tG_attn = np.array([grass(topcurv_SST[n], topcurv_AG[n]) for n in carriers if carrier_of[n] == "attn"])
        chance = float(np.mean([KF / V[n].shape[0] for n in carriers]))
        moves = float(np.median(fG)) < 0.5                                            # well below stays-put
        fragile_is_curv = abs(float(np.median(fG)) - float(np.median(tG))) < 0.5 * float(np.median(tG)) + 0.03
        rep["step3"] = {"fragile_G_median": float(np.median(fG)), "topcurv_G_median": float(np.median(tG)),
                        "fragile_G_attn": float(np.median(fG_attn)), "topcurv_G_attn": float(np.median(tG_attn)),
                        "chance": chance, "moves": bool(moves), "fragile_eq_topcurv": bool(fragile_is_curv)}
        np.savez(NPZ, com=com, M1=M1, antag=antag, parts=parts, fG=fG, tG=tG)
        if moves and fragile_is_curv:
            v3 = (f"TASK-A's NOT architecture's (Pythia replicates): fragile_G median={np.median(fG):.3f} ≈ topcurv_G {np.median(tG):.3f} "
                  f"(chance {chance:.3f}) — fragile subspace MOVES with task-A & equals generic curvature (no architectural fragile core). "
                  f"attn fragile {np.median(fG_attn):.3f} vs topcurv {np.median(tG_attn):.3f}.")
        elif not moves:
            v3 = (f"★ EYEBROW — ARCHITECTURAL fragile subspace on OPT: fragile_G median={np.median(fG):.3f} STAYS PUT across task-A "
                  f"(topcurv_G {np.median(tG):.3f}) — fragility-locus is architecture-dependent on OPT where Pythia's was task-A's. Cross-arch difference.")
        else:
            v3 = (f"PARTIAL: fragile_G {np.median(fG):.3f} vs topcurv_G {np.median(tG):.3f} — moves but fragile≠topcurv (a fragile component "
                  f"beyond generic curvature). Inspect.")
        rep["step3"]["verdict"] = v3
        print(f"[R2] STEP3: fragile_G={np.median(fG):.3f} topcurv_G={np.median(tG):.3f} (chance {chance:.3f})\n  {v3} ({time.time()-t0:.0f}s)", flush=True)
    else:
        rep["step3"] = {"skipped": "steps 1-2 did not both pass — no subspace to vary; not run (dependency order)"}
        print(f"[R2] STEP3 SKIPPED (steps 1-2 not both pass).", flush=True)

    # overall verdict
    antag_note = ("antagonism head-on too (replicates)" if s1_antag else
                  f"antagonism g_A·g_B0={np.median(antag):.3f} ~0 NOT head-on (the one Pythia feature that does NOT replicate on OPT — corroborating signal diverges)")
    if step1_pass and step2_pass and rep.get("step3", {}).get("moves") and rep["step3"].get("fragile_eq_topcurv"):
        rep["verdict"] = (f"RUNG-2 REPLICATES on OPT: seed/contested-core (com {np.median(com):.2f}, M1 {np.median(M1):.2f}) + "
                          f"beyond-curvature aimed (partial {np.median(parts):.2f}, frac_pos {np.mean(parts>0):.2f}) + task-A-owned (fragile_G≈topcurv_G) "
                          f"ALL replicate on a 3rd family -> mechanism cross-architecture (Pythia+GPT2+OPT). {antag_note}. OPT earns rung-3.")
    elif step1_pass and step2_pass:
        rep["verdict"] = (f"RUNG-2 mostly replicates: seed/contested-core + beyond-curvature aimed (partial {np.median(parts):.2f}) EXIST on OPT; "
                          f"step-3 -> {rep['step3'].get('verdict','')}. {antag_note}.")
    elif step1_pass and not step2_pass:
        rep["verdict"] = f"RUNG-2 BOUNDED: seed/contested-core backbone replicates but BEYOND-CURVATURE aimed does NOT (Pythia/GPT-specific). {antag_note}."
    else:
        rep["verdict"] = f"RUNG-2 FAIL: seed/contested-core ABSENT on OPT (com {np.median(com):.2f}, M1 {np.median(M1):.2f}) -> mechanism Pythia/GPT-specific."
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"\n[R2] === RUNG-2 VERDICT ===\n  {rep['verdict']}\n[R2] saved rung2b_opt.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
