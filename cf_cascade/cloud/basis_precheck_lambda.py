"""BASIS PRE-CHECK — the single gate on the basis-reframe Probes 1+2 (cross-task basis exists?).

The reframe: the beyond-curvature ~30% residual lives in a TASK-PAIR basis the curvature view is
blind to. Before either probe trusts such a basis, prove it exists as a usable object. Candidate
C = top-KPROJ eigvecs of task-B's Fisher F_B at w_A (task-pair geometry, NOT task-A projected).
Three gates, per substrate (OPT load-bearing-map / Pythia map-inert). ALL must pass for the probes.

  GATE 1 DISTINCT   — orthogonal-energy fraction of C outside V_A (=task-A curvature top-KPROJ).
                      C subset V_A (Grassmann ~1) => "task-pair basis" IS curvature => reframe pre-fails.
  GATE 2 CONDITIONED— F_B spectral concentration (effective rank, top-KPROJ energy frac) is a real
                      subspace not noise; + cross-term NON-TRIVIALITY (antagonism cosine of mean
                      grads in V_A). Expect OPT~0 / Pythia<0 (rung-2) -> scopes where contest exists.
  GATE 3 STABLE     — split task-B samples in half, build C on each, Grassmann(C1,C2) >> chance.

Verdict per substrate: all-3 pass => probes may run there. Gate1 fail => reframe premise dead.
Gate2 cross-term ~0 on OPT => basis degenerate where the map was (load-bearing) => probes Pythia-only,
itself a scoping finding. Read-only-ish (gradients + linalg at w_A; loads saved w_A, no training).
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"; AX = f"{G}/overlap_axis"
OUT = f"{G}/basis_precheck.json"
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
KPROJ = 128; N_FISHER = 240; CHANCE = None  # chance = KPROJ/in_dim, per carrier

SUBSTRATES = [
    {"tag": "OPT", "model": "facebook/opt-350m", "carrier_keys": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
     "ckpt": f"{G}/opt350m_sst2_wA.pt", "train_wa": False},
    {"tag": "PYTHIA", "model": "EleutherAI/pythia-410m", "carrier_keys": ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"],
     "ckpt": f"{G}/pythia410m_sst2_subsetmin.pt", "train_wa": False},
]


def load_jsonB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


def sst_rows(split, k, seed=0):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")[split]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))[:k]]


def b_pairs(tok, items):
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"][:256]; c = tok(co, add_special_tokens=False)["input_ids"][:256]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    Lm = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), Lm), pad, np.int64); lb = np.full((len(seqs), Lm), -100, np.int64); am = np.zeros((len(seqs), Lm), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


def fisher(model, tok, params, carriers, items, split_half=False):
    """Sum g g^T per carrier over items. If split_half, also return F on even/odd halves + mean grad."""
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in carriers}
    Fh = ({n: torch.zeros_like(F[n]) for n in carriers}, {n: torch.zeros_like(F[n]) for n in carriers}) if split_half else None
    gmean = {n: torch.zeros(params[n].shape, device=dev, dtype=torch.float64) for n in carriers}
    for idx, it in enumerate(items):
        inp, lb, am = b_pairs(tok, [it]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in carriers:
            g = params[n].grad.detach().double(); F[n] += g.T @ g; gmean[n] += g
            if split_half: Fh[idx % 2][n] += g.T @ g
    for n in carriers: gmean[n] /= max(len(items), 1)
    return F, Fh, gmean


def topk(M, k):
    _, vv = torch.linalg.eigh(M); return vv[:, -k:].float().contiguous()


def grass(A, B):
    k = min(A.shape[1], B.shape[1]); return float((A.T @ B).pow(2).sum() / max(k, 1))


def eff_rank(M):
    ev = torch.linalg.eigvalsh(M).clamp(min=0).double(); ev = ev / (ev.sum() + 1e-30)
    return float(torch.exp(-(ev * (ev + 1e-30).log()).sum()))   # participation/entropy effective rank


def run(cfg, t0):
    tag = cfg["tag"]; print(f"\n[{tag}] loading {cfg['model']}...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["model"]); tok.pad_token = tok.eos_token if tok.pad_token is None else tok.pad_token
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(cfg["ckpt"]))
    params = dict(model.named_parameters())
    carriers = [n for n in params if any(c in n for c in cfg["carrier_keys"]) and n.endswith("weight") and params[n].ndim == 2]
    print(f"[{tag}] {len(carriers)} carriers", flush=True)

    a_items = [(f"{s}\nSentiment:", LAB[l]) for s, l in sst_rows("train", N_FISHER, 0)]
    b_items = load_jsonB("medium_subj")[:N_FISHER]
    FA, _, gA = fisher(model, tok, params, carriers, a_items)
    FB, FBh, gB = fisher(model, tok, params, carriers, b_items, split_half=True)
    print(f"[{tag}] Fisher A+B done ({time.time()-t0:.0f}s)", flush=True)

    KSWEEP = [4, 8, 16, 32, 64, 128]; KWORK = 16
    # accumulators: per-K orthofrac (distinct) and split-half stability; plus per-carrier effrank/energy/antagonism
    od = {K: [] for K in KSWEEP}; st = {K: [] for K in KSWEEP}
    g2_eff, g2_top128, g2_cos, effrank_abs, dims = [], [], [], [], []
    for n in carriers:
        d = params[n].shape[1]; dims.append(d)
        kmax = min(KPROJ, d)
        VA = topk(FA[n], kmax); C = topk(FB[n], kmax); C1 = topk(FBh[0][n], kmax); C2 = topk(FBh[1][n], kmax)
        for K in KSWEEP:
            if K > kmax: continue
            od[K].append(1.0 - grass(C[:, -K:], VA[:, -K:]))      # distinct: energy of top-K C outside top-K V_A
            st[K].append(grass(C1[:, -K:], C2[:, -K:]))           # stable: split-half top-K
        er = eff_rank(FB[n]); effrank_abs.append(er); g2_eff.append(er / d)
        evB = torch.linalg.eigvalsh(FB[n]).clamp(min=0); g2_top128.append(float(evB[-kmax:].sum() / (evB.sum() + 1e-30)))
        aV = (gA[n] @ VA.double()); bV = (gB[n] @ VA.double()); g2_cos.append(float((aV * bV).sum() / (aV.norm() * bV.norm() + 1e-12)))

    def md(x): return float(np.median(x)) if len(x) else float("nan")
    chanceK = {K: float(np.mean([K / d for d in dims])) for K in KSWEEP}
    sweep = {K: {"distinct_orthofrac": md(od[K]), "stable_grassmann": md(st[K]), "chance": chanceK[K]} for K in KSWEEP}
    G2eff = md(g2_eff); G2top = md(g2_top128); G2cos = md(g2_cos); ER = md(effrank_abs)
    # gates at the WORKING dimension K=16 (>= the ~2-11 effrank, below the noise-padded 128)
    g1 = sweep[KWORK]["distinct_orthofrac"]; g3 = sweep[KWORK]["stable_grassmann"]; ck = chanceK[KWORK]
    gate1 = g1 > 0.40                                  # top-16 C >40% outside task-A curvature
    gate2 = (G2top > 0.50) and (G2eff < 0.50)          # F_B concentrated into a real (low-rank) subspace, NOT flat noise
    gate3 = g3 > 2 * ck                                # top-16 C reproducible across sample halves
    crossterm_trivial = abs(G2cos) < 0.05
    rep = {"tag": tag, "effrank_FB_abs": ER, "sweep": sweep, "KWORK": KWORK,
           "gate1_distinct_orthofrac_K16": g1, "gate2_effrank_over_d": G2eff, "gate2_topk_energy_frac": G2top,
           "gate2_antagonism_cos": G2cos, "gate3_splithalf_grassmann_K16": g3,
           "gates": {"distinct": bool(gate1), "conditioned": bool(gate2), "stable": bool(gate3),
                     "crossterm_trivial": bool(crossterm_trivial), "ALL_PASS": bool(gate1 and gate2 and gate3)}}
    print(f"[{tag}] F_B effrank≈{ER:.1f} (effrank/d {G2eff:.3f}; subspace is LOW-RANK -> working K={KWORK})", flush=True)
    print(f"[{tag}] distinct ortho-frac by K: " + " ".join(f"K{K}={sweep[K]['distinct_orthofrac']:.2f}" for K in KSWEEP if not np.isnan(sweep[K]['distinct_orthofrac'])), flush=True)
    print(f"[{tag}] stable grassmann by K:   " + " ".join(f"K{K}={sweep[K]['stable_grassmann']:.2f}(ch{chanceK[K]:.2f})" for K in KSWEEP if not np.isnan(sweep[K]['stable_grassmann'])), flush=True)
    print(f"[{tag}] GATE1 distinct@K16={g1:.3f}(>0.40?{gate1}) | GATE2 cond top128E={G2top:.3f} effr/d={G2eff:.3f}(?{gate2}) antag={G2cos:+.3f}(triv?{crossterm_trivial}) | GATE3 stable@K16={g3:.3f}(>{2*ck:.3f}?{gate3})", flush=True)
    print(f"[{tag}] === ALL PASS @ K16: {gate1 and gate2 and gate3} ===", flush=True)
    return rep


def main():
    t0 = time.time(); print("[BPC] BASIS PRE-CHECK (cross-task basis existence gate) on OPT + Pythia", flush=True)
    reps = {}
    for cfg in SUBSTRATES:
        try:
            reps[cfg["tag"]] = run(cfg, t0)
        except Exception as e:
            reps[cfg["tag"]] = {"error": traceback.format_exc()}; print(f"[{cfg['tag']}] FAILED: {e}", flush=True)
    # joint verdict
    try:
        opt = reps.get("OPT", {}).get("gates", {}); pyt = reps.get("PYTHIA", {}).get("gates", {})
        if opt.get("ALL_PASS") and pyt.get("ALL_PASS"):
            v = "BASIS EXISTS both substrates -> green-light Probe 1 then Probe 2."
        elif not opt.get("distinct", True) and not pyt.get("distinct", True):
            v = "GATE1 FAIL both -> cross-task basis IS curvature -> reframe premise dead, probes void."
        elif pyt.get("ALL_PASS") and not opt.get("ALL_PASS"):
            v = ("Basis exists on Pythia, degenerate on OPT (likely Gate2 antagonism~0) -> probes Pythia-only; "
                 "'no cross-task basis where the map was load-bearing' is itself the scoping finding.")
        else:
            v = "Mixed -> read per-substrate gates; run probes only where ALL_PASS."
        reps["verdict"] = v; print(f"\n[BPC] === JOINT VERDICT ===\n  {v}", flush=True)
    except Exception as e:
        reps["verdict_error"] = str(e)
    json.dump(reps, open(OUT, "w"), indent=2)
    print(f"[BPC] saved basis_precheck.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
