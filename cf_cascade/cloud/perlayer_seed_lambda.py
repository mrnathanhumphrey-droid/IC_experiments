"""PER-LAYER SEED re-run — settle whether there is BEYOND-CURVATURE seed<->contest structure that the dye1
layer-AVERAGE washed out. Re-extracts dye1's step-0 seed but stores it PER LAYER (24 layers x 3 carriers), in the
SAME top-64 Fisher basis deflation used, then pairs each layer's seed with ITS OWN contested core (no cross-layer
rank-smear) and computes the curvature-CONTROLLED partial correlation per layer -> distribution across 24 layers.

WHY per-layer has power the average lacked: (1) pairs seed_n with contest_n (same layer's structure, not a
rank-aligned average of 24 different bases); (2) the decisive stat is the across-LAYER distribution of partial corrs
(n=24 estimates) not one pooled 64-point corr; (3) deflation's M2 off-curvature contested dirs are per-layer -> only
visible at per-layer resolution.

BASIS ALIGNMENT: recompute FA identically to deflation/dye1 (150 SST-A, same checkpoint/fisher()), take PA=top-64
eigvecs (ascending, col 63 = highest curv) EXACTLY as deflation. deflation saved contest+lam in this PA but NOT PA
itself, so we verify alignment by matching our recomputed eigenvalues to the saved lam__ per layer (well-separated
spectrum -> matched eigenvalues => matched eigenvectors; projection is squared so sign is irrelevant).
CORRECTNESS CHECK: layer-mean of our per-layer seed (in rank space) must reproduce dye1's stored aggregated seed.

Seed = task-B (AG/low_agnews) gradient at w_A, accumulated over N_SEED=8 batches of 16 (dye1 convention, seed 0).
SAVE per-layer arrays BEFORE any readout. Pre-committed verdict in the launcher notes / printed at end.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
DEFL = f"{G}/deflation_artifacts.npz"; DYE1 = f"{G}/dye1_series.npz"
OUT = f"{G}/perlayer_seed_overlap.json"; NPZ = f"{G}/perlayer_seed.npz"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
KTOP = 64; CORE = 16; N_FISHER = 150; N_SEED = 8; BATCH = 16; LAB = {0: " negative", 1: " positive"}; dev = "cuda"
torch.manual_seed(0); np.random.seed(0); torch.backends.cudnn.benchmark = True


def sst_trA():
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")["train"]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(0).permutation(len(rows))[:max(N_FISHER, 200)]]


def load_taskB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


def b_sst(tok, items):
    seqs, labs = [], []
    for s, l in items:
        p = tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"]; c = tok(LAB[l], add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    return pack(tok, seqs, labs)


def b_pair(tok, items):
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"]; c = tok(co, add_special_tokens=False)["input_ids"]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    return pack(tok, seqs, labs)


def pack(tok, seqs, labs):
    L = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


def fisher(model, params, names, tok, items):
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    for it in items[:N_FISHER]:
        inp, lb, am = b_sst(tok, [it]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: g = params[n].grad.detach().double(); F[n] += g.T @ g
    return F


def pr(x):
    x = np.asarray(x, float); s1 = x.sum(); s2 = (x * x).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def spear(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def partial_spear(a, b, ctrl):
    ra, rb, rc = [np.argsort(np.argsort(x)).astype(float) for x in (a, b, ctrl)]
    def resid(y, x):
        x1 = np.c_[np.ones_like(x), x]; beta = np.linalg.lstsq(x1, y, rcond=None)[0]; return y - x1 @ beta
    return float(np.corrcoef(resid(ra, rc), resid(rb, rc))[0, 1])


def main():
    t0 = time.time(); print("[PLS] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev); model.load_state_dict(torch.load(CKPT))
    trA = sst_trA(); B = load_taskB("low_agnews"); params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]

    # FA -> PA(top-64) + eigenvalues, per layer; SAME convention as deflation (ascending, col63 highest curv)
    FA = fisher(model, params, names, tok, trA)
    PA = {}; LAM = {}
    for n in names:
        ev, V = torch.linalg.eigh(FA[n]); PA[n] = V[:, -KTOP:].contiguous(); LAM[n] = ev[-KTOP:].cpu().numpy()
    print(f"[PLS] FA + PA done ({time.time()-t0:.0f}s)", flush=True)

    # SEED (step 0): task-B grad at w_A, dye1 convention (8 batches of 16, mean), seed 0
    model.eval(); model.zero_grad(set_to_none=True); g = torch.Generator().manual_seed(0)
    for _ in range(N_SEED):
        inp, lb, am = b_pair(tok, [B[k] for k in torch.randperm(len(B), generator=g)[:BATCH].tolist()])
        (model(input_ids=inp, attention_mask=am, labels=lb).loss / N_SEED).backward()
    seed64 = {}; seed_full = {}
    for n in names:
        gd = params[n].grad.detach().double()                       # (out,in)
        seed64[n] = (gd @ PA[n].double()).pow(2).sum(0).cpu().numpy()        # (64,) PA-coords, j63=hi curv
        ev, Vf = torch.linalg.eigh(FA[n])
        seed_full[n] = (gd @ Vf.double()).pow(2).sum(0).cpu().numpy()        # (in,) full ascending
    model.zero_grad(set_to_none=True)
    print(f"[PLS] seed captured ({time.time()-t0:.0f}s)", flush=True)

    # ---- SAVE camera frame FIRST (per-layer arrays) ----
    arts = {}
    for c in CARRIERS:
        cn = [n for n in names if c in n]
        arts[f"seed64__{c}"] = np.stack([seed64[n] for n in cn])        # (24,64)
        arts[f"mylam64__{c}"] = np.stack([LAM[n] for n in cn])          # (24,64) my eigenvalues
        arts[f"seed_full__{c}"] = np.stack([seed_full[n] for n in cn])  # (24,in)
        arts[f"names__{c}"] = np.array(cn)
    np.savez(NPZ, **arts)
    print(f"[PLS] saved camera frame perlayer_seed.npz ({time.time()-t0:.0f}s)", flush=True)

    # ---- analysis (wrapped; arrays already safe) ----
    report = {}
    try:
        defl = np.load(DEFL); dye = np.load(DYE1)
        rng = np.random.default_rng(0); NPERM = 2000
        # alignment guard: my eigenvalues vs deflation saved lam__
        align = {}
        for c in CARRIERS:
            cn = [n for n in names if c in n]
            dmax = max(float(np.max(np.abs(LAM[n] - defl[f"lam__{n}"]) / (np.abs(defl[f"lam__{n}"]) + 1e-12))) for n in cn)
            align[c] = dmax
        # correctness: layer-mean of my seed64 (reversed to dye-rank) vs dye1 stored seed[:64]
        seedcheck = {}
        for c in CARRIERS:
            cn = [n for n in names if c in n]
            mine_rank = np.mean([seed64[n][::-1] for n in cn], axis=0)   # ->dye-rank (r0=hi curv)
            dye_seed = dye[f"seed__{c}"][:KTOP].astype(float)
            seedcheck[c] = spear(mine_rank, dye_seed)

        for c in CARRIERS:
            cn = [n for n in names if c in n]
            per = {"partial": [], "ovl": [], "raw_spear": [], "dissoc_n": [], "seed_E_dissoc_z": [],
                   "E_core": [], "E_out": []}
            for n in cn:
                s = seed64[n]; contest = defl[f"contest__{n}"].astype(float); lam = defl[f"lam__{n}"].astype(float)
                core16 = set(np.argsort(contest)[-CORE:].tolist())
                stop16 = set(np.argsort(s)[-CORE:].tolist())
                per["ovl"].append(len(stop16 & core16) / CORE)
                per["partial"].append(partial_spear(s, contest, lam))
                per["raw_spear"].append(spear(s, contest))
                # off-curvature dissociation: contested core dirs that are LOW curvature (PA cols 0..31, ascending=low)
                lowcurv = set(range(KTOP // 2))
                dissoc = sorted(core16 & lowcurv); per["dissoc_n"].append(len(dissoc))
                eft = seed_full[n].sum()
                per["E_core"].append(float(s[sorted(core16)].sum() / eft))
                per["E_out"].append(float(1.0 - s.sum() / eft))
                if dissoc:
                    ed = s[dissoc].sum() / eft
                    nd = np.array([s[rng.choice(sorted(lowcurv), len(dissoc), replace=False)].sum() / eft for _ in range(NPERM)])
                    per["seed_E_dissoc_z"].append(float((ed - nd.mean()) / (nd.std() + 1e-12)))
            pa = np.array(per["partial"]); npos = int((pa > 0).sum()); n = len(pa)
            tstat = float(pa.mean() / (pa.std(ddof=1) / np.sqrt(n) + 1e-12))
            report[c] = {
                "align_lam_maxreldiff": align[c], "seedcheck_spear_vs_dye1": seedcheck[c],
                "partial_per_layer": per["partial"], "partial_mean": float(pa.mean()), "partial_std": float(pa.std(ddof=1)),
                "partial_tstat": tstat, "partial_n_pos": npos, "partial_n": n,
                "ovl_mean": float(np.mean(per["ovl"])), "raw_spear_mean": float(np.mean(per["raw_spear"])),
                "dissoc_total": int(np.sum(per["dissoc_n"])), "dissoc_layers": int((np.array(per["dissoc_n"]) > 0).sum()),
                "seed_E_dissoc_z_mean": float(np.mean(per["seed_E_dissoc_z"])) if per["seed_E_dissoc_z"] else float("nan"),
                "E_core_mean": float(np.mean(per["E_core"])), "E_out_mean": float(np.mean(per["E_out"]))}
            r = report[c]
            print(f"  [{c:26s}] align={r['align_lam_maxreldiff']:.1e} seedchk={r['seedcheck_spear_vs_dye1']:+.2f} | "
                  f"partial mean={r['partial_mean']:+.2f}±{r['partial_std']:.2f} t={r['partial_tstat']:+.1f} pos={r['partial_n_pos']}/{r['partial_n']} | "
                  f"ovl={r['ovl_mean']:.2f} dissoc={r['dissoc_total']}({r['dissoc_layers']}L) Ediss_z={r['seed_E_dissoc_z_mean']:+.1f} Ecore={r['E_core_mean']:.2f}", flush=True)

        allpart = np.array([p for c in CARRIERS for p in report[c]["partial_per_layer"]])
        gmean = float(allpart.mean()); gt = float(allpart.mean() / (allpart.std(ddof=1) / np.sqrt(len(allpart)) + 1e-12))
        gpos = int((allpart > 0).sum()); gdiss = int(np.sum([report[c]["dissoc_total"] for c in CARRIERS]))
        if gt > 3 and gpos > 0.75 * len(allpart):
            verdict = (f"BEYOND-CURVATURE: per-layer partial mean={gmean:+.2f} t={gt:+.1f} ({gpos}/{len(allpart)} layers positive) "
                       f"-> seed tracks contest structure beyond curvature; layer-average HID it. Deflation doesn't fully absorb; "
                       f"re-opens novel-mechanism reading. dissoc dirs total={gdiss}.")
        elif gt < 2:
            verdict = (f"CURVATURE-MEDIATED FIRM: per-layer partial mean={gmean:+.2f} t={gt:+.1f} ({gpos}/{len(allpart)} pos) ~ null at "
                       f"powered resolution -> the z+1.4 was REAL not an averaging artifact. Seed=core only through shared high-curvature; "
                       f"no beyond-curvature structure. Deflation-corroboration STANDS. dissoc total={gdiss}.")
        else:
            verdict = (f"MIXED/LAYER-LOCALIZED: per-layer partial mean={gmean:+.2f} t={gt:+.1f} ({gpos}/{len(allpart)} pos) -> "
                       f"structure carried by a subset of layers; see per-layer arrays for which. dissoc total={gdiss}.")
        report["grand"] = {"partial_mean": gmean, "partial_tstat": gt, "partial_n_pos": gpos, "partial_n": len(allpart),
                           "dissoc_total": gdiss, "verdict": verdict}
        print(f"\n[PLS] GRAND partial mean={gmean:+.2f} t={gt:+.1f} pos={gpos}/{len(allpart)} dissoc={gdiss}\n[PLS] VERDICT: {verdict}", flush=True)
    except Exception as e:
        import traceback; report["error"] = traceback.format_exc(); print(f"[PLS] analysis error (camera frame saved): {e}", flush=True)
    json.dump({"report": report, "ktop": KTOP, "core": CORE, "wall_s": time.time() - t0}, open(OUT, "w"), indent=2)
    print(f"[PLS] saved perlayer_seed_overlap.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
