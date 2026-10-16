"""CONTROL TEST — timed+targeted forgetting protection. THE culmination: does protecting the fragile attention
subspace ONLY during the early seeding transient retain task-A as well as full-time GPM? 7-arm timing x targeting
matrix, Wilson-tightened (curvature-matched targeting control + per-window gradient-mass + per-task-B E).

ARMS (per task-B, per seed): protection = project task-B grad off subspace V during a timing window.
  1 naive           target=none      timing=none   -> baseline forgetting (floor) + yields E + mass profile
  2 fragile_full     target=fragile   timing=full   -> GPM upper bound (the known lever = GUARD: must retain)
  3 fragile_early    target=fragile   timing=[0,E)  -> THE HYPOTHESIS
  4 fragile_late     target=fragile   timing=[E,2E) -> TIMING control (matched step count)
  5a randhi_early    target=highcurv  timing=[0,E)  -> TARGETING control, CURVATURE-MATCHED (random 16 of top-64 curv \ fragile)
  5b randany_early   target=anywhere  timing=[0,E)  -> random-anywhere floor
  6 randany_full     target=anywhere  timing=full   -> does ANY full projection help
DECISIVE: 3v2 (as cheap as full-time?) ; 3v4 (timing-specific? + report early/late gradient-MASS) ;
  3v5a (FRAGILE beyond curvature? -- the claim) ; 3v5b ; 2v6.  Verdict needs 3>=2-ish AND 3>5a AND 3>4.
GUARDS: arm1 must forget (ret drops), arm2 must retain (reproduces lever) -- else harness broken, verdict void.
E per task-B from its OWN naive forgetting curve (63% time-constant). Curvature of 5a's draw REPORTED vs fragile (match check).
Fragile subspace = universal seed over {yelp,subj} (DISJOINT from test task-Bs ag,code -> no circularity), top-16 per carrier.
Retention = SST-2 held-out accuracy; cost = task-B final loss. mean+-std over 3 seeds. Save before readout.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
OUT = f"{G}/control_test.json"; NPZ = f"{G}/control_test.npz"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]  # protect all (arm2 reproduces lever)
FRAG_PANEL = ["yelp", "subj"]; TESTBS = ["ag", "code"]; SEEDS = [0, 1, 2]
JSONK = {"ag": "low_agnews", "yelp": "high_yelp", "subj": "medium_subj"}
KTOP = 64; FRAG = 16; N_FISHER = 150; N_SEED = 8; STEPS = 300; LR = 5e-5; BATCH = 16
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
torch.manual_seed(0); np.random.seed(0)
ARMS = [("naive", "none", "none"), ("fragile_full", "fragile", "full"), ("fragile_early", "fragile", "early"),
        ("fragile_late", "fragile", "late"), ("randhi_early", "highcurv", "early"),
        ("randany_early", "anywhere", "early"), ("randany_full", "anywhere", "full")]


def sst_rows(split, k, seed=0):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")[split]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    idx = np.random.default_rng(seed).permutation(len(rows))[:k]
    return [rows[i] for i in idx]


def load_code():
    from datasets import load_dataset
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca")["train"]
    rows = []
    for r in ds:
        instr = (r.get("instruction") or "").strip(); inp = (r.get("input") or "").strip(); out = (r.get("output") or "").strip()
        if instr and out: rows.append((instr + (("\n" + inp) if inp else "") + "\n", " " + out))
    assert len(rows) >= 500, f"code identity {len(rows)}"; return rows[:1500]


def load_jsonB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


def taskdata(key):
    if key == "code": return load_code()
    d = load_jsonB(JSONK[key]); assert len(d) > 50, f"{key} identity {len(d)}"; return d


def b_pairs(tok, items):
    seqs, labs = [], []
    for pr, co in items:
        p = tok(pr, add_special_tokens=False)["input_ids"][:256]; c = tok(co, add_special_tokens=False)["input_ids"][:256]
        seqs.append(p + c); labs.append([-100] * len(p) + c)
    L = max(len(x) for x in seqs); pad = tok.pad_token_id
    inp = np.full((len(seqs), L), pad, np.int64); lb = np.full((len(seqs), L), -100, np.int64); am = np.zeros((len(seqs), L), np.int64)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        inp[i, :len(s)] = s; lb[i, :len(l)] = l; am[i, :len(s)] = 1
    return torch.tensor(inp).to(dev), torch.tensor(lb).to(dev), torch.tensor(am).to(dev)


def fisher(model, params, names, tok, items):
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    for s, l in items[:N_FISHER]:
        inp, lb, am = b_pairs(tok, [(f"{s}\nSentiment:", LAB[l])]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: g = params[n].grad.detach().double(); F[n] += g.T @ g
    return F


def eval_sst(model, tok, evalset, posid, negid):
    model.eval(); correct = 0; n = 0
    with torch.no_grad():
        for i in range(0, len(evalset), 32):
            chunk = evalset[i:i + 32]
            seqs = [tok(f"{s}\nSentiment:", add_special_tokens=False)["input_ids"][:256] for s, _ in chunk]
            L = max(len(x) for x in seqs); pad = tok.pad_token_id
            inp = np.full((len(seqs), L), pad, np.int64); pos = np.zeros(len(seqs), np.int64)
            for j, sq in enumerate(seqs): inp[j, :len(sq)] = sq; pos[j] = len(sq) - 1
            inp = torch.tensor(inp).to(dev)
            lg = model(input_ids=inp).logits
            last = lg[torch.arange(len(seqs)), torch.tensor(pos).to(dev)]
            pred = (last[:, posid] > last[:, negid]).long().cpu().numpy()
            for j, (_, lab) in enumerate(chunk): correct += int(pred[j] == lab); n += 1
    return correct / max(n, 1)


def train_arm(model, tok, B, Vmap, timing, E, seed, names_prot, posid, negid, sst_eval, want_massprofile=False):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed); params = dict(model.named_parameters())
    mass = {"early": 0.0, "late": 0.0, "rest": 0.0}; massprof = []
    lastloss = float("nan")
    for step in range(STEPS):
        idx = torch.randperm(len(B), generator=g)[:BATCH].tolist()
        inp, lb, am = b_pairs(tok, [B[k] for k in idx])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): loss = model(input_ids=inp, attention_mask=am, labels=lb).loss
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        active = (timing == "full") or (timing == "early" and step < E) or (timing == "late" and E <= step < 2 * E)
        # measure fragile-mass for naive profile (no subtraction); project for protected arms
        if want_massprofile or active:
            step_mass = 0.0
            for n in names_prot:
                Vn = Vmap[n]; gd = params[n].grad
                comp = (gd @ Vn) @ Vn.T                      # projected component onto V
                step_mass += float(comp.norm())
                if active: gd.sub_(comp)                      # GPM: remove it (protect)
            bucket = "early" if step < E else ("late" if step < 2 * E else "rest")
            mass[bucket] += step_mass
            if want_massprofile: massprof.append(step_mass)
        opt.step(); lastloss = float(loss.item())
    ret = eval_sst(model, tok, sst_eval, posid, negid)
    return ret, lastloss, mass, massprof


def main():
    t0 = time.time(); print("[CTL] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    base.load_state_dict(torch.load(CKPT)); ckpt_sd = {k: v.detach().clone() for k, v in base.state_dict().items()}
    params = dict(base.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    posid = tok(" positive", add_special_tokens=False)["input_ids"][0]; negid = tok(" negative", add_special_tokens=False)["input_ids"][0]
    sst_eval = sst_rows("validation", 300, 0)

    # curvature basis + fragile subspace (universal seed over {yelp,subj}, DISJOINT from test task-Bs)
    FA = fisher(base, params, names, tok, sst_rows("train", 200, 0))
    PA = {}; LAM = {}
    for n in names:
        ev, V = torch.linalg.eigh(FA[n]); PA[n] = V[:, -KTOP:].contiguous(); LAM[n] = ev[-KTOP:].cpu().numpy()
    useed = {n: np.zeros(KTOP) for n in names}
    for pk in FRAG_PANEL:
        Bp = taskdata(pk); base.zero_grad(set_to_none=True); gg = torch.Generator().manual_seed(0)
        for _ in range(N_SEED):
            inp, lb, am = b_pairs(tok, [Bp[k] for k in torch.randperm(len(Bp), generator=gg)[:BATCH].tolist()])
            (base(input_ids=inp, attention_mask=am, labels=lb).loss / N_SEED).backward()
        for n in names: useed[n] += (params[n].grad.detach().double() @ PA[n].double()).pow(2).sum(0).cpu().numpy()
        base.zero_grad(set_to_none=True)
    frag_ranks = {n: np.argsort(useed[n])[-FRAG:] for n in names}
    V_frag = {n: PA[n][:, frag_ranks[n]].contiguous().float() for n in names}
    # curvature-match readout: mean eigenvalue of fragile vs the top-64 band
    fr_lam = float(np.mean([LAM[n][frag_ranks[n]].mean() for n in names]))
    band_lam = float(np.mean([LAM[n].mean() for n in names]))
    pre_acc = eval_sst(base, tok, sst_eval, posid, negid)
    print(f"[CTL] setup done: pre SST-2 acc={pre_acc:.3f}, fragile mean-lam={fr_lam:.2e} band mean-lam={band_lam:.2e} ({time.time()-t0:.0f}s)", flush=True)

    def make_V(target, seed):
        if target == "fragile": return V_frag, None
        if target == "highcurv":  # curvature-DISTRIBUTION-matched non-fragile (greedy nearest-eigenvalue) -> isolates fragile from stiff
            Vm = {}; lams = []
            for n in names:
                fr = set(frag_ranks[n].tolist()); cand = [r for r in range(KTOP) if r not in fr]
                clam = LAM[n][cand]; used = set(); pick = []
                for fl in LAM[n][frag_ranks[n]]:
                    for o in np.argsort(np.abs(clam - fl)):
                        if o not in used: used.add(o); pick.append(cand[o]); break
                pick = np.array(pick); Vm[n] = PA[n][:, pick].contiguous().float(); lams.append(LAM[n][pick].mean())
            return Vm, float(np.mean(lams))
        if target == "anywhere":  # random 16-dim subspace of R^in
            rng = np.random.default_rng(2000 + seed); Vm = {}
            for n in names:
                Q, _ = torch.linalg.qr(torch.tensor(rng.standard_normal((PA[n].shape[0], FRAG))).to(dev).float()); Vm[n] = Q.contiguous()
            return Vm, None
        return {n: V_frag[n] for n in names}, None  # none (unused for naive)

    results = {}; massreport = {}
    for tb in TESTBS:
        B = taskdata(tb); results[tb] = {}
        # arm1 naive seed0 -> determine E + mass profile
        base.load_state_dict(ckpt_sd)
        ret0, bl0, mass0, prof = train_arm(base, tok, B, V_frag, "none", 0, 0, names, posid, negid, sst_eval, want_massprofile=True)
        prof = np.array(prof); pre = pre_acc
        # E = step where forgetting reaches 63% of final (naive). measure via running SST eval cheaply? use loss proxy: use mass-cumulative as transient proxy + fallback
        # principled: re-eval SST every 20 steps would be costly; approximate transient by cumulative fragile-mass reaching 63%
        cum = np.cumsum(prof) / (prof.sum() + 1e-9); E = int(np.searchsorted(cum, 0.63)) + 1; E = max(10, min(E, STEPS // 3))
        massreport[tb] = {"E": E, "mass_early": float(prof[:E].sum()), "mass_late": float(prof[E:2 * E].sum()),
                          "mass_rest": float(prof[2 * E:].sum()), "naive_ret": ret0, "forgetting": pre - ret0}
        print(f"[CTL] {tb}: E={E} (mass early {prof[:E].sum():.1f} vs late {prof[E:2*E].sum():.1f}); naive ret={ret0:.3f} forget={pre-ret0:.3f} ({time.time()-t0:.0f}s)", flush=True)
        for (aname, target, timing) in ARMS:
            rets, bls, me, ml = [], [], [], []; armlam = None
            for sd in SEEDS:
                base.load_state_dict(ckpt_sd)
                Vmap, armlam = make_V(target, sd) if target != "none" else (V_frag, None)
                ret, bl, mass, _ = train_arm(base, tok, B, Vmap, timing, E, sd, names, posid, negid, sst_eval)
                rets.append(ret); bls.append(bl); me.append(mass["early"]); ml.append(mass["late"])
            results[tb][aname] = {"ret_mean": float(np.mean(rets)), "ret_std": float(np.std(rets)), "rets": rets,
                                  "bloss_mean": float(np.mean(bls)), "mass_early": float(np.mean(me)), "mass_late": float(np.mean(ml)),
                                  "draw_lam": armlam}
            r = results[tb][aname]
            print(f"  [{tb}][{aname:14s}] ret={r['ret_mean']:.3f}+-{r['ret_std']:.3f} bloss={r['bloss_mean']:.2f}" + (f" drawlam={armlam:.2e}" if armlam else ""), flush=True)

    # SAVE before readout (raw per-seed rets = binary camera frame)
    sd_arts = {}
    for tb in TESTBS:
        for (a, _, _) in ARMS:
            if a in results[tb]: sd_arts[f"rets__{tb}__{a}"] = np.array(results[tb][a]["rets"])
    np.savez(NPZ, **sd_arts)
    rep = {"pre_acc": pre_acc, "fragile_mean_lam": fr_lam, "band_mean_lam": band_lam, "E_and_mass": massreport,
           "results": results, "steps": STEPS}
    try:
        verdicts = {}
        for tb in TESTBS:
            R = {a: results[tb][a]["ret_mean"] for (a, _, _) in ARMS}
            naive, ffull, fearly, flate, rhi, ranyE, ranyF = R["naive"], R["fragile_full"], R["fragile_early"], R["fragile_late"], R["randhi_early"], R["randany_early"], R["randany_full"]
            guard = (naive < pre_acc - 0.03) and (ffull > naive + 0.03)   # arm1 forgets, arm2 retains
            cheap = fearly >= ffull - 0.03                                # 3 as good as 2
            timingsp = fearly > flate + 0.02                             # 3 beats 4 (timing)
            fragsp = fearly > rhi + 0.02                                  # 3 beats 5a (FRAGILE beyond curvature -- the claim)
            if not guard:
                v = f"GUARD FAILED (naive {naive:.3f} vs pre {pre_acc:.3f}; fragile_full {ffull:.3f}) -> harness broken, verdict VOID"
            elif cheap and timingsp and fragsp:
                v = f"HYPOTHESIS HELD: fragile_early {fearly:.3f} ~ fragile_full {ffull:.3f} (cheap) AND > late {flate:.3f} (timing) AND > randhi_early {rhi:.3f} (FRAGILE beyond curvature). Timed+targeted protection works."
            elif cheap and timingsp and not fragsp:
                v = f"STIFF-NOT-FRAGILE: fragile_early {fearly:.3f} ~ randhi_early {rhi:.3f} -> protecting STIFF (not specifically fragile) dirs is what helps; reduces to known lever, novel claim NOT supported."
            elif cheap and not timingsp:
                v = f"NOT-TIMING-SPECIFIC: fragile_early {fearly:.3f} ~ late {flate:.3f} -> WHEN doesn't matter (or both windows work); protection helps but not the seeding-transient story."
            else:
                v = f"early NOT as cheap as full (early {fearly:.3f} < full {ffull:.3f}) -> timed protection insufficient; needs full-time."
            verdicts[tb] = {"verdict": v, "guard": guard, "cheap": cheap, "timing_specific": timingsp, "fragile_specific": fragsp,
                            "vals": {"naive": naive, "fragile_full": ffull, "fragile_early": fearly, "fragile_late": flate, "randhi_early": rhi, "randany_early": ranyE, "randany_full": ranyF}}
            print(f"\n[CTL] === {tb} (pre={pre_acc:.3f}) ===\n  {v}\n  mass early/late = {massreport[tb]['mass_early']:.1f}/{massreport[tb]['mass_late']:.1f} (E={massreport[tb]['E']}); 5a draw-lam vs fragile-lam = {results[tb]['randhi_early']['draw_lam']:.2e} vs {fr_lam:.2e}", flush=True)
        rep["verdicts"] = verdicts
    except Exception as e:
        import traceback; rep["error"] = traceback.format_exc(); print(f"[CTL] readout error (results saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[CTL] saved control_test.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
