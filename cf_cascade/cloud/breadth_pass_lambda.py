"""BREADTH-PASS v3 (LEAVE-ONE-TYPE-OUT) — is the map's task-specific signal CROSS-TYPE or just within-type similarity?

v2 found a +0.171 task-specific lift, but the press (Wilson): diff-type pairwise (0.544) sat AT the language floor (nltext
0.561), so the lift may be carried entirely by WITHIN-type near-duplicates (sentiment x3, topic x2). The decisive test: hold
out an ENTIRE type, predict the held-out task from ONLY OTHER-type members (cross-type map), read vs the language floor.
  cross-type >> floor -> genuine broad task-relational neighborhood (#3 strong).
  cross-type ~ floor  -> SIMILARITY-LOCAL: map generalizes by similarity only; the +0.17 was within-type near-duplicates.
  graded -> near-text types generalize cross, structurally-far (code/ucdp) fall to floor (the honest developmental shape:
            a childhood map helps with tasks NEAR what it saw; novel-kind tasks still yank). Per-task cross printed to see which.
Singletons (subj/ucdp/code/emotion: no same-type sibling) give the PUREST cross-type read (their LOO IS cross-type already).

--- v2 design (unchanged below) ---
is task-A's neighborhood a learnable map, and is it TASK-SPECIFIC or generic model geometry?

v1 found held-out partial-spearman 0.66 (>>shuffle, beyond curvature) BUT the only null (permuted-SST) hit 0.54 -- and it
was a DIRTY null (same SST inputs, scrambled labels -> sentiment-shaped by construction -> lands in the real region), and
null members polluted the leave-one-out map. v2 fixes both:
  (a) MAP FIT = REAL task-Bs only (the "childhood" stream). Nulls are NEVER in the stream; they are held-out probes only.
  (b) TWO TRUE NULLS that share neither inputs nor classification format:
        randtok  -- synthetic random-token sequences (pure model gradient geometry, no language at all)
        nltext   -- natural-language next-token LM, no classification cue/label (TinyStories) -> language geometry w/o task format
      + keep nullperm (permuted-SST) as a THIRD, informative dirty-null: sentiment-shaped, should predict HIGH (sanity).

DECOMPOSITION: generic floor = partial(real-map, true-null). task-specific lift = real held-out partial - true-null floor.
  lift clears margin -> the map carries TASK-NEIGHBORHOOD structure beyond generic geometry (the part a childhood learns).
  real ~ true-null floor -> the "map" is generic model/format geometry, not a learned neighborhood. Bounds the idea.
Expect ordering if task-specific: real(~0.66) > nullperm(sentiment-shaped) > nltext > randtok(generic floor).

READOUTS (per layer, NEVER pooled): raw spearman vs shuffle; curvature-controlled partial-spearman(map, seed | lam);
per-carrier (attn/mlp); graded same-type vs diff-type (real only). Held-out is the control; seed not contest (conv-clean).
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
OUT = f"{G}/breadth_typeout.json"; NPZ = f"{G}/breadth_typeout.npz"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
KBAND = 128; N_FISHER = 200; N_SEED = 16; BATCH = 16; NSHUF = 200
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
torch.manual_seed(0); np.random.seed(0)

JSON_MEMBERS = [("agnews", "low_agnews", "topic"), ("yelp", "high_yelp", "sentiment"),
                ("subj", "medium_subj", "subjectivity"), ("ucdp", "ucdp_violence", "violence"),
                ("nullperm", "null_sst2permuted", "NULLperm")]  # nullperm = dirty null (sentiment-shaped)


def load_jsonB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


def hf_members():
    from datasets import load_dataset
    out = {}
    def take(ds, n=600): return list(ds)[:n]
    cands = [
        ("code", "code", lambda: [((r["instruction"] or "").strip() + "\n", " " + (r["output"] or "").strip())
                                  for r in take(load_dataset("iamtarun/python_code_instructions_18k_alpaca")["train"])
                                  if r.get("instruction") and r.get("output")]),
        ("emotion", "emotion", lambda: [(f"{r['text'].strip()}\nEmotion:", " " + ["sadness","joy","love","anger","fear","surprise"][r["label"]])
                                        for r in take(load_dataset("dair-ai/emotion")["train"])]),
        ("dbpedia", "topic", lambda: [(f"{r['content'].strip()[:400]}\nCategory:", " " + ["company","school","artist","athlete","politics","transport","building","nature","village","animal","plant","album","film","book"][r["label"]])
                                      for r in take(load_dataset("fancyzhx/dbpedia_14")["train"].shuffle(seed=0))]),
        ("imdb", "sentiment", lambda: [(f"{r['text'].strip()[:400]}\nSentiment:", LAB[r["label"]])
                                       for r in take(load_dataset("stanfordnlp/imdb")["train"].shuffle(seed=0))]),
        ("tweet", "sentiment", lambda: [(f"{r['text'].strip()}\nSentiment:", " " + ["negative","neutral","positive"][r["label"]])
                                        for r in take(load_dataset("mteb/tweet_sentiment_extraction")["train"]) if r.get("text")]),
    ]
    for key, typ, fn in cands:
        try:
            rows = [(p, c) for (p, c) in fn() if p and c]; assert len(rows) >= 50, f"only {len(rows)}"
            out[key] = (typ, rows); print(f"[BP]   +HF {key} ({typ}) n={len(rows)}", flush=True)
        except Exception as e:
            print(f"[BP]   xHF {key} FAILED: {str(e)[:100]}", flush=True)
    return out


def load_nltext():
    """natural-language next-token LM, NO classification cue/label -> language geometry without task format."""
    from datasets import load_dataset
    rows = []
    for r in list(load_dataset("roneneldan/TinyStories")["train"])[:1500]:
        w = (r.get("text") or "").split()
        if len(w) >= 24: rows.append((" ".join(w[:8]), " " + " ".join(w[8:32])))
    assert len(rows) >= 50, len(rows); return rows


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


def sst_rows(split, k, seed=0):
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/sst2")[split]
    rows = [(r["sentence"].strip(), int(r["label"])) for r in ds if r["label"] in (0, 1)]
    return [rows[i] for i in np.random.default_rng(seed).permutation(len(rows))[:k]]


def fisher_taskA(model, params, names, tok):
    F = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    for s, l in sst_rows("train", N_FISHER, 0):
        inp, lb, am = b_pairs(tok, [(f"{s}\nSentiment:", LAB[l])]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: g = params[n].grad.detach().double(); F[n] += g.T @ g
    return F


def seedcov(model, params, names, tok, items):
    S = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    gg = torch.Generator().manual_seed(0)
    for _ in range(N_SEED):
        idx = torch.randperm(len(items), generator=gg)[:BATCH].tolist()
        inp, lb, am = b_pairs(tok, [items[k] for k in idx]); model.zero_grad(set_to_none=True)
        model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: g = params[n].grad.detach().double(); S[n] += g.T @ g
    return S


def seedcov_rand(model, params, names, vocab):
    """pure random-token LM gradient -> generic model geometry floor (no language, no shared input, no format)."""
    S = {n: torch.zeros(params[n].shape[1], params[n].shape[1], device=dev, dtype=torch.float64) for n in names}
    g = torch.Generator().manual_seed(0)
    for _ in range(N_SEED):
        inp = torch.randint(0, vocab, (BATCH, 48), generator=g).to(dev); am = torch.ones_like(inp)
        lb = inp.clone(); lb[:, :32] = -100
        model.zero_grad(set_to_none=True); model(input_ids=inp, attention_mask=am, labels=lb).loss.backward()
        for n in names: gg = params[n].grad.detach().double(); S[n] += gg.T @ gg
    return S


def ranks(a): return np.argsort(np.argsort(np.asarray(a, float))).astype(float)


def spearman(x, y):
    rx, ry = ranks(x), ranks(y); rx -= rx.mean(); ry -= ry.mean()
    d = (rx @ rx) ** 0.5 * (ry @ ry) ** 0.5; return float(rx @ ry / d) if d > 0 else 0.0


def partial_spearman(x, y, z):
    rx, ry, rz = ranks(x), ranks(y), ranks(z); Z = np.column_stack([np.ones_like(rz), rz])
    def res(r): beta, *_ = np.linalg.lstsq(Z, r, rcond=None); return r - Z @ beta
    ex, ey = res(rx), res(ry); d = (ex @ ex) ** 0.5 * (ey @ ey) ** 0.5
    return float(ex @ ey / d) if d > 0 else 0.0


def main():
    t0 = time.time(); print("[BP] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    carrier_of = {n: ("attn" if "query_key_value" in n else "mlp") for n in names}
    vocab = model.get_input_embeddings().weight.shape[0]

    print("[BP] building panel...", flush=True)
    panel = {}; types = {}
    for key, jname, typ in JSON_MEMBERS:
        try:
            rows = load_jsonB(jname); assert len(rows) >= 50, len(rows)
            panel[key] = rows; types[key] = typ; print(f"[BP]   +JSON {key} ({typ}) n={len(rows)}", flush=True)
        except Exception as e:
            print(f"[BP]   xJSON {key} FAILED: {str(e)[:100]}", flush=True)
    for key, (typ, rows) in hf_members().items():
        panel[key] = rows; types[key] = typ
    try:
        panel["nltext"] = load_nltext(); types["nltext"] = "NULLnl"; print(f"[BP]   +nltext (NULLnl) n={len(panel['nltext'])}", flush=True)
    except Exception as e:
        print(f"[BP]   xnltext FAILED: {str(e)[:100]}", flush=True)

    real_keys = sorted([k for k in panel if not types[k].startswith("NULL")])
    assert len(real_keys) >= 6, f"only {len(real_keys)} REAL task-Bs, need >=6"
    null_keys = sorted([k for k in panel if types[k].startswith("NULL")]) + ["randtok"]
    types["randtok"] = "NULLrand"
    print(f"[BP] panel: {len(real_keys)} real {real_keys} | nulls {null_keys} ({time.time()-t0:.0f}s)", flush=True)

    FA = fisher_taskA(model, params, names, tok)
    V = {}; LAM = {}
    for n in names:
        ev, vv = torch.linalg.eigh(FA[n]); V[n] = vv[:, -KBAND:].contiguous(); LAM[n] = ev[-KBAND:].cpu().numpy()
    model.load_state_dict(torch.load(CKPT)); params = dict(model.named_parameters())
    print(f"[BP] FA basis done ({time.time()-t0:.0f}s)", flush=True)

    def energy(S):
        e = {}
        for n in names:
            Vn = V[n]; ev = (Vn.T @ S[n] @ Vn).diagonal().clamp(min=0).cpu().numpy(); s = ev.sum()
            e[n] = (ev / s) if s > 0 else ev
        return e

    E = {}
    for key in real_keys + [k for k in null_keys if k != "randtok"]:
        E[key] = energy(seedcov(model, params, names, tok, panel[key])); print(f"[BP] seed {key} done ({time.time()-t0:.0f}s)", flush=True)
    E["randtok"] = energy(seedcov_rand(model, params, names, vocab)); print(f"[BP] seed randtok done ({time.time()-t0:.0f}s)", flush=True)

    # ===== full LOO + leave-one-TYPE-out (cross-type) + same-type, per layer (NEVER pooled) =====
    rng = np.random.default_rng(0)
    typ = {k: types[k] for k in real_keys}
    sib = {h: [g for g in real_keys if g != h and typ[g] == typ[h]] for h in real_keys}  # same-type siblings
    crs = {h: [g for g in real_keys if typ[g] != typ[h]] for h in real_keys}             # OTHER-type only
    PL = {n: {} for n in names}
    for n in names:
        lam = LAM[n]; allreal = np.mean([E[g][n] for g in real_keys], axis=0)
        for h in real_keys:
            loo = np.mean([E[g][n] for g in real_keys if g != h], axis=0)
            cr = np.mean([E[g][n] for g in crs[h]], axis=0)
            sm = np.mean([E[g][n] for g in sib[h]], axis=0) if sib[h] else None
            PL[n][h] = {"loo": partial_spearman(loo, E[h][n], lam), "cross": partial_spearman(cr, E[h][n], lam),
                        "same": (partial_spearman(sm, E[h][n], lam) if sm is not None else float("nan")),
                        "raw": spearman(loo, E[h][n]),
                        "shuf": float(np.mean([spearman(loo[rng.permutation(KBAND)], E[h][n]) for _ in range(NSHUF)]))}
        for k in null_keys:  # nulls scored vs full REAL map (never in stream)
            PL[n][k] = {"loo": partial_spearman(allreal, E[k][n], lam), "cross": partial_spearman(allreal, E[k][n], lam),
                        "same": float("nan"), "raw": spearman(allreal, E[k][n]), "shuf": 0.0}

    def m(ks, ns, fld):
        vals = [PL[n][h][fld] for n in ns for h in ks]; vals = [x for x in vals if x == x]  # drop nan
        return {"mean": float(np.mean(vals)), "se": float(np.std(vals) / len(vals) ** 0.5),
                "frac_pos": float(np.mean(np.array(vals) > 0)), "n": len(vals)} if vals else None
    attn_n = [n for n in names if carrier_of[n] == "attn"]; mlp_n = [n for n in names if carrier_of[n] == "mlp"]
    singletons = [h for h in real_keys if not sib[h]]; hassib = [h for h in real_keys if sib[h]]
    A = {"real_loo": m(real_keys, names, "loo"), "real_cross": m(real_keys, names, "cross"),
         "cross_singletons": m(singletons, names, "cross"), "cross_hassib": m(hassib, names, "cross"),
         "same_hassib": m(hassib, names, "same"), "cross_attn": m(real_keys, attn_n, "cross"), "cross_mlp": m(real_keys, mlp_n, "cross")}
    for k in null_keys: A[f"null_{k}"] = m([k], names, "loo")
    true_nulls = [k for k in null_keys if types[k] in ("NULLrand", "NULLnl")]
    floor = float(np.mean([A[f"null_{k}"]["mean"] for k in true_nulls])); floor_se = float(np.mean([A[f"null_{k}"]["se"] for k in true_nulls]))
    floor_nl = A["null_nltext"]["mean"] if "null_nltext" in A and A["null_nltext"] else floor  # conservative language floor

    np.savez(NPZ, **{f"E__{k}__{i}": E[k][n] for i, n in enumerate(names) for k in E}, names=np.array(names),
             real_keys=np.array(real_keys), null_keys=np.array(null_keys))
    pertask = {h: {"loo": float(np.nanmean([PL[n][h]["loo"] for n in names])), "cross": float(np.nanmean([PL[n][h]["cross"] for n in names])),
                   "same": float(np.nanmean([PL[n][h]["same"] for n in names])), "type": typ[h], "siblings": sib[h]} for h in real_keys}
    rep = {"real_keys": real_keys, "null_keys": null_keys, "types": types, "KBAND": KBAND, "agg": A,
           "floor_mean": floor, "floor_se": floor_se, "floor_nltext": floor_nl, "singletons": singletons, "pertask": pertask}
    try:
        cx = A["real_cross"]["mean"]; cse = A["real_cross"]["se"]
        cxs = A["cross_singletons"]["mean"] if A["cross_singletons"] else float("nan")  # purest cross-type
        cxh = A["cross_hassib"]["mean"] if A["cross_hassib"] else float("nan")
        smh = A["same_hassib"]["mean"] if A["same_hassib"] else float("nan")
        lift_mean = cx - floor; lift_nl = cx - floor_nl
        if lift_nl > 0.15 and (cx - floor) > 3 * (cse + floor_se):
            v = (f"GENUINE CROSS-TYPE generalization: cross-type held-out partial={cx:.3f} vs language floor={floor_nl:.3f} "
                 f"(lift {lift_nl:+.3f} >>SE) -- a map from OTHER task types predicts a held-out type beyond generic language "
                 f"geometry. The neighborhood map is broadly task-relational, NOT just same-task similarity. #3 path strong.")
        elif lift_nl > 0.05:
            v = (f"GRADED / PARTLY-LOCAL: cross-type={cx:.3f} vs language floor={floor_nl:.3f} (lift {lift_nl:+.3f}) -- modest "
                 f"cross-type generalization, mostly similarity-local + a small genuine cross-type increment. Graded by structural "
                 f"distance (see per-task: near-text types generalize, structurally-far at floor). same-type {smh:.3f} > cross {cxh:.3f}.")
        else:
            v = (f"SIMILARITY-LOCAL: cross-type={cx:.3f} ~ language floor={floor_nl:.3f} (lift {lift_nl:+.3f}) -- the map generalizes "
                 f"by SIMILARITY only; across genuinely different task types it collapses to generic language geometry. The v2 +0.17 "
                 f"was carried by within-type near-duplicates (same-type {smh:.3f} vs cross-type {cxh:.3f} for multi-member types). "
                 f"Honest developmental shape: a childhood map helps NEAR what it saw; novel-kind tasks still yank.")
        rep["verdict"] = v; rep["lift_vs_floor_mean"] = lift_mean; rep["lift_vs_nltext"] = lift_nl
        print(f"\n[BP] === LEAVE-ONE-TYPE-OUT VERDICT ===\n  {v}", flush=True)
        print(f"  real LOO={A['real_loo']['mean']:.3f} | real CROSS-type={cx:.3f} | floor mean={floor:.3f} nltext={floor_nl:.3f}", flush=True)
        print(f"  cross singletons(purest)={cxs:.3f} | cross has-sibling={cxh:.3f} | SAME-type has-sibling={smh:.3f}", flush=True)
        print(f"  cross attn={A['cross_attn']['mean']:.3f} mlp={A['cross_mlp']['mean']:.3f}", flush=True)
        print("  PER-TASK cross-type partial (structural-distance check, high->low):", flush=True)
        for h in sorted(real_keys, key=lambda x: -pertask[x]["cross"]):
            print(f"    {h:9s} [{typ[h]:12s}] cross={pertask[h]['cross']:.3f} loo={pertask[h]['loo']:.3f} same={pertask[h]['same']:.3f} sibs={sib[h]}", flush=True)
        for k in null_keys: print(f"  NULL[{k:9s} {types[k]:9s}] partial={A[f'null_{k}']['mean']:.3f}", flush=True)
    except Exception as e:
        rep["error"] = traceback.format_exc(); print(f"[BP] readout error (saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[BP] saved breadth_typeout.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
