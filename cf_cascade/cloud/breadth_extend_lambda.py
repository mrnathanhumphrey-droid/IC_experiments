"""BREADTH-EXTEND (experiment #1) — does a BROADER CHILDHOOD extend the task-A neighborhood to reach ALIEN task-kinds?

v3 leave-one-type-out established: the learnable neighborhood is CENTERED ON TASK-A (sentiment) and DECAYS with structural
distance — affect/text-classification tasks generalize cross-type above the language floor, but the two ALIEN kinds fall to/below
it (code 0.582 ≈ floor, ucdp/violence-events 0.509 BELOW floor). THE BOUND THIS OPENS: is that a FIXED limit, or does widening
the childhood (a richer breadth stream with BRIDGE kinds) extend the neighborhood to reach the alien tasks?

DESIGN (clean A/B, one run, same held-out tasks):
  ORIG stream    = the original 9 real task-Bs (agnews/yelp/subj/ucdp/code/emotion/dbpedia/imdb/tweet).
  NEW bridges    = math (gsm8k), science-QA (sciq), NLI (snli)  [structurally between sentiment & CODE]
                 + news-category incl crime/politics/world (heegyu) + yahoo-topics  [toward CONFLICT-EVENTS / broad world].
  ENRICHED stream = ORIG + NEW.
  For each held-out task h (focus code, ucdp): cross-type partial from ORIG-other-type map vs ENRICHED-other-type map.
  Δ = cross_enriched − cross_orig, read against the language floor (nltext) and generic floor (randtok).

PRE-COMMITTED VERDICT (per alien task):
  cross_enriched rises and clears the language floor (Δ>0, enr−floor>~0.05) -> NEIGHBORHOOD EXTENSIBLE BY BREADTH: a richer
    childhood maps more of the world (the whole developmental claim). #3's decidable region grows with the breadth stream.
  cross_enriched stays at/below floor despite bridges -> HARD STRUCTURAL LIMIT: some task-kinds are unreachable from a
    sentiment-rooted map no matter how broad the childhood (the map extends within a family, not across deep kinds) — also a finding.
  (Caveat to log: bridge QUALITY — if a bridge kind itself lands at floor, it isn't actually bridging; report new-task cross too.)
Per-layer NEVER pooled; curvature-controlled partial(map,seed|λ); seed-not-contest; held-out is the control; dataset-identity logged.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "8"
os.environ.setdefault("HF_HOME", "/home/ubuntu/hf_cache")
import json, time, traceback, numpy as np, torch
G = "/home/ubuntu/gateA"
MODEL = "EleutherAI/pythia-410m"; CKPT = f"{G}/pythia410m_sst2_subsetmin.pt"; AX = f"{G}/overlap_axis"
OUT = f"{G}/breadth_extend.json"; NPZ = f"{G}/breadth_extend.npz"
CARRIERS = ["attention.query_key_value", "mlp.dense_h_to_4h", "mlp.dense_4h_to_h"]
KBAND = 128; N_FISHER = 200; N_SEED = 16; BATCH = 16
LAB = {0: " negative", 1: " positive"}; dev = "cuda"
torch.manual_seed(0); np.random.seed(0)

JSON_MEMBERS = [("agnews", "low_agnews", "topic", "orig"), ("yelp", "high_yelp", "sentiment", "orig"),
                ("subj", "medium_subj", "subjectivity", "orig"), ("ucdp", "ucdp_violence", "violence", "orig"),
                ("nullperm", "null_sst2permuted", "NULLperm", "null")]


def load_jsonB(name):
    d = json.load(open(f"{AX}/{name}.json")); t = d["prompt_template"]
    return [(t.format(text=e["text"]), e["completion"]) for e in d["examples"]]


def win(ds, k=600, w=3000):
    n = min(len(ds), w); return list(ds.shuffle(seed=0).select(range(n)))[: k * 3]


def hf_members():
    """(key, type, era, loader); era 'orig' = the original panel, 'new' = bridge kinds. per-loader try/except."""
    from datasets import load_dataset
    out = {}
    cands = [
        # ---- ORIG panel (reproduce v3) ----
        ("code", "code", "orig", lambda: [((r["instruction"] or "").strip() + "\n", " " + (r["output"] or "").strip())
                                          for r in win(load_dataset("iamtarun/python_code_instructions_18k_alpaca")["train"]) if r.get("instruction") and r.get("output")]),
        ("emotion", "emotion", "orig", lambda: [(f"{r['text'].strip()}\nEmotion:", " " + ["sadness","joy","love","anger","fear","surprise"][r["label"]])
                                                for r in win(load_dataset("dair-ai/emotion")["train"])]),
        ("dbpedia", "topic", "orig", lambda: [(f"{r['content'].strip()[:400]}\nCategory:", " " + ["company","school","artist","athlete","politics","transport","building","nature","village","animal","plant","album","film","book"][r["label"]])
                                              for r in win(load_dataset("fancyzhx/dbpedia_14")["train"])]),
        ("imdb", "sentiment", "orig", lambda: [(f"{r['text'].strip()[:400]}\nSentiment:", LAB[r["label"]]) for r in win(load_dataset("stanfordnlp/imdb")["train"])]),
        ("tweet", "sentiment", "orig", lambda: [(f"{r['text'].strip()}\nSentiment:", " " + ["negative","neutral","positive"][r["label"]])
                                                for r in win(load_dataset("mteb/tweet_sentiment_extraction")["train"]) if r.get("text")]),
        # ---- NEW bridges toward CODE (structured/symbolic/reasoning) ----
        ("gsm8k", "math", "new", lambda: [(r["question"].strip() + "\n", " " + r["answer"].strip()[:200]) for r in win(load_dataset("openai/gsm8k", "main")["train"])]),
        ("sciq", "science", "new", lambda: [(r["question"].strip() + "\n", " " + r["correct_answer"].strip()) for r in win(load_dataset("allenai/sciq")["train"]) if r.get("correct_answer")]),
        ("snli", "nli", "new", lambda: [(f"{r['premise'].strip()} / {r['hypothesis'].strip()}\nRelation:", " " + ["entailment","neutral","contradiction"][r["label"]])
                                        for r in win(load_dataset("stanfordnlp/snli")["train"]) if r["label"] in (0, 1, 2)]),
        # ---- NEW bridges toward CONFLICT-EVENTS / broad world ----
        ("newscat", "newscat", "new", lambda: [(f"{(r.get('headline') or '').strip()}\nCategory:", " " + (r.get("category") or "").strip().lower())
                                               for r in win(load_dataset("heegyu/news-category-dataset")["train"]) if r.get("headline") and r.get("category")]),
        ("yahoo", "qatopic", "new", lambda: [(f"{(r.get('question_title') or '').strip()}\nTopic:", " " + ["society","science","health","education","computers","sports","business","entertainment","family","politics"][r["topic"]])
                                             for r in win(load_dataset("community-datasets/yahoo_answers_topics")["train"]) if r.get("question_title")]),
    ]
    for key, typ, era, fn in cands:
        try:
            rows = [(p, c) for (p, c) in fn() if p and c][:600]; assert len(rows) >= 50, f"only {len(rows)}"
            out[key] = (typ, era, rows); print(f"[BX]   +HF {key} ({typ},{era}) n={len(rows)}", flush=True)
        except Exception as e:
            print(f"[BX]   xHF {key} FAILED: {str(e)[:120]}", flush=True)
    return out


def load_nltext():
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
    t0 = time.time(); print("[BX] loading...", flush=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    model.load_state_dict(torch.load(CKPT)); params = dict(model.named_parameters())
    names = [n for n in params if any(c in n for c in CARRIERS) and n.endswith("weight") and params[n].ndim == 2]
    carrier_of = {n: ("attn" if "query_key_value" in n else "mlp") for n in names}
    vocab = model.get_input_embeddings().weight.shape[0]

    print("[BX] building panel...", flush=True)
    panel = {}; types = {}; era = {}
    for key, jname, typ, er in JSON_MEMBERS:
        try:
            rows = load_jsonB(jname); assert len(rows) >= 50
            panel[key] = rows; types[key] = typ; era[key] = er; print(f"[BX]   +JSON {key} ({typ},{er}) n={len(rows)}", flush=True)
        except Exception as e:
            print(f"[BX]   xJSON {key} FAILED: {str(e)[:100]}", flush=True)
    for key, (typ, er, rows) in hf_members().items():
        panel[key] = rows; types[key] = typ; era[key] = er
    try:
        panel["nltext"] = load_nltext(); types["nltext"] = "NULLnl"; era["nltext"] = "null"; print(f"[BX]   +nltext n={len(panel['nltext'])}", flush=True)
    except Exception as e:
        print(f"[BX]   xnltext FAILED: {str(e)[:100]}", flush=True)

    orig_real = sorted([k for k in panel if era[k] == "orig"])
    new_real = sorted([k for k in panel if era[k] == "new"])
    enr_real = sorted(orig_real + new_real)
    null_keys = sorted([k for k in panel if era[k] == "null"]) + ["randtok"]
    types["randtok"] = "NULLrand"; era["randtok"] = "null"
    assert len(orig_real) >= 6 and len(new_real) >= 2, f"orig {orig_real} new {new_real}"
    print(f"[BX] ORIG={orig_real}\n[BX] NEW={new_real}\n[BX] nulls={null_keys} ({time.time()-t0:.0f}s)", flush=True)

    FA = fisher_taskA(model, params, names, tok)
    V = {}; LAM = {}
    for n in names:
        ev, vv = torch.linalg.eigh(FA[n]); V[n] = vv[:, -KBAND:].contiguous(); LAM[n] = ev[-KBAND:].cpu().numpy()
    model.load_state_dict(torch.load(CKPT)); params = dict(model.named_parameters())
    print(f"[BX] FA basis done ({time.time()-t0:.0f}s)", flush=True)

    def energy(S):
        e = {}
        for n in names:
            Vn = V[n]; ev = (Vn.T @ S[n] @ Vn).diagonal().clamp(min=0).cpu().numpy(); s = ev.sum(); e[n] = (ev / s) if s > 0 else ev
        return e

    E = {}
    for key in enr_real + [k for k in null_keys if k != "randtok"]:
        E[key] = energy(seedcov(model, params, names, tok, panel[key])); print(f"[BX] seed {key} done ({time.time()-t0:.0f}s)", flush=True)
    E["randtok"] = energy(seedcov_rand(model, params, names, vocab)); print(f"[BX] seed randtok done ({time.time()-t0:.0f}s)", flush=True)

    typ = {k: types[k] for k in enr_real}
    def crossmap(stream, h, n):
        src = [g for g in stream if g != h and typ[g] != typ[h]]
        return np.mean([E[g][n] for g in src], axis=0) if src else None

    # A/B: cross-type prediction of each held-out task under ORIG vs ENRICHED stream
    PL = {n: {} for n in names}
    allreal = {}
    for n in names:
        allreal[n] = np.mean([E[g][n] for g in enr_real], axis=0)
        for h in enr_real:
            co = crossmap(orig_real, h, n); ce = crossmap(enr_real, h, n)
            PL[n][h] = {"cross_orig": (partial_spearman(co, E[h][n], LAM[n]) if co is not None else float("nan")),
                        "cross_enr": (partial_spearman(ce, E[h][n], LAM[n]) if ce is not None else float("nan"))}
        for k in null_keys:
            PL[n][k] = {"cross_orig": partial_spearman(allreal[n], E[k][n], LAM[n]), "cross_enr": partial_spearman(allreal[n], E[k][n], LAM[n])}

    def mean_over(ks, fld):
        vals = [PL[n][h][fld] for n in names for h in ks]; vals = [x for x in vals if x == x]
        return float(np.mean(vals)) if vals else float("nan")
    floor_nl = mean_over(["nltext"], "cross_orig"); floor_rt = mean_over(["randtok"], "cross_orig")
    pertask = {h: {"cross_orig": float(np.nanmean([PL[n][h]["cross_orig"] for n in names])),
                   "cross_enr": float(np.nanmean([PL[n][h]["cross_enr"] for n in names])),
                   "type": typ[h], "era": era[h]} for h in enr_real}
    for h in pertask: pertask[h]["delta"] = pertask[h]["cross_enr"] - pertask[h]["cross_orig"]

    np.savez(NPZ, **{f"E__{k}__{i}": E[k][n] for i, n in enumerate(names) for k in E}, names=np.array(names),
             orig_real=np.array(orig_real), new_real=np.array(new_real))
    rep = {"orig_real": orig_real, "new_real": new_real, "null_keys": null_keys, "types": types, "KBAND": KBAND,
           "floor_nltext": floor_nl, "floor_randtok": floor_rt, "pertask": pertask}
    try:
        def line(h):
            p = pertask[h]; return f"{h:9s}[{typ[h]:11s}] orig={p['cross_orig']:.3f} -> enr={p['cross_enr']:.3f} (Δ{p['delta']:+.3f})  vs nl-floor {floor_nl:.3f}"
        verds = {}
        for h in ["code", "ucdp"]:
            if h not in pertask: verds[h] = "ABSENT"; continue
            p = pertask[h]; extensible = (p["delta"] > 0.03) and (p["cross_enr"] - floor_nl > 0.05)
            if extensible: verds[h] = f"EXTENSIBLE: cross rose {p['cross_orig']:.3f}->{p['cross_enr']:.3f} (Δ{p['delta']:+.3f}) & clears language floor {floor_nl:.3f} -> a broader childhood reaches it."
            elif p["cross_enr"] - floor_nl <= 0.0: verds[h] = f"HARD LIMIT: cross stays at/below language floor ({p['cross_enr']:.3f} vs {floor_nl:.3f}, Δ{p['delta']:+.3f}) despite bridges -> unreachable from a sentiment-rooted childhood."
            else: verds[h] = f"PARTIAL: cross moved {p['delta']:+.3f} to {p['cross_enr']:.3f} (floor {floor_nl:.3f}) -- weak extension, not decisive."
        rep["verdicts"] = verds
        print(f"\n[BX] === EXTEND VERDICT (does a broader childhood reach the alien kinds?) ===", flush=True)
        for h in ["code", "ucdp"]:
            if h in pertask: print(f"  [{h}] {verds[h]}", flush=True)
        print(f"  floors: nltext(language)={floor_nl:.3f} randtok(generic)={floor_rt:.3f}", flush=True)
        print("  --- ORIG task-Bs (alien-kind focus) ---", flush=True)
        for h in sorted(orig_real, key=lambda x: -pertask[x]["cross_enr"]): print(f"    {line(h)}", flush=True)
        print("  --- NEW bridge task-Bs (where they land = bridge quality) ---", flush=True)
        for h in sorted(new_real, key=lambda x: -pertask[x]["cross_enr"]): print(f"    {line(h)}", flush=True)
    except Exception as e:
        rep["error"] = traceback.format_exc(); print(f"[BX] readout error (saved): {e}", flush=True)
    json.dump(rep, open(OUT, "w"), indent=2)
    print(f"[BX] saved breadth_extend.json wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
