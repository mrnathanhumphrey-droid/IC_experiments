"""
probe_fantasy_rules_direction.py — PHASE-3(2): is FANTASY-vs-RULES a CONTESTED direction (route to poles) or a
constrained one (hard-neg fix)?  (SANDBOXED, $0 local GPU)

Applies the detector diagnostic (cos of discriminative directions predicted swap<->diffusion antagonism) to the
Librarian's one open gap — the keeper collision (fantasy "keeper" intrudes on soccer "goalkeeper"). Three diagnostics:
  1. DIRECTIONAL COSINE: base mean-relevance-gradient direction per pole + fine-tuned update Δθ direction per pole;
     cos(rules, fantasy) vs within-pole split-half consistency (the scale) and a random-label null.
  2. CF SURGE both orders: base->RULES->+FANTASY and base->FANTASY->+RULES; retention erosion + inV vs inSwap
     (directional landing) reusing the Phase-1 Fisher machinery.
  3. KEEPER INTRUSION: after JOINT training, rank of cross-pole keeper chunks against the true positive for
     goalkeeper(rules) and keeper(fantasy) queries, vs a same-pole-only baseline.

ISOLATION: reads only sandbox-copied inputs; writes only ./out/. Never touches production Librarian / checkpoints / Fly.

Run:  python probe_fantasy_rules_direction.py --smoke   # tiny plumbing
      python probe_fantasy_rules_direction.py           # full
"""
from __future__ import annotations
import json, sys, hashlib, glob
from pathlib import Path
import numpy as np, torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_metrics import evaluate

SMOKE = "--smoke" in sys.argv
BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAXLEN = 256
EPOCHS = 1 if SMOKE else 4
BATCH = 8 if SMOKE else 32
LR = 2e-5
K = 16 if SMOKE else 64
N_FISHER = 12 if SMOKE else 120
N_DIR = 24 if SMOKE else 240          # balanced pairs per pole for the gradient-direction estimate
NEG_PER_POS = 3 if SMOKE else 6
RETENTION_FRAC = 0.25
SWAP_SEED = 12345
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = HERE / "out" / ("fantasy_rules_direction_smoke.json" if SMOKE else "fantasy_rules_direction.json")
torch.manual_seed(0); np.random.seed(0)
RNG = np.random.RandomState(0)


# ── data ──────────────────────────────────────────────────────────────────────
def load_ctext():
    ctext, pole = {}, {}
    for s in ("soccer", "nba", "nfl"):
        for l in (HERE / "corpus" / f"rules_{s}.jsonl").open(encoding="utf-8"):
            r = json.loads(l); ctext[r["doc_id"]] = ((r.get("title", "") or "") + ". " + (r.get("body", "") or "")).strip()[:2000]; pole[r["doc_id"]] = "rules"
    for f in glob.glob(str(HERE / "corpus_fantasy" / "*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l); ctext[r["doc_id"]] = ((r.get("title", "") or "") + ". " + (r.get("body", "") or "")).strip()[:2000]; pole[r["doc_id"]] = "fantasy"
    return ctext, pole


def _split_of(qid, frac=RETENTION_FRAC):
    return "retention" if (int(hashlib.md5(qid.encode()).hexdigest(), 16) % 1000) < int(frac * 1000) else "train"


def build_rows(task_files, ctext, pole_name, pole_map, neg_per_pos=NEG_PER_POS):
    """Self-contained rows: positives = resolved task positives; negatives = sampled from the COMBINED corpus
    (shared cross-domain negative pool). Returns {'train':[...],'retention':[...]} of (qid, question, docs[], labels[])."""
    allids = list(ctext.keys())
    out = {"train": [], "retention": []}
    for tf in task_files:
        for l in (HERE / "data" / tf).open(encoding="utf-8"):
            e = json.loads(l)
            pos = [c["doc_id"] for c in e["candidates"] if str(c.get("relevant")) == "1" and c["doc_id"] in ctext]
            pos = list(dict.fromkeys(pos))
            if not pos:
                continue
            nneg = max(neg_per_pos * len(pos), neg_per_pos)
            posset = set(pos)
            # deterministic per-qid negative sample from the combined corpus (cross-domain distractors included)
            r = np.random.RandomState(int(hashlib.md5(e["qid"].encode()).hexdigest(), 16) % (2**32))
            negs = []
            while len(negs) < nneg:
                d = allids[r.randint(len(allids))]
                if d not in posset and d not in negs:
                    negs.append(d)
            docs = pos + negs; labs = [1] * len(pos) + [0] * len(negs)
            out[_split_of(e["qid"])].append((e["qid"], e["question"], [ctext[d] for d in docs], labs, docs))
    return out


def to_pairs(rows):
    return [(q, dt, int(l)) for _, q, docs, labs, _ in rows for dt, l in zip(docs, labs)]


# ── model plumbing ────────────────────────────────────────────────────────────
def fresh():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(DEVICE)
    return model, tok


@torch.no_grad()
def ce_scores(model, tok, q, docs):
    model.eval(); out = []
    for i in range(0, len(docs), 64):
        ch = docs[i:i + 64]
        enc = tok([q] * len(ch), ch, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEVICE)
        out.extend(model(**enc).logits.squeeze(-1).float().cpu().tolist())
    return out


def eval_rows(model, tok, rows):
    return evaluate([(ce_scores(model, tok, q, docs), labs) for _, q, docs, labs, _ in rows if docs])


def train_on(model, tok, rows, epochs=EPOCHS):
    pairs = to_pairs(rows); labels = [float(p[2]) for p in pairs]
    n_pos = sum(labels); pw = torch.tensor([(len(labels) - n_pos) / max(1.0, n_pos)], device=DEVICE)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw); opt = torch.optim.AdamW(model.parameters(), lr=LR)
    idx = np.arange(len(pairs))
    for ep in range(epochs):
        model.train(); np.random.shuffle(idx)
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            qs = [pairs[j][0] for j in b]; ds = [pairs[j][1] for j in b]
            y = torch.tensor([labels[j] for j in b], device=DEVICE)
            enc = tok(qs, ds, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEVICE)
            loss = lossf(model(**enc).logits.squeeze(-1), y)
            opt.zero_grad(); loss.backward(); opt.step()


def carrier_names(model):
    return [n for n, p in model.named_parameters()
            if "bert.encoder.layer." in n and n.endswith(".weight") and p.ndim == 2]


# ── Diagnostic 1: relevance-gradient direction per pole ───────────────────────
def mean_grad_dir(model, tok, pairs, carriers, n, relabel=False):
    """Mean SIGNED relevance gradient (the direction the loss pushes theta), per carrier, over n balanced pairs.
    relabel=True shuffles labels (null)."""
    pmap = dict(model.named_parameters()); lossf = nn.BCEWithLogitsLoss()
    pos = [p for p in pairs if p[2] == 1]; neg = [p for p in pairs if p[2] == 0]
    m = min(len(pos), len(neg), n // 2)
    sub = pos[:m] + neg[:m]
    labs = [p[2] for p in sub]
    if relabel:
        labs = list(RNG.permutation(labs))
    acc = {c: torch.zeros(pmap[c].numel(), dtype=torch.float32) for c in carriers}
    model.eval()
    for (q, d, _), y in zip(sub, labs):
        model.zero_grad(set_to_none=True)
        enc = tok([q], [d], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEVICE)
        lossf(model(**enc).logits.squeeze(-1), torch.tensor([float(y)], device=DEVICE)).backward()
        for c in carriers:
            acc[c] += pmap[c].grad.detach().flatten().to("cpu", torch.float32)
    return {c: (acc[c] / max(1, len(sub))) for c in carriers}


def delta_theta(base_sd, model, carriers):
    """Fine-tuned update direction Δθ = θ_after − θ_base, per carrier (cpu float32 flat)."""
    cur = dict(model.named_parameters())
    return {c: (cur[c].detach().flatten().to("cpu", torch.float32) - base_sd[c]) for c in carriers}


def cos_per_carrier(A, B, carriers):
    per = {}
    for c in carriers:
        a, b = A[c], B[c]
        den = (a.norm() * b.norm()).item() + 1e-30
        per[c] = float((a @ b).item() / den)
    va = torch.cat([A[c] for c in carriers]); vb = torch.cat([B[c] for c in carriers])
    pooled = float((va @ vb).item() / ((va.norm() * vb.norm()).item() + 1e-30))
    return pooled, per


# ── Diagnostic 2: Fisher basis + surge overlap (from Phase-1) ─────────────────
def fisher_basis(model, tok, pairs, carriers):
    lossf = nn.BCEWithLogitsLoss(); pmap = dict(model.named_parameters()); grads = {c: [] for c in carriers}
    model.eval()
    for q, d, y in pairs[:N_FISHER]:
        model.zero_grad(set_to_none=True)
        enc = tok([q], [d], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEVICE)
        lossf(model(**enc).logits.squeeze(-1), torch.tensor([float(y)], device=DEVICE)).backward()
        for c in carriers:
            grads[c].append(pmap[c].grad.detach().flatten().to("cpu", torch.float32))
    V, Vswap = {}, {}; gen = torch.Generator(device="cpu").manual_seed(SWAP_SEED)
    for c in carriers:
        G = torch.stack(grads[c], 0).to(DEVICE)
        _, _, Vh = torch.linalg.svd(G, full_matrices=False); k = min(K, Vh.shape[0])
        V[c] = Vh[:k].detach().to("cpu", torch.float16)
        dd = G.shape[1]; Rr = torch.randn(dd, k, generator=gen); Q, _ = torch.linalg.qr(Rr)
        Vswap[c] = Q.t().contiguous().to(torch.float16)
        del G, grads[c]
        if DEVICE == "cuda": torch.cuda.empty_cache()
    return V, Vswap


def inproj(Vc, g):
    gn = float(g.dot(g)) + 1e-30; p = Vc.to(torch.float32) @ g
    return float(p.dot(p)) / gn


def surge_overlap(model, tok, pairs, carriers, V, Vswap):
    pmap = dict(model.named_parameters()); lossf = nn.BCEWithLogitsLoss(); iv, isw = [], []
    model.eval()
    for q, d, y in pairs[:N_FISHER]:
        model.zero_grad(set_to_none=True)
        enc = tok([q], [d], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEVICE)
        lossf(model(**enc).logits.squeeze(-1), torch.tensor([float(y)], device=DEVICE)).backward()
        g = {c: pmap[c].grad.detach().flatten().to("cpu", torch.float32) for c in carriers}
        iv.append(np.mean([inproj(V[c], g[c]) for c in carriers]))
        isw.append(np.mean([inproj(Vswap[c], g[c]) for c in carriers]))
    return {"inV": float(np.mean(iv)), "inSwap": float(np.mean(isw))}


def rep(r):
    return {"auc": r.auc, "recall_at_5": r.recall_at_5, "recall_at_3": r.recall_at_3}


# ── Diagnostic 3: keeper intrusion ────────────────────────────────────────────
def keeper_chunks(ctext, pole_map):
    soc_gk = [d for d in ctext if pole_map[d] == "rules" and "law" in d.lower()
              and ("keeper" in ctext[d].lower())]
    fan_keep = [d for d in ctext if pole_map[d] == "fantasy" and "keeper" in d.lower()]
    return soc_gk, fan_keep


def intrusion(model, tok, rows, target_pole, intruders, ctext, pole_map, limit=40):
    """For queries whose TRUE positive is in target_pole, build candidates = true positive + all cross-pole
    `intruders` + a few same-pole negs; report fraction where an intruder outranks the true positive, and mean
    best-intruder rank."""
    beat = 0; n = 0; ranks = []
    for qid, q, docs, labs, dids in rows:
        pos = [d for d, y in zip(dids, labs) if y == 1 and pole_map.get(d) == target_pole]
        if not pos:
            continue
        cand = list(dict.fromkeys(pos[:1] + list(intruders)))
        if len(cand) < 2:
            continue
        sc = ce_scores(model, tok, q, [ctext[d] for d in cand])
        order = np.argsort(sc)[::-1]
        rank_pos = int(np.where(order == 0)[0][0])          # rank of the true positive (0 = top)
        ranks.append(rank_pos)
        if rank_pos > 0:                                     # an intruder outranked it
            beat += 1
        n += 1
        if n >= limit:
            break
    return {"n": n, "intruder_beats_positive_rate": (beat / n if n else None),
            "mean_true_positive_rank": (float(np.mean(ranks)) if ranks else None)}


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ctext, pole_map = load_ctext()
    print(f"[data] combined corpus: {len(ctext)} chunks "
          f"({sum(v=='rules' for v in pole_map.values())} rules / {sum(v=='fantasy' for v in pole_map.values())} fantasy) dev={DEVICE}", flush=True)
    rules = build_rows(["task_rules_soccer.jsonl", "task_rules_nba.jsonl", "task_rules_nfl.jsonl"], ctext, "rules", pole_map)
    fant = build_rows(["task_a_fantasy.jsonl"], ctext, "fantasy", pole_map)
    print(f"[data] RULES rows: train={len(rules['train'])} ret={len(rules['retention'])} | "
          f"FANTASY rows: train={len(fant['train'])} ret={len(fant['retention'])}", flush=True)

    result = {"config": {"base": BASE, "K": K, "epochs": EPOCHS, "N_DIR": N_DIR, "neg_per_pos": NEG_PER_POS,
                         "smoke": SMOKE, "device": DEVICE, "prereg_lock": "b52e2928"}}

    m0, tok = fresh(); carriers = carrier_names(m0)
    base_sd = {c: dict(m0.named_parameters())[c].detach().flatten().to("cpu", torch.float32).clone() for c in carriers}

    # ── Diagnostic 1a: base gradient-direction cosine (+ within-pole consistency + null) ──
    print("\n=== D1: base relevance-gradient direction cosine ===", flush=True)
    rp = to_pairs(rules["train"]); fp = to_pairs(fant["train"])
    g_r = mean_grad_dir(m0, tok, rp, carriers, N_DIR)
    g_f = mean_grad_dir(m0, tok, fp, carriers, N_DIR)
    g_r_a = mean_grad_dir(m0, tok, rp[::2], carriers, N_DIR)
    g_r_b = mean_grad_dir(m0, tok, rp[1::2], carriers, N_DIR)
    g_f_a = mean_grad_dir(m0, tok, fp[::2], carriers, N_DIR)
    g_f_b = mean_grad_dir(m0, tok, fp[1::2], carriers, N_DIR)
    g_null = mean_grad_dir(m0, tok, rp, carriers, N_DIR, relabel=True)
    cross_g, _ = cos_per_carrier(g_r, g_f, carriers)
    within_r, _ = cos_per_carrier(g_r_a, g_r_b, carriers)
    within_f, _ = cos_per_carrier(g_f_a, g_f_b, carriers)
    null_g, _ = cos_per_carrier(g_r, g_null, carriers)
    print(f"  cross-pole cos(rules,fantasy)  = {cross_g:+.3f}", flush=True)
    print(f"  within-pole cos(rules_a,rules_b)= {within_r:+.3f}   within(fantasy)= {within_f:+.3f}  (the reliability scale)", flush=True)
    print(f"  null cos(rules, rules-relabel)  = {null_g:+.3f}", flush=True)
    result["D1_base_gradient"] = {"cross": cross_g, "within_rules": within_r, "within_fantasy": within_f, "null": null_g}

    # ── Diagnostic 2 + Δθ: CF surge both orders ──
    def surge(order):
        A, B = order
        rowsA = rules if A == "rules" else fant
        rowsB = rules if B == "rules" else fant
        m, tk = fresh()
        train_on(m, tk, rowsA["train"])
        retA_solo = rep(eval_rows(m, tk, rowsA["retention"]))
        dth = delta_theta(base_sd, m, carriers)                       # Δθ for pole A (from base)
        V, Vswap = fisher_basis(m, tk, to_pairs(rowsA["train"]), carriers)
        ov = surge_overlap(m, tk, to_pairs(rowsB["train"]), carriers, V, Vswap)   # B's surge on A's curvature
        train_on(m, tk, rowsB["train"])                               # continue-train on B
        retA_after = rep(eval_rows(m, tk, rowsA["retention"]))
        retB_after = rep(eval_rows(m, tk, rowsB["retention"]))
        eros = retA_solo["auc"] - retA_after["auc"]
        print(f"  [{A}->+{B}] {A} solo AUC={retA_solo['auc']:.3f} -> after +{B}={retA_after['auc']:.3f} "
              f"(erosion {eros:+.3f}) | surge inV={ov['inV']:.3f}/swap={ov['inSwap']:.6f} | {B} AUC={retB_after['auc']:.3f}", flush=True)
        return {"prior": A, "added": B, "prior_solo": retA_solo, "prior_after": retA_after,
                "added_after": retB_after, "erosion_auc": eros, "surge_overlap": ov}, dth

    print("\n=== D2: CF surge (both orders) + dtheta ===", flush=True)
    rf, dth_rules = surge(("rules", "fantasy"))
    fr, dth_fant = surge(("fantasy", "rules"))
    cross_dth, per_dth = cos_per_carrier(dth_rules, dth_fant, carriers)
    print(f"  dtheta cross-pole cos(dtheta_rules, dtheta_fantasy) = {cross_dth:+.3f}   (the faithful update-direction analog)", flush=True)
    result["D2_surge"] = {"rules_then_fantasy": rf, "fantasy_then_rules": fr}
    result["D1_delta_theta"] = {"cross": cross_dth, "per_carrier_min": min(per_dth.values()), "per_carrier_max": max(per_dth.values())}

    # ── Diagnostic 3: keeper intrusion (after JOINT) ──
    print("\n=== D3: keeper intrusion (joint model) ===", flush=True)
    soc_gk, fan_keep = keeper_chunks(ctext, pole_map)
    print(f"  soccer-goalkeeper chunks={len(soc_gk)} | fantasy-keeper chunks={len(fan_keep)}", flush=True)
    jm, jtok = fresh(); train_on(jm, jtok, rules["train"] + fant["train"])
    # do fantasy-keeper chunks intrude on soccer-goalkeeper (rules) queries?
    intr_on_rules = intrusion(jm, jtok, rules["retention"] + rules["train"], "rules", fan_keep, ctext, pole_map)
    # do soccer-goalkeeper chunks intrude on fantasy-keeper queries?
    intr_on_fant = intrusion(jm, jtok, fant["retention"] + fant["train"], "fantasy", soc_gk, ctext, pole_map)
    print(f"  fantasy-keeper intruding on rules queries : beats_positive={intr_on_rules['intruder_beats_positive_rate']} (n={intr_on_rules['n']})", flush=True)
    print(f"  rules-goalkeeper intruding on fantasy Qs   : beats_positive={intr_on_fant['intruder_beats_positive_rate']} (n={intr_on_fant['n']})", flush=True)
    result["D3_keeper_intrusion"] = {"n_soccer_gk": len(soc_gk), "n_fantasy_keeper": len(fan_keep),
                                     "fantasy_keeper_on_rules": intr_on_rules, "rules_gk_on_fantasy": intr_on_fant}

    # ── DECISION ──
    contested = (min(cross_g, cross_dth) < -0.10) or \
                (max(rf["erosion_auc"], fr["erosion_auc"]) > 0.02 and
                 (rf["surge_overlap"]["inV"] > 10 * max(rf["surge_overlap"]["inSwap"], 1e-9) or
                  fr["surge_overlap"]["inV"] > 10 * max(fr["surge_overlap"]["inSwap"], 1e-9)))
    verdict = "CONTESTED -> ROUTE to separate poles" if contested else "CONSTRAINED/NOT-CONTESTED -> HARD-NEG fix (no routing)"
    print("\n" + "=" * 72)
    print(f"DECISION: {verdict}")
    print(f"  cross-pole cos: base_grad={cross_g:+.3f}  dtheta={cross_dth:+.3f}  (contested if < -0.10)")
    print(f"  max erosion: {max(rf['erosion_auc'], fr['erosion_auc']):+.3f} AUC  (contested if > 0.02 w/ directional inV)")
    result["DECISION"] = {"contested": bool(contested), "verdict": verdict}

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
