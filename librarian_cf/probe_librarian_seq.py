"""
probe_librarian_seq.py — PHASE-1: sequential-domain-surge CF probe on the Librarian reranker (SANDBOXED).

Question: as you SEQUENTIALLY add a sport's rules to the fine-tuned cross-encoder, does it ERODE retrieval on the
sports already there — and is the erosion DIRECTIONAL (lands in the prior sport's Fisher curvature)? And does JOINT
training (production's approach) avoid it?

ISOLATION (hard constraint): reads NOTHING from D:/SportsThought; all inputs were copied into this sandbox; writes ONLY
to ./out/. Never touches the production Librarian, its checkpoints, or the Fly app. This is the sequential COUNTERFACTUAL
that production (joint re-train) deliberately avoids.

Task = rules-only reranking. ctext = all rules chunk bodies (soccer+nba+nfl). Each sport's questions keep the
rules-resolvable candidates; negatives include OTHER sports' chunks -> cross-sport interference is measured directly.
Metric = AUC / R@5 (NEVER P@1 — saturates). Reuses the D:/IC_experiments discipline: per-(layer,carrier) Fisher V +
mandatory random-SWAP control, inV vs inSwap.

Run:  python probe_librarian_seq.py --smoke   # tiny plumbing check (few Qs, 1 epoch, K=16)
      python probe_librarian_seq.py           # full
"""
from __future__ import annotations
import json, sys, hashlib
from pathlib import Path
import numpy as np, torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_metrics import evaluate

SMOKE = "--smoke" in sys.argv
SEQ = ["soccer", "nba", "nfl"]                 # ordered add sequence
BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAXLEN = 256
EPOCHS = 1 if SMOKE else 4
BATCH = 8 if SMOKE else 32
LR = 2e-5
K = 16 if SMOKE else 64                         # Fisher curvature-subspace rank
N_FISHER = 12 if SMOKE else 120                 # pairs for the Fisher basis
RETENTION_FRAC = 0.25                            # per-sport held-out retention slice (deterministic)
SWAP_SEED = 12345
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = HERE / "out" / ("seq_result_smoke.json" if SMOKE else "seq_result.json")
torch.manual_seed(0); np.random.seed(0)


# ── data ─────────────────────────────────────────────────────────────────────
def load_ctext():
    ctext = {}
    for s in SEQ:
        for l in (HERE / "corpus" / f"rules_{s}.jsonl").open(encoding="utf-8"):
            r = json.loads(l)
            ctext[r["doc_id"]] = ((r.get("title", "") or "") + ". " + (r.get("body", "") or "")).strip()[:2000]
    return ctext


def _split_of(qid, frac=RETENTION_FRAC):
    """Deterministic per-question train/retention assignment (hash → stable, no leakage)."""
    h = int(hashlib.md5(qid.encode()).hexdigest(), 16) % 1000
    return "retention" if h < int(frac * 1000) else "train"


def sport_rows(sport, ctext):
    """Return {'train':[...], 'retention':[...]} of (qid, question, [doctext], [label]) with rules-resolvable candidates."""
    out = {"train": [], "retention": []}
    for l in (HERE / "data" / f"task_rules_{sport}.jsonl").open(encoding="utf-8"):
        e = json.loads(l)
        cand = [(c["doc_id"], int(c["relevant"])) for c in e["candidates"]
                if str(c["relevant"]) in ("0", "1") and c["doc_id"] in ctext]
        pos = [d for d, y in cand if y == 1]; neg = [d for d, y in cand if y == 0]
        if not pos or not neg:
            continue
        docs = [ctext[d] for d, _ in cand]; labs = [y for _, y in cand]
        out[_split_of(e["qid"])].append((e["qid"], e["question"], docs, labs))
    return out


def to_pairs(rows):
    return [(q, d, int(l)) for _, q, docs, labs in rows for d, l in zip(docs, labs)]


# ── model plumbing (reranker) ─────────────────────────────────────────────────
@torch.no_grad()
def ce_scores(model, tok, q, docs):
    model.eval(); out = []
    for i in range(0, len(docs), 64):
        ch = docs[i:i + 64]
        enc = tok([q] * len(ch), ch, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEVICE)
        out.extend(model(**enc).logits.squeeze(-1).float().cpu().tolist())
    return out


def eval_rows(model, tok, rows):
    return evaluate([(ce_scores(model, tok, q, docs), labs) for _, q, docs, labs in rows if docs])


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


# ── Fisher curvature basis V (+ swap control) for a sport, at the current weights ──
def carrier_names(model):
    return [n for n, p in model.named_parameters()
            if "bert.encoder.layer." in n and n.endswith(".weight") and p.ndim == 2]


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
        d = G.shape[1]; R = torch.randn(d, k, generator=gen); Q, _ = torch.linalg.qr(R)
        Vswap[c] = Q.t().contiguous().to(torch.float16)
        del G, grads[c]
        if DEVICE == "cuda": torch.cuda.empty_cache()
    return V, Vswap


def inproj(Vc, g):
    gn = float(g.dot(g)) + 1e-30; p = Vc.to(torch.float32) @ g
    return float(p.dot(p)) / gn


def surge_overlap(model, tok, rows, carriers, prior_V):
    """Mean inV vs inSwap of this sport's training grads projected onto EACH prior sport's Fisher basis."""
    pmap = dict(model.named_parameters()); pairs = to_pairs(rows)[:N_FISHER]
    lossf = nn.BCEWithLogitsLoss(); acc = {s: {"inV": [], "inSwap": []} for s in prior_V}
    model.eval()
    for q, d, y in pairs:
        model.zero_grad(set_to_none=True)
        enc = tok([q], [d], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEVICE)
        lossf(model(**enc).logits.squeeze(-1), torch.tensor([float(y)], device=DEVICE)).backward()
        for s, (V, Vswap) in prior_V.items():
            iv = np.mean([inproj(V[c], pmap[c].grad.detach().flatten().to("cpu", torch.float32)) for c in carriers])
            isw = np.mean([inproj(Vswap[c], pmap[c].grad.detach().flatten().to("cpu", torch.float32)) for c in carriers])
            acc[s]["inV"].append(iv); acc[s]["inSwap"].append(isw)
    return {s: {"inV": float(np.mean(a["inV"])), "inSwap": float(np.mean(a["inSwap"]))} for s, a in acc.items()}


def fresh():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=1).to(DEVICE)
    return model, tok


def rep(r):
    return {"auc": r.auc, "recall_at_5": r.recall_at_5, "recall_at_3": r.recall_at_3}


def main():
    ctext = load_ctext()
    data = {s: sport_rows(s, ctext) for s in SEQ}
    for s in SEQ:
        print(f"[data] {s}: train={len(data[s]['train'])} retention={len(data[s]['retention'])}", flush=True)

    # ══ ARM 1: SEQUENTIAL surge ══════════════════════════════════════════════
    print(f"\n[SEQ] sequential add: {' -> '.join(SEQ)} (dev={DEVICE} K={K} epochs={EPOCHS})", flush=True)
    model, tok = fresh(); carriers = carrier_names(model)
    seq_log = []; prior_V = {}          # {sport: (V, Vswap)} accumulated as sports are learned
    for i, sport in enumerate(SEQ):
        # instrument the surge overlap of THIS sport's grads onto already-learned sports (before training on it)
        overlap = surge_overlap(model, tok, data[sport]["train"], carriers, prior_V) if prior_V else {}
        train_on(model, tok, data[sport]["train"])
        # retention on THIS sport + all prior sports (after learning `sport`)
        ret = {s: rep(eval_rows(model, tok, data[s]["retention"])) for s in SEQ[:i + 1]}
        seq_log.append({"after_adding": sport, "retention": ret, "surge_overlap_on_prior": overlap})
        line = " | ".join(f"{s}:AUC={ret[s]['auc']:.3f}" for s in ret)
        ov = " | ".join(f"{s}:inV={o['inV']:.3f}/swap={o['inSwap']:.3f}" for s, o in overlap.items())
        print(f"  +{sport:6}  {line}    surge-overlap[{ov}]", flush=True)
        # snapshot this sport's Fisher basis for future surge-overlap probes
        prior_V[sport] = fisher_basis(model, tok, to_pairs(data[sport]["train"]), carriers)

    # ══ ARM 2: JOINT baseline (production's approach) ════════════════════════
    print(f"\n[JOINT] train all {len(SEQ)} sports together (production approach)", flush=True)
    jm, jtok = fresh()
    all_train = [r for s in SEQ for r in data[s]["train"]]
    train_on(jm, jtok, all_train)
    joint_ret = {s: rep(eval_rows(jm, jtok, data[s]["retention"])) for s in SEQ}
    print("  " + " | ".join(f"{s}:AUC={joint_ret[s]['auc']:.3f}" for s in SEQ), flush=True)

    # ══ readout: sequential final retention vs joint, per sport ══════════════
    seq_final = seq_log[-1]["retention"]
    print(f"\n{'='*72}\nPHASE-1 READOUT — retention AUC per sport (higher=less forgotten)")
    print(f"  {'sport':8} {'seq_final':>10} {'joint':>8} {'seq-joint':>10}")
    for s in SEQ:
        print(f"  {s:8} {seq_final[s]['auc']:>10.3f} {joint_ret[s]['auc']:>8.3f} {seq_final[s]['auc']-joint_ret[s]['auc']:>+10.3f}")

    OUT.write_text(json.dumps({
        "config": {"SEQ": SEQ, "base": BASE, "K": K, "epochs": EPOCHS, "lr": LR, "retention_frac": RETENTION_FRAC,
                   "smoke": SMOKE, "n_rules_chunks": len(ctext), "device": DEVICE},
        "sequential": seq_log, "joint_retention": joint_ret,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
