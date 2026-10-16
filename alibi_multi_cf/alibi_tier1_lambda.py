"""
alibi_tier1_lambda.py — Thread #2 real-text Tier-1: does the ALiBi x multilingual CONJUNCTION flip the CF childhood-protection
sign in from-scratch small LMs trained on REAL text? Runs the full 2x2 = {ALiBi,RoPE} x {mono(en), multi(en+de+es+ru+hi+ar)} on
ONE instance and writes a 2x2 protection matrix.

Faithful to probe13's protection sign: PROTECTION = retA_childhood - retA_nochild on a downstream A->B interference (task-A=SST-2
sentiment, task-B=AG-News topic). NEG = childhood erodes (BLOOM signature). Shared BPE tokenizer across all cells (only pretrain
DATA + PE differ). Non-saturation guard: phase-B steps tuned so retA_noc lands in a partial-forgetting band.

MODE=smoke -> tiny (1 cell, tiny model/data/steps) for local plumbing validation.
Env: MODE(full|smoke), OUT(json path). Writes gateA/alibi_tier1.json.
"""
from __future__ import annotations
import os, sys, json, time, math, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("MODE", "full") == "smoke"
OUT = os.environ.get("OUT", ("gateA/alibi_tier1_smoke.json" if SMOKE else "gateA/alibi_tier1.json"))
torch.manual_seed(0); np.random.seed(0); random.seed(0)

# model / training config (env-overridable so a medium local competence check + the full Lambda run share one script)
def _i(k, d): return int(os.environ.get(k, d))
def _f(k, d): return float(os.environ.get(k, d))
D_MODEL = _i("D_MODEL", 128 if SMOKE else 384)
N_LAYERS = _i("N_LAYERS", 2 if SMOKE else 6)
N_HEADS = _i("N_HEADS", 4 if SMOKE else 6)
SEQ = _i("SEQ", 64 if SMOKE else 256)
VOCAB = _i("VOCAB", 4000 if SMOKE else 16000)
PRETRAIN_STEPS = _i("PRETRAIN_STEPS", 60 if SMOKE else 6000)
PRETRAIN_BATCH = _i("PRETRAIN_BATCH", 16 if SMOKE else 32)
PT_CHARS = _i("PT_CHARS", 200_000 if SMOKE else 40_000_000)   # chars of corpus per language bucket
LR_PT = _f("LR_PT", 3e-4)
DS_STEPS = _i("DS_STEPS", 40 if SMOKE else 300)               # per downstream train phase
CHILD_STEPS = _i("CHILD_STEPS", 20 if SMOKE else 150)
LR_DS = _f("LR_DS", 1e-4)                                     # from-scratch small model: higher than pretrained-FT 2e-5
DS_N = _i("DS_N", 400 if SMOKE else 3000)                     # examples per downstream task
MULTI_LANGS = (["en", "de", "es"] if SMOKE else ["en", "de", "es", "ru", "hi", "ar"])
_ONECELL = os.environ.get("ONECELL", "")                     # e.g. "alibi_multi" for a single-cell competence check
if _ONECELL:
    _pe, _lg = _ONECELL.split("_"); CELLS = [(_pe, _lg)]
elif SMOKE:
    CELLS = [("alibi", "multi")]
else:
    CELLS = [("alibi", "mono"), ("alibi", "multi"), ("rope", "mono"), ("rope", "multi")]


# ── corpus (HF wikipedia streaming; fallback wikitext) ────────────────────────
def get_text(langs, n_chars):
    from datasets import load_dataset
    buf = []; per = n_chars // len(langs)
    for lg in langs:
        got = 0
        try:
            ds = load_dataset("wikimedia/wikipedia", f"20231101.{lg}", split="train", streaming=True)
            for ex in ds:
                t = ex.get("text", "")
                if len(t) < 200: continue
                buf.append(t); got += len(t)
                if got >= per: break
        except Exception as e:
            print(f"[corpus] {lg} failed ({str(e)[:80]}); fallback wikitext", flush=True)
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
            for ex in ds:
                t = ex.get("text", "")
                if len(t) < 200: continue
                buf.append(t); got += len(t)
                if got >= per: break
        print(f"[corpus] {lg}: {got/1e6:.1f}M chars", flush=True)
    random.shuffle(buf); return "\n".join(buf)


def train_tokenizer(text, vocab):
    from tokenizers import ByteLevelBPETokenizer
    import tempfile
    p = tempfile.mktemp(suffix=".txt"); open(p, "w", encoding="utf-8").write(text[:50_000_000])
    tok = ByteLevelBPETokenizer(); tok.train([p], vocab_size=vocab, min_frequency=2,
                                             special_tokens=["<pad>", "<s>", "</s>", "<unk>"])
    return tok


def encode_ids(tok, text, seq):
    ids = tok.encode(text).ids
    n = (len(ids) // seq) * seq
    return torch.tensor(ids[:n], dtype=torch.long).view(-1, seq)


# ── model: GPT-ish, switchable ALiBi/RoPE ─────────────────────────────────────
def alibi_slopes(h): return torch.tensor([2 ** (-8.0 * (i + 1) / h) for i in range(h)], device=DEV)
def rope_cache(dh, L):
    th = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=DEV).float() / dh))
    a = torch.outer(torch.arange(L, device=DEV).float(), th); return torch.cos(a), torch.sin(a)
def apply_rope(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]; c = cos[None, None]; s = sin[None, None]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Attn(nn.Module):
    def __init__(s, d, h, pe):
        super().__init__(); s.h = h; s.dh = d // h; s.pe = pe; s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d)
        if pe == "alibi": s.register_buffer("sl", alibi_slopes(h), persistent=False)
    def forward(s, x, rope):
        B, L, _ = x.shape
        qkv = s.qkv(x).view(B, L, 3, s.h, s.dh).permute(2, 0, 3, 1, 4); q, k, v = qkv[0], qkv[1], qkv[2]
        if s.pe == "rope":
            cos, sin = rope; q = apply_rope(q, cos[:L], sin[:L]); k = apply_rope(k, cos[:L], sin[:L])
        att = (q @ k.transpose(-2, -1)) / math.sqrt(s.dh)
        pos = torch.arange(L, device=x.device); causal = pos[None, :] > pos[:, None]
        if s.pe == "alibi":
            dist = (pos[None, :] - pos[:, None]).float(); att = att + s.sl[None, :, None, None] * dist[None, None]
        att = att.masked_fill(causal[None, None], float("-inf")); att = Fun.softmax(att, -1)
        return s.o((att @ v).transpose(1, 2).reshape(B, L, -1))


class Blk(nn.Module):
    def __init__(s, d, h, pe):
        super().__init__(); s.ln1 = nn.LayerNorm(d); s.at = Attn(d, h, pe); s.ln2 = nn.LayerNorm(d)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x, r): x = x + s.at(s.ln1(x), r); return x + s.mlp(s.ln2(x))


class GPT(nn.Module):
    def __init__(s, pe, ncls=0):
        super().__init__(); s.pe = pe; s.emb = nn.Embedding(VOCAB, D_MODEL)
        s.blocks = nn.ModuleList([Blk(D_MODEL, N_HEADS, pe) for _ in range(N_LAYERS)])
        s.lnf = nn.LayerNorm(D_MODEL); s.lm = nn.Linear(D_MODEL, VOCAB)
        s.cls = nn.Linear(D_MODEL, ncls) if ncls else None
        s.rope = rope_cache(D_MODEL // N_HEADS, SEQ) if pe == "rope" else None
    def hidden(s, x):
        h = s.emb(x)
        for b in s.blocks: h = b(h, s.rope)
        return s.lnf(h)
    def forward_lm(s, x): return s.lm(s.hidden(x))
    def forward_cls(s, x, mask):
        h = s.hidden(x); pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return s.cls(pooled)


def pretrain(model, data, steps, bs, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr); model.train(); n = data.shape[0]
    for st in range(steps):
        idx = torch.randint(0, n, (bs,)); x = data[idx].to(DEV)
        logits = model.forward_lm(x[:, :-1]); loss = Fun.cross_entropy(logits.reshape(-1, VOCAB), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if st % max(1, steps // 4) == 0: print(f"    pretrain {st}/{steps} loss={float(loss):.3f}", flush=True)
    return model


# ── downstream classification (fine-tune LM+cls head) ─────────────────────────
def load_task(name, tok, n):
    from datasets import load_dataset
    if name == "sst2":
        ds = load_dataset("stanfordnlp/sst2", split="train"); txt, lab = ds["sentence"][:n], ds["label"][:n]; ncls = 2
    else:
        ds = load_dataset("fancyzhx/ag_news", split="train"); txt, lab = ds["text"][:n], ds["label"][:n]; ncls = 4
    X = torch.zeros(len(txt), SEQ, dtype=torch.long); M = torch.zeros(len(txt), SEQ)
    for i, t in enumerate(txt):
        ids = tok.encode(t).ids[:SEQ]; X[i, :len(ids)] = torch.tensor(ids); M[i, :len(ids)] = 1
    return X, M, torch.tensor(lab), ncls


def set_cls(model, ncls):
    model.cls = nn.Linear(D_MODEL, ncls).to(DEV); return model


def ft(model, X, M, y, steps, lr, bs=32):
    opt = torch.optim.AdamW(model.parameters(), lr=lr); model.train(); n = len(y)
    for st in range(steps):
        idx = torch.randint(0, n, (bs,)); logits = model.forward_cls(X[idx].to(DEV), M[idx].to(DEV))
        loss = Fun.cross_entropy(logits, y[idx].to(DEV))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model


@torch.no_grad()
def acc(model, X, M, y, ncls):
    model.eval(); model.cls = model.cls if model.cls.out_features == ncls else model.cls
    preds = []
    for i in range(0, len(y), 128):
        preds.append(model.forward_cls(X[i:i+128].to(DEV), M[i:i+128].to(DEV)).argmax(-1).cpu())
    return float((torch.cat(preds) == y).float().mean())


def clone(model):
    m = GPT(model.pe, ncls=(model.cls.out_features if model.cls is not None else 0)).to(DEV)
    m.load_state_dict(model.state_dict()); return m


def me(x):
    x = np.array(x); return [float(x.mean()), float(x.std(ddof=1) / max(len(x), 1) ** 0.5) if len(x) > 1 else 0.0]


def main():
    t0 = time.time()
    print(f"[TIER1] dev={DEV} smoke={SMOKE} cells={CELLS} d={D_MODEL} L={N_LAYERS} vocab={VOCAB}", flush=True)
    # shared tokenizer from the MULTI mix (so tokenization is identical across cells; only pretrain data differs)
    tok_text = get_text(MULTI_LANGS, PT_CHARS if SMOKE else 8_000_000)
    tok = train_tokenizer(tok_text, VOCAB); print(f"[TIER1] tokenizer trained ({time.time()-t0:.0f}s)", flush=True)

    # downstream data (shared, tokenized once)
    Xa, Ma, ya, ncls_a = load_task("sst2", tok, DS_N)      # task-A = SST-2
    Xb, Mb, yb, ncls_b = load_task("agnews", tok, DS_N)    # task-B = AG-News
    # fixed retention/test split for task-A
    na = len(ya); perm = torch.randperm(na); tr_a, te_a = perm[:int(na*0.8)], perm[int(na*0.8):]
    print(f"[TIER1] downstream loaded: SST={na} AGNews={len(yb)} ({time.time()-t0:.0f}s)", flush=True)

    # corpora per lingual setting (mono reuses first slice; multi uses the mix) — SAME token budget
    corpora = {}
    corpora["mono"] = encode_ids(tok, get_text(["en"], PT_CHARS), SEQ)
    corpora["multi"] = encode_ids(tok, tok_text if SMOKE else get_text(MULTI_LANGS, PT_CHARS), SEQ)
    print(f"[TIER1] corpora: mono={corpora['mono'].shape} multi={corpora['multi'].shape} ({time.time()-t0:.0f}s)", flush=True)

    def protection_for(pe, lingual, phaseb, seed=0):
        torch.manual_seed(seed); np.random.seed(seed)
        base = GPT(pe).to(DEV)
        pretrain(base, corpora[lingual], PRETRAIN_STEPS, PRETRAIN_BATCH, LR_PT)
        set_cls(base, ncls_a)
        ft(base, Xa[tr_a], Ma[tr_a], ya[tr_a], DS_STEPS, LR_DS)      # learn task-A (SST)
        accA0 = acc(base, Xa[te_a], Ma[te_a], ya[te_a], ncls_a)
        base_state = {k: v.clone() for k, v in base.state_dict().items()}
        # no-childhood: -> task-B -> retention A
        noc = clone(base); set_cls(noc, ncls_b); ft(noc, Xb, Mb, yb, phaseb, LR_DS)
        set_cls_back(noc, base_state, ncls_a); retA_noc = acc(noc, Xa[te_a], Ma[te_a], ya[te_a], ncls_a)
        # childhood: extra task-A -> task-B -> retention A
        ch = clone(base); ft(ch, Xa[tr_a], Ma[tr_a], ya[tr_a], CHILD_STEPS, LR_DS)
        set_cls(ch, ncls_b); ft(ch, Xb, Mb, yb, phaseb, LR_DS)
        set_cls_back(ch, None, ncls_a); retA_ch = acc(ch, Xa[te_a], Ma[te_a], ya[te_a], ncls_a)
        return accA0, retA_noc, retA_ch

    def set_cls_back(model, _state, ncls):
        # restore an A-head classifier trained on the model's current features (re-probe A after interference)
        set_cls(model, ncls); ft(model, Xa[tr_a], Ma[tr_a], ya[tr_a], 40, LR_DS)

    # non-saturation tuner on the first cell: pick phaseB so retA_noc in [0.55,0.80] (SST chance=0.5)
    _fixed = os.environ.get("PHASEB", "")
    if _fixed:
        chosen = int(_fixed)
        a0, rnoc, rch = protection_for(*CELLS[0], chosen)
        print(f"[TIER1] PHASEB fixed={chosen}  (cell0 accA0={a0:.3f} retA_noc={rnoc:.3f} retA_ch={rch:.3f})", flush=True)
    else:
        pe0, lg0 = CELLS[0]; chosen = None
        for pb in ([DS_STEPS] if SMOKE else [80, 150, 250, 400]):
            a0, rnoc, _ = protection_for(pe0, lg0, pb)
            print(f"[TIER1] tuner phaseB={pb} accA0={a0:.3f} retA_noc={rnoc:.3f}", flush=True)
            if 0.55 <= rnoc <= 0.80 and chosen is None: chosen = pb
        chosen = chosen or (DS_STEPS if SMOKE else 250)
    print(f"[TIER1] chosen phaseB={chosen}", flush=True)

    cells = {}
    for pe, lg in CELLS:
        runs = [protection_for(pe, lg, chosen, seed=si) for si in range(1 if SMOKE else 2)]
        prot = me([r[2] - r[1] for r in runs])
        cells[f"{pe}_{lg}"] = {"accA0": me([r[0] for r in runs]), "retA_noc": me([r[1] for r in runs]),
                               "retA_ch": me([r[2] for r in runs]), "protection": prot}
        c = cells[f"{pe}_{lg}"]
        print(f"[TIER1] {pe:5} {lg:5}: accA0={c['accA0'][0]:.3f} retA_noc={c['retA_noc'][0]:.3f} "
              f"retA_ch={c['retA_ch'][0]:.3f} PROTECTION={prot[0]:+.3f}±{prot[1]:.3f}", flush=True)

    conj = None
    others = {k: v["protection"] for k, v in cells.items() if k != "alibi_multi"}
    if "alibi_multi" in cells and others:
        am = cells["alibi_multi"]["protection"]
        pse = (am[1] ** 2 + max(o[1] for o in others.values()) ** 2) ** 0.5
        margin = min(o[0] for o in others.values()) - am[0]
        conj = bool(am[0] < -0.02 and margin > 2 * pse)
        print(f"\n[TIER1] ALiBi×multi protection={am[0]:+.3f}±{am[1]:.3f}  others_min={min(o[0] for o in others.values()):+.3f} "
              f"margin={margin:+.3f} (2·SE={2*pse:.3f})  CONJUNCTION={conj}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump({"config": {"d": D_MODEL, "L": N_LAYERS, "vocab": VOCAB, "langs": MULTI_LANGS, "smoke": SMOKE,
                          "phaseb": chosen}, "cells": cells, "conjunction": conj},
              open(OUT, "w"), indent=1)
    print(f"[TIER1] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
