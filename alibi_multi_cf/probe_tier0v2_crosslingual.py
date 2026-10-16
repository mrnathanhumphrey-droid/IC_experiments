"""
probe_tier0v2_crosslingual.py — Thread #2 Tier-0 v2: cross-lingual interference + multilingual-BASE knob. ($0 local GPU)

Fixes v1's saturation: shared copy@2 rule across LANGUAGES (disjoint vocab blocks) so learning a NEW language only PARTIALLY erodes
an old one (dynamic range). 2x2 = {ALiBi,RoPE} x {mono-base(lang0 only), multi-base(langs0-3)}. task-A=copy@2 lang0 (protected),
task-B=copy@2 lang5 (new). PROTECTION = retA_childhood - retA_nochild; NEG = childhood erodes = BLOOM signature. Test: only ALiBi x
multi-base erodes?

Run:  python probe_tier0v2_crosslingual.py --smoke   # 1 seed tiny
      python probe_tier0v2_crosslingual.py           # full
"""
from __future__ import annotations
import sys, json, time, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
SMOKE = "--smoke" in sys.argv
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = HERE / "out" / ("tier0v2_crosslingual_smoke.json" if SMOKE else "tier0v2_crosslingual.json")
(HERE / "out").mkdir(parents=True, exist_ok=True)

D_MODEL, N_LAYERS, N_HEADS = 128, 3, 8
SEQ_LEN, BLOCK, K_LANG = 24, 32, 6
VOCAB = BLOCK * K_LANG            # 192
D_COPY = 2
LANG_A, LANG_B, BASE_MULTI = 0, 5, [0, 1, 2, 3]
MONO_BASE = 300 if SMOKE else 700
MULTI_BASE = 500 if SMOKE else 1600
CHILD_STEPS = 100 if SMOKE else 400
LR = 3e-3
SEEDS = [0] if SMOKE else [0, 1, 2, 3]


def gen_batch(D, langs, n, rng):
    x = np.zeros((n, SEQ_LEN), np.int64)
    for i in range(n):
        blk = langs[rng.randint(len(langs))]
        x[i] = rng.randint(blk * BLOCK, (blk + 1) * BLOCK, SEQ_LEN)
    xt = torch.from_numpy(x).to(DEV); y = xt.clone(); y[:, D:] = xt[:, :-D]; y[:, :D] = -100
    return xt, y


def alibi_slopes(h): return torch.tensor([2 ** (-8.0 * (i + 1) / h) for i in range(h)], device=DEV)
def rope_cache(dh, L):
    theta = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=DEV).float() / dh))
    ang = torch.outer(torch.arange(L, device=DEV).float(), theta); return torch.cos(ang), torch.sin(ang)
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
        if s.pe == "rope": q = apply_rope(q, *rope); k = apply_rope(k, *rope)
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


class LM(nn.Module):
    def __init__(s, pe):
        super().__init__(); s.pe = pe; s.emb = nn.Embedding(VOCAB, D_MODEL)
        s.blocks = nn.ModuleList([Blk(D_MODEL, N_HEADS, pe) for _ in range(N_LAYERS)])
        s.lnf = nn.LayerNorm(D_MODEL); s.head = nn.Linear(D_MODEL, VOCAB)
        s.rope = rope_cache(D_MODEL // N_HEADS, SEQ_LEN) if pe == "rope" else None
    def forward(s, x):
        h = s.emb(x)
        for b in s.blocks: h = b(h, s.rope)
        return s.head(s.lnf(h))


def train(m, langs, steps, lr, seed):
    opt = torch.optim.AdamW(m.parameters(), lr=lr); m.train(); rng = np.random.RandomState(seed + 777)
    for _ in range(steps):
        x, y = gen_batch(D_COPY, langs, 128, rng); lo = m(x)
        loss = Fun.cross_entropy(lo.view(-1, VOCAB), y.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return m


@torch.no_grad()
def acc(m, lang):
    x, y = gen_batch(D_COPY, [lang], 512, np.random.RandomState(999)); pred = m(x).argmax(-1); msk = y != -100
    return float((pred[msk] == y[msk]).float().mean())


def clone(m): n = LM(m.pe).to(DEV); n.load_state_dict(m.state_dict()); return n


def make_base(pe, mode, seed):
    torch.manual_seed(seed); np.random.seed(seed); m = LM(pe).to(DEV)
    if mode == "mono": train(m, [LANG_A], MONO_BASE, LR, seed)
    else: train(m, BASE_MULTI, MULTI_BASE, LR, seed)
    return m


def cell(pe, mode, phaseb, seed):
    base = make_base(pe, mode, seed); accA_base = acc(base, LANG_A)
    noc = clone(base); train(noc, [LANG_B], phaseb, LR, seed); retA_noc = acc(noc, LANG_A)
    ch = clone(base); train(ch, [LANG_A], CHILD_STEPS, LR, seed + 50); train(ch, [LANG_B], phaseb, LR, seed)
    retA_ch = acc(ch, LANG_A)
    return {"accA_base": accA_base, "retA_noc": retA_noc, "retA_ch": retA_ch, "protection": retA_ch - retA_noc}


def me(x):
    x = np.array(x); return [float(x.mean()), float(x.std(ddof=1) / max(len(x), 1) ** 0.5) if len(x) > 1 else 0.0]


def main():
    t0 = time.time()
    print(f"[TIER0v2] cross-lingual: task-A=copy@2 lang{LANG_A} (protected), task-B=lang{LANG_B} (new); "
          f"base mono=[{LANG_A}] vs multi={BASE_MULTI}  d={D_MODEL} dev={DEV}", flush=True)

    # non-saturation tuner: pick phase-B steps so retA_noc (multi-alibi, seed0) lands in [0.30,0.75]
    print("[TIER0v2] phase-B tuner (multi-alibi seed0): target retA_noc in [0.30,0.75]", flush=True)
    chosen = None
    for pb in ([40] if SMOKE else [20, 40, 70, 110, 170]):
        r = cell("alibi", "multi", pb, 0)["retA_noc"]
        print(f"    phaseB={pb:4d}  retA_noc={r:.3f}", flush=True)
        if 0.30 <= r <= 0.75 and chosen is None: chosen = pb
    if chosen is None:
        chosen = 70 if not SMOKE else 40
        print(f"    [guard] no phaseB gave retA_noc in target; using {chosen} (report as partial-limit)", flush=True)
    print(f"[TIER0v2] chosen phaseB={chosen}", flush=True)

    cells = {}
    for pe in ("alibi", "rope"):
        for mode in ("mono", "multi"):
            rs = [cell(pe, mode, chosen, s) for s in SEEDS]
            cells[f"{pe}_{mode}"] = {k: me([r[k] for r in rs]) for k in ("accA_base", "retA_noc", "retA_ch", "protection")}
            c = cells[f"{pe}_{mode}"]
            print(f"  {pe:5} {mode:5}: accA_base={c['accA_base'][0]:.3f} retA_noc={c['retA_noc'][0]:.3f} "
                  f"retA_ch={c['retA_ch'][0]:.3f} PROTECTION={c['protection'][0]:+.3f}±{c['protection'][1]:.3f}", flush=True)

    print("\n=== 2x2 PROTECTION (childhood - nochild; NEG = erodes = BLOOM signature) ===", flush=True)
    print(f"  {'':12}{'mono-base':>12}{'multi-base':>12}")
    for pe in ("alibi", "rope"):
        print(f"  {pe:12}" + "".join(f"{cells[f'{pe}_{m}']['protection'][0]:+8.3f}    " for m in ("mono", "multi")), flush=True)

    am = cells["alibi_multi"]["protection"]; others = {k: cells[k]["protection"] for k in cells if k != "alibi_multi"}
    pooled_se = (am[1] ** 2 + max(o[1] for o in others.values()) ** 2) ** 0.5
    margin = min(o[0] for o in others.values()) - am[0]
    conj = am[0] < -0.02 and margin > 2 * pooled_se
    interp = ("CONJUNCTION REPRODUCES -> fire Lambda Tier-1" if conj else
              "NOT a clean conjunction (single-knob or null) -> Lambda a judgment call")
    print(f"\n  ALiBi×multi-base protection = {am[0]:+.3f}±{am[1]:.3f}", flush=True)
    print(f"  others min = {min(o[0] for o in others.values()):+.3f} | margin={margin:+.3f} (2·pooledSE={2*pooled_se:.3f})", flush=True)
    print(f"  VERDICT: {interp}", flush=True)

    OUT.write_text(json.dumps({"config": {"d": D_MODEL, "layers": N_LAYERS, "heads": N_HEADS, "copy": D_COPY,
                                          "langA": LANG_A, "langB": LANG_B, "base_multi": BASE_MULTI, "phaseb": chosen,
                                          "seeds": SEEDS, "smoke": SMOKE, "prereg_lock": "690a5b42"},
                               "cells": cells, "conjunction": bool(conj), "margin": margin, "verdict": interp}, indent=1))
    print(f"\nwrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
