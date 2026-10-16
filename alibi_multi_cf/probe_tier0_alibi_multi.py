"""
probe_tier0_alibi_multi.py — Thread #2 Tier-0: does the ALiBi x multilingual CONJUNCTION flip the CF retention-sign
in a controlled from-scratch 2x2? ($0 local GPU)

2x2 = {ALiBi, RoPE} x {mono, multi}. Tiny decoder transformer trained from scratch on a position-sensitive COPY@D task
(target[i]=input[i-D], so PE matters). PROTECTION = retA_childhood - retA_nochildhood: positive=childhood protects task-A
against task-B interference; NEGATIVE = childhood ERODES (the BLOOM signature). We test whether ONLY the ALiBi x multi cell erodes.

Run:  python probe_tier0_alibi_multi.py --smoke   # 1 seed, tiny
      python probe_tier0_alibi_multi.py           # full 2x2 x seeds
"""
from __future__ import annotations
import sys, json, time, math
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun

HERE = Path(__file__).resolve().parent
SMOKE = "--smoke" in sys.argv
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = HERE / "out" / ("tier0_alibi_multi_smoke.json" if SMOKE else "tier0_alibi_multi.json")
(HERE / "out").mkdir(parents=True, exist_ok=True)

# config — d128/H8/L3 confirmed to bring BOTH ALiBi & RoPE to base competence on copy@D (test_capacity.py)
D_MODEL, N_LAYERS, N_HEADS = 128, 3, 8
SEQ_LEN = 24
BLOCK = 32                    # tokens per "language" block
K_LANG = 4                    # languages in multi mode
VOCAB = BLOCK * K_LANG        # 128
D_A, D_B = 2, 5               # copy distances (task-A vs task-B) — interfere
BASE_STEPS = 200 if SMOKE else 800
CHILD_STEPS = 100 if SMOKE else 400
PHASEB_STEPS = 200 if SMOKE else 800
BATCH = 128
LR = 3e-3
SEEDS = [0] if SMOKE else [0, 1, 2, 3]
EVAL_BATCH = 512


# ── data: copy@D, mono (block0) or multi (per-sequence random language) ───────
def gen_batch(D, mode, n, rng):
    x = np.zeros((n, SEQ_LEN), np.int64)
    for i in range(n):
        blk = 0 if mode == "mono" else rng.randint(K_LANG)      # multi: each SEQUENCE is one language
        x[i] = rng.randint(blk * BLOCK, (blk + 1) * BLOCK, SEQ_LEN)
    xt = torch.from_numpy(x).to(DEV)
    y = xt.clone()
    y[:, D:] = xt[:, :-D]        # target[i] = input[i-D]  (copy from D back)
    y[:, :D] = -100             # ignore first D positions
    return xt, y


# ── model: tiny decoder, switchable ALiBi / RoPE ──────────────────────────────
def alibi_slopes(h):
    return torch.tensor([2 ** (-8.0 * (i + 1) / h) for i in range(h)], device=DEV)


def rope_cache(dh, L):
    theta = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=DEV).float() / dh))
    pos = torch.arange(L, device=DEV).float()
    ang = torch.outer(pos, theta)                       # (L, dh/2)
    return torch.cos(ang), torch.sin(ang)


def apply_rope(x, cos, sin):
    # x: (B, H, L, dh)
    x1, x2 = x[..., ::2], x[..., 1::2]
    c = cos[None, None]; s = sin[None, None]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class Attn(nn.Module):
    def __init__(self, d, h, pe):
        super().__init__(); self.h = h; self.dh = d // h; self.pe = pe
        self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        if pe == "alibi": self.register_buffer("slopes", alibi_slopes(h), persistent=False)

    def forward(self, x, rope):
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                # (B,H,L,dh)
        if self.pe == "rope":
            cos, sin = rope; q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)   # (B,H,L,L)
        pos = torch.arange(L, device=x.device)
        causal = (pos[None, :] > pos[:, None])
        if self.pe == "alibi":
            dist = (pos[None, :] - pos[:, None]).float()        # j - i (negative for past)
            att = att + self.slopes[None, :, None, None] * dist[None, None]  # bias ~ -slope*|i-j| for past
        att = att.masked_fill(causal[None, None], float("-inf"))
        att = Fun.softmax(att, -1)
        out = (att @ v).transpose(1, 2).reshape(B, L, -1)
        return self.o(out)


class Block(nn.Module):
    def __init__(self, d, h, pe):
        super().__init__(); self.ln1 = nn.LayerNorm(d); self.at = Attn(d, h, pe)
        self.ln2 = nn.LayerNorm(d); self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, rope):
        x = x + self.at(self.ln1(x), rope); x = x + self.mlp(self.ln2(x)); return x


class TinyLM(nn.Module):
    def __init__(self, pe):
        super().__init__(); self.pe = pe
        self.emb = nn.Embedding(VOCAB, D_MODEL)
        self.blocks = nn.ModuleList([Block(D_MODEL, N_HEADS, pe) for _ in range(N_LAYERS)])
        self.lnf = nn.LayerNorm(D_MODEL); self.head = nn.Linear(D_MODEL, VOCAB)
        self.rope = rope_cache(D_MODEL // N_HEADS, SEQ_LEN) if pe == "rope" else None

    def forward(self, x):
        h = self.emb(x)
        for b in self.blocks: h = b(h, self.rope)
        return self.head(self.lnf(h))


def train(model, D, mode, steps, lr, seed):
    rng = np.random.RandomState(seed + 777); opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        x, y = gen_batch(D, mode, BATCH, rng)
        logits = model(x)
        loss = Fun.cross_entropy(logits.view(-1, VOCAB), y.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_acc(model, D, mode, seed=999):
    rng = np.random.RandomState(seed); model.eval()
    x, y = gen_batch(D, mode, EVAL_BATCH, rng)
    pred = model(x).argmax(-1)
    m = (y != -100)
    return float((pred[m] == y[m]).float().mean())


def clone_of(model):
    m = TinyLM(model.pe).to(DEV); m.load_state_dict(model.state_dict()); return m


def run_cell(pe, mode, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    base = TinyLM(pe).to(DEV)
    train(base, D_A, mode, BASE_STEPS, LR, seed)
    accA_base = eval_acc(base, D_A, mode)
    # no-childhood: base -> phase-B -> retention on A
    m_noc = clone_of(base); train(m_noc, D_B, mode, PHASEB_STEPS, LR, seed)
    retA_noc = eval_acc(m_noc, D_A, mode)
    # childhood: base -> extra A warmup -> phase-B -> retention on A
    m_ch = clone_of(base); train(m_ch, D_A, mode, CHILD_STEPS, LR, seed + 50)
    train(m_ch, D_B, mode, PHASEB_STEPS, LR, seed)
    retA_ch = eval_acc(m_ch, D_A, mode)
    return {"accA_base": accA_base, "retA_noc": retA_noc, "retA_ch": retA_ch,
            "protection": retA_ch - retA_noc}


def me(x):
    x = np.array(x); return [float(x.mean()), float(x.std(ddof=1) / max(len(x), 1) ** 0.5) if len(x) > 1 else 0.0]


def main():
    t0 = time.time()
    print(f"[TIER0] 2x2 {{ALiBi,RoPE}}x{{mono,multi}}  d={D_MODEL} L={N_LAYERS} H={N_HEADS} copy@{D_A}->{D_B} "
          f"seeds={SEEDS} dev={DEV}", flush=True)
    cells = {}
    for pe in ("alibi", "rope"):
        for mode in ("mono", "multi"):
            rs = [run_cell(pe, mode, s) for s in SEEDS]
            prot = me([r["protection"] for r in rs])
            cells[f"{pe}_{mode}"] = {"protection": prot, "retA_noc": me([r["retA_noc"] for r in rs]),
                                     "retA_ch": me([r["retA_ch"] for r in rs]), "accA_base": me([r["accA_base"] for r in rs]),
                                     "seeds": rs}
            print(f"  {pe:5} {mode:5}: accA_base={cells[f'{pe}_{mode}']['accA_base'][0]:.3f} "
                  f"retA_noc={cells[f'{pe}_{mode}']['retA_noc'][0]:.3f} retA_ch={cells[f'{pe}_{mode}']['retA_ch'][0]:.3f} "
                  f"PROTECTION={prot[0]:+.3f}±{prot[1]:.3f}", flush=True)

    am = cells["alibi_multi"]["protection"]; others = {k: cells[k]["protection"] for k in cells if k != "alibi_multi"}
    print("\n=== 2x2 PROTECTION (childhood - nochildhood; NEG = erodes = BLOOM signature) ===", flush=True)
    print(f"  {'':12}{'mono':>12}{'multi':>12}")
    for pe in ("alibi", "rope"):
        print(f"  {pe:12}" + "".join(f"{cells[f'{pe}_{m}']['protection'][0]:+8.3f}    " for m in ("mono", "multi")), flush=True)
    # conjunction test: alibi_multi negative AND clearly below the other three
    pooled_se = (am[1] ** 2 + max(o[1] for o in others.values()) ** 2) ** 0.5
    margin = min(o[0] for o in others.values()) - am[0]
    conj = am[0] < 0 and margin > 2 * pooled_se
    interp = ("CONJUNCTION REPRODUCES -> fire Lambda Tier-1" if conj else
              "NOT a clean conjunction (single-knob or null) -> Lambda is a judgment call")
    print(f"\n  ALiBi×multi protection = {am[0]:+.3f}±{am[1]:.3f}", flush=True)
    print(f"  others min protection  = {min(o[0] for o in others.values()):+.3f}  | margin={margin:+.3f} (2·pooledSE={2*pooled_se:.3f})", flush=True)
    print(f"  VERDICT: {interp}", flush=True)

    OUT.write_text(json.dumps({"config": {"d": D_MODEL, "layers": N_LAYERS, "heads": N_HEADS, "seq": SEQ_LEN,
                                          "D_A": D_A, "D_B": D_B, "seeds": SEEDS, "smoke": SMOKE, "prereg_lock": "140d742c"},
                               "cells": cells, "conjunction": bool(conj), "margin": margin, "verdict": interp}, indent=1))
    print(f"\nwrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
