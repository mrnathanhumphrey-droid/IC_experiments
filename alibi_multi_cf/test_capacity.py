"""Quick capacity test: can ALiBi learn copy@D to competence with more heads/capacity? (task-design gate)"""
import math, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fun
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN, BLOCK, K_LANG = 24, 32, 4
VOCAB = BLOCK * K_LANG


def gen_batch(D, mode, n, rng, d_model, seq=SEQ_LEN):
    x = np.zeros((n, seq), np.int64)
    for i in range(n):
        blk = 0 if mode == "mono" else rng.randint(K_LANG)
        x[i] = rng.randint(blk * BLOCK, (blk + 1) * BLOCK, seq)
    xt = torch.from_numpy(x).to(DEV); y = xt.clone()
    y[:, D:] = xt[:, :-D]; y[:, :D] = -100
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
    def __init__(s, pe, d, L, H):
        super().__init__(); s.pe = pe; s.emb = nn.Embedding(VOCAB, d)
        s.blocks = nn.ModuleList([Blk(d, H, pe) for _ in range(L)]); s.lnf = nn.LayerNorm(d); s.head = nn.Linear(d, VOCAB)
        s.rope = rope_cache(d // H, SEQ_LEN) if pe == "rope" else None
    def forward(s, x):
        h = s.emb(x)
        for b in s.blocks: h = b(h, s.rope)
        return s.head(s.lnf(h))


def test(pe, d, L, H, steps, D, mode="mono"):
    torch.manual_seed(0); np.random.seed(0); m = LM(pe, d, L, H).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3); m.train(); rng = np.random.RandomState(1)
    for _ in range(steps):
        x, y = gen_batch(D, mode, 128, rng, d); lo = m(x)
        loss = Fun.cross_entropy(lo.view(-1, VOCAB), y.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        x, y = gen_batch(D, mode, 512, np.random.RandomState(999), d); pred = m(x).argmax(-1); msk = y != -100
        return float((pred[msk] == y[msk]).float().mean())


for (d, H, L, steps) in [(128, 8, 3, 800), (192, 12, 4, 1500), (256, 16, 4, 2500)]:
    for D in (2, 5):
        a = test("alibi", d, L, H, steps, D); r = test("rope", d, L, H, steps, D)
        print(f"d={d} H={H} L={L} steps={steps} copy@{D}:  alibi={a:.3f}  rope={r:.3f}", flush=True)
