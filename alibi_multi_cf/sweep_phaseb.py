"""Non-saturation sweep: find phase-B strength giving PARTIAL task-A forgetting (retA_noc mid-range),
then measure PROTECTION (childhood - nochildhood) per cell there. Reuses test_capacity model defs."""
import numpy as np, torch, torch.nn.functional as Fun
import importlib.util
spec = importlib.util.spec_from_file_location("tc", "test_capacity.py"); tc = importlib.util.module_from_spec(spec); spec.loader.exec_module(tc)
DEV = tc.DEV; LM = tc.LM; gen_batch = tc.gen_batch; VOCAB = tc.VOCAB
D_A, D_B = 2, 5
d, L, H = 128, 3, 8
BASE_STEPS = 800


def train(m, D, mode, steps, lr, seed):
    opt = torch.optim.AdamW(m.parameters(), lr=lr); m.train(); rng = np.random.RandomState(seed + 777)
    for _ in range(steps):
        x, y = gen_batch(D, mode, 128, rng, d); lo = m(x)
        loss = Fun.cross_entropy(lo.view(-1, VOCAB), y.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return m


@torch.no_grad()
def acc(m, D, mode):
    x, y = gen_batch(D, mode, 512, np.random.RandomState(999), d); pred = m(x).argmax(-1); msk = y != -100
    return float((pred[msk] == y[msk]).float().mean())


def clone(m):
    n = LM(m.pe, d, L, H).to(DEV); n.load_state_dict(m.state_dict()); return n


def cell(pe, mode, pbsteps, seed, child=400):
    torch.manual_seed(seed); np.random.seed(seed)
    base = LM(pe, d, L, H).to(DEV); train(base, D_A, mode, BASE_STEPS, 3e-3, seed)
    noc = clone(base); train(noc, D_B, mode, pbsteps, 3e-3, seed); rnoc = acc(noc, D_A, mode)
    ch = clone(base); train(ch, D_A, mode, child, 3e-3, seed + 50); train(ch, D_B, mode, pbsteps, 3e-3, seed); rch = acc(ch, D_A, mode)
    return rnoc, rch


print("phase-B sweep: retA_noc (partial-forgetting regime) + PROTECTION per cell (seed avg of 2)")
for pb in (30, 60, 100, 150, 250):
    row = {}
    for pe in ("alibi", "rope"):
        for mode in ("mono", "multi"):
            rn = []; pr = []
            for s in (0, 1):
                a, b = cell(pe, mode, pb, s); rn.append(a); pr.append(b - a)
            row[f"{pe}_{mode}"] = (np.mean(rn), np.mean(pr))
    print(f"\n pb={pb}:")
    for k, (rn, pr) in row.items():
        print(f"    {k:12} retA_noc={rn:.3f}  PROTECTION={pr:+.3f}")
