"""CPU dry-run for Probe-10 post-hoc (LOCK 080dbe88). No GPU/model. Validates: unit-cosine overlap, self-consistency
(mean pairwise cosine among K dirs), probe-column extraction, TASKS/PROBES wiring."""
import os, torch
os.environ.setdefault("CELL", "opt350m"); os.environ.setdefault("KDIR", "8")
import probe10post_dirsim_lambda as PP
ok = True
def check(name, cond, detail=""):
    global ok; ok = ok and bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")

print("=== (1) unit-cosine overlap ===")
D = 500
a = torch.randn(D, dtype=torch.float64); a = a / a.norm()
check("cos(a,a)=1", abs(float(a @ a) - 1.0) < 1e-9, f"{float(a@a):.4f}")
b = torch.randn(D, dtype=torch.float64); b = b - (b @ a) * a; b = b / b.norm()  # orthogonalize
check("cos(a,b_perp)~0", abs(float(a @ b)) < 1e-9, f"{float(a@b):.2e}")
# partial overlap monotone
c1 = (a + 0.2 * b); c1 = c1 / c1.norm(); c2 = (a + 2.0 * b); c2 = c2 / c2.norm()
check("cos decreases as we tilt away", float(a @ c1) > float(a @ c2), f"near={float(a@c1):.3f} far={float(a@c2):.3f}")

print("=== (2) self-consistency = mean pairwise cosine among K unit dirs ===")
K = 8
base = torch.randn(D, dtype=torch.float64); base = base / base.norm()
tight = torch.stack([(base + 0.01 * torch.randn(D, dtype=torch.float64)) for _ in range(K)])  # eps*sqrt(D)<<1
tight = tight / tight.norm(dim=1, keepdim=True)
loose = torch.randn(K, D, dtype=torch.float64); loose = loose / loose.norm(dim=1, keepdim=True)
def selfcon(M):
    G = M @ M.T; iu = torch.triu_indices(K, K, offset=1); return float(G[iu[0], iu[1]].mean())
check("tight cluster high self-consistency (>0.9)", selfcon(tight) > 0.9, f"{selfcon(tight):.3f}")
check("random dirs low self-consistency (<0.2)", abs(selfcon(loose)) < 0.2, f"{selfcon(loose):.3f}")

print("=== (3) probe-column extraction + mean-direction normalize ===")
TASKS = PP.TASKS
meandir = {X: (lambda v: v / v.norm())(torch.randn(D, dtype=torch.float64)) for X in TASKS}
overlap = {X: {Y: float(meandir[X] @ meandir[Y]) for Y in TASKS} for X in TASKS}
check("diagonal = 1", all(abs(overlap[X][X] - 1.0) < 1e-9 for X in TASKS), "")
check("symmetric", all(abs(overlap[X][Y] - overlap[Y][X]) < 1e-9 for X in TASKS for Y in TASKS), "")
pc = {T: {X: overlap[X][T] for X in TASKS} for T in PP.PROBES}
check("probe columns cover all tasks", all(set(pc[T]) == set(TASKS) for T in PP.PROBES), str(list(pc)))

print("=== (4) wiring ===")
check("6 tasks incl 4 swap-slots + backbone", set(["agnews","dbpedia","yahoo","emotion","trec","subj"]) == set(TASKS), str(TASKS))
check("probes = agnews,yahoo", PP.PROBES == ["agnews","yahoo"], str(PP.PROBES))
check("KDIR>=5", PP.KDIR >= 5, str(PP.KDIR))

print(f"\n=== DRY-RUN {'ALL PASS' if ok else 'FAILED'} ===")
