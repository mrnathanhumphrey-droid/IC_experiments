"""
probe_ic_dose.py -- PHASE 0 dose-response decider (NOT pre-registered, no claim). Reuses probe_ic_feasibility machinery.
Fix A-demos at k=8, GROW the intervening B block (dose = 8,16,32,64), measure retroactive interference:
  prompt = [A demos x8] + [B demos x dose] + [A query]     (A distant, B block grows adjacent to query)
Two questions per dose:
  (1) does mean AUC-drop GROW with dose?  -> measurable regime found + Direction-C stream-length axis alive
  (2) does per-pair drop track -cos(d_A,d_B)?  -> the Phase-1 law preview (anti-parallel forget > aligned)
$0 local, ASCII-only prints.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_ic_feasibility as F   # LM, load_tasks, demo_block, query_str, eval_acc, auc

OUT = HERE / "out" / "ic_dose.json"
MODEL = os.environ.get("IC_MODEL", "Qwen/Qwen2.5-1.5B")
KSHOT_A = int(os.environ.get("IC_KSHOT_A", "8"))
DOSES = [int(x) for x in os.environ.get("IC_DOSES", "8,16,32,64").split(",")]
N_EVAL = int(os.environ.get("IC_NEVAL", "160"))
N_DIR = int(os.environ.get("IC_NDIR", "120"))
LAYER = int(os.environ.get("IC_LAYER", "15"))
A_OFF, B_OFF, EVAL_OFF, DIR_OFF = 0, 1500, 3500, 5000   # must fit smallest pool (length ~6666)
SEED = int(os.environ.get("IC_SEED", "0"))
torch.manual_seed(SEED); np.random.seed(SEED)


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx*ry).sum()/d) if d > 0 else float("nan")


def main():
    t0 = time.time()
    print(f"[DOSE] model={MODEL} kA={KSHOT_A} doses={DOSES} N_eval={N_EVAL} layer={LAYER}", flush=True)
    tasks = F.load_tasks(); tn = list(tasks.keys()); n = len(tn)
    lm = F.LM(MODEL); print(f"  loaded ({time.time()-t0:.0f}s)", flush=True)

    # fixed A-demo block per task (k=8), drawn from A_OFF
    demoA = {}
    for t in tn:
        txt, _ = F.demo_block(tasks[t], tasks[t]["pos"][A_OFF:], tasks[t]["neg"][A_OFF:], KSHOT_A, 0)
        demoA[t] = txt

    # cosine matrix at LAYER (directions from queries under own A-demos) -- BATCHED
    act = {}
    for t in tn:
        pool = tasks[t]["pos"][DIR_OFF:] + tasks[t]["neg"][DIR_OFF:]; np.random.RandomState(SEED).shuffle(pool)
        prompts = [demoA[t] + "\n\n" + F.query_str(tasks[t], q) for q in pool[:N_DIR]]
        act[t] = lm.acts_batch(prompts, [LAYER])[LAYER]
    dirs = {}
    for t in tn:
        X = act[t]; others = np.concatenate([act[o] for o in tn if o != t])
        d = X.mean(0) - others.mean(0); dirs[t] = d/(np.linalg.norm(d)+1e-8)
    cos = {(i, j): float(dirs[i] @ dirs[j]) for i in tn for j in tn if i != j}
    print(f"  cos range [{min(cos.values()):+.2f},{max(cos.values()):+.2f}] ({time.time()-t0:.0f}s)", flush=True)

    # clean AUC per task (A demos only, k=8)
    clean = {}
    for a in tn:
        pa = tasks[a]["pos"][EVAL_OFF:]; na = tasks[a]["neg"][EVAL_OFF:]
        _, clean[a] = F.eval_acc(lm, tasks[a], pa, na, demoA[a], N_EVAL)
    print(f"  clean AUC {[round(clean[a],3) for a in tn]} ({time.time()-t0:.0f}s)", flush=True)

    # dose sweep
    results = {}
    for dose in DOSES:
        demoB = {}
        for t in tn:
            txt, _ = F.demo_block(tasks[t], tasks[t]["pos"][B_OFF:], tasks[t]["neg"][B_OFF:], dose, 0)
            demoB[t] = txt
        pairs = []; drops = []; coss = []
        for a in tn:
            pa = tasks[a]["pos"][EVAL_OFF:]; na = tasks[a]["neg"][EVAL_OFF:]
            for b in tn:
                if b == a: continue
                pre = demoA[a] + "\n\n" + demoB[b]                 # A(8) then B(dose)
                _, auc_i = F.eval_acc(lm, tasks[a], pa, na, pre, N_EVAL)
                drop = clean[a] - auc_i
                pairs.append((a, b)); drops.append(drop); coss.append(cos[(a, b)])
        drops = np.array(drops); coss = np.array(coss)
        mean_drop = float(drops.mean())
        rho = spearman(coss, drops)                                 # law preview: expect NEGATIVE (drop up as cos down)
        aln = drops[coss > 0.3]; anti = drops[coss < -0.3]
        ma = float(aln.mean()) if len(aln) else float("nan"); mn = float(anti.mean()) if len(anti) else float("nan")
        results[dose] = {"mean_drop": round(mean_drop, 4), "rho_cos_drop": round(rho, 3),
                         "aligned_drop": round(ma, 4), "antipar_drop": round(mn, 4),
                         "max_drop": round(float(drops.max()), 4)}
        print(f"  dose={dose:3d} | mean_drop={mean_drop:+.4f} max={drops.max():+.4f} | rho(cos,drop)={rho:+.3f} "
              f"| aligned={ma:+.4f} antipar={mn:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
        OUT.write_text(json.dumps({"model": MODEL, "kA": KSHOT_A, "layer": LAYER, "clean": {a: round(clean[a],3) for a in tn},
                                   "cos_range": [round(min(cos.values()),3), round(max(cos.values()),3)],
                                   "doses": results}, indent=1))

    md = [results[d]["mean_drop"] for d in DOSES]
    grows = bool(md[-1] > md[0] + 0.01 and md[-1] > 0.02)
    lawful = bool(results[DOSES[-1]]["rho_cos_drop"] <= -0.30 or (results[DOSES[-1]]["antipar_drop"] - results[DOSES[-1]]["aligned_drop"]) >= 0.03)
    print("\n=== DOSE-RESPONSE (feasibility decider) ===", flush=True)
    print(f"  interference GROWS with dose : {grows}  (mean_drop {md[0]:+.4f} @8 -> {md[-1]:+.4f} @{DOSES[-1]})", flush=True)
    print(f"  drop tracks -cos at top dose : {lawful}  (rho {results[DOSES[-1]]['rho_cos_drop']:+.3f}, "
          f"antipar-aligned {results[DOSES[-1]]['antipar_drop']-results[DOSES[-1]]['aligned_drop']:+.4f})", flush=True)
    verdict = ("MEASURABLE + LAWFUL -> green-light Phase 1 prereg (dose = C stream-length axis)" if grows and lawful else
               "MEASURABLE, ordering unclear -> Phase 1 viable but law preview weak; consider harder tasks" if grows else
               "interference ~flat even at high dose -> in-context forgetting near-zero on this model; rethink substrate")
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"[DOSE] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
