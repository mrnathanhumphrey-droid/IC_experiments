"""
probe_ic_facilitation.py -- characterize the LOW-DOSE facilitation reversal (in-context arc, option 2).
Observation (2 runs): at short intervening streams (dose 4-8), anti-parallel B demos HELP task-A recall
(rho(cos,drop) ~ +0.4; antipar drop ~ -0.04 = AUC UP) MORE than aligned B. Is it cosine/semantic, label-space,
or generic warm-up? MATCHED-LABEL task recast (uniform cue "Answer" + yes/no -> label-space identical, cosine
varies by content only) + NEUTRAL-FILLER control (any block helps? = warm-up) + A-REPLAY control (help upper bound).
Low doses only (no OOM). $0 concept, runs on Lambda a100. ASCII-only prints.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_ic_feasibility as F   # LM (+batched), demo_block, query_str, eval_acc, auc

OUT = HERE / "out" / "ic_facilitation.json"
MODEL = os.environ.get("IC_MODEL", "Qwen/Qwen2.5-1.5B")
KA = int(os.environ.get("IC_KSHOT_A", "8"))
DOSES = [int(x) for x in os.environ.get("IC_DOSES", "4,8,16").split(",")]
N_EVAL = int(os.environ.get("IC_NEVAL", "200"))
N_DIR = int(os.environ.get("IC_NDIR", "120"))
LAYER = int(os.environ.get("IC_LAYER", "15"))
A_OFF, B_OFF, RP_OFF, EVAL_OFF, DIR_OFF, FIL_OFF = 0, 1200, 2400, 3600, 5000, 0   # FIL_OFF indexes the separate 6000-snippet filler pool
SEED = int(os.environ.get("IC_SEED", "0"))
torch.manual_seed(SEED); np.random.seed(SEED)
CUE = "Answer"; LABELS = ("yes", "no")


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean(); d = np.sqrt((rx**2).sum()*(ry**2).sum())
    return float((rx*ry).sum()/d) if d > 0 else float("nan")


def clip(s, n=220): return " ".join(str(s).split())[:n]


def load_tasks_ml():
    from datasets import load_dataset
    ag = load_dataset("fancyzhx/ag_news", split="train")
    txt = [clip(x) for x in ag["text"]]; lab = list(ag["label"])
    by = {c: [txt[i] for i in range(len(lab)) if lab[i] == c] for c in range(4)}
    for c in by: np.random.RandomState(SEED).shuffle(by[c])
    def ovr(c):
        pos = by[c][:]; neg = [t for o in range(4) if o != c for t in by[o]]
        np.random.RandomState(SEED+1).shuffle(neg); return pos, neg
    tasks = {}
    for c, nm in {0: "world", 1: "sports", 2: "business", 3: "technology"}.items():
        pos, neg = ovr(c); tasks[nm] = {"cue": CUE, "labels": LABELS, "pos": pos, "neg": neg}
    allt = txt[:20000]; wc = sorted(allt, key=lambda t: len(t.split())); third = len(wc)//3
    longs = wc[-third:]; shorts = wc[:third]
    np.random.RandomState(SEED+2).shuffle(longs); np.random.RandomState(SEED+3).shuffle(shorts)
    tasks["length"] = {"cue": CUE, "labels": LABELS, "pos": longs, "neg": shorts}
    # filler pool = raw article text (no Text/Answer structure), reserved slice
    filler = txt[20000:26000]
    return tasks, filler


def filler_block(filler, dose, off):
    # ~dose raw snippets joined as plain text (matched-length, NO task structure) = warm-up control
    snips = filler[off:off+dose]
    return "\n\n".join(snips)


def main():
    t0 = time.time()
    print(f"[FAC] model={MODEL} kA={KA} doses={DOSES} N_eval={N_EVAL} layer={LAYER} (matched-label)", flush=True)
    tasks, filler = load_tasks_ml(); tn = list(tasks.keys()); n = len(tn)
    lm = F.LM(MODEL); print(f"  loaded ({time.time()-t0:.0f}s)", flush=True)

    demoA = {t: F.demo_block(tasks[t], tasks[t]["pos"][A_OFF:], tasks[t]["neg"][A_OFF:], KA, 0)[0] for t in tn}

    # cosine (matched-label directions) at LAYER
    act = {}
    for t in tn:
        pool = tasks[t]["pos"][DIR_OFF:] + tasks[t]["neg"][DIR_OFF:]; np.random.RandomState(SEED).shuffle(pool)
        prompts = [demoA[t] + "\n\n" + F.query_str(tasks[t], q) for q in pool[:N_DIR]]
        act[t] = lm.acts_batch(prompts, [LAYER])[LAYER]
    dirs = {}
    for t in tn:
        d = act[t].mean(0) - np.concatenate([act[o] for o in tn if o != t]).mean(0); dirs[t] = d/(np.linalg.norm(d)+1e-8)
    cos = {(i, j): float(dirs[i] @ dirs[j]) for i in tn for j in tn if i != j}
    print(f"  cos range [{min(cos.values()):+.2f},{max(cos.values()):+.2f}] ({time.time()-t0:.0f}s)", flush=True)

    clean = {}
    for a in tn:
        pa = tasks[a]["pos"][EVAL_OFF:]; na = tasks[a]["neg"][EVAL_OFF:]
        _, clean[a] = F.eval_acc(lm, tasks[a], pa, na, demoA[a], N_EVAL)
    print(f"  clean AUC {[round(clean[a],3) for a in tn]} ({time.time()-t0:.0f}s)", flush=True)

    results = {}
    for dose in DOSES:
        demoB = {t: F.demo_block(tasks[t], tasks[t]["pos"][B_OFF:], tasks[t]["neg"][B_OFF:], dose, 0)[0] for t in tn}
        replayA = {t: F.demo_block(tasks[t], tasks[t]["pos"][RP_OFF:], tasks[t]["neg"][RP_OFF:], dose, 0)[0] for t in tn}
        fill = filler_block(filler, dose, FIL_OFF)
        pairs = []; drops = []; coss = []; d_neu = []; d_rep = []
        for a in tn:
            pa = tasks[a]["pos"][EVAL_OFF:]; na = tasks[a]["neg"][EVAL_OFF:]
            for b in tn:
                if b == a: continue
                _, ai = F.eval_acc(lm, tasks[a], pa, na, demoA[a] + "\n\n" + demoB[b], N_EVAL)
                pairs.append((a, b)); drops.append(clean[a]-ai); coss.append(cos[(a, b)])
            _, an = F.eval_acc(lm, tasks[a], pa, na, demoA[a] + "\n\n" + fill, N_EVAL); d_neu.append(clean[a]-an)
            _, ar = F.eval_acc(lm, tasks[a], pa, na, demoA[a] + "\n\n" + replayA[a], N_EVAL); d_rep.append(clean[a]-ar)
        drops = np.array(drops); coss = np.array(coss)
        rho = spearman(coss, drops)
        aln = drops[coss > 0.3]; anti = drops[coss < -0.3]
        ma = float(aln.mean()) if len(aln) else float("nan"); mn = float(anti.mean()) if len(anti) else float("nan")
        neu = float(np.mean(d_neu)); rep = float(np.mean(d_rep))
        results[dose] = {"rho_cos_drop": round(rho, 3), "aligned_drop": round(ma, 4), "antipar_drop": round(mn, 4),
                         "neutral_drop": round(neu, 4), "replay_drop": round(rep, 4),
                         "antipar_minus_neutral": round(mn-neu, 4), "mean_drop": round(float(drops.mean()), 4)}
        print(f"  dose={dose:3d} | rho={rho:+.3f} | antipar={mn:+.4f} aligned={ma:+.4f} | NEUTRAL={neu:+.4f} REPLAY={rep:+.4f} "
              f"| antipar-neutral={mn-neu:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
        OUT.write_text(json.dumps({"model": MODEL, "kA": KA, "layer": LAYER, "matched_label": True,
                                   "clean": {a: round(clean[a], 3) for a in tn},
                                   "cos_range": [round(min(cos.values()), 3), round(max(cos.values()), 3)],
                                   "doses": results}, indent=1))

    # readout
    lo = DOSES[0]
    r = results[lo]
    cosine_real = bool(r["rho_cos_drop"] >= 0.25)              # facilitation ordering survives matched labels -> cosine/semantic
    specific = bool(r["antipar_minus_neutral"] <= -0.01)       # antipar-B helps MORE (more negative drop) than neutral filler
    print("\n=== FACILITATION CHARACTERIZATION (matched-label) ===", flush=True)
    print(f"  cosine-REAL (rho survives matched labels) : {cosine_real}  (dose{lo} rho={r['rho_cos_drop']:+.3f})", flush=True)
    print(f"  SPECIFIC (antipar-B beats neutral filler)  : {specific}  (antipar {r['antipar_drop']:+.4f} vs neutral {r['neutral_drop']:+.4f}, diff {r['antipar_minus_neutral']:+.4f})", flush=True)
    verdict = ("COSINE-REAL + SPECIFIC: low-dose anti-parallel demos give genuine semantic facilitation beyond warm-up" if cosine_real and specific else
               "WARM-UP: any intervening block helps ~equally -> facilitation is generic, not cosine" if cosine_real and not specific else
               "LABEL/FORMAT artifact: matched labels kill the ordering -> earlier +0.4 was label-space distinctness, not cosine" if not cosine_real else
               "mixed")
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"[FAC] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
