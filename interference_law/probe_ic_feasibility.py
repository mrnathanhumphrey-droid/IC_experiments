"""
probe_ic_feasibility.py -- PHASE 0 feasibility gate for Directions A (in-context cosine twin) + C (stream composition).
NOT pre-registered, produces NO claim. Instrument-only checks on a LOCAL BASE model doing few-shot ICL classification:

  GATE 0.1  ICL INTERFERENCE IS MEASURABLE  -- task-A few-shot accuracy DROPS when task-B demos are prepended.
  GATE 0.2  ACTIVATION DIRECTIONS ARE REAL  -- per-task direction d_p = normalize(mean(act|task p) - mean(act|ref))
            separates held-out task-p queries from other-task queries (AUC). Sweep layers, pick best.
  HEADROOM  clean k-shot AUC/accuracy UNSATURATED (not ~chance, not ~1.0) -- the recurring saturation trap.

3 binary capabilities from CACHED data (no download): sentiment (sst2), world/sports (ag_news 0v1), biz/tech (ag_news 2v3).
$0 local, ASCII-only prints (Windows cp1252). Reports numbers, fires nothing downstream.
"""
from __future__ import annotations
import os, sys, json, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "out" / "ic_feasibility.json"
OUT.parent.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = os.environ.get("IC_MODEL", "Qwen/Qwen2.5-0.5B")
KSHOT = int(os.environ.get("IC_KSHOT", "8"))
N_EVAL = int(os.environ.get("IC_NEVAL", "120"))     # queries per task for accuracy
N_DIR = int(os.environ.get("IC_NDIR", "160"))       # queries per task for direction extraction
SEED = int(os.environ.get("IC_SEED", "0"))
torch.manual_seed(SEED); np.random.seed(SEED)
rng = np.random.RandomState(SEED)


def auc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUC
    order = np.argsort(scores); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    r_pos = ranks[labels == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---- data (cached; no download) : Option 1 = SAME ag_news manifold, different LABEL RULES ----
# 4 one-vs-rest topic capabilities + 1 non-topic (length) capability, all queries drawn from the SAME
# article distribution -> discriminative directions differ by RULE not by corpus -> graded cosine, real competition.
def load_tasks():
    from datasets import load_dataset
    ag = load_dataset("fancyzhx/ag_news", split="train")
    def clip(s, n=220): return " ".join(str(s).split())[:n]
    txt = [clip(x) for x in ag["text"]]; lab = list(ag["label"])
    by = {c: [txt[i] for i in range(len(lab)) if lab[i] == c] for c in range(4)}
    for c in by: rng.shuffle(by[c])
    def ovr(c):  # one-vs-rest: pos = class c, neg = mix of the other three
        pos = by[c][:]
        neg = []
        for o in range(4):
            if o != c: neg += by[o][:]
        rng.shuffle(neg)
        return pos, neg
    names = {0: ("world", "Topic"), 1: ("sports", "Topic"), 2: ("business", "Topic"), 3: ("technology", "Topic")}
    tasks = {}
    for c, (nm, cue) in names.items():
        pos, neg = ovr(c)
        tasks[nm] = {"cue": cue, "labels": (nm, "other"), "pos": pos, "neg": neg}
    # non-topic capability over the SAME articles: long vs short (tertile word-count split), classes mixed
    allt = txt[:20000]; wc = sorted(allt, key=lambda t: len(t.split()))
    third = len(wc) // 3
    shorts = wc[:third]; longs = wc[-third:]
    rng.shuffle(shorts); rng.shuffle(longs)
    tasks["length"] = {"cue": "Length", "labels": ("long", "short"), "pos": longs, "neg": shorts}
    return tasks


# ---- prompt construction ----
def demo_block(task, pos_pool, neg_pool, k, taken):
    """k/2 pos + k/2 neg interleaved demos, drawn past `taken` offset. Returns text + new offset."""
    lab = task["labels"]; cue = task["cue"]; lines = []
    half = max(1, k // 2)
    ps = pos_pool[taken:taken + half]; ns = neg_pool[taken:taken + half]
    demos = []
    for i in range(half):
        if i < len(ps): demos.append((ps[i], lab[0]))
        if i < len(ns): demos.append((ns[i], lab[1]))
    rng.shuffle(demos)
    for txt, y in demos:
        lines.append(f"Text: {txt}\n{cue}: {y}")
    return "\n\n".join(lines), taken + half


def query_str(task, txt):
    return f"Text: {txt}\n{task['cue']}:"


# ---- model ----
class LM:
    def __init__(self, name):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(DEV).eval()
        self.nlayer = self.model.config.num_hidden_layers

    def label_ids(self, labels):
        # first token id of " <label>" for each label word
        out = []
        for w in labels:
            ids = self.tok.encode(" " + w, add_special_tokens=False)
            out.append(ids[0])
        return out

    @torch.no_grad()
    def classify(self, prompt, lids):
        ids = self.tok(prompt, return_tensors="pt").to(DEV)
        logits = self.model(**ids).logits[0, -1]        # next-token logits
        s = logits[lids]                                 # [2]
        return int(torch.argmax(s).item()), float((s[0] - s[1]).item())

    @torch.no_grad()
    def acts(self, prompt, layers):
        ids = self.tok(prompt, return_tensors="pt").to(DEV)
        hs = self.model(**ids, output_hidden_states=True).hidden_states   # tuple len nlayer+1
        return {L: hs[L][0, -1].float().cpu().numpy() for L in layers}

    def _pad(self):
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"    # real tokens at 0..len-1 (natural positions); gather last real token

    @torch.no_grad()
    def classify_batch(self, prompts, lids, bs=12):
        self._pad(); out = []
        for i in range(0, len(prompts), bs):
            enc = self.tok(prompts[i:i+bs], return_tensors="pt", padding=True).to(DEV)
            B = enc["input_ids"].shape[0]; idx = enc["attention_mask"].sum(-1) - 1
            logits = self.model(**enc).logits[torch.arange(B, device=DEV), idx]   # [B,V] next-token at last real tok
            s = logits[:, lids]
            out.append((s[:, 0] - s[:, 1]).float().cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def acts_batch(self, prompts, layers, bs=16):
        self._pad(); out = {L: [] for L in layers}
        for i in range(0, len(prompts), bs):
            enc = self.tok(prompts[i:i+bs], return_tensors="pt", padding=True).to(DEV)
            B = enc["input_ids"].shape[0]; idx = enc["attention_mask"].sum(-1) - 1
            hs = self.model(**enc, output_hidden_states=True).hidden_states
            ar = torch.arange(B, device=DEV)
            for L in layers: out[L].append(hs[L][ar, idx].float().cpu().numpy())
        return {L: np.concatenate(out[L]) for L in layers}


def eval_acc(lm, task, pos, neg, k_demos_text, n):
    """accuracy + AUC on n queries (half pos/half neg) given a fixed demo preamble. BATCHED."""
    lids = lm.label_ids(task["labels"]); half = n // 2
    qs = [(pos[i], 0) for i in range(half)] + [(neg[i], 1) for i in range(half)]
    prompts = [((k_demos_text + "\n\n") if k_demos_text else "") + query_str(task, txt) for txt, _ in qs]
    m = lm.classify_batch(prompts, lids)                 # margin = logit(label0) - logit(label1)
    yidx = np.array([y for _, y in qs])
    pred = (m < 0).astype(int)                           # pred index 0 when margin>0
    acc = float((pred == yidx).mean())
    ytrue = (yidx == 0).astype(int)                      # label0 (=pos examples) is AUC-positive class
    return acc, auc(m, ytrue)


def main():
    t0 = time.time()
    print(f"[IC0] model={MODEL} kshot={KSHOT} n_eval={N_EVAL} n_dir={N_DIR} dev={DEV}", flush=True)
    tasks = load_tasks()
    for k, v in tasks.items():
        print(f"  task {k}: pos={len(v['pos'])} neg={len(v['neg'])} cue={v['cue']} labels={v['labels']}", flush=True)
    lm = LM(MODEL); nlayer = lm.nlayer
    print(f"  loaded {MODEL}: {nlayer} layers ({time.time()-t0:.0f}s)", flush=True)

    tnames = list(tasks.keys())
    # reserve disjoint pools: demos from front, eval from a later offset, direction from another
    DEMO_OFF = 0; EVAL_OFF = 400; DIR_OFF = 800

    # ---- GATE 0.1 : interference + headroom (accuracy/AUC clean vs task-B-prepended) ----
    print("\n[IC0] GATE 0.1 -- interference (clean k-shot vs task-B demos prepended)", flush=True)
    demo_text = {};
    for tn in tnames:
        txt, _ = demo_block(tasks[tn], tasks[tn]["pos"][DEMO_OFF:], tasks[tn]["neg"][DEMO_OFF:], KSHOT, 0)
        demo_text[tn] = txt
    clean = {}; interfered = {}
    for a in tnames:
        pa = tasks[a]["pos"][EVAL_OFF:]; na = tasks[a]["neg"][EVAL_OFF:]
        acc_c, auc_c = eval_acc(lm, tasks[a], pa, na, demo_text[a], N_EVAL)
        clean[a] = {"acc": round(acc_c, 3), "auc": round(auc_c, 3)}
        drops = {}
        for b in tnames:
            if b == a: continue
            # RETROACTIVE interference: A demos THEN B demos THEN A query -> A distant, B adjacent = forgetting analog
            # (this A->B->recall-A unit is also the atomic step of a Direction-C stream)
            pre = demo_text[a] + "\n\n" + demo_text[b]
            _, auc_i = eval_acc(lm, tasks[a], pa, na, pre, N_EVAL)
            drops[b] = round(auc_c - auc_i, 3)             # AUC-drop = continuous forgetting metric
        interfered[a] = drops
        print(f"  A={a:13} clean auc={auc_c:.3f} (acc={acc_c:.3f}) | AUC-drop when B prepended: {drops}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- GATE 0.2 : activation directions separate (layer sweep) ----
    print("\n[IC0] GATE 0.2 -- activation directions (layer sweep, AUC task-p vs other-tasks)", flush=True)
    sweep = sorted(set([int(nlayer * f) for f in (0.4, 0.55, 0.7, 0.85, 1.0)]))
    # collect query activations per task (few-shot preamble = task's own demos, so 'task p is active')
    act = {L: {tn: [] for tn in tnames} for L in sweep}
    for tn in tnames:
        pool = tasks[tn]["pos"][DIR_OFF:] + tasks[tn]["neg"][DIR_OFF:]; rng.shuffle(pool)
        for txt in pool[:N_DIR]:
            prompt = demo_text[tn] + "\n\n" + query_str(tasks[tn], txt)
            a = lm.acts(prompt, sweep)
            for L in sweep: act[L][tn].append(a[L])
    dir_auc = {}; cos_by_L = {}
    for L in sweep:
        X = {tn: np.array(act[L][tn]) for tn in tnames}
        half = N_DIR // 2
        aucs = []; dirs = {}
        for tn in tnames:
            tr = X[tn][:half]; others = np.concatenate([X[o][:half] for o in tnames if o != tn])
            d = tr.mean(0) - others.mean(0); d = d / (np.linalg.norm(d) + 1e-8); dirs[tn] = d
            te_pos = X[tn][half:]; te_neg = np.concatenate([X[o][half:] for o in tnames if o != tn])
            sc = np.concatenate([te_pos @ d, te_neg @ d]); yy = np.concatenate([np.ones(len(te_pos)), np.zeros(len(te_neg))])
            aucs.append(auc(sc, yy))
        dir_auc[L] = round(float(np.mean(aucs)), 3)
        cm = np.array([[float(dirs[i] @ dirs[j]) for j in tnames] for i in tnames])
        off = cm[~np.eye(len(tnames), dtype=bool)]
        cos_by_L[L] = {"matrix": np.round(cm, 3).tolist(), "off_min": round(float(off.min()), 3),
                       "off_max": round(float(off.max()), 3), "off_range": round(float(off.max() - off.min()), 3)}
        print(f"  layer {L:3d}/{nlayer}: dirAUC={dir_auc[L]:.3f} | cos(d_i,d_j) off-diag range [{off.min():+.2f},{off.max():+.2f}] spread={off.max()-off.min():.2f}", flush=True)
    best_L = max(dir_auc, key=dir_auc.get)
    # pick the layer with the WIDEST cosine spread among non-saturated-direction layers (that's what Phase 1 needs)
    cand = [L for L in sweep if 0.70 <= dir_auc[L] <= 0.995]
    cos_L = max(cand, key=lambda L: cos_by_L[L]["off_range"]) if cand else best_L
    print(f"\n  [cosine matrix @ layer {cos_L}] tasks={tnames}", flush=True)
    for i, row in enumerate(cos_by_L[cos_L]["matrix"]):
        print(f"    {tnames[i]:12} {['%+.2f' % x for x in row]}", flush=True)

    # ---- verdict summary (feasibility, not a claim) ----
    clean_accs = [clean[a]["acc"] for a in tnames]; clean_aucs = [clean[a]["auc"] for a in tnames]
    max_drop = max(max(d.values()) for d in interfered.values())
    mean_drop = float(np.mean([v for d in interfered.values() for v in d.values()]))
    n_unsat = sum(1 for a in clean_aucs if 0.62 <= a <= 0.98)      # capabilities with AUC headroom
    cos_spread = cos_by_L[cos_L]["off_range"]
    g01 = bool(max_drop >= 0.03)                                   # measurable interference exists (AUC-drop)
    g02 = bool(dir_auc[cos_L] >= 0.70)                             # directions reliable (graded-ness is G03's job)
    g03 = bool(cos_spread >= 0.20)                                 # cosine matrix is GRADED (Phase-1 needs range)
    headroom = bool(n_unsat >= 3)                                  # >=3 capabilities unsaturated (AUC band)
    res = {
        "model": MODEL, "kshot": KSHOT, "nlayer": nlayer, "layers_swept": sweep, "tasks": tnames,
        "clean": clean, "interference_drop": interfered,
        "max_drop": round(max_drop, 3), "mean_drop": round(mean_drop, 3),
        "dir_auc_by_layer": dir_auc, "best_layer": best_L, "cos_layer": cos_L,
        "cos_by_layer": cos_by_L, "cos_spread_at_cos_L": cos_spread,
        "clean_acc": [round(a, 3) for a in clean_accs], "clean_auc": [round(a, 3) for a in clean_aucs],
        "n_unsaturated": n_unsat,
        "gates": {"G01_interference": g01, "G02_directions": g02, "G03_cosine_graded": g03, "headroom": headroom},
    }
    OUT.write_text(json.dumps(res, indent=1))
    print("\n=== PHASE 0 FEASIBILITY (instrument-only, no claim) ===", flush=True)
    print(f"  GATE 0.1 interference measurable : {g01}  (max AUC-drop {max_drop:+.3f}, mean {mean_drop:+.3f})", flush=True)
    print(f"  GATE 0.2 directions real+unsat   : {g02}  (cos-layer {cos_L}, AUC {dir_auc[cos_L]:.3f})", flush=True)
    print(f"  GATE 0.3 cosine matrix GRADED    : {g03}  (off-diag spread {cos_spread:.2f} @ layer {cos_L})", flush=True)
    print(f"  HEADROOM  >=3 unsaturated caps    : {headroom}  ({n_unsat}/{len(tnames)} in AUC band; clean AUC {[round(a,2) for a in clean_aucs]})", flush=True)
    go = g01 and g02 and g03 and headroom
    print(f"\n[IC0] ALL GATES PASS = {go}  ->  {'A/C alive on this model; lock layer=%d, proceed to Phase 1 PREREG' % best_L if go else 'gate(s) failed; retune model/kshot/tasks before any claim'}", flush=True)
    print(f"[IC0] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
