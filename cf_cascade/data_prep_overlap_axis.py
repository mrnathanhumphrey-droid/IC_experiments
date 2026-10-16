"""Overlap-axis task-B data prep (CPU-only, answer-blind, no GPU, no freeze dependency).
Downloads + formats the four constructed-overlap task-B sets cast-as-generation, reusing
A = SST-2 as the fixed anchor. Overlap declared by a-priori task identity, NOT by forgetting.

  ~0 null : SST-2 with PERMUTED labels (incoherent task; shared null for sigma_cp=0,
            c2 B=A, OGD orthogonal limit; controls movement-magnitude)
  low     : AG-News (news topic; diff domain + diff skill)  [FK6-distinct main pair]
  medium  : SUBJ (subjectivity on movie-related text; shared inputs, diff readout)
  high    : Yelp polarity (sentiment on other reviews; maximal feature/skill sharing)

Saves train(2000)+val(400) per level to gateA/overlap_axis/<level>.json as
{prompt_template, examples:[{text,label,completion}]}. Verifies downloadability + format.
"""
import os, json
os.environ.setdefault("HF_HOME", "D:/IC_experiments/cf_cascade/hf_cache")
import numpy as np
from datasets import load_dataset

OUT = "D:/IC_experiments/cf_cascade/gateA/overlap_axis"
os.makedirs(OUT, exist_ok=True)
NTR, NVAL = 2000, 400
rng = np.random.default_rng(0)


def save(level, template, examples, meta):
    json.dump({"level": level, "prompt_template": template, "meta": meta,
               "n": len(examples), "examples": examples}, open(f"{OUT}/{level}.json", "w"))
    labs = [e["label"] for e in examples]
    print(f"  [{level}] {len(examples)} ex, label dist {dict(zip(*np.unique(labs, return_counts=True)))} "
          f"-> {OUT}/{level}.json", flush=True)


def take(rows, n):
    idx = rng.permutation(len(rows))[:n]
    return [rows[i] for i in idx]


def main():
    print("[prep] overlap-axis B-data (CPU, answer-blind)", flush=True)

    # ---- A anchor + ~0 null: SST-2 permuted labels ----
    sst = load_dataset("stanfordnlp/sst2")
    sst_tr = [(r["sentence"].strip(), int(r["label"])) for r in sst["train"] if r["label"] in (0, 1)]
    sst_va = [(r["sentence"].strip(), int(r["label"])) for r in sst["validation"] if r["label"] in (0, 1)]
    LAB_SENT = {0: " negative", 1: " positive"}
    # permuted = shuffle the label column so input->label mapping is destroyed
    def perm(rows, n):
        sub = take(rows, n); labs = [l for _, l in sub]; rng.shuffle(labs)
        return [{"text": t, "label": int(l), "completion": LAB_SENT[l]} for (t, _), l in zip(sub, labs)]
    save("null_sst2permuted", "{text}\nSentiment:", perm(sst_tr, NTR) + perm(sst_va, NVAL),
         {"overlap": "~0", "src": "SST-2 labels permuted", "skill": "incoherent"})

    # ---- low: AG-News (topic) ----
    ag = load_dataset("fancyzhx/ag_news")
    AG = {0: " World", 1: " Sports", 2: " Business", 3: " Sci/Tech"}
    def ag_fmt(split, n):
        rows = [(r["text"].strip().replace("\n", " "), int(r["label"])) for r in ag[split]]
        return [{"text": t, "label": l, "completion": AG[l]} for t, l in take(rows, n)]
    save("low_agnews", "{text}\nTopic:", ag_fmt("train", NTR) + ag_fmt("test", NVAL),
         {"overlap": "low", "src": "AG-News", "skill": "topic"})

    # ---- medium: SUBJ (subjectivity) ----
    try:
        subj = load_dataset("SetFit/subj")
        SUBJ = {0: " objective", 1: " subjective"}
        def subj_fmt(split, n):
            rows = [(r["text"].strip().replace("\n", " "), int(r["label"])) for r in subj[split]]
            return [{"text": t, "label": l, "completion": SUBJ[l]} for t, l in take(rows, n)]
        save("medium_subj", "{text}\nType:", subj_fmt("train", NTR) + subj_fmt("test", NVAL),
             {"overlap": "medium", "src": "SUBJ", "skill": "subjectivity"})
    except Exception as e:
        print(f"  [medium_subj] FAILED ({type(e).__name__}: {e}); fallback id needed", flush=True)

    # ---- high: Yelp polarity (sentiment) ----
    yelp = load_dataset("fancyzhx/yelp_polarity")
    YELP = {0: " negative", 1: " positive"}
    def yelp_fmt(split, n):
        rows = [(r["text"].strip().replace("\n", " ")[:600], int(r["label"])) for r in yelp[split]]
        return [{"text": t, "label": l, "completion": YELP[l]} for t, l in take(rows, n)]
    save("high_yelp", "{text}\nSentiment:", yelp_fmt("train", NTR) + yelp_fmt("test", NVAL),
         {"overlap": "high", "src": "Yelp polarity", "skill": "sentiment"})

    print("[prep] DONE — overlap axis built, answer-blind, no GPU touched", flush=True)


if __name__ == "__main__":
    main()
