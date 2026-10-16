"""
pr_steering_lambda.py — #3 SUBSPACE DIMENSIONALITY (PR predicts steerable-axis-vs-broad-subspace).
Pre-reg: cf_cascade/PREREG_SUBSPACE_DIMENSIONALITY_PR_2026_07_27.md (LOCK 10d04137). Exec plan EXEC_PLAN_SUBSPACE_DIMENSIONALITY_PR.

Per concept: harvest last-token residual-stream acts for matched pos/neg prompt pairs -> difference matrix D (N x d) at each layer.
  PR = (Σσ²)²/Σσ⁴ of D (participation ratio; PR≈1 steerable axis, PR≫1 broad subspace).  stable_rank = ‖D‖_F²/σ1².
Causal steering: v_full = mean(D). v_k = rank-k reconstruction of v_full in D's top-k right-singular basis.
  EFFECT(v) = mean over NEG prompts of Δ(logit[pos_tok] - logit[neg_tok]) when +α·v is added at the analysis layer (steer neg->pos).
  k* = smallest k with EFFECT(v_k) >= 0.9 * EFFECT(v_full).  Analysis layer = argmax separability (|v_full| vs within-class scatter), frozen per concept.
CORE Q (P0): Spearman(PR, k*) across concepts + perm-null + Haar-null band. HIT |ρ|>=0.30. NULL -> KILL (dead scalar).
Baselines: stable_rank, top1-effect-frac. SMOKE=1 -> 3 concepts, Gemma-2-2B, fewer pairs.

Models: google/gemma-2-2b (gated -> HF_TOKEN), Qwen/Qwen2.5-1.5B. Activations only, no training. bf16, single GPU.
"""
from __future__ import annotations
import os, sys, json, time, math
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
OUTDIR = Path(os.environ.get("OUTDIR", str(HERE / "gateA"))); OUTDIR.mkdir(parents=True, exist_ok=True)
SMOKE = int(os.environ.get("SMOKE", "0"))
MODEL = os.environ.get("MODEL", "google/gemma-2-2b")
TAG = os.environ.get("TAG", MODEL.split("/")[-1].replace(".", "_"))
OUT = OUTDIR / os.environ.get("OUTNAME", f"pr_steering_{TAG}.json")
ALPHA = float(os.environ.get("ALPHA", "4.0"))          # steering scale: steer vector = ALPHA * (rank-k reconstruction of mean-diff)
PERM = 50 if SMOKE else int(os.environ.get("PERM", "5000"))
HAAR = 30 if SMOKE else int(os.environ.get("HAAR", "200"))
SEED = int(os.environ.get("SEED", "0"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)


# ---------------- concept registry (uniform frame-generator; matched N via NPAIRS) ----------------
# Each concept = (pos_words, neg_words, pos_tok, neg_tok, desc). Contrastive pair (p,n) = same frame w/ pos vs neg word.
# Pairs built words-outer/frames-inner so truncation to NPAIRS spans all words. Probe tokens give a uniform
# causal logit-diff readout (no per-concept classifier / generation). Concepts span simple antonyms -> contextual valence.
NPAIRS = int(os.environ.get("NPAIRS", "16" if SMOKE else "48"))
FRAMES = ["The thing was truly {w}.", "Honestly it all felt really {w}.",
          "My friend said it seemed quite {w}.", "In the end everyone agreed it was {w}."]

_W = {
  "sentiment": (["wonderful","excellent","fantastic","delightful","superb","brilliant","amazing","great",
                 "lovely","outstanding","marvelous","terrific","gorgeous","impressive","charming","beautiful"],
                ["terrible","awful","horrible","dreadful","disgusting","atrocious","lousy","bad",
                 "miserable","appalling","hideous","abysmal","ghastly","unpleasant","repulsive","ugly"], " great", " terrible"),
  "size":      (["enormous","gigantic","massive","huge","colossal","immense","vast","giant",
                 "towering","monumental","mammoth","hulking","sizable","substantial","big","large"],
                ["tiny","minuscule","miniature","microscopic","petite","diminutive","little","minute",
                 "slight","compact","teeny","dwarfed","small","modest","puny","wee"], " huge", " tiny"),
  "temperature":(["scorching","boiling","blazing","sweltering","fiery","burning","torrid","roasting",
                  "searing","sizzling","hot","warm","heated","molten","tropical","blistering"],
                 ["freezing","icy","frigid","frosty","chilly","glacial","wintry","arctic",
                  "cold","cool","frozen","bitter","polar","snowy","chilled","numbing"], " hot", " cold"),
  "speed":     (["rapid","swift","quick","speedy","fast","brisk","hurried","fleet",
                 "nimble","hasty","zippy","express","supersonic","breakneck","lightning","accelerated"],
                ["sluggish","slow","leisurely","plodding","crawling","lethargic","gradual","unhurried",
                 "dawdling","ponderous","languid","creeping","tardy","laggard","idle","dragging"], " fast", " slow"),
  "brightness":(["dazzling","radiant","brilliant","luminous","gleaming","glowing","bright","shining",
                 "vivid","blazing","glaring","sparkling","sunlit","lustrous","beaming","incandescent"],
                ["gloomy","murky","shadowy","dim","dark","dull","dusky","obscure",
                 "dingy","somber","black","unlit","cloudy","overcast","muted","shaded"], " bright", " dark"),
  "difficulty":(["difficult","hard","challenging","tough","arduous","demanding","grueling","strenuous",
                 "complicated","tricky","laborious","formidable","daunting","punishing","taxing","onerous"],
                ["easy","simple","effortless","straightforward","trivial","painless","manageable","elementary",
                 "basic","gentle","light","breezy","undemanding","facile","smooth","uncomplicated"], " hard", " easy"),
  "wealth":    (["wealthy","rich","affluent","opulent","prosperous","lavish","luxurious","moneyed",
                 "posh","upscale","plush","extravagant","privileged","flush","loaded","sumptuous"],
                ["poor","destitute","impoverished","penniless","broke","needy","bankrupt","deprived",
                 "indigent","struggling","humble","meager","strapped","hard-up","pauper","threadbare"], " rich", " poor"),
  "strength":  (["powerful","strong","mighty","muscular","forceful","robust","sturdy","potent",
                 "vigorous","herculean","formidable","dominant","stalwart","brawny","rugged","tough"],
                ["weak","feeble","frail","flimsy","fragile","delicate","puny","faint",
                 "powerless","brittle","infirm","spindly","sickly","helpless","soft","wobbly"], " strong", " weak"),
  "age":       (["ancient","old","aged","elderly","antique","archaic","decrepit","venerable",
                 "vintage","weathered","timeworn","seasoned","hoary","geriatric","primeval","olden"],
                ["new","fresh","modern","recent","novel","contemporary","youthful","young",
                 "pristine","current","newborn","nascent","budding","untouched","newfangled","fledgling"], " old", " new"),
  "loudness":  (["deafening","thunderous","booming","blaring","loud","roaring","raucous","earsplitting",
                 "noisy","clamorous","resounding","piercing","blasting","screeching","rowdy","uproarious"],
                ["silent","quiet","hushed","muffled","faint","soft","noiseless","muted",
                 "whispered","subdued","still","tranquil","peaceful","inaudible","low","gentle"], " loud", " quiet"),
  "cleanliness":(["spotless","pristine","immaculate","clean","gleaming","sanitary","polished","fresh",
                  "tidy","sparkling","hygienic","laundered","scrubbed","dustless","neat","unspoiled"],
                 ["filthy","dirty","grimy","muddy","grubby","soiled","stained","squalid",
                  "dingy","murky","polluted","greasy","dusty","messy","foul","sullied"], " clean", " dirty"),
  "toxicity":  (["hateful","disgusting","worthless","pathetic","vile","repulsive","contemptible","loathsome",
                 "odious","despicable","revolting","wretched","abhorrent","detestable","obnoxious","monstrous"],
                ["kind","wonderful","lovely","delightful","gracious","admirable","thoughtful","gentle",
                 "warm","caring","pleasant","charming","generous","compassionate","friendly","sweet"], " hateful", " kind"),
}


def _mk(pos, neg, ptok, ntok, desc):
    pairs = [(f.format(w=p), f.format(w=n)) for p, n in zip(pos, neg) for f in FRAMES]  # words-outer -> truncation spans words
    return (pairs[:NPAIRS], ptok, ntok, desc)

CONCEPTS = {k: _mk(v[0], v[1], v[2], v[3], k) for k, v in _W.items()}
SMOKE_SET = ["sentiment", "size", "toxicity"]


def concept_list():
    names = os.environ.get("CONCEPTS", "")
    if names: return [n for n in names.split(",") if n in CONCEPTS]
    return SMOKE_SET if SMOKE else list(CONCEPTS)


# ---------------- model ----------------
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN"))
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"     # last real token at attention_mask.sum(1)-1
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, token=os.environ.get("HF_TOKEN"),
                                             attn_implementation="eager").to(DEV).eval()
    layers = m.model.layers; d = m.config.hidden_size
    print(f"[MODEL] {MODEL} nlayers={len(layers)} d={d}", flush=True)
    return m, tok, layers, d


@torch.no_grad()
def last_tok_acts(m, tok, prompts, bs=16):
    """Return hidden states at EVERY layer for the last non-pad token: array [nlayers+1, N, d]."""
    outs = None
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEV)
        hs = m(**enc, output_hidden_states=True).hidden_states           # tuple (nlayers+1) of [b,t,d]
        last = enc["attention_mask"].sum(1) - 1                          # last non-pad idx per row
        step = np.stack([h[torch.arange(h.shape[0]), last].float().cpu().numpy() for h in hs])  # [L+1,b,d]
        outs = step if outs is None else np.concatenate([outs, step], axis=1)
    return outs                                                          # [L+1, N, d]


def eff_metrics(D):
    """D: [N,d] UNCENTERED paired diffs (row i = pos_i - neg_i). PR of D measures concept dimensionality:
       aligned diffs -> σ1 dominates -> PR≈1 (steerable axis); diverse diffs -> PR≫1 (broad subspace).
       vfull = mean-diff lies in D's row space, so its top-k right-singular reconstruction converges to it."""
    U, s, Vt = np.linalg.svd(D, full_matrices=False)                    # Vt: [r,d] right singular (concept subspace)
    s2 = s**2; PR = float((s2.sum()**2) / (np.square(s2).sum() + 1e-30))
    stable_rank = float(s2.sum() / (s[0]**2 + 1e-30))
    vfull = D.mean(0)                                                    # mean-diff steering direction (in row space of D)
    sep = float(np.linalg.norm(vfull) / (np.sqrt(np.mean(np.sum(D**2, 1))) + 1e-30))   # mean-diff vs rms row (axis-ness)
    return PR, stable_rank, Vt, vfull, sep


class Steer:
    """Additive steering (+vec) at a block's output — used only as a causality sanity check."""
    def __init__(self, layer): self.layer = layer; self.vec = None; self.h = None
    def __enter__(self):
        def hook(mod, inp, out):
            if self.vec is None: return out
            h = out[0] if isinstance(out, tuple) else out
            v = torch.tensor(self.vec, dtype=h.dtype, device=h.device)
            return (h + v,) + out[1:] if isinstance(out, tuple) else h + v
        self.h = self.layer.register_forward_hook(hook); return self
    def __exit__(self, *a): self.h.remove()


class Ablate:
    """Project the top-k concept directions (Vk: [k,d]) OUT of a block's output at ALL positions.
       k* = smallest k that destroys the concept = the causal dimensionality (Arditi single-dir vs Wollschläger cone)."""
    def __init__(self, layer): self.layer = layer; self.Vk = None; self.h = None
    def __enter__(self):
        def hook(mod, inp, out):
            if self.Vk is None: return out
            h = out[0] if isinstance(out, tuple) else out
            V = torch.tensor(self.Vk, dtype=h.dtype, device=h.device)   # [k,d], orthonormal rows
            h2 = h - (h @ V.T) @ V                                       # remove top-k concept subspace
            return (h2,) + out[1:] if isinstance(out, tuple) else h2
        self.h = self.layer.register_forward_hook(hook); return self
    def __exit__(self, *a): self.h.remove()


@torch.no_grad()
def measure_ld(m, tok, prompts, pos_id, neg_id, bs=16):
    """mean over prompts of (logit[pos]-logit[neg]) at last token, under whatever hook is currently armed."""
    vals = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i+bs], return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEV)
        lg = m(**enc).logits
        last = enc["attention_mask"].sum(1) - 1
        ll = lg[torch.arange(lg.shape[0]), last].float()
        vals.append((ll[:, pos_id] - ll[:, neg_id]).cpu().numpy())
    return float(np.concatenate(vals).mean())


def first_token_id(tok, s):
    ids = tok(s, add_special_tokens=False)["input_ids"]; return ids[0]


def run_concept(m, tok, layers, name):
    pairs, ptok, ntok, desc = CONCEPTS[name]
    pos_p = [p for p, _ in pairs]; neg_p = [n for _, n in pairs]
    Apos = last_tok_acts(m, tok, pos_p); Aneg = last_tok_acts(m, tok, neg_p)   # [L+1,N,d]
    L1 = Apos.shape[0]; pos_id = first_token_id(tok, ptok); neg_id = first_token_id(tok, ntok)
    # candidate layers by separability (cheap, no forward), then pick among them by CAUSAL ablatable signal (denom)
    seps = sorted(((eff_metrics(Apos[li] - Aneg[li])[4], li) for li in range(1, L1)), reverse=True)
    best = None
    for li in [l for _, l in seps[:6]]:
        PR, sr, Vt, vfull, sep = eff_metrics(Apos[li] - Aneg[li]); Vt = Vt[:min(len(Vt), 40)]
        with Ablate(layers[li - 1]) as ab:
            ab.Vk = None; LD0 = measure_ld(m, tok, pos_p, pos_id, neg_id)
            ab.Vk = Vt.astype(np.float32); LDfull = measure_ld(m, tok, pos_p, pos_id, neg_id)
        denom = LD0 - LDfull
        if best is None or denom > best["denom"]:
            best = dict(layer=li, PR=PR, stable_rank=sr, Vt=Vt, vfull=vfull, sep=sep, denom=denom, LD0=LD0, LDfull=LDfull)
    li = best["layer"]; Vt = best["Vt"]; vfull = best["vfull"]; denom = best["denom"]
    block = layers[li - 1]                                              # hidden_states[li] == output of block li-1
    vn = float(np.linalg.norm(vfull)); scale = ALPHA * vn
    # per-direction CAUSAL EFFECT: steer each SVD direction Vi individually at EQUAL magnitude -> e_i (readout response).
    # causal dimensionality = participation ratio of the |e_i| distribution: ~1 if one dir carries the effect (steerable
    # axis), ~K if spread (broad). Monotone & well-defined, UNLIKE ablation-rank (which overshot non-monotonically).
    # Equal magnitude (not s_i-weighted) so causal_PR is INDEPENDENT of the spectral weighting -> non-circular test vs spec_PR.
    dmodel = Vt.shape[1]; rng = np.random.RandomState(12345)
    Gr = rng.randn(len(Vt), dmodel); Gr = Gr - (Gr @ Vt.T) @ Vt        # random dirs ORTHOGONAL to concept subspace (control)
    Gr = Gr / (np.linalg.norm(Gr, axis=1, keepdims=True) + 1e-30)
    with Steer(block) as st:
        st.vec = None; base_neg = measure_ld(m, tok, neg_p, pos_id, neg_id)
        st.vec = (ALPHA * vfull).astype(np.float32); steer_full = measure_ld(m, tok, neg_p, pos_id, neg_id) - base_neg
        e = []
        for i in range(len(Vt)):
            st.vec = (scale * Vt[i]).astype(np.float32); e.append(measure_ld(m, tok, neg_p, pos_id, neg_id) - base_neg)
        r = []
        for j in range(len(Gr)):
            st.vec = (scale * Gr[j]).astype(np.float32); r.append(measure_ld(m, tok, neg_p, pos_id, neg_id) - base_neg)
        st.vec = None
    e = np.asarray(e, float); a = np.abs(e); tot = float(a.sum())
    rabs = np.abs(np.asarray(r, float)); rand_mean = float(rabs.mean()); rand_p95 = float(np.percentile(rabs, 95))
    causal_PR = float((tot ** 2) / (np.square(a).sum() + 1e-30))
    order = np.argsort(-a); cum = np.cumsum(a[order]) / (tot + 1e-30)
    causal_k90 = int(np.searchsorted(cum, 0.9) + 1) if tot > 1e-6 else None
    top1_causal = float(a.max() / (tot + 1e-30)) if tot > 1e-6 else None
    # CONTROL: concept-direction effects vs random off-concept effects (same magnitude). Floor-correct at rand p95.
    ratio = float(a.mean() / (rand_mean + 1e-9))
    sig = np.clip(a - rand_p95, 0, None); tots = float(sig.sum())
    causal_PR_signal = float((tots ** 2) / (np.square(sig).sum() + 1e-30)) if tots > 1e-6 else 0.0
    n_signal = int((a > rand_p95).sum())
    valid = bool((denom > 0.5) and (abs(steer_full) > 1.0))            # concept must be genuinely expressed & steerable
    res = {"concept": name, "desc": desc, "layer": li, "n_pairs": len(pairs), "n_dirs": len(Vt),
           "PR": round(best["PR"], 3), "stable_rank": round(best["stable_rank"], 3), "separability": round(best["sep"], 3),
           "ablatable": round(denom, 4), "steer_full": round(steer_full, 4),
           "causal_PR": round(causal_PR, 3), "causal_k90": causal_k90,
           "top1_causal_frac": round(top1_causal, 3) if top1_causal is not None else None,
           "rand_mean_abs": round(rand_mean, 3), "concept_mean_abs": round(float(a.mean()), 3), "concept_over_rand": round(ratio, 3),
           "causal_PR_signal": round(causal_PR_signal, 3), "n_signal_dirs": n_signal, "valid": valid,
           "e_top5": [round(float(x), 3) for x in np.sort(a)[::-1][:5]]}
    print(f"  [{name}] L{li} specPR={res['PR']} causalPR={res['causal_PR']} PRsig={res['causal_PR_signal']} nsig={n_signal} c/rand={ratio:.2f} steer={res['steer_full']} valid={valid}", flush=True)
    return res


def spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx @ ry) / (np.sqrt((rx@rx)*(ry@ry)) + 1e-30))


def main():
    t0 = time.time(); print(f"[PR] #3 subspace-dim | SMOKE={SMOKE} model={MODEL} alpha={ALPHA} dev={DEV}", flush=True)
    m, tok, layers, d = load_model()
    names = concept_list(); rows = []
    for nm in names:
        try: rows.append(run_concept(m, tok, layers, nm))
        except Exception as e:
            print(f"  [{nm}] ERROR {type(e).__name__}: {str(e)[:160]}", flush=True)
    valid = [r for r in rows if r.get("valid")]
    summary = {"model": MODEL, "tag": TAG, "smoke": bool(SMOKE), "alpha": ALPHA, "npairs": NPAIRS, "n_concepts": len(rows),
               "n_valid": len(valid), "prereg_lock": "10d04137", "concepts": rows}
    if len(valid) >= 5:
        sPR = [r["PR"] for r in valid]; cPR = [r["causal_PR"] for r in valid]; cPRs = [r["causal_PR_signal"] for r in valid]
        ck = [r["causal_k90"] for r in valid]; nsig = [r["n_signal_dirs"] for r in valid]; cor = [r["concept_over_rand"] for r in valid]
        rho = spearman(sPR, cPR); rho_s = spearman(sPR, cPRs); rho_ns = spearman(sPR, nsig); rho_k = spearman(sPR, ck)
        rng = np.random.RandomState(0); nul = [spearman(sPR, list(rng.permutation(cPRs))) for _ in range(PERM)]
        p_s = float((np.abs(nul) >= abs(rho_s)).mean())
        mcor = float(np.mean(cor))
        summary["core"] = {"n": len(valid), "mean_concept_over_rand": round(mcor, 3),
                           "spearman_specPR_causalPR": round(rho, 3), "spearman_specPR_causalPRsignal": round(rho_s, 3), "perm_p_signal": round(p_s, 4),
                           "spearman_specPR_nsignal": round(rho_ns, 3), "spearman_specPR_causalk90": round(rho_k, 3),
                           "CONTROL_confounded": bool(mcor < 1.5), "HIT": bool(abs(rho_s) >= 0.30 and p_s < 0.01 and mcor >= 1.5)}
        print(f"\n[PR] CONTROL mean concept/rand={mcor:.2f} (>=1.5 concept-specific; <1.5 CONFOUNDED) | "
              f"CORE rho(specPR,causalPRsignal)={rho_s:+.3f} p={p_s:.4f} rho(specPR,nsig)={rho_ns:+.3f} | HIT={summary['core']['HIT']}", flush=True)
    else:
        print(f"\n[PR] {len(valid)}/{len(rows)} valid concepts — need >=5 for correlation (smoke validates pipeline).", flush=True)
    OUT.write_text(json.dumps(summary, indent=1))
    print(f"[PR] wrote {OUT}  wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
