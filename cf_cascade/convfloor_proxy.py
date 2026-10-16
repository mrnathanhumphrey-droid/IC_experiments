"""Convergence-floor PROXY (the feasibility-gate before the feasibility-gate).

Question (pre-committed): at the component counts a Pythia matrix offers, up to
what moment order p is <|x|^p> reliably estimable? c2 is the CURVATURE of zeta(p)
across p, so if the usable-p cap is too low (<=3), there is too little leverage to
fit c2 robustly per matrix -> provision Pythia-1B (d=2048) from the start instead
of wasting a 410M run.

The damage tail is UNKNOWN until Gate B, so this BRACKETS across candidate tails:
  Gaussian       -> all moments finite, fast convergence (best case)
  lognormal      -> all moments finite but heavy, slow high-p convergence
  Student-t(nu)  -> tail index = nu; population moment <|x|^p> diverges for p>=nu
  Pareto(a)      -> tail index = a; same
Heavy-tailed damage is plausible because damage is projected onto a heavy-tailed
(alpha~3.7, Martin-Hill ESD) spectral basis. alpha grid brackets 2.5..5.

COMPONENT COUNTS:
  1024   single 410M square matrix (d=1024)
  2048   single 1B square matrix   (d=2048)
  24576  pooled across 24 layers, 410M (homogeneity-pool, for reference only)

METRIC (decision-relevant): across-trial relative scatter of the sample moment
  s(p,N) = std_T( mean_N(|x|^p) ) / median_T( mean_N(|x|^p) )    over T independent draws.
A single matrix gives ONE such estimate; s is how unreliable that one estimate is.
usable-p cap = largest p with s(p,N) < S_THRESH.   (report at 0.25 and 0.50)
Cross-check: probe1-style running-mean plateau (stable within +/-2% over a 30% N-extension).
PRE-REGISTERED READ: cap<=3 at N=1024 (alpha~3.7) -> 410M component-starved -> go 1B.
"""
import numpy as np

RNG = np.random.default_rng(20260617)
T = 600                      # independent trials (independent "matrices")
P_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
N_LIST = [1024, 2048, 24576]
S_THRESH = [0.25, 0.50]


def draw(dist, alpha, n):
    """Return |x| samples, shape (T, n). alpha = tail index where applicable."""
    if dist == "gaussian":
        x = RNG.standard_normal((T, n))
    elif dist == "lognormal":
        x = RNG.lognormal(mean=0.0, sigma=1.0, size=(T, n))
    elif dist == "student_t":
        x = RNG.standard_t(df=alpha, size=(T, n))        # tail index = df = alpha
    elif dist == "pareto":
        x = (RNG.pareto(a=alpha, size=(T, n)) + 1.0)      # tail index = alpha
    else:
        raise ValueError(dist)
    return np.abs(x)


def usable_cap(scatter_by_p, thresh):
    """Largest p (contiguous from low end) with scatter < thresh."""
    cap = 0.0
    for p in P_GRID:
        if scatter_by_p[p] < thresh:
            cap = p
        else:
            break
    return cap


def plateau_p(absx_big, p, N):
    """probe1 cross-check on ONE big draw: running mean of |x|^p; is it stable
    within +/-2% from n=N to n=1.3N?"""
    xp = absx_big ** p
    run = np.cumsum(xp) / np.arange(1, xp.size + 1)
    n2 = int(min(1.3 * N, xp.size - 1))
    if n2 <= N:
        return False
    base = run[N - 1]
    seg = run[N - 1:n2]
    return bool(np.all(np.abs(seg - base) <= 0.02 * abs(base)))


def main():
    print("=== CONVERGENCE-FLOOR PROXY ===  T=%d trials" % T)
    print("usable-p cap = largest p with across-trial relative scatter < threshold\n")
    cases = [("gaussian", None), ("lognormal", None),
             ("student_t", 5.0), ("student_t", 3.7), ("student_t", 3.0), ("student_t", 2.5),
             ("pareto", 3.7), ("pareto", 3.0)]

    big = {}  # for plateau cross-check
    for dist, alpha in cases:
        label = f"{dist}" + (f"(a={alpha})" if alpha else "")
        print(f"--- {label} ---")
        # one big draw for the probe1 plateau cross-check
        bigdraw = draw(dist, alpha if alpha else 4.0, 60000)[0]
        for N in N_LIST:
            scatter = {}
            absx = draw(dist, alpha if alpha else 4.0, N)   # (T, N)
            for p in P_GRID:
                m = np.mean(absx ** p, axis=1)              # (T,)
                med = np.median(m)
                scatter[p] = float(np.std(m) / med) if med > 0 else float("inf")
            caps = {th: usable_cap(scatter, th) for th in S_THRESH}
            # probe1 plateau cap on the big draw
            plat = 0.0
            for p in P_GRID:
                if plateau_p(bigdraw, p, N):
                    plat = p
                else:
                    break
            sc_str = " ".join(f"p{p:g}:{scatter[p]:.2f}" for p in [2.0, 3.0, 4.0, 6.0])
            print(f"  N={N:6d}  cap@.25={caps[0.25]:>4g}  cap@.50={caps[0.50]:>4g}  "
                  f"plateau_cap={plat:>4g}   [{sc_str}]")
        print()

    print("=== PRE-REGISTERED READ ===")
    print("  If cap@.25 <= 3 at N=1024 for student_t/pareto(a~3.7):")
    print("  -> 410M (1024 comp) is component-starved for robust c2 curvature -> provision 1B (2048).")
    print("  Compare N=1024 (410M) vs N=2048 (1B): does doubling components lift the cap past 3-4?")


if __name__ == "__main__":
    main()
