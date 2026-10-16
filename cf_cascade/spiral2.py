"""SPIRAL-2: is the depth-collapse self-similar (fractal/nested) or a smooth monotone funnel?
Profile = the SPIRAL-1 curvature-controlled collapse (b_effrank_curvres), per carrier, per substrate.
PRE-REG (weakened prior, honest): SPIRAL-1 was monotone-smooth -> EXPECT layered-but-NOT-recursive.
A fractal must EXCEED a phase-randomized surrogate (preserves marginal + power-spectrum/autocorrelation;
destroys higher-order phase structure = genuine nesting). Hurst/spectrum are PRESERVED by the surrogate
by construction, so they cannot be the discriminator -> use BLOCK RE-COLLAPSE statistics that need
phase structure (each block re-expands then re-collapses, phase-locked to block boundaries = nesting).
Per-layer, carriers separate, NEVER pooled. Effect size vs surrogate, not t.
"""
import numpy as np, json
from scipy.stats import rankdata

rng=np.random.default_rng(7)
CARR=['qkv','h4h','4hh']
NLAY=24

# ---- load ratified SPIRAL-1 collapse profiles ----
S1=json.load(open('gateA/spiral1.json'))
def profile(sub,carr): return np.array(S1['substrates'][sub][carr]['b_effrank_curvres']['series'])

# GUARD: reproduce SPIRAL-1's banked rho on the loaded profile (bit-consistency)
d=np.arange(NLAY)
g=np.corrcoef(rankdata(profile('breadth9','qkv')),rankdata(d))[0,1]
assert abs(g-S1['substrates']['breadth9']['qkv']['b_effrank_curvres']['rho_vs_depth'])<1e-12, 'guard fail'
print(f"GUARD ok: reloaded breadth9/qkv rho={g:.6f} == banked SPIRAL-1\n")

def detrend_lin(y):
    x=np.arange(len(y))
    c=np.polyfit(x,y,1)
    return y-np.polyval(c,x)

def folded_descent(y,k):
    """remove global linear funnel, fold residual into blocks of size k, average by within-block
    position -> folded f(pos). descent = mean(first third) - mean(last third). >0 & large = each
    block re-expands at start, collapses by end = NESTED. ~0 = smooth funnel (no block structure)."""
    r=detrend_lin(y)
    nb=len(y)//k
    r=r[:nb*k].reshape(nb,k)
    f=r.mean(0)
    t=max(1,k//3)
    return f[:t].mean()-f[-t:].mean()

def reexpansion(y,k):
    """sum of within-block drops / global drop. >1 = blocks drop MORE than net (re-expand between
    blocks = sawtooth/nested). ~baseline = smooth funnel. baseline is scale-dependent -> use surrogate."""
    x=np.arange(len(y)); nb=len(y)//k
    gdrop=-np.polyfit(x,y,1)[0]*(len(y)-1)
    tot=0.0
    for b in range(nb):
        seg=y[b*k:(b+1)*k]; xs=np.arange(k)
        tot+=-np.polyfit(xs,seg,1)[0]*(k-1)
    return tot/(gdrop+1e-30)

def phase_surrogate(y):
    """preserve marginal + power spectrum, randomize phases (IAAFT-lite: phase-randomize then
    rank-remap to original marginal)."""
    yc=y-y.mean()
    F=np.fft.rfft(yc)
    ph=rng.uniform(0,2*np.pi,len(F))
    ph[0]=0
    if len(yc)%2==0: ph[-1]=0
    Fs=np.abs(F)*np.exp(1j*ph)
    s=np.fft.irfft(Fs,n=len(yc))
    # rank-remap to original marginal
    out=np.empty_like(y)
    out[np.argsort(s)]=np.sort(y)
    return out

def dfa_hurst(y):
    yi=np.cumsum(y-y.mean())
    scales=[4,6,8,12]; F=[]
    for s in scales:
        nb=len(yi)//s; rms=[]
        for b in range(nb):
            seg=yi[b*s:(b+1)*s]; xs=np.arange(s)
            rms.append(np.sqrt(np.mean((seg-np.polyval(np.polyfit(xs,seg,1),xs))**2)))
        F.append(np.mean(rms))
    return np.polyfit(np.log(scales),np.log(np.array(F)+1e-30),1)[0]

def test(y,niter=4000):
    res={}
    for k in [6,8,12]:
        fr=folded_descent(y,k); rx=reexpansion(y,k)
        fr_n=np.array([folded_descent(phase_surrogate(y),k) for _ in range(niter)])
        rx_n=np.array([reexpansion(phase_surrogate(y),k) for _ in range(niter)])
        res[f'k{k}']={
            'folded_descent':fr,'folded_z':(fr-fr_n.mean())/(fr_n.std()+1e-30),
            'folded_p':(np.sum(fr_n>=fr)+1)/(niter+1),
            'reexpansion':rx,'reexp_z':(rx-rx_n.mean())/(rx_n.std()+1e-30),
            'reexp_p':(np.sum(rx_n>=rx)+1)/(niter+1)}
    h=dfa_hurst(y)
    h_n=np.array([dfa_hurst(phase_surrogate(y)) for _ in range(niter)])
    res['dfa_hurst']={'H':h,'H_surr_mean':h_n.mean(),'H_z':(h-h_n.mean())/(h_n.std()+1e-30),
                      'note':'Hurst PRESERVED by surrogate by construction -> descriptive only, not the discriminator'}
    return res

out={'pre_reg':'expect layered-but-NOT-recursive (smooth funnel); fractal must exceed phase-randomized surrogate',
     'guard':f'reloaded SPIRAL-1 rho={g:.6f}','substrates':{}}
for sub in ['breadth9','crosspair4']:
    out['substrates'][sub]={c:test(profile(sub,c)) for c in CARR}
json.dump(out,open('gateA/spiral2.json','w'),indent=1)

print("=== SPIRAL-2: self-similar nesting vs smooth funnel (phase-randomized surrogate null) ===")
print("discriminator = BLOCK RE-COLLAPSE (folded-residual descent + re-expansion); >surrogate => recursive\n")
for sub in ['breadth9','crosspair4']:
    print(f"[{sub}]")
    for c in CARR:
        r=out['substrates'][sub][c]
        fz=[r[f'k{k}']['folded_z'] for k in [6,8,12]]
        rz=[r[f'k{k}']['reexp_z'] for k in [6,8,12]]
        print(f"  {c:4s} folded_z(k6/8/12)={fz[0]:+5.2f}/{fz[1]:+5.2f}/{fz[2]:+5.2f}   "
              f"reexp_z={rz[0]:+5.2f}/{rz[1]:+5.2f}/{rz[2]:+5.2f}   "
              f"DFA-H={r['dfa_hurst']['H']:.2f}(surr {r['dfa_hurst']['H_surr_mean']:.2f})")
    print()
# overall: max |folded_z| across all carrier/scale/substrate vs the +2 bar
allz=[abs(out['substrates'][s][c][f'k{k}']['folded_z']) for s in ['breadth9','crosspair4'] for c in CARR for k in [6,8,12]]
print(f"max |folded_z| over all carrier/scale/substrate = {max(allz):.2f}  (bar ~+2 for one-sided recursion)")
print("saved gateA/spiral2.json")
