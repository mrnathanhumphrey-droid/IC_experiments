"""SPIRAL-3: is the depth-collapse LOAD-BEARING (does it explain things the flat map couldn't)?
Two HARD pre-registered predictions from the inversion (seed concentrates where the map MERGES tasks):
 (A) per-layer SEED energy concentrates in the DEEP collapsed layers:
       pre-reg  rho(seed_energy, depth) > 0   AND   rho(seed_energy, eff_rank_collapse) < 0
       (seed is high where effective-rank is low = where tasks have merged into the contested core)
 (B) carrier-split = DIFFERENTIAL collapse: attention collapses harder than MLP -> attention more
       universal than MLP. pre-reg: per-carrier collapse-strength order == per-carrier universality order.
Per-layer, carriers separate, never pooled. Depth-shuffle null. Effect size vs SPIRAL-1 benchmark
(rho ~ -0.72, z ~ -3.5). Guard: SPIRAL-1 partial reproduces bit-exact on these same arrays (done).
"""
import numpy as np, json
from scipy.stats import rankdata

rng=np.random.default_rng(11)
CARRf={'qkv':'attention.query_key_value','h4h':'mlp.dense_h_to_4h','4hh':'mlp.dense_4h_to_h'}
CARR=['qkv','h4h','4hh']; NLAY=24

def srho(a,b): return np.corrcoef(rankdata(a),rankdata(b))[0,1]
def shuf_null(series, target, niter=5000):
    real=srho(series,target)
    nl=np.array([srho(series,rng.permutation(target)) for _ in range(niter)])
    return real,(real-nl.mean())/(nl.std()+1e-30),(np.sum(np.abs(nl)>=abs(real))+1)/(niter+1)

# ---- GUARD: reproduce SPIRAL-1 partial bit-exact on perlayer arrays ----
ps=np.load('gateA/perlayer_seed.npz',allow_pickle=True)
da=np.load('gateA/deflation_artifacts.npz',allow_pickle=True)
J=json.load(open('gateA/perlayer_seed_overlap.json'))
def partial_spear(x,y,z):
    rx,ry,rz=rankdata(x),rankdata(y),rankdata(z)
    rxy=np.corrcoef(rx,ry)[0,1];rxz=np.corrcoef(rx,rz)[0,1];ryz=np.corrcoef(ry,rz)[0,1]
    return (rxy-rxz*ryz)/np.sqrt((1-rxz**2)*(1-ryz**2)+1e-30)
seed64=ps['seed64__attention.query_key_value']; lam64=ps['mylam64__attention.query_key_value']
ct=lambda L: da[f'contest__gpt_neox.layers.{L}.attention.query_key_value.weight']
rec=np.array([partial_spear(seed64[L],ct(L),lam64[L]) for L in range(24)])
gd=np.max(np.abs(rec-np.array(J['report']['attention.query_key_value']['partial_per_layer'])))
assert gd<1e-12; print(f"GUARD ok: SPIRAL-1 partial reproduced max|diff|={gd:.2e}\n")

# ---- SPIRAL-1 collapse profiles (eff_rank; low = collapsed/merged) ----
S1=json.load(open('gateA/spiral1.json'))
collapse=lambda carr: np.array(S1['substrates']['breadth9'][carr]['b_effrank_curvres']['series'])

# ---- per-layer SEED energy, basis-consistent (top-64 + full) ----
cr=np.load('gateA/crosspair_residual.npz',allow_pickle=True)
co=np.load('gateA/crosspair_opt3.npz',allow_pickle=True)
CPT=['agnews','yelp','subj','code']
def seedE_perlayer(carr):
    sf=ps['seed_full__'+CARRf[carr]]; s64=ps['seed64__'+CARRf[carr]]
    return np.linalg.norm(sf,axis=1), np.linalg.norm(s64,axis=1)
def cp_seedE(carr):
    mats=[ (co if t=='code' else cr)[f'seed__{t}__{CARRf[carr]}'] for t in CPT]  # each (24,64)
    return np.mean([np.linalg.norm(m,axis=1) for m in mats],axis=0)

d=np.arange(NLAY)
A={}
for carr in CARR:
    efull,e64=seedE_perlayer(carr); ecp=cp_seedE(carr); col=collapse(carr)
    rec={}
    for nm,ser in [('seedfull',efull),('seed64',e64),('crosspair_seed',ecp)]:
        rD,zD,pD=shuf_null(ser,d)              # vs depth (pre-reg >0)
        rC=srho(ser,col)                       # vs collapse eff_rank (pre-reg <0)
        _,zC,pC=shuf_null(ser,col)
        rec[nm]={'rho_vs_depth':rD,'z_depth':zD,'p_depth':pD,
                 'rho_vs_collapse':rC,'z_collapse':zC,'p_collapse':pC,
                 'series':ser.tolist()}
    A[carr]=rec

# ---- (B) carrier-split: collapse strength vs universality, per carrier ----
crj=json.load(open('gateA/crosspair_residual.json'))
collapse_strength={c:-S1['substrates']['breadth9'][c]['b_effrank_curvres']['rho_vs_depth'] for c in CARR}  # +=stronger collapse
universality={}
for c in CARR:
    pairs=crj['report']['contest'][CARRf[c]]['pairs']
    universality[c]=float(np.mean([pairs[pk]['partial_mean'] for pk in pairs]))
B={'collapse_strength':collapse_strength,'universality':universality,
   'collapse_order':sorted(CARR,key=lambda c:-collapse_strength[c]),
   'universality_order':sorted(CARR,key=lambda c:-universality[c])}

out={'guard':f'SPIRAL-1 partial reproduced {gd:.2e}','A_seed_location':A,'B_carrier_split':B}
json.dump(out,open('gateA/spiral3.json','w'),indent=1)

print("=== SPIRAL-3 (A): does SEED concentrate in the DEEP collapsed layers? ===")
print("pre-reg: rho(seed,depth)>0  AND  rho(seed,collapse_effrank)<0\n")
for carr in CARR:
    print(f"[{carr}]")
    for nm in ['seedfull','seed64','crosspair_seed']:
        r=A[carr][nm]
        print(f"   {nm:15s} rho_depth={r['rho_vs_depth']:+.3f}(z={r['z_depth']:+5.2f})   "
              f"rho_collapse={r['rho_vs_collapse']:+.3f}(z={r['z_collapse']:+5.2f})")
    print()
print("=== SPIRAL-3 (B): carrier-split = differential collapse? (n=3 carriers, qualitative) ===")
for c in CARR:
    print(f"   {c:4s} collapse_strength={collapse_strength[c]:+.3f}   universality={universality[c]:.3f}")
print(f"   collapse order     : {' > '.join(B['collapse_order'])}")
print(f"   universality order : {' > '.join(B['universality_order'])}")
print(f"   ORDERS MATCH: {B['collapse_order']==B['universality_order']}")
print("\nsaved gateA/spiral3.json")
