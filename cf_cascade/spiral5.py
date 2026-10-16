"""SPIRAL-5: is the TWIST the load-bearing half of the depth-geometry?
SPIRAL-3 asked "does the CONTRACTION (funnel) predict seed-location/carrier-split" -> no/ambiguous.
SPIRAL-5 asks the same of the ROTATION (twist), with a better prior: the twist already reproduced the
carrier-split CRISPLY (qkv/h4h spiral, 4hh incoherent) where the funnel couldn't.
PRE-REG (committed before measuring):
 1. CURVATURE CONTROL is decisive — rotation must predict BEYOND curvature (twist rides the backbone;
    if rotation predicts only because curvature does, that's load-bearing-curvature, already known).
 2. DIRECTION: rotation-coherence tracks the broad-flow/universal split — MORE coherence -> MORE
    universal/shared (spiraling carriers qkv/up = broad-flow universal; incoherent 4hh = channel).
Per-(layer,carrier) for power; dimension-normalized coherence (vs each carrier's own random floor) so
the R^4096 down-proj isn't penalized for living in a bigger space. Carriers separate where between-carrier.
Effect size vs program benchmark (partials ~0.45-0.64), not t.
"""
import numpy as np, json
from scipy.stats import rankdata
rng=np.random.default_rng(31)
CARRf={'qkv':'attention.query_key_value','h4h':'mlp.dense_h_to_4h','4hh':'mlp.dense_4h_to_h'}
COFF={'qkv':0,'h4h':1,'4hh':2}; CARR=['qkv','h4h','4hh']; NLAY=24

def srho(a,b): return np.corrcoef(rankdata(a),rankdata(b))[0,1]
def partial_spear(x,y,z):
    rx,ry,rz=rankdata(x),rankdata(y),rankdata(z)
    rxy=np.corrcoef(rx,ry)[0,1];rxz=np.corrcoef(rx,rz)[0,1];ryz=np.corrcoef(ry,rz)[0,1]
    return (rxy-rxz*ryz)/np.sqrt((1-rxz**2)*(1-ryz**2)+1e-30)
def ov(A,B,k): return np.linalg.norm(A.T@B,'fro')**2/k

# ---- load ----
tv=np.load('gateA/taskA_variation.npz',allow_pickle=True)
ps=np.load('gateA/perlayer_seed.npz',allow_pickle=True)
bz=np.load('gateA/breadth_pass.npz',allow_pickle=True)
cr=np.load('gateA/crosspair_residual.npz',allow_pickle=True)
co=np.load('gateA/crosspair_opt3.npz',allow_pickle=True)
REAL=[str(x) for x in bz['real_keys']]
def E(t,d,carr): return bz[f'E__{t}__{d*3+COFF[carr]}']

# ---- GUARD: reproduce SPIRAL-4 consec (qkv 0.534) + Cut-1 G(sst,ag)=0.126 ----
Vq=tv['fragile__sst2__attention.query_key_value']; kq=Vq.shape[2]
consec_q=np.mean([ov(Vq[L],Vq[L+1],kq) for L in range(NLAY-1)])
Gag=np.mean([ov(Vq[L],tv['fragile__ag__attention.query_key_value'][L],kq) for L in range(NLAY)])
assert abs(consec_q-0.534)<2e-3 and abs(Gag-0.126)<2e-3, (consec_q,Gag)
print(f"GUARD ok: SPIRAL-4 consec_qkv={consec_q:.3f}(==0.534), Cut-1 G(sst,ag)={Gag:.3f}(==0.126)\n")

# ---- per-(layer,carrier) fields ----
def lam_layer(carr,L): return ps['mylam64__'+CARRf[carr]][L]            # (64,) curvature eigenvalues
def carrier_O(carr):
    V=tv[f'fragile__sst2__{CARRf[carr]}']; k=V.shape[2]
    O=np.array([[ov(V[i],V[j],k) for j in range(NLAY)] for i in range(NLAY)])
    return O
def cross_task_sim(carr,d):  # universality at this layer-carrier = mean pairwise sim of 9 task-Bs' seed
    M=[E(t,d,carr) for t in REAL]
    return np.mean([srho(M[i],M[j]) for i in range(len(M)) for j in range(i+1,len(M))])
def seed_energy(carr):  # per-layer seed energy (crosspair-seed, 4 task-Bs avg) -- SPIRAL-3 object
    CPT=['agnews','yelp','subj','code']
    return np.mean([np.linalg.norm((co if t=='code' else cr)[f'seed__{t}__{CARRf[carr]}'],axis=1) for t in CPT],axis=0)

# build long table over 72 (layer,carrier) units + per-carrier summaries
rows={'carr':[],'L':[],'rotcoh_norm':[],'curv':[],'univ':[],'seedE':[]}
percarr={}
for carr in CARR:
    O=carrier_O(carr); offdiag=np.mean([O[i,j] for i in range(NLAY) for j in range(NLAY) if abs(i-j)>1])
    consec=np.mean([O[L,L+1] for L in range(NLAY-1)])
    se=seed_energy(carr)
    for L in range(NLAY):
        # local rotation-coherence = mean of the (up to two) consecutive overlaps, normed by carrier floor
        nb=[O[L,L+1]] if L==0 else ([O[L-1,L]] if L==NLAY-1 else [O[L-1,L],O[L,L+1]])
        lc=np.mean(nb)/(offdiag+1e-30)
        rows['carr'].append(carr); rows['L'].append(L)
        rows['rotcoh_norm'].append(lc)
        rows['curv'].append(float(np.sum(lam_layer(carr,L))))
        rows['univ'].append(cross_task_sim(carr,L))
        rows['seedE'].append(float(se[L]))
    percarr[carr]={'consec':consec,'offdiag':offdiag,'coh_ratio':consec/offdiag,
                   'mean_curv':float(np.mean([np.sum(lam_layer(carr,L)) for L in range(NLAY)])),
                   'mean_univ':float(np.mean([cross_task_sim(carr,L) for L in range(NLAY)]))}
for k in rows: rows[k]=np.array(rows[k]) if k!='carr' else np.array(rows[k])

# ---- TEST A: rotation-coherence vs universality (carrier-split), BEYOND curvature ----
# A2 per-(layer,carrier) n=72 (powered, dimension-normalized coherence)
rc=rows['rotcoh_norm'].astype(float); cv=rows['curv'].astype(float); uv=rows['univ'].astype(float)
A2_raw=srho(rc,uv); A2_partial=partial_spear(rc,uv,cv)
# permutation null for the partial
NIT=5000
nl=np.array([partial_spear(rng.permutation(rc),uv,cv) for _ in range(NIT)])
A2_z=(A2_partial-nl.mean())/(nl.std()+1e-30); A2_p=(np.sum(np.abs(nl)>=abs(A2_partial))+1)/(NIT+1)
# does curvature itself predict universality (the thing rotation must beat)?
A2_curv_only=srho(cv,uv)

# A2 WITHIN-carrier decomposition (is the pooled partial real, or a between-carrier pooling artifact?)
A2_within={}
for carr in CARR:
    m=rows['carr']==carr
    A2_within[carr]={'raw':srho(rows['rotcoh_norm'][m].astype(float),rows['univ'][m].astype(float)),
                     'partial_beyond_curv':partial_spear(rows['rotcoh_norm'][m].astype(float),rows['univ'][m].astype(float),rows['curv'][m].astype(float))}
# A1 between-carrier (n=3, honest): coherence ratio vs universality (2 measures) vs curvature
contest_univ={'qkv':0.569,'h4h':0.636,'4hh':0.717}  # crosspair contest (SPIRAL-3)
A1={c:{'coh_ratio':percarr[c]['coh_ratio'],'univ_crosstask':percarr[c]['mean_univ'],
       'univ_contest':contest_univ[c],'curv':percarr[c]['mean_curv']} for c in CARR}

# ---- TEST B: rotation-rate vs seed-location, BEYOND curvature (per carrier, 24 layers) ----
B={}
for carr in CARR:
    m=rows['carr']==carr
    rrate=1.0/(rows['rotcoh_norm'][m].astype(float)+1e-9)  # higher = faster rotation (less coherence)
    se=rows['seedE'][m].astype(float); cvv=rows['curv'][m].astype(float)
    raw=srho(rrate,se); part=partial_spear(rrate,se,cvv)
    nlb=np.array([partial_spear(rng.permutation(rrate),se,cvv) for _ in range(2000)])
    z=(part-nlb.mean())/(nlb.std()+1e-30)
    B[carr]={'raw':raw,'partial_vs_curv':part,'z':z,'curv_only':srho(cvv,se)}

out={'guard':f'SPIRAL-4 consec {consec_q:.3f}, Cut-1 G {Gag:.3f}',
     'A2_rotation_vs_universality':{'raw':A2_raw,'partial_beyond_curv':A2_partial,'z':A2_z,'p':A2_p,
                                    'curv_only_predicts_univ':A2_curv_only,'n':72},
     'A2_within_carrier':A2_within,
     'A1_between_carrier':A1,
     'B_rotation_vs_seedloc':B}
json.dump(out,open('gateA/spiral5.json','w'),indent=1)

print("=== TEST A2: does rotation-coherence predict UNIVERSALITY beyond curvature? (n=72 layer-carrier) ===")
print(f"  raw rho(rotcoh, univ)        = {A2_raw:+.3f}")
print(f"  curvature-only rho(curv,univ)= {A2_curv_only:+.3f}   <- the backbone the twist must beat")
print(f"  PARTIAL rho(rotcoh,univ|curv)= {A2_partial:+.3f}   z={A2_z:+.2f}  p={A2_p:.4f}   (pre-reg: >0)")
print("  --- WITHIN-carrier decomposition (is the pooled partial real or a between-carrier pooling artifact?) ---")
for c in CARR: print(f"     {c:4s} raw={A2_within[c]['raw']:+.3f}  partial|curv={A2_within[c]['partial_beyond_curv']:+.3f}")
print("\n=== TEST A1: between-carrier (n=3, qualitative) ===")
print(f"  {'carr':5s}{'coh_ratio':>11s}{'univ_xtask':>11s}{'univ_contest':>13s}{'curv':>11s}")
for c in CARR: print(f"  {c:5s}{A1[c]['coh_ratio']:11.2f}{A1[c]['univ_crosstask']:11.3f}{A1[c]['univ_contest']:13.3f}{A1[c]['curv']:11.2e}")
print("\n=== TEST B: does rotation-RATE predict SEED-LOCATION beyond curvature? (per carrier, 24 layers) ===")
print(f"  {'carr':5s}{'raw':>8s}{'partial|curv':>14s}{'z':>7s}{'curv_only':>11s}")
for c in CARR:
    b=B[c]; print(f"  {c:5s}{b['raw']:+8.3f}{b['partial_vs_curv']:+14.3f}{b['z']:+7.2f}{b['curv_only']:+11.3f}")
print("\nsaved gateA/spiral5.json")
