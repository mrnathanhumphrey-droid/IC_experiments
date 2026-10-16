"""SPIRAL-4: does the contested subspace ROTATE (twist) with depth, or just contract (cone)?
The funnel (SPIRAL-1/2/3) measured the RADIUS (how collapsed) at each depth. This measures the
ORTHOGONAL property: the ORIENTATION of a FIXED-rank (16-dim) contested subspace across depth.
Object: fragile__sst2__<carrier> = per-layer top-16 contested/seed eigenvectors in a common
per-carrier coordinate space (qkv/h4h in R^1024, 4hh in R^4096). Carriers separate, never pooled.

PRE-REG (committed before measuring): the subspace rotates COHERENTLY toward the deep core's
orientation as it descends -> overlap with the SHALLOW (L0) orientation DECREASES with depth
(rho<0), overlap with the DEEP (L23) orientation INCREASES (rho>0), and overlap decays monotonically
with depth-gap. Primary null: SHUFFLE layer order (twist must live in the DEPTH ORDER).

Forms verdict:
 CONE        : overlap stays HIGH at all gaps (fixed axis, no reorientation).
 SPIRAL      : overlap decays graded+monotone with gap AND drift is one-directional (shuffle-killed).
 INCOHERENT  : overlap ~ random floor at gap=1 already (no depth-smooth drift; shuffle survives).
"""
import numpy as np, json
from scipy.stats import rankdata
rng=np.random.default_rng(23)
CARRf={'qkv':'attention.query_key_value','h4h':'mlp.dense_h_to_4h','4hh':'mlp.dense_4h_to_h'}
CARR=['qkv','h4h','4hh']; NLAY=24
z=np.load('gateA/taskA_variation.npz',allow_pickle=True)

def srho(a,b): return np.corrcoef(rankdata(a),rankdata(b))[0,1]

def overlap(A,B,k):  # mean sq cosine of principal angles in [floor,1]; 1=identical
    return np.linalg.norm(A.T@B,'fro')**2/k

# GUARD already done externally: orth 1.2e-15, G(sst,ag)=0.126==banked 0.13.
out={'guard':'fragile vectors orthonormal 1.2e-15; G(sst,ag) reproduces Cut-1 0.13','carriers':{}}
for carr in CARR:
    V=z[f'fragile__sst2__{CARRf[carr]}']   # (24, D, 16)
    D=V.shape[1]; k=V.shape[2]; floor=k/D
    # full overlap matrix
    O=np.array([[overlap(V[i],V[j],k) for j in range(NLAY)] for i in range(NLAY)])
    # (1) overlap vs depth-gap
    Ogap=np.array([np.mean([O[L,L+g] for L in range(NLAY-g)]) for g in range(1,NLAY)])
    rho_Ogap=srho(Ogap,np.arange(1,NLAY))
    # (2) drift from shallow / toward deep
    o_from0=np.array([O[0,L] for L in range(1,NLAY)])      # vs L
    o_tolast=np.array([O[NLAY-1,L] for L in range(NLAY-1)])
    rho_from0=srho(o_from0,np.arange(1,NLAY))
    rho_tolast=srho(o_tolast,np.arange(NLAY-1))
    # (3) nearest-orientation-neighbor is a depth-neighbor?
    nn=[]
    for L in range(NLAY):
        order=np.argsort(-O[L]); order=order[order!=L]
        nn.append(1.0 if order[0] in (L-1,L+1) else 0.0)
    nn_frac=np.mean(nn); nn_chance=2/(NLAY-1)
    consec=np.mean([O[L,L+1] for L in range(NLAY-1)])
    offdiag=np.mean([O[i,j] for i in range(NLAY) for j in range(NLAY) if abs(i-j)>1])

    # SHUFFLE-ORDER NULL (twist must live in depth order)
    NIT=5000
    nl_rho=np.empty(NIT); nl_consec=np.empty(NIT); nl_nn=np.empty(NIT)
    for it in range(NIT):
        p=rng.permutation(NLAY)
        Os=O[np.ix_(p,p)]
        Og=np.array([np.mean([Os[L,L+g] for L in range(NLAY-g)]) for g in range(1,NLAY)])
        nl_rho[it]=srho(Og,np.arange(1,NLAY))
        nl_consec[it]=np.mean([Os[L,L+1] for L in range(NLAY-1)])
        nns=[]
        for L in range(NLAY):
            o=np.argsort(-Os[L]); o=o[o!=L]; nns.append(1.0 if o[0] in (L-1,L+1) else 0.0)
        nl_nn[it]=np.mean(nns)
    def zp(real,nl,side):  # side=-1 expect below, +1 expect above
        z=(real-nl.mean())/(nl.std()+1e-30)
        p=(np.sum((nl<=real) if side<0 else (nl>=real))+1)/(NIT+1)
        return z,p
    z_rho,p_rho=zp(rho_Ogap,nl_rho,-1)        # decay -> rho more negative than shuffle
    z_consec,p_consec=zp(consec,nl_consec,+1) # consecutive more aligned than shuffle
    z_nn,p_nn=zp(nn_frac,nl_nn,+1)

    out['carriers'][carr]={
        'D':int(D),'k':int(k),'random_floor':floor,
        'consec_overlap':consec,'offdiag_overlap':offdiag,'O_gap1':float(Ogap[0]),'O_gapmax':float(Ogap[-1]),
        'rho_Ogap_vs_gap':rho_Ogap,'z_shuf':z_rho,'p_shuf':p_rho,
        'consec_z_shuf':z_consec,'consec_p':p_consec,
        'rho_drift_from_L0':rho_from0,'rho_drift_to_L23':rho_tolast,
        'nn_depth_frac':nn_frac,'nn_chance':nn_chance,'nn_z_shuf':z_nn,'nn_p':p_nn,
        'Ogap':Ogap.tolist()}
json.dump(out,open('gateA/spiral4.json','w'),indent=1)

print("=== SPIRAL-4: twist (orientation drift) of the contested subspace vs depth ===")
print("PRE-REG: coherent rotation toward deep core -> O decays w/ gap, drift_from_L0<0, drift_to_L23>0, shuffle-killed\n")
print(f"{'carr':5s}{'floor':>7s}{'consec':>8s}{'offdiag':>8s}{'O_g1':>7s}{'O_gmax':>8s}{'rhoGap':>8s}{'zShuf':>7s}{'driftL0':>8s}{'driftL23':>9s}{'nnFrac':>7s}{'nnChe':>7s}{'nnZ':>6s}")
for c in CARR:
    r=out['carriers'][c]
    print(f"{c:5s}{r['random_floor']:7.3f}{r['consec_overlap']:8.3f}{r['offdiag_overlap']:8.3f}{r['O_gap1']:7.3f}{r['O_gapmax']:8.3f}"
          f"{r['rho_Ogap_vs_gap']:+8.3f}{r['z_shuf']:+7.2f}{r['rho_drift_from_L0']:+8.3f}{r['rho_drift_to_L23']:+9.3f}"
          f"{r['nn_depth_frac']:7.2f}{r['nn_chance']:7.2f}{r['nn_z_shuf']:+6.2f}")
print("\nreads: consec>>floor & consec>>offdiag & rhoGap<0(zShuf<-2) & drift_from_L0<0 & drift_to_L23>0 & nnFrac>>chance = SPIRAL")
print("       consec~offdiag~high & O_gmax high = CONE ; consec~floor & shuffle survives = INCOHERENT")
print("saved gateA/spiral4.json")
