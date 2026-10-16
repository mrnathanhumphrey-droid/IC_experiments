"""SPIRAL-1: is task-A's neighborhood map layered coarse->fine by network depth?
CPU-only, on existing artifacts. Per-layer, never pooled (carriers separate too).
Pre-registered direction: early=broad/shared/low-res -> deep=specific/separated/high-res.
  (a) cross-task-B resolution  = mean pairwise sim among task-Bs' seed profiles  -> predict DOWN with depth
  (b) effective dim of task-stack = effective rank of (task x dir) seed matrix    -> predict UP with depth
  (c) per-layer cross-pair partial (universality strength)                        -> predict DOWN with depth
Primary null: SHUFFLE depth index, recompute trend spearman. Trend must live in the ORDER.
"""
import numpy as np, json
from scipy.stats import rankdata

rng = np.random.default_rng(0)
CARRIERS = {'qkv':'attention.query_key_value','h4h':'mlp.dense_h_to_4h','4hh':'mlp.dense_4h_to_h'}
COFF = {'qkv':0,'h4h':1,'4hh':2}
NLAY = 24

def prank(x): return rankdata(x)
def pcorr(a,b):  # pearson on ranks = spearman
    return np.corrcoef(prank(a),prank(b))[0,1]
def partial_spear(x,y,z):
    rx,ry,rz=prank(x),prank(y),prank(z)
    rxy=np.corrcoef(rx,ry)[0,1]; rxz=np.corrcoef(rx,rz)[0,1]; ryz=np.corrcoef(ry,rz)[0,1]
    return (rxy-rxz*ryz)/np.sqrt((1-rxz**2)*(1-ryz**2)+1e-30)

def eff_rank(M):  # M: (n_task, n_dir) -> participation ratio of singular spectrum
    s=np.linalg.svd(M-M.mean(0,keepdims=True), compute_uv=False)
    ev=s**2
    if ev.sum()<=0: return np.nan
    return (ev.sum()**2)/(ev**2).sum()

def trend_null(measure, niter=5000):
    d=np.arange(len(measure))
    real=np.corrcoef(prank(measure),prank(d))[0,1]
    null=np.empty(niter)
    for i in range(niter):
        null[i]=np.corrcoef(prank(measure),prank(rng.permutation(d)))[0,1]
    z=(real-null.mean())/(null.std()+1e-30)
    p=(np.sum(np.abs(null)>=abs(real))+1)/(niter+1)
    return real, z, p, null.std()

# ---------- substrate 1: breadth E (9 real task-Bs, 128 curvature-ordered dirs) ----------
bz=np.load('gateA/breadth_pass.npz',allow_pickle=True)
REAL=[str(x) for x in bz['real_keys']]   # 9 tasks
def E(task, d, carr): return bz[f'E__{task}__{d*3+COFF[carr]}']  # (128,) curvature-ordered
NDIR_B=128
dir_idx=np.arange(NDIR_B)  # curvature-rank proxy (dirs sorted by lambda desc)

# ---------- substrate 2: crosspair (4 task-Bs, 64 dirs, explicit lam) ----------
cr=np.load('gateA/crosspair_residual.npz',allow_pickle=True)
co=np.load('gateA/crosspair_opt3.npz',allow_pickle=True)
CP_TASKS=['agnews','yelp','subj','code']
def CPseed(task,carr):
    src = co if task=='code' else cr
    return src[f'seed__{task}__{CARRIERS[carr]}']   # (24,64)
def CPlam(carr): return cr[f'lam__{CARRIERS[carr]}']  # (24,64)

out={'guard':'partial reproduced 4.4e-16 (see spiral run log)','substrates':{}}

def run_substrate(name, get_layer_taskmat, lamfn, ntask):
    """get_layer_taskmat(d,carr)->(ntask,ndir); lamfn(d,carr)->ctrl vector(ndir)"""
    res={}
    for carr in CARRIERS:
        a_raw=np.zeros(NLAY); a_res=np.zeros(NLAY); b_raw=np.zeros(NLAY); b_res=np.zeros(NLAY)
        for d in range(NLAY):
            M=get_layer_taskmat(d,carr)          # (ntask,ndir)
            ctrl=lamfn(d,carr)
            # (a) mean pairwise sim among tasks
            raw_p=[]; res_p=[]
            for i in range(ntask):
                for j in range(i+1,ntask):
                    raw_p.append(pcorr(M[i],M[j]))
                    res_p.append(partial_spear(M[i],M[j],ctrl))
            a_raw[d]=np.mean(raw_p); a_res[d]=np.mean(res_p)
            # (b) eff rank of task-stack (rank-normalized rows so curvature backbone doesn't dominate scale)
            Mr=np.vstack([prank(M[i]) for i in range(ntask)])
            b_raw[d]=eff_rank(Mr)
            # curvature-residualized rows for (b)
            Mres=np.vstack([prank(M[i])-np.polyval(np.polyfit(prank(ctrl),prank(M[i]),1),prank(ctrl)) for i in range(ntask)])
            b_res[d]=eff_rank(Mres)
        carr_res={}
        for label,series in [('a_meanpairsim_raw',a_raw),('a_meanpairsim_curvres',a_res),
                             ('b_effrank_raw',b_raw),('b_effrank_curvres',b_res)]:
            real,z,p,ns=trend_null(series)
            carr_res[label]={'series':series.tolist(),'rho_vs_depth':real,'z_vs_shuffle':z,'p':p,'null_std':ns}
        res[carr]=carr_res
    return res

# breadth
out['substrates']['breadth9'] = run_substrate(
    'breadth9',
    lambda d,carr: np.vstack([E(t,d,carr) for t in REAL]),
    lambda d,carr: dir_idx.astype(float),
    len(REAL))
# crosspair
out['substrates']['crosspair4'] = run_substrate(
    'crosspair4',
    lambda d,carr: np.vstack([CPseed(t,carr)[d] for t in CP_TASKS]),
    lambda d,carr: CPlam(carr)[d],
    len(CP_TASKS))

# ---------- measure (c): per-layer cross-pair partial (universality strength) ----------
crj=json.load(open('gateA/crosspair_residual.json'))
c_res={}
for carr in CARRIERS:
    pairs=crj['report']['contest'][CARRIERS[carr]]['pairs']
    # average the 3 pair partial-per-layer profiles (contest = universality)
    prof=np.mean([np.array(pairs[pk]['partial_per_layer']) for pk in pairs],axis=0)
    real,z,p,ns=trend_null(prof)
    c_res[carr]={'series':prof.tolist(),'rho_vs_depth':real,'z_vs_shuffle':z,'p':p,'null_std':ns}
out['c_crosspair_partial_universality']=c_res

json.dump(out, open('gateA/spiral1.json','w'), indent=1)

# ---------- report ----------
def line(carr,lab,d): r=d['rho_vs_depth']; print(f"   {carr:4s} {lab:28s} rho={r:+.3f}  z_shuf={d['z_vs_shuffle']:+6.2f}  p={d['p']:.4f}")
print("\n=== SPIRAL-1: depth trend (rho vs depth) + shuffle null ===")
print("PRE-REG: (a) DOWN, (b) UP, (c) DOWN  with depth = coarse->fine\n")
for sub in ['breadth9','crosspair4']:
    print(f"[{sub}]")
    for carr in CARRIERS:
        for lab in ['a_meanpairsim_raw','a_meanpairsim_curvres','b_effrank_raw','b_effrank_curvres']:
            line(carr,lab,out['substrates'][sub][carr][lab])
    print()
print("[c] per-layer cross-pair partial (universality strength)")
for carr in CARRIERS: line(carr,'c_universality',out['c_crosspair_partial_universality'][carr])
print("\nsaved gateA/spiral1.json")
