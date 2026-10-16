"""CPU dry-run for Probe-13 dose-response (LOCK 49d14b48). No GPU/model. Validates: me() aggregation, per-dose row building
(N_phase2, N_random, N_conc, retA_drop), DOSES parsing, per-cell anchor, and a synthetic dose-response shape."""
import os, numpy as np
os.environ.setdefault("CELL", "opt350m"); os.environ.setdefault("DOSES", "1e-6,3e-6,1e-5,3e-5"); os.environ.setdefault("NSEEDS", "50")
import probe13_doseresponse_lambda as P
ok = True
def check(name, cond, detail=""):
    global ok; ok = ok and bool(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")

print("=== (1) me() aggregation ===")
check("me(list) = [mean, se]", (lambda m: abs(m[0]-2.0)<1e-9 and m[1]>0)(P.me([1.0,2.0,3.0])), str(P.me([1.0,2.0,3.0])))
check("me([]) safe", P.me([]) == [0.0,0.0], str(P.me([])))
check("me(single) se=0", P.me([1.5])==[1.5,0.0], str(P.me([1.5])))

print("=== (2) per-dose row building (paired N_phase2/N_random/N_conc/retA_drop) ===")
pre_sst=0.823; rng=np.random.default_rng(0); n=50
# synthetic: NoCH dp2 damage ~0.13; CH reduces it more at higher dose (dose-response). random ~0.
dp2_n=rng.normal(0.13,0.02,n)
def row_for(dose_reduce, retA):
    dp2_c=dp2_n - rng.normal(dose_reduce,0.02,n)   # childhood cuts phase2 damage by ~dose_reduce
    drk_n=rng.normal(0.00,0.01,n); drk_c=rng.normal(0.00,0.01,n)
    a=np.array
    Np=P.me((a(dp2_n)-a(dp2_c)).tolist()); Nr=P.me((a(drk_n)-a(drk_c)).tolist())
    rt=[retA,0.002]
    return {"lr":0,"retA":rt,"retA_drop":[pre_sst-rt[0],rt[1]],"N_phase2":Np,"N_random":Nr,
            "N_conc":[Np[0]-Nr[0],(Np[1]**2+Nr[1]**2)**0.5]}
r_lo=row_for(0.02,0.815); r_hi=row_for(0.12,0.760)   # gentle dose small protection; strong dose big protection+more cost
check("N_phase2 rises with dose", r_hi["N_phase2"][0] > r_lo["N_phase2"][0], f"lo={r_lo['N_phase2'][0]:.3f} hi={r_hi['N_phase2'][0]:.3f}")
check("retA_drop rises with dose", r_hi["retA_drop"][0] > r_lo["retA_drop"][0], f"lo_drop={r_lo['retA_drop'][0]:.3f} hi_drop={r_hi['retA_drop'][0]:.3f}")
check("N_conc = N_phase2 - N_random", abs(r_hi["N_conc"][0]-(r_hi["N_phase2"][0]-r_hi["N_random"][0]))<1e-9, f"{r_hi['N_conc'][0]:.4f}")
check("random control ~0", abs(r_hi["N_random"][0])<0.02, f"N_random={r_hi['N_random'][0]:+.4f}")

print("=== (3) DOSE-ARTIFACT vs REAL-ARCH read logic (two synthetic cells on retA_drop axis) ===")
# cell A and cell B: if they fall on ONE curve in (retA_drop -> N_phase2), it's dose; if B is systematically lower, arch diff.
drops=np.array([0.01,0.03,0.06,0.10]); NA=1.2*drops+0.005; NB_dose=1.2*drops+0.005; NB_arch=0.5*drops+0.005
def slope(x,y): return float(np.polyfit(x,y,1)[0])
check("DOSE-artifact: A and B slopes ~equal", abs(slope(drops,NA)-slope(drops,NB_dose))<0.05, f"sA={slope(drops,NA):.2f} sB={slope(drops,NB_dose):.2f}")
check("REAL-arch: B slope < A slope", slope(drops,NB_arch) < slope(drops,NA)-0.3, f"sA={slope(drops,NA):.2f} sB_arch={slope(drops,NB_arch):.2f}")

print("=== (4) wiring ===")
check("DOSES parsed to floats", P.DOSES==[1e-6,3e-6,1e-5,3e-5], str(P.DOSES))
check("4 cells incl opt13b w/ per-cell anchor 0.93", P.CELLS["opt13b"].get("anchor")==0.93, str(P.CELLS["opt13b"].get("anchor")))
check("no alttask/calibration in module", not any("alttask" in k.lower() for k in dir(P)), "clean (N_phase2 vs N_random only)")
check("K_RAND small (sweep)", P.K_RAND==3, str(P.K_RAND))

print(f"\n=== DRY-RUN {'ALL PASS' if ok else 'FAILED'} ===")
