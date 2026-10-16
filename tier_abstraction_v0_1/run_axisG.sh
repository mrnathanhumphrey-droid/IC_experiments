#!/bin/bash
set -e
ulimit -n 65536
cd /home/ubuntu/dt
if [ -f /lambda/nfs/iNat/inat_feats.npz ]; then echo "[boot] feats cache present"; else echo "[boot] WARN: no feats cache"; fi
python3 granularity_axisG.py --seeds 0 1 2 --cache /lambda/nfs/iNat/inat_feats.npz --out_dir /lambda/nfs/iNat
echo "[boot] DONE"
