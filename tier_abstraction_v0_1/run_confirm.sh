#!/bin/bash
set -e
ulimit -n 65536
cd /home/ubuntu/dt
if [ -f /lambda/nfs/iNat/inat_feats.npz ]; then echo "[boot] feats cache present, no extract needed"; else echo "[boot] WARN: no feats cache!"; fi
python3 deeptree_inat_confirm.py --seeds 0 1 2 --cache /lambda/nfs/iNat/inat_feats.npz --out_dir /lambda/nfs/iNat
echo "[boot] DONE"
