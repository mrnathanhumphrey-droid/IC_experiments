#!/bin/bash
set -e
ulimit -n 65536
python3 -c "import h5py" 2>/dev/null || pip install -q h5py
cd /home/ubuntu/dt
H5=/lambda/nfs/iNat/3dshapes.h5
if [ ! -f "$H5" ]; then
  echo "[boot] downloading 3DShapes h5 (~255MB, one-time to FS)..."
  wget -q -O "$H5" https://storage.googleapis.com/3d-shapes/3dshapes.h5
fi
echo "[boot] 3dshapes present ($(du -h $H5 | cut -f1)); running form3"
python3 form3_compositional.py --h5 "$H5" --n_sub 120000 --splits 5 --seeds 0 1 2 --out_dir /lambda/nfs/iNat
echo "[boot] DONE"
