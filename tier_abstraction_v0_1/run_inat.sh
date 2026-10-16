#!/bin/bash
set -e
mkdir -p /home/ubuntu/inat
echo "[boot] copying tarball FS->local SSD (heartbeat every 20s)..."
if [ ! -f /home/ubuntu/inat/2021_valid.tgz ]; then
  ( cp /lambda/nfs/iNat/2021_valid.tgz /home/ubuntu/inat/ ) &
  CPPID=$!
  while kill -0 $CPPID 2>/dev/null; do
    echo "[boot] ...copying $(du -h /home/ubuntu/inat/2021_valid.tgz 2>/dev/null | cut -f1)"
    sleep 20
  done
  wait $CPPID
fi
echo "[boot] tarball local. launching deeptree (extract local SSD, features -> FS npz)"
ulimit -n 65536
cd /home/ubuntu/dt
python3 deeptree_inat.py --seeds 0 1 2 --data_root /home/ubuntu/inat --out_dir /lambda/nfs/iNat
echo "[boot] DONE"
