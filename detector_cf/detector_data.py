"""Face-crop dataset for the learned detector — replaces tier_abstraction's CIFAR loader.

Reads pre-cropped face PNGs per family (prep_crops.py output). On each __getitem__:
  - random JPEG-compress at a quality drawn from QUALITY_POOL  (== compression AUGMENTATION:
    the model learns the COMPRESSED regime directly, the whole point of Mark B)
  - high-pass RESIDUAL (gray - Gaussian blur)  -> 1-channel forensic input
  - resize to INPUT, light flip aug
Returns (residual[1,H,W], label{0,1}, gen_id{0=real,1=insw,2=sim,3=ghost}).
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

FAM_LABEL = {"real": (0, 0), "inswap": (1, 1), "simswap": (1, 2), "ghost": (1, 3),
             "ffppfake": (1, 5)}
FAM_DIR = {"real": "real", "inswap": "inswap", "simswap": "simswap", "ghost": "ghost",
           "ffppfake": "fake"}
QUALITY_POOL = [100, 100, 90, 70, 50, 30, 18]   # 100 twice = include clean ~30% of the time


def jpeg(img, q):
    if q >= 100:
        return img
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def residual(bgr, ksize=5):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(g, (ksize, ksize), 0)
    return g - blur


class FaceCropDS(Dataset):
    def __init__(self, root, input_size=160, train=True, fixed_quality=None, families=None):
        self.root = Path(root); self.input = input_size; self.train = train
        self.fixed_quality = fixed_quality
        fams = families or list(FAM_LABEL.keys())
        self.items = []   # (path, label, gen_id)
        for fam in fams:
            label, gid = FAM_LABEL[fam]
            for p in sorted(glob.glob(str(self.root / FAM_DIR[fam] / "*.png"))):
                self.items.append((p, label, gid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, gid = self.items[i]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.input, self.input, 3), np.uint8)
        if self.fixed_quality is not None:
            q = self.fixed_quality
        else:
            q = QUALITY_POOL[np.random.randint(len(QUALITY_POOL))] if self.train else 100
        img = jpeg(img, q)
        res = residual(img)
        res = cv2.resize(res, (self.input, self.input), interpolation=cv2.INTER_AREA)
        if self.train and np.random.rand() < 0.5:
            res = res[:, ::-1].copy()
        t = torch.from_numpy(res).unsqueeze(0).float()
        return t, float(label), int(gid)


def stems_by_gen(ds):
    """Return gen_id array aligned with dataset order (for LOGO splitting)."""
    return np.array([it[2] for it in ds.items])
