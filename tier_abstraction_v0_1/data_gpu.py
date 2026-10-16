"""GPU-resident CIFAR-100: the whole dataset lives on the GPU as uint8; augmentation runs
on-device. The model is tiny (~121k params) so a CPU DataLoader starves the 5070 — this
removes the host<->device bottleneck entirely (each epoch becomes a few seconds).

Augmentation matches torchvision semantics: pad-0 + per-image random 32-crop + random
hflip applied on the [0,1] image, THEN normalize. Eval/probe passes normalize only.
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from data import FINE_TO_COARSE, CIFAR100_MEAN, CIFAR100_STD, SPLIT_SEED


class GPUData:
    def __init__(self, device, data_root="./data"):
        self.device = device
        tr = torchvision.datasets.CIFAR100(data_root, train=True, download=True)
        te = torchvision.datasets.CIFAR100(data_root, train=False, download=True)

        def to_uint8(ds):
            x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).contiguous()  # [N,3,32,32] uint8
            y = torch.tensor(ds.targets, dtype=torch.long)
            return x, y

        x_tr, y_tr = to_uint8(tr)
        x_te, y_te = to_uint8(te)

        g = torch.Generator().manual_seed(SPLIT_SEED)
        perm = torch.randperm(len(x_tr), generator=g)
        tr_idx, val_idx = perm[:45000], perm[45000:]

        f2c = torch.tensor(FINE_TO_COARSE, dtype=torch.long)

        def pack(x, yf):
            return {"x": x.to(device), "fine": yf.to(device), "coarse": f2c[yf].to(device)}

        self.splits = {
            "train": pack(x_tr[tr_idx], y_tr[tr_idx]),
            "val": pack(x_tr[val_idx], y_tr[val_idx]),
            "test": pack(x_te, y_te),
        }
        self.mean = torch.tensor(CIFAR100_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR100_STD, device=device).view(1, 3, 1, 1)

    def _normalize(self, xb_uint8):
        return (xb_uint8.float().div_(255.0) - self.mean) / self.std

    def _augment(self, xf01):
        """xf01: [B,3,32,32] float in [0,1]. pad-0(4) + per-image random crop + random hflip."""
        B = xf01.size(0)
        xp = F.pad(xf01, (4, 4, 4, 4))  # [B,3,40,40], zero pad (matches RandomCrop padding=4)
        oy = torch.randint(0, 9, (B,), device=xf01.device)
        ox = torch.randint(0, 9, (B,), device=xf01.device)
        ar = torch.arange(32, device=xf01.device)
        rows = (oy.view(B, 1) + ar.view(1, 32))          # [B,32] over H
        cols = (ox.view(B, 1) + ar.view(1, 32))          # [B,32] over W
        xp = torch.gather(xp, 2, rows.view(B, 1, 32, 1).expand(B, 3, 32, 40))
        xp = torch.gather(xp, 3, cols.view(B, 1, 1, 32).expand(B, 3, 32, 32))
        flip = torch.rand(B, device=xf01.device) < 0.5
        xp[flip] = torch.flip(xp[flip], dims=[3])
        return xp

    def iter_train(self, batch_size, generator):
        s = self.splits["train"]
        n = s["x"].size(0)
        order = torch.randperm(n, generator=generator, device=self.device)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            xf01 = s["x"][idx].float().div(255.0)
            xa = self._augment(xf01)
            xn = (xa - self.mean) / self.std
            yield xn, s["fine"][idx], s["coarse"][idx]

    def iter_eval(self, split, batch_size=1024):
        s = self.splits[split]
        n = s["x"].size(0)
        for i in range(0, n, batch_size):
            xb = s["x"][i:i + batch_size]
            yield self._normalize(xb), s["fine"][i:i + batch_size], s["coarse"][i:i + batch_size]


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = GPUData(dev)
    g = torch.Generator(device=dev).manual_seed(0)
    xn, f, c = next(d.iter_train(128, g))
    print("train batch", tuple(xn.shape), "mean~0?", round(float(xn.mean()), 3),
          "fine", tuple(f.shape), "coarse range", int(c.min()), int(c.max()))
    xe, fe, ce = next(d.iter_eval("test", 512))
    print("eval batch", tuple(xe.shape), "n_test", d.splits["test"]["x"].size(0))
