"""CIFAR-100 loaders with the standard 100->20 superclass map and a fixed 45k/5k/10k split.

torchvision's CIFAR100 only returns the fine label; the coarse (superclass) label is
attached here via the canonical Krizhevsky fine->coarse lookup. The 45k/5k split of the
50k train set uses a FIXED seed so probe/eval features are comparable across runs.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import torchvision
import torchvision.transforms as T

# Canonical CIFAR-100 fine(0-99) -> coarse(0-19) map, indexed by torchvision's fine id order.
FINE_TO_COARSE = np.array([
    4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
    3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
    6, 11,  5, 10,  7,  6, 13, 15,  3, 15,
    0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
    5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
    16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
    10, 3,  2, 12, 12, 16, 12,  1,  9, 19,
    2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
    16, 19, 2,  4,  6, 19,  5,  5,  8, 19,
    18, 1,  2, 15,  6,  0, 17,  8, 14, 13,
], dtype=np.int64)

CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)

SPLIT_SEED = 1234  # fixed: train/val carve is identical across all runs


class CoarseWrapper(Dataset):
    """Wraps a CIFAR100 dataset/subset to return (img, fine, coarse)."""
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, fine = self.base[i]
        coarse = int(FINE_TO_COARSE[fine])
        return img, fine, coarse


def _transforms():
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    return train_tf, eval_tf


def get_loaders(batch_size=128, data_root="./data", num_workers=4):
    """Returns dict of loaders:
      train       : 45k, augmented, shuffled  (for training)
      train_eval  : 45k, NO aug, not shuffled (for fitting linear probes on frozen features)
      val         : 5k,  NO aug                (model selection / monitoring)
      test        : 10k, NO aug                (probe + abstraction-metric evaluation)
    """
    train_tf, eval_tf = _transforms()

    full_train_aug = torchvision.datasets.CIFAR100(data_root, train=True, download=True, transform=train_tf)
    full_train_eval = torchvision.datasets.CIFAR100(data_root, train=True, download=True, transform=eval_tf)
    test_eval = torchvision.datasets.CIFAR100(data_root, train=False, download=True, transform=eval_tf)

    n = len(full_train_aug)  # 50000
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx, val_idx = perm[:45000], perm[45000:]

    train_ds = CoarseWrapper(Subset(full_train_aug, train_idx))
    train_eval_ds = CoarseWrapper(Subset(full_train_eval, train_idx))
    val_ds = CoarseWrapper(Subset(full_train_eval, val_idx))
    test_ds = CoarseWrapper(test_eval)

    pin = torch.cuda.is_available()
    persist = num_workers > 0
    common = dict(num_workers=num_workers, pin_memory=pin, persistent_workers=persist)
    if persist:
        common["prefetch_factor"] = 4
    return {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **common),
        "train_eval": DataLoader(train_eval_ds, batch_size=512, shuffle=False, **common),
        "val": DataLoader(val_ds, batch_size=512, shuffle=False, **common),
        "test": DataLoader(test_ds, batch_size=512, shuffle=False, **common),
    }


if __name__ == "__main__":
    loaders = get_loaders(batch_size=128)
    xb, fb, cb = next(iter(loaders["train"]))
    print("batch img", tuple(xb.shape), "fine", tuple(fb.shape), "coarse", tuple(cb.shape))
    print("fine range", int(fb.min()), int(fb.max()), "coarse range", int(cb.min()), int(cb.max()))
    print("n_coarse_unique", len(set(FINE_TO_COARSE.tolist())))
    print("split sizes", len(loaders["train"].dataset), len(loaders["val"].dataset), len(loaders["test"].dataset))
