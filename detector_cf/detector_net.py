"""Borrowed backbone: tier_abstraction_v0_1/models.py::LowerTierCNN (3-block plain CNN,
AdaptiveAvgPool -> any input size). Adapted for binary deepfake detection: drop the VIB +
tier heads, add one 4th block (larger input than CIFAR-32) + a single binary logit head.

Input = 1-channel high-pass RESIDUAL of the face crop (forces forensic signal, not identity/
content). This is OUR harness repurposed, not a from-scratch build.
"""
import torch
import torch.nn as nn


class DetectorCNN(nn.Module):
    def __init__(self, in_ch=1, width=32):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(w, 2*w, 3, padding=1), nn.BatchNorm2d(2*w), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(2*w, 4*w, 3, padding=1), nn.BatchNorm2d(4*w), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(4*w, 4*w, 3, padding=1), nn.BatchNorm2d(4*w), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(4*w, 1)

    def forward(self, x):
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.head(self.drop(h)).squeeze(1)   # raw logit


if __name__ == "__main__":
    m = DetectorCNN(in_ch=1)
    for hw in (128, 160, 256):
        y = m(torch.randn(2, 1, hw, hw))
        print(f"input {hw}x{hw} -> logits {tuple(y.shape)}")
    print("params", sum(p.numel() for p in m.parameters()))
