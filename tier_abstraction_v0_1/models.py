"""Two-tier model: plain-CNN LowerTier -> VIB TranslationModule -> UpperTier.

MoE dropped for v0.1 (orthogonal to the tier-abstraction question; adds gating
instability). The load-bearing piece is the variational information bottleneck in
the translation module. UpperTier optionally carries a superclass head (Arm B);
Arm A omits it (lam=0) to test whether abstraction emerges from the bottleneck alone.
"""
import torch
import torch.nn as nn


class LowerTierCNN(nn.Module):
    """Plain 3-block CNN -> 64-dim z_low."""
    def __init__(self, z_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 32->16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 16->8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),# 8->4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, z_dim)

    def forward(self, x):
        h = self.features(x)
        h = self.pool(h).flatten(1)
        return self.fc(h)  # z_low


class TranslationVIB(nn.Module):
    """64 -> 128 -> (mu, logvar) of width K. Reparameterized sample is z_trans."""
    def __init__(self, in_dim=64, hidden=128, K=8):
        super().__init__()
        self.K = K
        self.body = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True))
        self.head = nn.Linear(hidden, 2 * K)

    def forward(self, z_low):
        h = self.body(z_low)
        mu, logvar = self.head(h).chunk(2, dim=-1)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z_trans = mu + std * torch.randn_like(std)
        else:
            z_trans = mu  # deterministic at eval (probes/metrics use the mean)
        return z_trans, mu, logvar


class UpperTier(nn.Module):
    """MLP adjudicator on z_trans: fine head always, superclass head optional (Arm B)."""
    def __init__(self, K=8, n_fine=100, n_coarse=20, use_super=True):
        super().__init__()
        self.use_super = use_super
        self.fine_head = nn.Sequential(nn.Linear(K, 64), nn.ReLU(inplace=True), nn.Linear(64, n_fine))
        if use_super:
            self.super_head = nn.Sequential(nn.Linear(K, 64), nn.ReLU(inplace=True), nn.Linear(64, n_coarse))

    def forward(self, z_trans):
        fine_logits = self.fine_head(z_trans)
        super_logits = self.super_head(z_trans) if self.use_super else None
        return fine_logits, super_logits


class FullModel(nn.Module):
    def __init__(self, z_low_dim=64, K=8, n_fine=100, n_coarse=20, use_super=True):
        super().__init__()
        self.lower = LowerTierCNN(z_low_dim)
        self.translate = TranslationVIB(z_low_dim, 128, K)
        self.upper = UpperTier(K, n_fine, n_coarse, use_super)

    def forward(self, x):
        z_low = self.lower(x)
        z_trans, mu, logvar = self.translate(z_low)
        fine_logits, super_logits = self.upper(z_trans)
        return {
            "z_low": z_low, "z_trans": z_trans, "mu": mu, "logvar": logvar,
            "fine_logits": fine_logits, "super_logits": super_logits,
        }


if __name__ == "__main__":
    m = FullModel(use_super=True)
    x = torch.randn(4, 3, 32, 32)
    m.train()
    out = m(x)
    for k, v in out.items():
        print(k, None if v is None else tuple(v.shape))
    n_params = sum(p.numel() for p in m.parameters())
    print("params", n_params)
