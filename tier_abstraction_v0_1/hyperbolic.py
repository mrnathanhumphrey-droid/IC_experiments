"""Minimal Poincare-ball ops (curvature c=1) with the numerical guards that hyperbolic
training needs. Used by Form 1 (hyperbolic z_trans). float32-safe via clamping.
"""
import torch

EPS = 1e-5


def project(x, eps=EPS):
    """Project into the open unit ball (norm < 1-eps)."""
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    maxnorm = 1 - eps
    return torch.where(norm > maxnorm, x / norm * maxnorm, x)


def expmap0(v, eps=EPS):
    """Exponential map at the origin: tangent vector v in R^K -> point in the ball."""
    norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    z = torch.tanh(norm.clamp(max=15.0)) * v / norm
    return project(z, eps)


def dist(x, y, eps=EPS):
    """Poincare distance. x,y broadcastable [..., K] -> [...]."""
    x = project(x, eps); y = project(y, eps)
    diff2 = (x - y).pow(2).sum(-1)
    xn = x.pow(2).sum(-1); yn = y.pow(2).sum(-1)
    denom = (1 - xn).clamp_min(eps) * (1 - yn).clamp_min(eps)
    arg = (1 + 2 * diff2 / denom).clamp_min(1 + eps)
    return torch.acosh(arg)


if __name__ == "__main__":
    torch.manual_seed(0)
    v = torch.randn(5, 8) * 2
    z = expmap0(v)
    print("ball norms < 1 ?", bool((z.norm(dim=-1) < 1).all()), z.norm(dim=-1).max().item())
    p = expmap0(torch.randn(3, 8) * 0.1)
    d = dist(z.unsqueeze(1), p.unsqueeze(0))   # [5,3]
    print("dist shape", tuple(d.shape), "finite?", bool(torch.isfinite(d).all()), "d>=0?", bool((d >= 0).all()))
