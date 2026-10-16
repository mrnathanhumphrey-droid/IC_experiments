"""Loss terms: analytic VIB KL (the rate term, used as the MI proxy) + combined objective.

The KL(q(z_trans|z_low) || N(0,I)) is the variational information-bottleneck *rate*: an
analytic upper bound on I(z_low; z_trans). We log its mean as the mutual-information proxy
(no MINE needed). Combined loss: L = L_fine + lam*L_super + beta*KL.
"""
import torch
import torch.nn.functional as F


def kl_standard_normal(mu, logvar):
    """Per-sample KL(N(mu,sigma^2) || N(0,I)), summed over dims. Returns shape (B,)."""
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1.0, dim=-1)


def combined_loss(out, fine_y, coarse_y, beta, lam, use_super):
    fine_logits = out["fine_logits"]
    l_fine = F.cross_entropy(fine_logits, fine_y)

    if use_super and out["super_logits"] is not None:
        l_super = F.cross_entropy(out["super_logits"], coarse_y)
    else:
        l_super = torch.zeros((), device=fine_logits.device)

    kl = kl_standard_normal(out["mu"], out["logvar"]).mean()

    total = l_fine + lam * l_super + beta * kl
    parts = {
        "total": float(total.detach()),
        "l_fine": float(l_fine.detach()),
        "l_super": float(l_super.detach()),
        "kl": float(kl.detach()),
    }
    return total, parts


if __name__ == "__main__":
    mu = torch.zeros(8, 8)
    logvar = torch.zeros(8, 8)
    print("KL at N(0,I) should be ~0:", float(kl_standard_normal(mu, logvar).mean()))
    mu2 = torch.ones(8, 8)
    print("KL with mu=1:", float(kl_standard_normal(mu2, logvar).mean()))  # expect 0.5*8 = 4.0
