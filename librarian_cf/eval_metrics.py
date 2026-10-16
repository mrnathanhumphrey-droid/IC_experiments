"""
eval_metrics.py — ranking metrics for retrieval (Phase B1+). Pure, CI-pinned.

Each example is a candidate pool with binary relevance. We score the pool (by the net, or
by the embedding baseline), rank by score desc, and measure how well the relevant ones float
to the top:

  P@1        — is the top-ranked candidate relevant?
  MRR        — 1 / rank of the FIRST relevant candidate (reciprocal rank, averaged)
  Recall@k   — fraction of all relevant candidates landing in the top-k
  AUC        — P(a random positive outranks a random negative), per-example then averaged

Deterministic tie-break: equal scores keep input order (stable sort), so a baseline that
returns a flat/constant score degrades gracefully to pool order rather than randomly.
Examples without BOTH a positive and a negative are not rankable and are skipped by `evaluate`.
"""
from __future__ import annotations

from dataclasses import dataclass


def _ranked_labels(scores: list, labels: list) -> list:
    """Labels reordered by score DESC, stable on ties (input order preserved)."""
    order = sorted(range(len(scores)), key=lambda i: -scores[i])  # stable: equal -> input order
    return [labels[i] for i in order]


def p_at_1(scores: list, labels: list) -> float:
    return float(_ranked_labels(scores, labels)[0]) if scores else 0.0


def reciprocal_rank(scores: list, labels: list) -> float:
    for r, lab in enumerate(_ranked_labels(scores, labels), start=1):
        if lab:
            return 1.0 / r
    return 0.0


def recall_at_k(scores: list, labels: list, k: int) -> float:
    total = sum(1 for x in labels if x)
    if total == 0:
        return 0.0
    hit = sum(1 for lab in _ranked_labels(scores, labels)[:k] if lab)
    return hit / total


def example_auc(scores: list, labels: list) -> float | None:
    """Per-example ranking AUC via the rank-sum (Mann-Whitney) identity; ties get 0.5
    credit. None when the example has no positive or no negative (AUC undefined)."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


@dataclass
class Report:
    n: int
    p_at_1: float
    mrr: float
    recall_at_3: float
    recall_at_5: float
    auc: float

    def __str__(self) -> str:
        return (f"n={self.n}  P@1={self.p_at_1:.3f}  MRR={self.mrr:.3f}  "
                f"R@3={self.recall_at_3:.3f}  R@5={self.recall_at_5:.3f}  AUC={self.auc:.3f}")


def evaluate(scored: list) -> Report:
    """`scored` = list of (scores, labels) per example. Skips unrankable examples
    (no positive or no negative). Returns macro-averaged metrics."""
    rankable = [(s, l) for s, l in scored if any(l) and not all(l)]
    if not rankable:
        return Report(0, 0, 0, 0, 0, 0)
    n = len(rankable)
    aucs = [a for a in (example_auc(s, l) for s, l in rankable) if a is not None]
    return Report(
        n=n,
        p_at_1=sum(p_at_1(s, l) for s, l in rankable) / n,
        mrr=sum(reciprocal_rank(s, l) for s, l in rankable) / n,
        recall_at_3=sum(recall_at_k(s, l, 3) for s, l in rankable) / n,
        recall_at_5=sum(recall_at_k(s, l, 5) for s, l in rankable) / n,
        auc=(sum(aucs) / len(aucs)) if aucs else 0.0,
    )


if __name__ == "__main__":
    # a perfect ranker vs a reversed one
    perfect = [([3, 2, 1, 0], [1, 1, 0, 0])]
    reversed_ = [([0, 1, 2, 3], [1, 1, 0, 0])]
    print("perfect :", evaluate(perfect))
    print("reversed:", evaluate(reversed_))
