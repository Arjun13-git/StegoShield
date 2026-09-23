"""Small, dependency-light statistics helpers for evaluation reporting.

Kept separate from training code: these operate only on already-computed
detector scores and never touch a model.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auc_rank(negative_scores: np.ndarray, positive_scores: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U rank formula (average ranks for ties).

    An implementation independent of sklearn's curve-based roc_auc_score,
    used to cross-check reported AUC values.
    """
    neg = np.asarray(negative_scores, dtype=np.float64)
    pos = np.asarray(positive_scores, dtype=np.float64)
    if len(neg) == 0 or len(pos) == 0:
        raise ValueError("both classes must be non-empty")
    ranks = rankdata(np.concatenate([neg, pos]))  # average ranks for ties
    rank_sum_pos = ranks[len(neg):].sum()
    u = rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order.

    Controls the family-wise error rate across the given family of tests.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * p_values[idx]
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)
    return [float(x) for x in adjusted]


def paired_auc_bootstrap(
    cover: np.ndarray,
    stego: np.ndarray,
    *,
    baseline_cover: np.ndarray | None = None,
    baseline_stego: np.ndarray | None = None,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap over SOURCES (a cover and its stego are resampled together).

    Returns the AUC and its (1-alpha) percentile interval. If a baseline
    (cover, stego) pair over the same sources is given, also returns the AUC
    difference (condition - baseline) with an interval computed on the SAME
    resampled sources for both conditions.
    """
    cover = np.asarray(cover, dtype=np.float64)
    stego = np.asarray(stego, dtype=np.float64)
    if len(cover) != len(stego):
        raise ValueError("cover and stego must be paired (same length)")
    n = len(cover)
    rng = np.random.default_rng(seed)
    aucs, deltas = [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        a = auc_rank(cover[idx], stego[idx])
        aucs.append(a)
        if baseline_cover is not None:
            deltas.append(a - auc_rank(baseline_cover[idx], baseline_stego[idx]))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out = {"auc": auc_rank(cover, stego), "auc_ci": [float(np.percentile(aucs, lo)), float(np.percentile(aucs, hi))]}
    if baseline_cover is not None:
        out["delta_auc"] = out["auc"] - auc_rank(baseline_cover, baseline_stego)
        out["delta_auc_ci"] = [float(np.percentile(deltas, lo)), float(np.percentile(deltas, hi))]
    return out
