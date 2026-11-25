"""Ranking metrics and bootstrap intervals over customers."""
from __future__ import annotations

import numpy as np


def recall_at_k(top: np.ndarray, target: np.ndarray, k: int = 10) -> float:
    """Hits in the top k over min(k, number of target products). With large B2B orders the
    plain recall (hits over all targets) is capped far below one, so the denominator is
    capped at k, as in Liang et al. (2018)."""
    if len(target) == 0:
        return np.nan
    hits = np.isin(top[:k], target).sum()
    return hits / min(k, len(target))


def ndcg_at_k(top: np.ndarray, target: np.ndarray, k: int = 10) -> float:
    if len(target) == 0:
        return np.nan
    rel = np.isin(top[:k], target).astype(float)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (rel * disc[:len(rel)]).sum()
    idcg = disc[:min(k, len(target))].sum()
    return dcg / idcg


def per_user(tops: np.ndarray, targets: list[np.ndarray], k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    rec = np.array([recall_at_k(t, g, k) for t, g in zip(tops, targets)])
    nd = np.array([ndcg_at_k(t, g, k) for t, g in zip(tops, targets)])
    return rec, nd


def bootstrap_ci(values: np.ndarray, n: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and percentile interval, resampling customers with replacement."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n, len(v)))
    means = v[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def top_k(scores: np.ndarray, k: int = 10, exclude=None) -> np.ndarray:
    """Indices of the k highest scores per row; excluded positions get minus infinity."""
    s = np.array(scores, dtype=np.float64, copy=True)
    if exclude is not None:
        for row, cols in enumerate(exclude):
            if len(cols):
                s[row, cols] = -np.inf
    part = np.argpartition(-s, kth=min(k, s.shape[1] - 1), axis=1)[:, :k]
    order = np.take_along_axis(s, part, axis=1).argsort(axis=1)[:, ::-1]
    return np.take_along_axis(part, order, axis=1)
