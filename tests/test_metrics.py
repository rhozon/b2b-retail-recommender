import numpy as np

from b2bretail import metrics


def test_recall_caps_denominator_at_k():
    top = np.arange(10)
    assert metrics.recall_at_k(top, np.array([0, 1, 50]), 10) == 2 / 3
    assert metrics.recall_at_k(top, np.arange(100), 10) == 1.0


def test_ndcg_perfect_and_order():
    target = np.array([3, 4])
    assert np.isclose(metrics.ndcg_at_k(np.array([3, 4, 0, 1]), target, 4), 1.0)
    early = metrics.ndcg_at_k(np.array([3, 0, 1, 4]), target, 4)
    late = metrics.ndcg_at_k(np.array([0, 1, 3, 4]), target, 4)
    assert 0 < late < early < 1


def test_top_k_excludes_and_sorts():
    s = np.array([[0.1, 0.9, 0.5, 0.7], [1.0, 0.0, 0.2, 0.3]])
    top = metrics.top_k(s, 2, exclude=[np.array([1]), np.array([])])
    assert top.tolist() == [[3, 2], [0, 3]]


def test_bootstrap_interval_contains_mean():
    v = np.random.default_rng(1).normal(0.3, 0.1, 500)
    m, lo, hi = metrics.bootstrap_ci(v)
    assert lo < m < hi and hi - lo < 0.05
