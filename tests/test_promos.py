import numpy as np

from b2bretail import promos


def confounded(n=4000, effect=2.0, seed=0, heterogeneous=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    p = 1 / (1 + np.exp(-(1.2 * X[:, 0] - 0.5)))
    t = rng.binomial(1, p)
    tau = effect * (1 + X[:, 1]) if heterogeneous else np.full(n, effect)
    y = 5 + 3 * X[:, 0] + X[:, 2] + t * tau + rng.normal(size=n)
    return X, t, y, tau


def test_aipw_removes_selection_that_fools_the_naive_difference():
    X, t, y, _ = confounded()
    n = promos.nuisance(X, t, y)
    est, se = promos.aipw(t, y, n)
    naive = y[t == 1].mean() - y[t == 0].mean()
    assert abs(naive - 2.0) > 1.0
    assert abs(est - 2.0) < 3 * se + 0.15


def test_placebo_with_no_effect_is_close_to_zero():
    X, t, y, _ = confounded(effect=0.0, seed=1)
    est, se = promos.aipw(t, y, promos.nuisance(X, t, y))
    assert abs(est) < 3 * se + 0.15


def test_blp_slope_is_near_one_when_the_model_sees_real_heterogeneity():
    X, t, y, tau = confounded(seed=2, heterogeneous=True, n=6000)
    n = promos.nuisance(X, t, y)
    b = promos.blp(promos.dr_learner(X, t, y, n), promos.pseudo_outcome(t, y, n))
    assert b["slope"] > 0.6 and b["slope"] - 2 * b["slope_se"] > 0


def test_policy_value_of_treating_everyone_is_the_mean_with_treatment():
    X, t, y, tau = confounded(seed=3)
    n = promos.nuisance(X, t, y)
    v1, _ = promos.policy_value(np.ones(len(y), int), t, y, n)
    v0, _ = promos.policy_value(np.zeros(len(y), int), t, y, n)
    assert abs((v1 - v0) - promos.aipw(t, y, n)[0]) < 1e-9


def test_budget_choice_respects_the_budget_and_skips_losses():
    gain = np.array([5.0, 4.0, -1.0, 3.0, 0.5])
    cost = np.array([2.0, 2.0, 0.1, 1.0, 1.0])
    pick = promos.choose_budget(gain, cost, budget=3.0)
    assert (pick * cost).sum() <= 3.0 + 1e-9 and pick[2] == 0
    assert pick.tolist() == [1, 0, 0, 1, 0]
