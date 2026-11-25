"""Promotion targeting on a semi-synthetic experiment.

The data has no promotions, so a randomised promotion is simulated on top of real customers.
Each customer's spend in the test window without the promotion is the real spend. The effect
of the promotion is a known function of the customer's real history: a voucher brings back
lapsed customers who used to spend, and on frequent heavy buyers it mostly gives away a
discount on orders they would place anyway. Half the customers are assigned the voucher at
random. Because the true effect is known for every customer, any targeting rule can be
scored exactly. This tests whether the uplift models recover who responds; it says nothing
about how real customers of this wholesaler respond to real promotions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

X_COLS = ["log_recency", "log_frequency", "log_monetary", "log_lines_per_invoice", "log_distinct_products", "uk",
          "first_days"]

REACTIVATION = 0.6   # extra spend on a lapsed customer, as a share of past monthly revenue
DISCOUNT = 0.10      # voucher value given away on spend that would have happened anyway
NOISE = 0.25         # noise on the outcome, as a share of past monthly revenue


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def true_effect(f: pd.DataFrame, scale: float = 1.0) -> np.ndarray:
    """Net effect of the voucher on revenue in the window (currency units)."""
    m = f["monthly_revenue"].clip(lower=0).to_numpy()
    lapsed = sigmoid((f["recency_days"].to_numpy() - 90) / 30)
    active = sigmoid((30 - f["recency_days"].to_numpy()) / 15)
    loyal = np.clip(np.log1p(f["frequency"].to_numpy()) / np.log1p(50), 0, 1)
    return scale * (REACTIVATION * m * lapsed - DISCOUNT * 2.0 * m * active * loyal)


def simulate(f: pd.DataFrame, y0: pd.Series, seed: int, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tau = true_effect(f, scale)
    t = rng.integers(0, 2, len(f))
    m = f["monthly_revenue"].clip(lower=0).to_numpy()
    y = y0.reindex(f.index).fillna(0).to_numpy() + t * tau + rng.normal(0, NOISE * m + 1.0)
    return t, y, tau


def _gbm(seed):
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=40, random_state=seed)


def t_learner(X, t, y, seed=0, folds=5):
    """Out-of-fold effect estimates from one outcome model per arm."""
    est = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        m1 = _gbm(seed).fit(X[tr][t[tr] == 1], y[tr][t[tr] == 1])
        m0 = _gbm(seed).fit(X[tr][t[tr] == 0], y[tr][t[tr] == 0])
        est[te] = m1.predict(X[te]) - m0.predict(X[te])
    return est


def x_learner(X, t, y, seed=0, folds=5, p=0.5):
    """X-learner (Kunzel et al., 2019) with the known assignment probability."""
    est = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        Xt, tt, yt = X[tr], t[tr], y[tr]
        m1 = _gbm(seed).fit(Xt[tt == 1], yt[tt == 1])
        m0 = _gbm(seed).fit(Xt[tt == 0], yt[tt == 0])
        d1 = yt[tt == 1] - m0.predict(Xt[tt == 1])
        d0 = m1.predict(Xt[tt == 0]) - yt[tt == 0]
        g1 = _gbm(seed).fit(Xt[tt == 1], d1)
        g0 = _gbm(seed).fit(Xt[tt == 0], d0)
        est[te] = p * g0.predict(X[te]) + (1 - p) * g1.predict(X[te])
    return est


def dr_learner(X, t, y, seed=0, folds=5, p=0.5):
    """Doubly robust learner (Kennedy, 2023): regress the doubly robust pseudo-outcome on X,
    with cross-fitted outcome models and the known assignment probability."""
    est = np.zeros(len(y))
    kf = list(KFold(folds, shuffle=True, random_state=seed).split(X))
    phi = np.zeros(len(y))
    for tr, te in kf:
        m1 = _gbm(seed).fit(X[tr][t[tr] == 1], y[tr][t[tr] == 1])
        m0 = _gbm(seed).fit(X[tr][t[tr] == 0], y[tr][t[tr] == 0])
        mu1, mu0 = m1.predict(X[te]), m0.predict(X[te])
        phi[te] = mu1 - mu0 + t[te] * (y[te] - mu1) / p - (1 - t[te]) * (y[te] - mu0) / (1 - p)
    for tr, te in kf:
        est[te] = _gbm(seed).fit(X[tr], phi[tr]).predict(X[te])
    return est


GRID = np.round(np.linspace(0, 1, 51), 2)


def gain_curve(score: np.ndarray, tau: np.ndarray, grid=GRID) -> np.ndarray:
    """True incremental revenue when the top share q of customers by `score` get the voucher."""
    order = np.argsort(-score, kind="stable")
    cum = np.concatenate([[0.0], np.cumsum(tau[order])])
    n = len(tau)
    return np.array([cum[int(round(q * n))] for q in grid])


def run(f: pd.DataFrame, y0: pd.Series, reps: int = 10, seed: int = 0, scale: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Curves (mean over replications) and the area under each curve relative to the oracle."""
    X = f[X_COLS].to_numpy(dtype=float)
    curves, areas = [], []
    for r in range(reps):
        t, y, tau = simulate(f, y0, seed + r, scale)
        scores = {
            "x_learner": x_learner(X, t, y, seed + r),
            "t_learner": t_learner(X, t, y, seed + r),
            "dr_learner": dr_learner(X, t, y, seed + r),
            "top_spenders": f["monetary"].to_numpy(),
            "most_lapsed": f["recency_days"].to_numpy(),
            "oracle": tau,
        }
        rand = GRID * tau.sum()
        oracle_area = np.trapezoid(gain_curve(tau, tau) - rand, GRID)
        for name, s in scores.items():
            g = gain_curve(s, tau)
            curves.append(pd.DataFrame({"rep": r, "policy": name, "q": GRID, "gain": g}))
            areas.append({"rep": r, "policy": name, "area_vs_oracle": float(np.trapezoid(g - rand, GRID) / oracle_area)})
        curves.append(pd.DataFrame({"rep": r, "policy": "random", "q": GRID, "gain": rand}))
        areas.append({"rep": r, "policy": "random", "area_vs_oracle": 0.0})
    c = pd.concat(curves).groupby(["policy", "q"], as_index=False)["gain"].mean()
    a = pd.DataFrame(areas).groupby("policy")["area_vs_oracle"].agg(["mean", "min", "max"]).reset_index()
    return c, a


def sensitivity(f: pd.DataFrame, y0: pd.Series, scales=(0.5, 1.0, 2.0, 4.0), reps: int = 5, seed: int = 100) -> pd.DataFrame:
    """Area relative to the oracle when the true effect is scaled up or down: how much signal
    the learners need before they beat simple rules."""
    rows = []
    for sc in scales:
        _, a = run(f, y0, reps=reps, seed=seed, scale=sc)
        rows.append(a.assign(scale=sc))
    return pd.concat(rows, ignore_index=True)
