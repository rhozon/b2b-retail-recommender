"""Who to send a promotion to, on real campaigns: The Complete Journey (84.51°).

Online Retail II has no promotions, so this module works on another public panel: a year of
grocery purchases by 2,469 households, with the households each campaign reached and the
coupons they redeemed. The three "Type A" campaigns each reached about 1,100 households.

The campaigns were not randomised, so every estimate here is observational:

* the effect of a campaign on spend during the campaign is estimated with augmented inverse
  probability weighting (AIPW), cross-fitted, adjusting for each household's behaviour before;
* a placebo runs the same estimator on spend in the weeks just BEFORE the campaign, which the
  campaign cannot have caused: if it is far from zero, the adjustment has not removed the
  selection, and the campaign estimate cannot be trusted;
* the same estimator on the change in spend (campaign weeks minus the weeks before) combines
  the adjustment with a difference in differences, which removes selection on stable levels;
* "mde" is the minimum detectable effect, 2.8 standard errors (5% test, 80% power);
* heterogeneous effects come from a doubly robust learner (Kennedy, 2023), and targeting rules
  are compared by their AIPW policy value, the standard off-policy estimate;
* a budget split chooses households to maximise estimated net gain (margin on extra spend
  minus the coupon cost) under a coupon budget, as a 0-1 knapsack solved with scipy's MILP.

    python -m b2bretail.promos [--dir ~/data/completejourney] [--out results] [--site DIR]
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

DATA_DIR = Path(os.environ.get("CJ_DIR", Path.home() / "data" / "completejourney"))
TYPE_A = (8, 13, 18)
FEATURES = ["spend_wk", "trips_wk", "products", "weeks_active", "retail_disc_share", "coupon_disc_wk",
            "recency_days", "stores", "prior_campaigns", "prior_redemptions"]
CLIP = 0.02


def load(path: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    t = pd.read_csv(path / "transactions.csv.gz", parse_dates=["transaction_timestamp"],
                    usecols=["household_id", "store_id", "basket_id", "product_id", "sales_value", "retail_disc",
                             "coupon_disc", "transaction_timestamp"])
    d = pd.read_csv(path / "campaign_descriptions.csv.gz", parse_dates=["start_date", "end_date"])
    c = pd.read_csv(path / "campaigns.csv.gz")
    r = pd.read_csv(path / "coupon_redemptions.csv.gz", parse_dates=["redemption_date"])
    return {"tx": t, "desc": d, "camp": c, "redeem": r}


def _window(tx: pd.DataFrame, a: pd.Timestamp, b: pd.Timestamp) -> pd.DataFrame:
    return tx[(tx["transaction_timestamp"] >= a) & (tx["transaction_timestamp"] < b)]


def design(D: dict, campaign_id: int, history_weeks: int = 10) -> pd.DataFrame:
    """One row per household active before the campaign.

    Time line, for a campaign that starts at s and lasts L weeks:
    features from [s - L - history, s - L), placebo outcome on [s - L, s), outcome on [s, s + L).
    Features never see the placebo window, so the placebo is a fair test of the adjustment.
    """
    row = D["desc"].set_index("campaign_id").loc[campaign_id]
    s, e = row["start_date"], row["end_date"] + pd.Timedelta(days=1)
    L = e - s
    weeks = L / pd.Timedelta(weeks=1)
    f0, f1 = s - L - pd.Timedelta(weeks=history_weeks), s - L
    tx = D["tx"]
    h = _window(tx, f0, f1)
    g = h.groupby("household_id")
    X = pd.DataFrame({
        "spend_wk": g["sales_value"].sum() / history_weeks,
        "trips_wk": g["basket_id"].nunique() / history_weeks,
        "products": g["product_id"].nunique(),
        "weeks_active": g["transaction_timestamp"].apply(lambda x: x.dt.to_period("W").nunique()),
        "retail_disc_share": g["retail_disc"].sum() / g["sales_value"].sum().clip(lower=1e-9),
        "coupon_disc_wk": g["coupon_disc"].sum() / history_weeks,
        "recency_days": (f1 - g["transaction_timestamp"].max()).dt.days,
        "stores": g["store_id"].nunique(),
    })
    camp = D["camp"].merge(D["desc"], on="campaign_id")
    prior = camp[camp["start_date"] < f1].groupby("household_id").size()
    red = D["redeem"][D["redeem"]["redemption_date"] < f1].groupby("household_id").size()
    X["prior_campaigns"] = prior.reindex(X.index).fillna(0)
    X["prior_redemptions"] = red.reindex(X.index).fillna(0)
    X["treated"] = X.index.isin(D["camp"].loc[D["camp"]["campaign_id"] == campaign_id, "household_id"]).astype(int)
    pl = _window(tx, f1, s).groupby("household_id")["sales_value"].sum() / weeks
    y = _window(tx, s, e).groupby("household_id")
    X["placebo_wk"] = pl.reindex(X.index).fillna(0)
    X["spend_camp_wk"] = (y["sales_value"].sum() / weeks).reindex(X.index).fillna(0)
    X["coupon_camp_wk"] = (y["coupon_disc"].sum() / weeks).reindex(X.index).fillna(0)
    return X


def _gbr(seed):
    return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30,
                                         random_state=seed)


def _gbc(seed):
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30,
                                          random_state=seed)


@dataclass
class Nuisance:
    e: np.ndarray    # propensity, out of fold
    m0: np.ndarray   # outcome without the campaign
    m1: np.ndarray   # outcome with the campaign


def nuisance(X: np.ndarray, t: np.ndarray, y: np.ndarray, seed: int = 0, folds: int = 5) -> Nuisance:
    e, m0, m1 = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed).split(X):
        e[te] = _gbc(seed).fit(X[tr], t[tr]).predict_proba(X[te])[:, 1]
        m1[te] = _gbr(seed).fit(X[tr][t[tr] == 1], y[tr][t[tr] == 1]).predict(X[te])
        m0[te] = _gbr(seed).fit(X[tr][t[tr] == 0], y[tr][t[tr] == 0]).predict(X[te])
    return Nuisance(np.clip(e, CLIP, 1 - CLIP), m0, m1)


def pseudo_outcome(t: np.ndarray, y: np.ndarray, n: Nuisance) -> np.ndarray:
    """Doubly robust score: its mean is the AIPW effect, its regression on X the DR-learner."""
    return n.m1 - n.m0 + t * (y - n.m1) / n.e - (1 - t) * (y - n.m0) / (1 - n.e)


def aipw(t: np.ndarray, y: np.ndarray, n: Nuisance) -> tuple[float, float]:
    phi = pseudo_outcome(t, y, n)
    return float(phi.mean()), float(phi.std(ddof=1) / np.sqrt(len(phi)))


def dr_learner(X: np.ndarray, t: np.ndarray, y: np.ndarray, n: Nuisance, seed: int = 0, folds: int = 5) -> np.ndarray:
    phi = pseudo_outcome(t, y, n)
    out = np.zeros(len(y))
    for tr, te in KFold(folds, shuffle=True, random_state=seed + 1).split(X):
        out[te] = _gbr(seed).fit(X[tr], phi[tr]).predict(X[te])
    return out


def policy_value(pi: np.ndarray, t: np.ndarray, y: np.ndarray, n: Nuisance) -> tuple[float, float]:
    """AIPW estimate of mean outcome per household if households with pi = 1 got the campaign
    and the rest did not, and its standard error."""
    mu = np.where(pi == 1, n.m1, n.m0)
    w = np.where(pi == 1, t / n.e, (1 - t) / (1 - n.e))
    v = mu + w * (y - mu)
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def choose_budget(gain: np.ndarray, cost: np.ndarray, budget: float) -> np.ndarray:
    """0-1 knapsack: pick households maximising total gain with total cost within budget.
    Households with non-positive gain are never picked."""
    n = len(gain)
    c = -np.where(gain > 0, gain, 0.0)
    res = milp(c, constraints=[LinearConstraint(cost[None, :], -np.inf, budget)], integrality=np.ones(n),
               bounds=Bounds(np.zeros(n), np.where(gain > 0, 1.0, 0.0)), options={"time_limit": 60})
    if res.x is None:
        raise RuntimeError(res.message)
    return (res.x > 0.5).astype(int)


def analyse(D: dict, campaign_id: int, seed: int = 0) -> dict:
    df = design(D, campaign_id)
    X, t = df[FEATURES].to_numpy(float), df["treated"].to_numpy()
    out = {"campaign_id": campaign_id, "households": int(len(df)), "treated": int(t.sum())}
    df["change_wk"] = df["spend_camp_wk"] - df["placebo_wk"]
    for name, col in (("campaign", "spend_camp_wk"), ("placebo", "placebo_wk"), ("change", "change_wk")):
        y = df[col].to_numpy(float)
        n = nuisance(X, t, y, seed)
        est, se = aipw(t, y, n)
        out[name] = {"naive": float(y[t == 1].mean() - y[t == 0].mean()), "aipw": est, "se": se,
                     "mde": 2.8 * se}
        if name == "change":  # targeting works on the change, the outcome the placebo supports
            nu, yc = n, y
    out["propensity"] = {"q01": float(np.quantile(nu.e, 0.01)), "q99": float(np.quantile(nu.e, 0.99)),
                         "share_clipped": float(np.mean((nu.e <= CLIP + 1e-12) | (nu.e >= 1 - CLIP - 1e-12)))}
    tau = dr_learner(X, t, yc, nu, seed)
    df["cate_wk"] = tau
    df["propensity"] = nu.e
    # coupon cost if the household gets the campaign: predicted from the treated households
    cost = _gbr(seed).fit(X[t == 1], df["coupon_camp_wk"].to_numpy()[t == 1]).predict(X).clip(min=0)
    df["coupon_cost_wk"] = cost
    rules = {"Model (DR-learner)": tau, "Top spenders": df["spend_wk"].to_numpy(),
             "Most lapsed": df["recency_days"].to_numpy(), "Random": np.random.default_rng(seed).random(len(df))}
    curve = []
    for share in np.linspace(0, 1, 11):
        k = int(round(share * len(df)))
        row = {"share": float(share)}
        for name, score in rules.items():
            pi = np.zeros(len(df), int)
            if k:
                pi[np.argsort(-score)[:k]] = 1
            v, se = policy_value(pi, t, yc, nu)
            row[name], row[name + " se"] = v, se
        curve.append(row)
    out["policy_curve"] = curve
    out["blp"] = blp(tau, pseudo_outcome(t, yc, nu))
    out["baseline_spend_wk"] = float(df["spend_wk"].mean())
    out["coupon_cost_wk_treated"] = float(df.loc[t == 1, "coupon_camp_wk"].mean())
    out["window"] = {k: str(v.date()) for k, v in
                     D["desc"].set_index("campaign_id").loc[campaign_id, ["start_date", "end_date"]].items()}
    return out, df


def blp(tau: np.ndarray, phi: np.ndarray) -> dict:
    """Best linear predictor of the effect on the model's estimate (Chernozhukov et al.): regress
    the doubly robust score on 1 and the centred estimate. A slope near 1 and clearly above 0
    means the model captures real heterogeneity; near 0 means its ranking is noise."""
    Z = np.column_stack([np.ones(len(tau)), tau - tau.mean()])
    beta, *_ = np.linalg.lstsq(Z, phi, rcond=None)
    r = phi - Z @ beta
    bread = np.linalg.inv(Z.T @ Z)
    V = bread @ (Z.T * r ** 2) @ Z @ bread
    return {"ate": float(beta[0]), "ate_se": float(np.sqrt(V[0, 0])), "slope": float(beta[1]),
            "slope_se": float(np.sqrt(V[1, 1]))}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DATA_DIR))
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    D = load(Path(a.dir))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res, frames = [], []
    for cid in TYPE_A:
        r, df = analyse(D, cid)
        res.append(r)
        frames.append(df.assign(campaign=cid))
        print(cid, json.dumps({k: r[k] for k in ("treated", "campaign", "placebo", "change", "blp")}))
    (out / "promos.json").write_text(json.dumps(res, indent=2))
    pd.concat(frames).to_parquet(out / "promos_households.parquet")


if __name__ == "__main__":
    main()
