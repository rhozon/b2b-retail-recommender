import numpy as np
import pandas as pd

from b2bretail import assortment, impute, segment, uplift
from conftest import synthetic_sales


def features(seed=0):
    s = synthetic_sales(seed=seed)
    return segment.customer_features(s, pd.Timestamp("2011-12-01")), s


def test_customer_features_rfm():
    f, s = features()
    c = s["customer_id"].iloc[0]
    rows = s[s["customer_id"] == c]
    assert f.loc[c, "frequency"] == rows["invoice"].nunique()
    assert np.isclose(f.loc[c, "monetary"], rows["revenue"].sum())
    assert f.loc[c, "recency_days"] == (pd.Timestamp("2011-12-01") - rows["invoice_date"].max()).days


def test_segments_have_unique_readable_names():
    rng = np.random.default_rng(0)
    n = 300
    f = pd.DataFrame({"recency_days": np.r_[rng.integers(1, 20, n), rng.integers(200, 400, n)],
                      "frequency": np.r_[rng.integers(10, 30, n), rng.integers(1, 3, n)],
                      "monetary": np.r_[rng.uniform(5000, 9000, n), rng.uniform(50, 300, n)],
                      "lines_per_invoice": rng.uniform(5, 30, 2 * n),
                      "distinct_products": rng.integers(5, 100, 2 * n), "uk": 1})
    for c, src in (("log_recency", "recency_days"), ("log_monetary", "monetary")):
        f[c] = np.log1p(f[src])
    f["log_frequency"] = np.log(f["frequency"])
    f["log_lines_per_invoice"] = np.log(f["lines_per_invoice"])
    f["log_distinct_products"] = np.log(f["distinct_products"])
    seg, prof, info = segment.fit_segments(f, ks=range(2, 4))
    assert info["k"] == 2
    assert prof.index.is_unique
    top = seg.iloc[:n].mode().iat[0]
    assert top.startswith("Top") and "active" in top


def test_matcher_recovers_customers_with_distinct_baskets():
    s = synthetic_sales(n_groups=4, users_per_group=1, items_per_group=10, invoices=8, seed=2)
    hist = s[s["invoice_date"] < "2011-08-01"]
    hold = s[s["invoice_date"] >= "2011-08-01"]
    res = impute.evaluate(hist, hold, k=2)
    assert res["profile_match_acc@1_known"] == 1.0


def test_milp_respects_size_and_family_cap():
    exp = pd.Series({f"p{i}": 100 - i for i in range(20)})
    fam = pd.Series({f"p{i}": ("A" if i < 10 else "B") for i in range(20)})
    chosen = assortment.choose(exp, fam, k=6, per_family=3)
    assert len(chosen) == 6
    assert sum(fam[c] == "A" for c in chosen) == 3 and sum(fam[c] == "B" for c in chosen) == 3
    assert set(chosen) == {"p0", "p1", "p2", "p10", "p11", "p12"}


def test_gain_curve_oracle_beats_random():
    tau = np.array([5.0, -1.0, 3.0, 0.0, -2.0])
    g = uplift.gain_curve(tau, tau, grid=np.array([0, 0.4, 1.0]))
    assert g.tolist() == [0.0, 8.0, 5.0]


def test_uplift_learners_find_the_responders():
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 3))
    tau = np.where(X[:, 0] > 0, 10.0, -2.0)
    t = rng.integers(0, 2, n)
    y = X[:, 1] * 3 + t * tau + rng.normal(0, 1, n)
    for learner in (uplift.t_learner, uplift.x_learner, uplift.dr_learner):
        est = learner(X, t, y)
        assert np.corrcoef(est, tau)[0, 1] > 0.8


def test_true_effect_signs():
    f = pd.DataFrame({"monthly_revenue": [100.0, 100.0], "recency_days": [300, 5], "frequency": [3, 60]})
    tau = uplift.true_effect(f)
    assert tau[0] > 0 > tau[1]


def test_uplift_replications_do_not_leak_state():
    # replication r of a multi-replication run must equal a run started at seed r
    rng = np.random.default_rng(1)
    n = 1200
    rec = rng.integers(1, 400, n)
    freq = rng.integers(1, 60, n)
    m = rng.lognormal(5, 1, n)
    f = pd.DataFrame({"recency_days": rec, "frequency": freq, "monthly_revenue": m, "monetary": m * 12,
                      "log_recency": np.log1p(rec), "log_frequency": np.log(freq), "log_monetary": np.log1p(m * 12),
                      "log_lines_per_invoice": 1.0, "log_distinct_products": 1.0, "uk": 1, "first_days": 365})
    y0 = pd.Series(rng.exponential(0.2 * m), index=f.index)
    both = uplift.run(f, y0, reps=2, seed=0)[1].set_index("policy")
    alone = [uplift.run(f, y0, reps=1, seed=k)[1].set_index("policy").loc["x_learner", "mean"] for k in (0, 1)]
    assert np.allclose(sorted(alone), [both.loc["x_learner", "min"], both.loc["x_learner", "max"]])
