"""Customer features at the cutoff, RFM and k-means segments with readable names."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURES = ["log_recency", "log_frequency", "log_monetary", "log_lines_per_invoice", "log_distinct_products"]


def customer_features(train: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """One row per customer with purchases before the cutoff."""
    t = train.dropna(subset=["customer_id"])
    inv = t.groupby(["customer_id", "invoice"]).agg(date=("invoice_date", "min"), lines=("stock_code", "size"),
                                                   revenue=("revenue", "sum")).reset_index()
    g = inv.groupby("customer_id")
    f = pd.DataFrame({
        "recency_days": (cutoff - g["date"].max()).dt.days,
        "frequency": g.size(),
        "monetary": g["revenue"].sum(),
        "lines_per_invoice": g["lines"].mean(),
        "first_days": (cutoff - g["date"].min()).dt.days,
    })
    f["distinct_products"] = t.groupby("customer_id")["stock_code"].nunique()
    f["uk"] = (t.groupby("customer_id")["country"].agg(lambda s: s.mode().iat[0]) == "United Kingdom").astype(int)
    months = np.maximum(f["first_days"] / 30.44, 1.0)
    f["monthly_revenue"] = f["monetary"] / months
    f["log_recency"] = np.log1p(f["recency_days"])
    f["log_frequency"] = np.log(f["frequency"])
    f["log_monetary"] = np.log1p(f["monetary"].clip(lower=0))
    f["log_lines_per_invoice"] = np.log(f["lines_per_invoice"])
    f["log_distinct_products"] = np.log(f["distinct_products"])
    return f


def fit_segments(f: pd.DataFrame, ks=range(3, 7), seed: int = 0) -> tuple[pd.Series, pd.DataFrame, dict]:
    """k-means on standardised log RFM features, k chosen by silhouette."""
    X = StandardScaler().fit_transform(f[FEATURES])
    sil = {}
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(X), size=min(3000, len(X)), replace=False)
    fits = {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        fits[k] = km
        sil[k] = float(silhouette_score(X[sample], km.labels_[sample]))
    k = max(sil, key=sil.get)
    labels = pd.Series(fits[k].labels_, index=f.index, name="cluster")
    prof = profile(f, labels)
    names = name_segments(prof)
    seg = labels.map(names).rename("segment")
    prof.index = prof.index.map(names)
    return seg, prof, {"k": k, "silhouette": sil}


def profile(f: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    g = f.groupby(labels)
    p = pd.DataFrame({
        "customers": g.size(),
        "share_customers": g.size() / len(f),
        "share_revenue": g["monetary"].sum() / f["monetary"].sum(),
        "median_recency_days": g["recency_days"].median(),
        "median_invoices": g["frequency"].median(),
        "median_revenue": g["monetary"].median(),
        "median_lines_per_invoice": g["lines_per_invoice"].median(),
        "median_distinct_products": g["distinct_products"].median(),
        "share_uk": g["uk"].mean(),
    })
    return p


def name_segments(p: pd.DataFrame) -> dict:
    """Names from the medians: value tier from revenue rank, activity from recency."""
    order = p["median_revenue"].rank(ascending=False, method="first").astype(int)
    n = len(p)
    names = {}
    for c, row in p.iterrows():
        r = order[c]
        tier = "Top" if r == 1 else ("Low-value" if r == n else "Mid-value")
        if r not in (1, n) and n > 3:
            tier = "Mid-value" if r <= (n + 1) // 2 else "Small"
        rec = row["median_recency_days"]
        act = "active" if rec <= 30 else ("cooling" if rec <= 120 else "lapsed")
        names[c] = f"{tier}, {act}"
    # make names unique if two clusters land on the same words
    seen: dict[str, int] = {}
    for c in sorted(names, key=lambda c: order[c]):
        nm = names[c]
        if nm in seen:
            seen[nm] += 1
            names[c] = f"{nm} ({seen[nm]})"
        else:
            seen[nm] = 1
    return names
