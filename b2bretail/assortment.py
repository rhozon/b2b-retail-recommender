"""Which products should a segment's catalogue page lead with?

For each segment, every product gets an expected revenue from the segment's recent buying
(share of the segment's customers who bought it times the mean revenue per buyer). A
mixed-integer program picks K products that maximise expected revenue with at most a fixed
number from any one product family, so the page does not fill up with variants of one item.
Families come from clustering the product embeddings learned by the ALS recommender.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

STOP = {"OF", "AND", "THE", "SET", "IN", "WITH", "A", "TO", "SMALL", "LARGE", "MINI", "PACK", "BOX",
        "RED", "PINK", "BLUE", "WHITE", "GREEN", "BLACK", "IVORY", "CREAM", "VINTAGE", "RETROSPOT", "DESIGN",
        "OF4", "OF6", "OF3", "OF12", "12", "6", "4", "3", "2", "PARTY", "HANGING", "HEART", "GLASS", "METAL"}


def families(items: pd.Index, desc: pd.Series, n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Group products into families by the words of their descriptions (TF-IDF and k-means;
    the file has no category field) and name each family by its two most common words."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    text = pd.Series(items, index=items).map(desc).fillna("").astype(str).str.upper()
    vec = TfidfVectorizer(token_pattern=r"[A-Z]{3,}", lowercase=False, stop_words=sorted(STOP), min_df=3)
    T = vec.fit_transform(text)
    km = KMeans(n_clusters=n, n_init=10, random_state=seed).fit(normalize(T))
    fam = pd.DataFrame({"stock_code": items, "family_id": km.labels_, "description": text.to_numpy()})
    names = {}
    for f, g in fam.groupby("family_id"):
        words = Counter(w for d in g["description"] for w in set(re.findall(r"[A-Z]{3,}", d)) if w not in STOP)
        top = [w for w, _ in words.most_common(2)]
        names[f] = " / ".join(top).title() if top else f"Family {f}"
    seen: dict[str, int] = {}
    for f in sorted(names):
        if names[f] in seen:
            seen[names[f]] += 1
            names[f] = f"{names[f]} ({seen[names[f]]})"
        else:
            seen[names[f]] = 1
    fam["family"] = fam["family_id"].map(names)
    return fam


def expected_revenue(train: pd.DataFrame, members: pd.Index, cutoff: pd.Timestamp, days: int = 180) -> pd.Series:
    """Segment revenue on each product over the last `days` before the cutoff: the share of
    the segment's customers who bought it, times the mean revenue per buyer, times the
    segment size."""
    recent = train[(train["invoice_date"] >= cutoff - pd.Timedelta(days=days)) & train["customer_id"].isin(members)]
    return recent.groupby("stock_code")["revenue"].sum()


def choose(exp: pd.Series, family: pd.Series, k: int = 30, per_family: int = 4) -> list[str]:
    """K products maximising expected revenue, at most `per_family` from each family."""
    exp = exp[exp > 0]
    codes = exp.index.to_numpy()
    fam = family.reindex(codes).fillna(-1).to_numpy()
    c = -exp.to_numpy(dtype=float)
    cons = [LinearConstraint(np.ones((1, len(codes))), k, k)]
    fams = np.unique(fam)
    A = (fam[None, :] == fams[:, None]).astype(float)
    cons.append(LinearConstraint(A, 0, per_family))
    res = milp(c, constraints=cons, integrality=np.ones(len(codes)), bounds=Bounds(0, 1))
    if not res.success:
        raise RuntimeError(res.message)
    sel = codes[res.x > 0.5]
    return list(exp.loc[sel].sort_values(ascending=False).index)


def coverage(test: pd.DataFrame, members: pd.Index, chosen: list[str], family: pd.Series) -> dict:
    t = test[test["customer_id"].isin(members)]
    on = t["stock_code"].isin(chosen)
    buyers = t.loc[on, "customer_id"].nunique()
    return {"revenue_share": float(t.loc[on, "revenue"].sum() / max(t["revenue"].sum(), 1e-9)),
            "customers_reached": float(buyers / max(t["customer_id"].nunique(), 1)),
            "families": int(family.reindex(chosen).nunique())}
