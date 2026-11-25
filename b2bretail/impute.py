"""Who placed an invoice that has no customer id?

About a fifth of the lines have no customer. If some of those invoices come from known
customers, attributing them changes recency, frequency and the products each customer is
known to buy. The method matches the basket against each known customer's purchase profile
(TF-IDF over products, cosine similarity) within the same country. It is evaluated by hiding
the id of invoices whose customer is known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize


def _baskets(lines: pd.DataFrame, items: pd.Index) -> tuple[pd.DataFrame, sparse.csr_matrix]:
    l = lines[lines["stock_code"].isin(items)]
    meta = l.groupby("invoice").agg(customer_id=("customer_id", "first"), country=("country", "first"),
                                    date=("invoice_date", "min"), lines=("stock_code", "size")).reset_index()
    r = meta.set_index("invoice").index.get_indexer(l["invoice"])
    c = items.get_indexer(l["stock_code"])
    M = sparse.csr_matrix((np.ones(len(l), dtype=np.float32), (r, c)), shape=(len(meta), len(items)))
    M.data[:] = 1.0
    return meta, M


class Matcher:
    def __init__(self, history: pd.DataFrame):
        h = history.dropna(subset=["customer_id"])
        self.items = pd.Index(np.sort(h["stock_code"].unique()))
        self.customers = pd.Index(np.sort(h["customer_id"].unique()))
        cnt = h.groupby(["customer_id", "stock_code"]).size()
        r = self.customers.get_indexer(cnt.index.get_level_values(0))
        c = self.items.get_indexer(cnt.index.get_level_values(1))
        P = sparse.csr_matrix((cnt.to_numpy(dtype=float), (r, c)), shape=(len(self.customers), len(self.items)))
        self.tfidf = TfidfTransformer(sublinear_tf=True).fit(P)
        self.P = normalize(self.tfidf.transform(P))
        country = h.groupby("customer_id")["country"].agg(lambda s: s.mode().iat[0])
        self.country = country.reindex(self.customers).to_numpy()
        self.invoices = h.groupby("customer_id")["invoice"].nunique().reindex(self.customers).to_numpy()

    def rank(self, lines: pd.DataFrame, k: int = 10) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Top-k candidate customers per invoice and their similarity."""
        meta, M = _baskets(lines, self.items)
        Q = normalize(self.tfidf.transform(M))
        countries = meta["country"].to_numpy()
        tops, sims = [], []
        for s in range(0, Q.shape[0], 2000):
            S = (Q[s:s + 2000] @ self.P.T).toarray()
            same = countries[s:s + 2000, None] == self.country[None, :]
            S = np.where(same, S, S - 1.0)  # another country only if nothing matches at home
            top = np.argsort(-S, axis=1)[:, :k]
            tops.append(top)
            sims.append(np.take_along_axis(S, top, axis=1))
        return meta, np.vstack(tops), np.vstack(sims)

    def baseline(self, meta: pd.DataFrame, k: int = 10) -> np.ndarray:
        """The customers with the most invoices in the invoice's country."""
        out = np.zeros((len(meta), k), dtype=int)
        order = np.argsort(-self.invoices)
        for j, ctry in enumerate(meta["country"].to_numpy()):
            cand = order[self.country[order] == ctry]
            cand = np.concatenate([cand, order[self.country[order] != ctry]])
            out[j] = cand[:k]
        return out


def evaluate(history: pd.DataFrame, holdout: pd.DataFrame, k: int = 10) -> dict:
    """Hide the customer of every invoice in `holdout` and try to recover it."""
    m = Matcher(history)
    meta, top, sim = m.rank(holdout, k)
    truth = m.customers.get_indexer(meta["customer_id"].astype("Int64").to_numpy())
    known = truth >= 0
    res = {"invoices": int(len(meta)), "share_known_customer": float(known.mean())}
    base = m.baseline(meta, k)
    for name, t in (("profile_match", top), ("busiest_in_country", base)):
        for kk in (1, 5, 10):
            hit = (t[:, :kk] == truth[:, None]).any(axis=1)
            res[f"{name}_acc@{kk}_known"] = float(hit[known].mean())
            res[f"{name}_acc@{kk}_all"] = float(hit.mean())
    # precision of the first candidate when only confident matches are used
    s1 = sim[:, 0]
    hit1 = top[:, 0] == truth
    curve = []
    for cov in (0.1, 0.25, 0.5, 0.75, 1.0):
        thr = np.quantile(s1, 1 - cov)
        sel = s1 >= thr
        curve.append({"coverage": cov, "threshold": float(thr), "precision": float(hit1[sel].mean())})
    res["confidence_curve"] = curve
    return res


def attribute(history: pd.DataFrame, unknown: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Candidate customer for invoices without an id; only matches above the threshold."""
    m = Matcher(history)
    meta, top, sim = m.rank(unknown, 1)
    meta["candidate"] = m.customers[top[:, 0]]
    meta["similarity"] = sim[:, 0]
    meta["confident"] = meta["similarity"] >= threshold
    return meta
