"""Recommenders that score every product for a set of customers.

All of them take the customers x products matrix R (number of invoices with the product) and
return a dense score matrix for the requested customers. Higher is better.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from .split import Split


def popularity(split: Split, users: np.ndarray, days: int = 90) -> np.ndarray:
    """Number of distinct customers who bought the product in the last `days` before the cutoff."""
    recent = split.train[split.train["invoice_date"] >= split.cutoff - pd.Timedelta(days=days)]
    n = recent.groupby("stock_code")["customer_id"].nunique()
    pop = n.reindex(split.items).fillna(0).to_numpy(dtype=float)
    pop += 1e-6 * (split.R > 0).sum(axis=0).A1  # older sales break ties
    return np.tile(pop, (len(users), 1))


def buy_again(split: Split, users: np.ndarray) -> np.ndarray:
    """The customer's own products, by how many invoices had them, most recent first on ties."""
    last = split.train.groupby(["customer_id", "stock_code"])["invoice_date"].max()
    age = (split.cutoff - last).dt.days
    S = split.R[users].toarray().astype(float)
    r = split.users.get_indexer(age.index.get_level_values(0))
    c = split.items.get_indexer(age.index.get_level_values(1))
    A = sparse.csr_matrix((1.0 / (1.0 + age.to_numpy()), (r, c)), shape=split.R.shape)[users].toarray()
    return S + 0.01 * A + 1e-9 * popularity(split, users)


def item_similarity(R: sparse.csr_matrix, neighbours: int = 100) -> sparse.csr_matrix:
    """Cosine similarity between products over the customers who bought them, keeping the
    strongest `neighbours` per product."""
    B = (R > 0).astype(np.float32).tocsc()
    norms = np.sqrt(np.asarray(B.multiply(B).sum(axis=0)).ravel())
    norms[norms == 0] = 1.0
    C = (B.T @ B).toarray()
    C = C / norms[:, None] / norms[None, :]
    np.fill_diagonal(C, 0.0)
    if neighbours < C.shape[1]:
        kth = np.argpartition(-C, neighbours, axis=1)[:, neighbours:]
        np.put_along_axis(C, kth, 0.0, axis=1)
    return sparse.csr_matrix(C.astype(np.float32))


def item_knn(split: Split, users: np.ndarray, neighbours: int = 100, S: sparse.csr_matrix | None = None) -> np.ndarray:
    S = item_similarity(split.R, neighbours) if S is None else S
    B = (split.R[users] > 0).astype(np.float32)
    return np.asarray((B @ S).todense()) + 1e-6 * popularity(split, users)


def als(R: sparse.csr_matrix, factors: int = 64, reg: float = 0.1, alpha: float = 10.0,
        iters: int = 15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Implicit-feedback ALS (Hu, Koren and Volinsky, 2008). Confidence 1 + alpha*log(1+r)."""
    rng = np.random.default_rng(seed)
    n_u, n_i = R.shape
    X = rng.normal(0, 0.01, (n_u, factors))
    Y = rng.normal(0, 0.01, (n_i, factors))
    C = R.copy().astype(np.float64)
    C.data = alpha * np.log1p(C.data)
    Ct = C.T.tocsr()
    I = reg * np.eye(factors)

    def solve(A: sparse.csr_matrix, F: np.ndarray, out: np.ndarray) -> None:
        FtF = F.T @ F
        for u in range(A.shape[0]):
            s, e = A.indptr[u], A.indptr[u + 1]
            if s == e:
                out[u] = 0.0
                continue
            idx, conf = A.indices[s:e], A.data[s:e]
            Fu = F[idx]
            M = FtF + (Fu.T * conf) @ Fu + I
            b = (Fu.T * (1.0 + conf)).sum(axis=1)
            out[u] = np.linalg.solve(M, b)

    for _ in range(iters):
        solve(C, Y, X)
        solve(Ct, X, Y)
    return X, Y


def als_scores(X: np.ndarray, Y: np.ndarray, users: np.ndarray) -> np.ndarray:
    return X[users] @ Y.T


def seen(split: Split, users: np.ndarray) -> list[np.ndarray]:
    R = split.R
    return [R.indices[R.indptr[u]:R.indptr[u + 1]] for u in users]


def reasons(split: Split, user: int, recs: np.ndarray, S: sparse.csr_matrix) -> list[int | None]:
    """For each recommended product, the customer's own product most similar to it: the
    'often bought with' line shown next to a recommendation."""
    own = split.R.indices[split.R.indptr[user]:split.R.indptr[user + 1]]
    out = []
    if len(own) == 0:
        return [None] * len(recs)
    Sd = S[own][:, recs].toarray()
    for j in range(len(recs)):
        col = Sd[:, j]
        out.append(int(own[col.argmax()]) if col.max() > 0 else None)
    return out
