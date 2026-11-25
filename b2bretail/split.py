"""Temporal split and the customer x product matrices the recommenders work on."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

# Test window: the last 61 days of the data (9 October to 9 December 2011). Everything the
# models learn comes from before it. The validation window used to pick settings is the 61
# days before the test window.
CUTOFF = pd.Timestamp("2011-10-09")
VAL_CUTOFF = pd.Timestamp("2011-08-09")


@dataclass
class Split:
    train: pd.DataFrame          # sale lines with a customer, before the cutoff
    test: pd.DataFrame           # sale lines with a customer, in the window after the cutoff
    users: pd.Index              # customer ids with a purchase before the cutoff
    items: pd.Index              # stock codes sold before the cutoff
    R: sparse.csr_matrix         # users x items, number of invoices containing the item
    cutoff: pd.Timestamp
    end: pd.Timestamp


def make_split(sales: pd.DataFrame, cutoff: pd.Timestamp = CUTOFF,
               end: pd.Timestamp | None = None) -> Split:
    s = sales.dropna(subset=["customer_id"])
    end = end if end is not None else s["invoice_date"].max() + pd.Timedelta(seconds=1)
    train = s[s["invoice_date"] < cutoff]
    test = s[(s["invoice_date"] >= cutoff) & (s["invoice_date"] < end)]
    users = pd.Index(np.sort(train["customer_id"].unique()))
    items = pd.Index(np.sort(train["stock_code"].unique()))
    R = interaction_matrix(train, users, items)
    return Split(train, test, users, items, R, cutoff, end)


def interaction_matrix(lines: pd.DataFrame, users: pd.Index, items: pd.Index) -> sparse.csr_matrix:
    """Number of distinct invoices in which each customer bought each product."""
    ui = lines[["customer_id", "stock_code", "invoice"]].drop_duplicates()
    ui = ui[ui["customer_id"].isin(users) & ui["stock_code"].isin(items)]
    counts = ui.groupby(["customer_id", "stock_code"]).size()
    r = users.get_indexer(counts.index.get_level_values(0))
    c = items.get_indexer(counts.index.get_level_values(1))
    return sparse.csr_matrix((counts.to_numpy(dtype=np.float32), (r, c)),
                             shape=(len(users), len(items)))


def targets(split: Split, new_only: bool) -> tuple[np.ndarray, list[np.ndarray]]:
    """Customers bought from in both periods, and the products each bought in the test window
    (products unknown before the cutoff cannot be recommended and are left out). With
    new_only, only products the customer had never bought before the cutoff count."""
    t = split.test[split.test["customer_id"].isin(split.users) & split.test["stock_code"].isin(split.items)]
    g = t.groupby("customer_id")["stock_code"].unique()
    uidx, tg = [], []
    R = split.R
    for cid, codes in g.items():
        u = split.users.get_loc(cid)
        it = np.unique(split.items.get_indexer(codes))
        if new_only:
            seen = R.indices[R.indptr[u]:R.indptr[u + 1]]
            it = np.setdiff1d(it, seen)
        if len(it):
            uidx.append(u)
            tg.append(it)
    return np.array(uidx, dtype=int), tg


def cold_share(split: Split) -> float:
    """Share of test-window revenue on products never sold before the cutoff."""
    t = split.test
    return float(t.loc[~t["stock_code"].isin(split.items), "revenue"].sum() / t["revenue"].sum())
