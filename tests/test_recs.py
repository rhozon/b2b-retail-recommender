import numpy as np
import pandas as pd

from b2bretail import metrics, recs, seqmodel
from b2bretail.split import interaction_matrix, make_split, targets
from conftest import synthetic_sales


def block_of(split, item_idx):
    return int(split.items[item_idx][2])  # 20000 + g*100 + i -> digit g


def test_split_is_temporal_and_targets_are_new():
    s = synthetic_sales()
    sp = make_split(s, pd.Timestamp("2011-09-01"))
    assert sp.train["invoice_date"].max() < sp.cutoff <= sp.test["invoice_date"].min()
    users, tg = targets(sp, new_only=True)
    for u, t in zip(users, tg):
        seen = sp.R.indices[sp.R.indptr[u]:sp.R.indptr[u + 1]]
        assert not np.isin(t, seen).any()


def test_interaction_matrix_counts_invoices():
    s = synthetic_sales(n_groups=1, users_per_group=2, invoices=3)
    users = pd.Index(sorted(s["customer_id"].unique()))
    items = pd.Index(sorted(s["stock_code"].unique()))
    R = interaction_matrix(s, users, items)
    assert R.sum() == len(s[["customer_id", "stock_code", "invoice"]].drop_duplicates())


def test_item_knn_and_als_stay_inside_the_customer_block():
    s = synthetic_sales(seed=3)
    sp = make_split(s, pd.Timestamp("2011-09-01"))
    users = np.arange(len(sp.users))
    group = (sp.users.to_numpy() // 1000)
    for sc in (recs.item_knn(sp, users, 20), recs.als_scores(*recs.als(sp.R, factors=8, iters=8), users)):
        top = metrics.top_k(sc, 3)
        same = [all(block_of(sp, j) == group[u] for j in top[u]) for u in users]
        assert np.mean(same) > 0.95


def test_buy_again_ranks_own_products_first():
    s = synthetic_sales(seed=4)
    sp = make_split(s, pd.Timestamp("2011-09-01"))
    users = np.arange(5)
    top = metrics.top_k(recs.buy_again(sp, users), 3)
    for u in users:
        own = sp.R.indices[sp.R.indptr[u]:sp.R.indptr[u + 1]]
        assert np.isin(top[u], own).all()


def test_reasons_point_to_a_product_the_customer_bought():
    s = synthetic_sales(seed=5)
    sp = make_split(s, pd.Timestamp("2011-09-01"))
    S = recs.item_similarity(sp.R, 20)
    why = recs.reasons(sp, 0, np.array([0, 1, 2]), S)
    own = set(sp.R.indices[sp.R.indptr[0]:sp.R.indptr[1]])
    assert all(w is None or w in own for w in why)


def test_sequence_model_learns_a_fixed_order():
    # every customer buys basket A then B then C, again and again
    A, B, C = np.array([0, 1]), np.array([2, 3]), np.array([4, 5])
    seqs = [[A, B, C, A, B, C, A, B] for _ in range(40)]
    model = seqmodel.fit(seqs, 6, epochs=30, batch_size=20, lr=0.02, dim=8, hidden=16)
    sc = seqmodel.scores(model, seqs, np.arange(3))
    top = metrics.top_k(sc, 2)
    assert set(top[0]) == {4, 5}  # after B comes C
