import numpy as np

from b2bretail import metrics, sasrec


def test_item_sequences_keep_order_and_drop_repeats_inside_a_basket():
    baskets = [[np.array([3, 1]), np.array([2]), np.array([1, 1, 4])]]
    seq = sasrec.item_sequences(baskets)[0]
    assert seq.tolist() == [1, 3, 2, 1, 4]


def test_sasrec_learns_a_fixed_order():
    # every customer buys 0,1,2,3,4,5 in this order, again and again
    seqs = [np.tile(np.arange(6), 6)[: 30 + k % 6] for k in range(60)]
    model = sasrec.fit(seqs, 6, epochs=40, batch_size=20, lr=5e-3, dim=16, heads=1, layers=1, max_len=20)
    users = np.arange(6)
    sc = sasrec.scores(model, seqs, users)
    top = metrics.top_k(sc, 1)[:, 0]
    expected = np.array([(seqs[u][-1] + 1) % 6 for u in users])
    assert (top == expected).mean() >= 5 / 6


def test_scores_handle_customers_without_history():
    model = sasrec.SASRec(5, dim=8, heads=1, layers=1, max_len=10)
    sc = sasrec.scores(model, [np.array([], dtype=int), np.array([1, 2])], np.array([0, 1]))
    assert sc.shape == (2, 5) and np.isfinite(sc).all()


def test_basket_transformer_learns_a_fixed_basket_order():
    from b2bretail import seqmodel

    A, B, C = np.array([0, 1]), np.array([2, 3]), np.array([4, 5])
    seqs = [[A, B, C, A, B, C, A, B] for _ in range(40)]
    model = sasrec.fit_baskets(seqs, 6, epochs=30, batch_size=20, lr=0.01, dim=16, heads=1, layers=1, max_len=10)
    top = metrics.top_k(seqmodel.scores(model, seqs, np.arange(3)), 2)
    assert set(top[0]) == {4, 5}
