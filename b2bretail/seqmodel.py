"""A small sequential next-basket model in PyTorch.

Each invoice is a basket. A basket becomes the mean of its product embeddings, a GRU reads the
customer's baskets in order, and a linear layer scores every product for the next basket. The
loss is the multi-label softmax cross-entropy over the products of the next basket.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

MAX_BASKETS = 50


def basket_sequences(lines: pd.DataFrame, users: pd.Index, items: pd.Index) -> list[list[np.ndarray]]:
    """For each customer in `users`, the list of baskets (product indices) in date order,
    keeping the last MAX_BASKETS."""
    l = lines[lines["customer_id"].isin(users) & lines["stock_code"].isin(items)]
    l = l.assign(u=users.get_indexer(l["customer_id"]), i=items.get_indexer(l["stock_code"]))
    l = l[["invoice", "u", "i", "invoice_date"]].drop_duplicates(["invoice", "i"])
    t0 = l.groupby("invoice")["invoice_date"].transform("min")
    l = l.assign(t0=t0).sort_values(["u", "t0", "invoice", "i"])
    inv = l["invoice"].to_numpy()
    cuts = np.flatnonzero(inv[1:] != inv[:-1]) + 1
    starts = np.concatenate([[0], cuts])
    seqs: list[list[np.ndarray]] = [[] for _ in range(len(users))]
    us, its = l["u"].to_numpy(), l["i"].to_numpy()
    for s, basket in zip(starts, np.split(its, cuts)):
        seqs[us[s]].append(basket)
    return [s[-MAX_BASKETS:] for s in seqs]


class NextBasket(nn.Module):
    def __init__(self, n_items: int, dim: int = 64, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.emb = nn.EmbeddingBag(n_items + 1, dim, mode="mean", padding_idx=n_items)
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_items)
        self.n_items = n_items

    def forward(self, baskets: list[list[np.ndarray]]) -> torch.Tensor:
        """Logits for the basket after each position: shape (batch, max_len, n_items)."""
        lens = [len(s) for s in baskets]
        T = max(lens)
        flat, offsets = [], []
        pos = 0
        for s in baskets:
            for t in range(T):
                b = s[t] if t < len(s) else np.array([self.n_items])
                offsets.append(pos)
                flat.append(b)
                pos += len(b)
        x = self.emb(torch.as_tensor(np.concatenate(flat), dtype=torch.long),
                     torch.as_tensor(offsets, dtype=torch.long))
        x = x.view(len(baskets), T, -1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        h, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, total_length=T)
        return self.out(self.drop(h))


def _loss(model: NextBasket, batch: list[list[np.ndarray]]) -> torch.Tensor:
    inp = [s[:-1] for s in batch]
    logits = model(inp)
    logp = torch.log_softmax(logits, dim=-1)
    B, T, N = logp.shape
    rows, cols, vals = [], [], []
    mask = torch.zeros(B, T)
    for b, s in enumerate(batch):
        for t in range(len(s) - 1):
            nxt = s[t + 1]
            rows.append(np.full(len(nxt), b * T + t))
            cols.append(nxt)
            vals.append(np.full(len(nxt), 1.0 / len(nxt)))
            mask[b, t] = 1.0
    W = torch.zeros(B * T, N)
    W[torch.as_tensor(np.concatenate(rows)), torch.as_tensor(np.concatenate(cols))] = torch.as_tensor(
        np.concatenate(vals), dtype=torch.float32)
    per_step = -(logp.reshape(B * T, N) * W).sum(dim=1).view(B, T)
    return (per_step * mask).sum() / mask.sum().clamp(min=1.0)


def fit(seqs: list[list[np.ndarray]], n_items: int, epochs: int = 15, batch_size: int = 64,
        lr: float = 2e-3, seed: int = 0, dim: int = 64, hidden: int = 128,
        on_epoch=None) -> NextBasket:
    """Train on every customer with at least two baskets. `on_epoch(model, epoch)` is called
    after each epoch (used to pick the number of epochs on the validation window)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = NextBasket(n_items, dim, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    train = [s for s in seqs if len(s) >= 2]
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        for start in range(0, len(order), batch_size):
            batch = [train[j] for j in order[start:start + batch_size]]
            opt.zero_grad()
            loss = _loss(model, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if on_epoch is not None:
            on_epoch(model, ep)
    return model


@torch.no_grad()
def scores(model: NextBasket, seqs: list[list[np.ndarray]], users: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Scores for the basket that follows each customer's last known basket."""
    model.eval()
    out = np.zeros((len(users), model.n_items), dtype=np.float32)
    for start in range(0, len(users), batch_size):
        us = users[start:start + batch_size]
        batch = [seqs[u] if len(seqs[u]) else [np.array([model.n_items])] for u in us]
        logits = model(batch)
        last = torch.as_tensor([len(s) - 1 for s in batch])
        out[start:start + len(us)] = logits[torch.arange(len(us)), last].numpy()
    return out
