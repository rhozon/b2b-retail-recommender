"""SASRec: self-attentive sequential recommendation (Kang and McAuley, 2018), in PyTorch.

Each customer's history becomes one sequence of products, basket after basket (inside a basket
the products are sorted and repeats dropped). A causal transformer reads the sequence and, at
every position, scores every product as the next one; the loss is the full softmax
cross-entropy over the catalogue, which is small enough here (about 4,500 products) to skip
negative sampling. The score for the next order is the output at the last position.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

MAX_LEN = 200


def item_sequences(baskets: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Flatten each customer's baskets into one product sequence."""
    out = []
    for seq in baskets:
        parts = [np.unique(b) for b in seq if len(b)]
        out.append(np.concatenate(parts).astype(np.int64) if parts else np.zeros(0, dtype=np.int64))
    return out


class SASRec(nn.Module):
    def __init__(self, n_items: int, dim: int = 64, heads: int = 2, layers: int = 2, max_len: int = MAX_LEN,
                 dropout: float = 0.2):
        super().__init__()
        self.n_items, self.max_len = n_items, max_len
        self.pad = n_items
        self.item = nn.Embedding(n_items + 1, dim, padding_idx=n_items)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T) product indices, left-padded with `pad`. Returns logits (batch, T, n_items)."""
        B, T = x.shape
        h = self.item(x) * (self.item.embedding_dim ** 0.5) + self.pos(torch.arange(T, device=x.device))[None]
        h = self.drop(h)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        h = self.enc(h, mask=causal, src_key_padding_mask=(x == self.pad))
        h = self.norm(h)
        return h @ self.item.weight[: self.n_items].T


def _left_pad(seqs: list[np.ndarray], T: int, pad: int) -> np.ndarray:
    out = np.full((len(seqs), T), pad, dtype=np.int64)
    for i, s in enumerate(seqs):
        s = s[-T:]
        if len(s):
            out[i, T - len(s):] = s
    return out


def fit(seqs: list[np.ndarray], n_items: int, epochs: int = 20, batch_size: int = 64, lr: float = 1e-3,
        seed: int = 0, dim: int = 64, heads: int = 2, layers: int = 2, max_len: int = MAX_LEN,
        on_epoch=None) -> SASRec:
    """Train on every customer with at least two products. `on_epoch(model, epoch)` runs after
    each epoch (used to pick the number of epochs on the validation window)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SASRec(n_items, dim, heads, layers, max_len)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    train = [s[-(max_len + 1):] for s in seqs if len(s) >= 2]
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train))
        for start in range(0, len(order), batch_size):
            batch = [train[j] for j in order[start:start + batch_size]]
            T = max(len(s) for s in batch) - 1
            inp = torch.as_tensor(_left_pad([s[:-1] for s in batch], T, model.pad))
            tgt = torch.as_tensor(_left_pad([s[1:] for s in batch], T, -100))
            opt.zero_grad()
            logits = model(inp)
            loss = loss_fn(logits.reshape(-1, n_items), tgt.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if on_epoch is not None:
            on_epoch(model, ep)
    return model


@torch.no_grad()
def scores(model: SASRec, seqs: list[np.ndarray], users: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Scores for the product that follows each customer's last known product."""
    model.eval()
    out = np.zeros((len(users), model.n_items), dtype=np.float32)
    for start in range(0, len(users), batch_size):
        us = users[start:start + batch_size]
        batch = [seqs[u] if len(seqs[u]) else np.zeros(0, dtype=np.int64) for u in us]
        T = max(1, min(model.max_len, max(len(s) for s in batch)))
        x = _left_pad(batch, T, model.pad)
        empty = (x == model.pad).all(axis=1)
        x[empty, -1] = 0  # a placeholder token, so the attention has something to read
        logits = model(torch.as_tensor(x))[:, -1].numpy()
        logits[empty] = 0.0
        out[start:start + len(us)] = logits
    return out


class BasketTransformer(nn.Module):
    """SASRec read basket by basket: each basket is the mean of its product embeddings, a causal
    transformer reads the baskets in order, and the output scores every product for the next
    basket. Same interface and loss as seqmodel.NextBasket (the GRU), so the two differ only in
    the sequence encoder."""

    def __init__(self, n_items: int, dim: int = 64, heads: int = 2, layers: int = 2, max_len: int = 50,
                 dropout: float = 0.2):
        super().__init__()
        self.n_items, self.max_len = n_items, max_len
        self.emb = nn.EmbeddingBag(n_items + 1, dim, mode="mean", padding_idx=n_items)
        self.pos = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=2 * dim, dropout=dropout,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dim, n_items)

    def forward(self, baskets: list[list[np.ndarray]]) -> torch.Tensor:
        lens = [len(s) for s in baskets]
        T = max(lens)
        flat, offsets, pos = [], [], 0
        for s in baskets:
            for t in range(T):
                b = s[t] if t < len(s) else np.array([self.n_items])
                offsets.append(pos)
                flat.append(b)
                pos += len(b)
        x = self.emb(torch.as_tensor(np.concatenate(flat), dtype=torch.long),
                     torch.as_tensor(offsets, dtype=torch.long)).view(len(baskets), T, -1)
        x = self.drop(x + self.pos(torch.arange(T))[None])
        pad = torch.as_tensor([[t >= n for t in range(T)] for n in lens])
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        h = self.enc(x, mask=causal, src_key_padding_mask=pad)
        return self.out(self.drop(self.norm(h)))


def fit_baskets(seqs: list[list[np.ndarray]], n_items: int, epochs: int = 15, batch_size: int = 64, lr: float = 2e-3,
                seed: int = 0, dim: int = 64, heads: int = 2, layers: int = 2, max_len: int = 50,
                on_epoch=None) -> BasketTransformer:
    from .seqmodel import _loss

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = BasketTransformer(n_items, dim, heads, layers, max_len)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    train = [s[-max_len:] for s in seqs if len(s) >= 2]
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
