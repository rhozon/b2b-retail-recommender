"""Data file for the simulator on the case page (assets/case10/data.json)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import recs

SAMPLE = 300


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def site_data(site: Path, sp, desc, seg, prof, f, S, per, rm, assort, items, curves, areas, summary) -> Path:
    site.mkdir(parents=True, exist_ok=True)
    new_mode, all_mode = "new to the customer", "all products"
    best_new = summary["recs"][new_mode]["best"]
    users_n, tops_n, _, _ = per[(new_mode, best_new)]
    users_a, tops_a, _, _ = per[(all_mode, "Buy again")]
    pos_a = {u: j for j, u in enumerate(users_a)}
    test_items = sp.test[sp.test["stock_code"].isin(sp.items)].groupby("customer_id")["stock_code"].unique()

    # customers shown: evaluation customers, sampled in proportion to segment size
    rng = np.random.default_rng(7)
    cids = sp.users[users_n]
    segs = seg.reindex(cids)
    chosen = []
    for name, g in pd.Series(cids, index=cids).groupby(segs.to_numpy()):
        n = max(15, int(round(SAMPLE * len(g) / len(cids))))
        chosen += list(rng.choice(g.to_numpy(), size=min(n, len(g)), replace=False))
    rng.shuffle(chosen)
    pos_n = {u: j for j, u in enumerate(users_n)}
    customers = []
    for i, cid in enumerate(chosen):
        u = sp.users.get_loc(cid)
        bought_later = set(test_items.get(cid, []))
        rec = tops_n[pos_n[u]]
        why = recs.reasons(sp, u, rec, S)
        own = sp.R[u].toarray().ravel()
        fav = np.argsort(-own)[:3]
        ra = tops_a[pos_a[u]][:5] if u in pos_a else []
        row = f.loc[cid]
        customers.append({
            "id": f"C{i + 1:03d}",
            "segment": seg.get(cid),
            "country": sp.train.loc[sp.train["customer_id"] == cid, "country"].mode().iat[0],
            "invoices": int(row["frequency"]),
            "days_since_last": int(row["recency_days"]),
            "revenue": _r(row["monetary"], 0),
            "favourites": [desc.get(sp.items[j], sp.items[j]) for j in fav if own[j] > 0],
            "discover": [{"p": desc.get(sp.items[j], sp.items[j]),
                          "why": (desc.get(sp.items[w], sp.items[w]) if w is not None else None),
                          "hit": sp.items[j] in bought_later} for j, w in zip(rec, why)],
            "reorder": [{"p": desc.get(sp.items[j], sp.items[j]), "hit": sp.items[j] in bought_later} for j in ra],
        })

    segments = []
    for name, p in prof.iterrows():
        a = assort[assort["segment"] == name].set_index("plan")
        it = items[items["segment"] == name].sort_values("rank")
        segments.append({
            "name": name,
            "customers": int(p["customers"]),
            "share_customers": _r(p["share_customers"]),
            "share_revenue": _r(p["share_revenue"]),
            "median_recency_days": _r(p["median_recency_days"], 0),
            "median_invoices": _r(p["median_invoices"], 1),
            "median_revenue": _r(p["median_revenue"], 0),
            "median_lines_per_invoice": _r(p["median_lines_per_invoice"], 1),
            "share_uk": _r(p["share_uk"]),
            "plans": {k: {"revenue_share": _r(v["revenue_share"]), "customers_reached": _r(v["customers_reached"]),
                          "families": int(v["families"])} for k, v in a.iterrows()},
            "assortment": [{"p": r.description, "family": r.family, "rev": _r(r.expected_revenue_180d, 0)}
                           for r in it.itertuples()],
        })

    pol = {}
    for name, g in curves.groupby("policy"):
        pol[name] = [_r(x, 0) for x in g.sort_values("q")["gain"]]
    out = {
        "data": summary["data"],
        "split": summary["split"],
        "metrics": json.loads(rm.round(4).to_json(orient="records")),
        "best_new": best_new,
        "customers": customers,
        "segments": segments,
        "uplift": {"q": sorted(curves["q"].unique().round(2).tolist()), "curves": pol,
                   "customers": int(len(f)),
                   "area": {r.policy: _r(r.mean, 3) for r in areas.itertuples()}},
    }
    path = site / "data.json"
    path.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    return path
