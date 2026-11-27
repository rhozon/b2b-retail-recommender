"""Data for the version 2 tabs of the case page: model comparison and real campaigns.

    python -m b2bretail.export_v2 results path/to/assets/case10/data_v2.json
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pandas as pd


def _clean(x):
    if isinstance(x, float):
        return round(x, 6) if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def build(results: Path) -> dict:
    rm = pd.read_csv(results / "recs_metrics.csv")
    summary = json.loads((results / "summary.json").read_text())
    promos = json.loads((results / "promos.json").read_text())
    val = summary["tuning"]["validation_ndcg"]
    curves = {}
    for key, name in (("seq_epoch_", "Sequential GRU (PyTorch)"), ("sasrec_items_epoch_", "SASRec, product by product"),
                      ("sasrec_baskets_epoch_", "SASRec, basket by basket")):
        pts = sorted((int(k[len(key):]), v) for k, v in val.items() if re.fullmatch(re.escape(key) + r"\d+", k))
        curves[name] = [[e, v] for e, v in pts]
    return _clean({"recs": rm.to_dict(orient="records"), "validation": curves,
                   "baselines_validation": {"Implicit ALS": val.get("als_32_10"), "Item-to-item (co-purchase)": val.get("knn_50")},
                   "best": summary["tuning"]["best"], "promos": promos,
                   "lake": {k: {"rows": v["rows"], "version": v["version"]} for k, v in (summary.get("lake") or {}).items()}, "mlflow_run": summary.get("mlflow_run")})


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    out = Path(argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(Path(argv[0])), ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                   encoding="utf-8")
    print(out, f"{out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
