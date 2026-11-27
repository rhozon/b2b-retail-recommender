"""Bronze, silver and gold Delta tables, the layout a Databricks lakehouse would hold.

Written locally with delta-rs (the `deltalake` package), so they open the same way in Spark on
Databricks (`spark.read.format("delta").load(path)`) as in pandas here. Every write is an
overwrite that adds a new table version, so earlier runs stay readable by version number.

* bronze/lines       every raw invoice line, as read
* silver/sales       the sale lines that pass the quality rules
* gold/customers     one row per customer: behaviour features and segment
* gold/recs_metrics  the evaluation table of the recommenders
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from deltalake import DeltaTable, write_deltalake


def _write(df: pd.DataFrame, path: Path, partition_by: list[str] | None = None) -> dict:
    df = df.copy()
    for c in df.columns:
        if str(df[c].dtype) in ("object", "string"):
            df[c] = df[c].astype("string").astype(object).where(df[c].notna(), None)
        elif str(df[c].dtype) == "Int64":
            df[c] = df[c].astype("float64")
    write_deltalake(str(path), df, mode="overwrite", partition_by=partition_by, schema_mode="overwrite")
    dt = DeltaTable(str(path))
    return {"path": str(path), "rows": int(len(df)), "version": int(dt.version())}


def write_all(root: Path, raw: pd.DataFrame, sales: pd.DataFrame, features: pd.DataFrame, segments: pd.Series,
              recs_metrics: pd.DataFrame) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    month = lambda d: d["invoice_date"].dt.strftime("%Y-%m")
    gold = features.join(segments.rename("segment")).reset_index().rename(columns={"index": "customer_id"})
    return {
        "bronze/lines": _write(raw.assign(month=month(raw)), root / "bronze" / "lines", ["month"]),
        "silver/sales": _write(sales.assign(month=month(sales)), root / "silver" / "sales", ["month"]),
        "gold/customers": _write(gold, root / "gold" / "customers"),
        "gold/recs_metrics": _write(recs_metrics, root / "gold" / "recs_metrics"),
    }


def read(root: Path, table: str, version: int | None = None) -> pd.DataFrame:
    return DeltaTable(str(Path(root) / table), version=version).to_pandas()
