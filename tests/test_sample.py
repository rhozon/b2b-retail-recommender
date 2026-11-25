"""The committed sample goes through the pandas cleaning with sensible counts."""
from pathlib import Path

import pandas as pd

from b2bretail import data

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample_lines.csv"


def test_sample_cleaning():
    raw = data.standardise(pd.read_csv(SAMPLE, dtype={"Invoice": str, "StockCode": str}))
    s = data.clean_sales(raw)
    assert 0.85 < len(s) / len(raw) < 1.0
    assert raw["invoice"].str.startswith("C").any()
    assert not s["invoice"].str.startswith("C").any()
    assert s["customer_id"].isna().any() and s["customer_id"].notna().any()
