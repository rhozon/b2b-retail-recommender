"""Loading, data-quality checks and cleaning of the Online Retail II transactions.

The source is the UCI Online Retail II file (Chen, 2012; CC BY 4.0): every invoice line of a
UK online wholesaler of gift-ware between 1 December 2009 and 9 December 2011. Many of its
customers are wholesalers, which makes it one of the few public B2B transaction logs.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
RAW_DIR = Path(os.environ.get("ORII_DIR", Path.home() / "data" / "online-retail-ii"))
COLUMNS = {"Invoice": "invoice", "StockCode": "stock_code", "Description": "description",
           "Quantity": "quantity", "InvoiceDate": "invoice_date", "Price": "price",
           "Customer ID": "customer_id", "Country": "country"}

# A product line has a stock code that starts with five digits (85123A, 22423, ...). The rest
# are postage, bank charges, manual adjustments, samples, fees and gift vouchers.
PRODUCT_CODE = re.compile(r"^\d{5}")


def load_raw(raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    """Both sheets of the workbook as one frame, cached as Parquet next to the workbook."""
    raw_dir = Path(raw_dir)
    cache = raw_dir / "online_retail_II.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    sheets = pd.read_excel(raw_dir / "online_retail_II.xlsx", sheet_name=None, engine="calamine")
    df = pd.concat(sheets.values(), ignore_index=True).rename(columns=COLUMNS)
    df = standardise(df)
    df.to_parquet(cache, index=False)
    return df


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=COLUMNS).copy()
    df["invoice"] = df["invoice"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["description"] = df["description"].astype("string").str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype(float)
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    cid = pd.to_numeric(df["customer_id"], errors="coerce")
    df["customer_id"] = cid.astype("Int64")
    df["country"] = df["country"].astype(str).str.strip()
    return df


def flags(df: pd.DataFrame) -> pd.DataFrame:
    """One boolean column per data-quality rule. A line can break more than one."""
    f = pd.DataFrame(index=df.index)
    f["cancellation"] = df["invoice"].str.startswith("C")
    f["adjustment"] = df["invoice"].str.startswith("A")
    f["non_positive_quantity"] = df["quantity"] <= 0
    f["non_positive_price"] = df["price"] <= 0
    f["non_product_code"] = ~df["stock_code"].str.match(PRODUCT_CODE)
    f["missing_description"] = df["description"].isna()
    f["missing_customer"] = df["customer_id"].isna()
    f["exact_duplicate"] = df.duplicated(keep="first")
    # Price far from the usual price of the same product: more than ten times the median, or
    # below a tenth of it. Kept in the data but reported.
    med = df.loc[df["price"] > 0].groupby("stock_code")["price"].median()
    ratio = df["price"] / df["stock_code"].map(med)
    f["price_outlier"] = (ratio > 10) | (ratio < 0.1)
    return f.fillna(False).astype(bool)


SALE_EXCLUSIONS = ["cancellation", "adjustment", "non_positive_quantity", "non_positive_price",
                   "non_product_code", "exact_duplicate"]


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    f = flags(df)
    rows = [{"rule": "all lines", "lines": len(df), "share": 1.0}]
    for c in f.columns:
        rows.append({"rule": c, "lines": int(f[c].sum()), "share": float(f[c].mean())})
    sale = ~f[SALE_EXCLUSIONS].any(axis=1)
    rows.append({"rule": "kept as sale lines", "lines": int(sale.sum()), "share": float(sale.mean())})
    rows.append({"rule": "sale lines with a customer", "lines": int((sale & ~f["missing_customer"]).sum()),
                 "share": float((sale & ~f["missing_customer"]).mean())})
    return pd.DataFrame(rows)


def clean_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Sale lines: product codes with positive quantity and price, no cancellations, no exact
    duplicates. Lines without a customer are kept; the recommenders drop them, the imputation
    step works on them."""
    f = flags(df)
    out = df.loc[~f[SALE_EXCLUSIONS].any(axis=1)].copy()
    out["revenue"] = out["quantity"] * out["price"]
    out["price_outlier"] = f.loc[out.index, "price_outlier"]
    return out.reset_index(drop=True)


def descriptions(sales: pd.DataFrame) -> pd.Series:
    """Most frequent description of each stock code."""
    d = sales.dropna(subset=["description"])
    return (d.groupby(["stock_code", "description"]).size().reset_index(name="n")
            .sort_values(["stock_code", "n"], ascending=[True, False])
            .drop_duplicates("stock_code").set_index("stock_code")["description"])
