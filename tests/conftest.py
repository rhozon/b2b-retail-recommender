import numpy as np
import pandas as pd
import pytest


def line(inv, code, qty, date, price, cust, country="United Kingdom", desc=None):
    return {"Invoice": inv, "StockCode": code, "Description": desc or f"ITEM {code}", "Quantity": qty,
            "InvoiceDate": pd.Timestamp(date), "Price": price, "Customer ID": cust, "Country": country}


@pytest.fixture
def raw():
    from b2bretail import data

    rows = [
        line("500001", "10001", 6, "2011-01-03 10:00", 2.0, 1),
        line("500001", "10002", 12, "2011-01-03 10:00", 1.0, 1),
        line("500001", "10002", 12, "2011-01-03 10:00", 1.0, 1),            # exact duplicate
        line("C500002", "10001", -6, "2011-01-04 10:00", 2.0, 1),          # cancellation
        line("500003", "POST", 1, "2011-01-05 10:00", 18.0, 2),            # postage
        line("500004", "10003", 4, "2011-01-05 11:00", 0.0, 2),            # zero price
        line("500005", "10003", 4, "2011-01-06 11:00", 3.0, None),         # no customer
        line("A500006", "B", 1, "2011-01-06 12:00", -100.0, None),         # adjustment
        line("500007", "10004", 2, "2011-01-07 12:00", 500.0, 3),          # price far from usual
        line("500008", "10004", 2, "2011-01-08 12:00", 5.0, 3),
        line("500009", "10004", 2, "2011-01-09 12:00", 5.0, 3),
    ]
    return data.standardise(pd.DataFrame(rows))


def synthetic_sales(n_groups=3, users_per_group=30, items_per_group=8, invoices=6, seed=0,
                    start="2011-01-01", days=300):
    """Customers in groups that buy from their own block of products."""
    rng = np.random.default_rng(seed)
    rows = []
    inv = 0
    t0 = pd.Timestamp(start)
    for g in range(n_groups):
        block = [f"{20000 + g * 100 + i}" for i in range(items_per_group)]
        for u in range(users_per_group):
            cid = g * 1000 + u
            for k in range(invoices):
                inv += 1
                when = t0 + pd.Timedelta(days=int(rng.integers(0, days)))
                for code in rng.choice(block, size=3, replace=False):
                    rows.append({"invoice": f"{600000 + inv}", "stock_code": code, "description": f"ITEM {code}",
                                 "quantity": 1, "invoice_date": when, "price": 1.0, "customer_id": cid,
                                 "country": "United Kingdom", "revenue": 1.0})
    df = pd.DataFrame(rows)
    df["customer_id"] = df["customer_id"].astype("Int64")
    return df
