from b2bretail import data


def test_flags_catch_each_rule(raw):
    f = data.flags(raw)
    assert f["exact_duplicate"].sum() == 1
    assert f["cancellation"].sum() == 1
    assert f["adjustment"].sum() == 1
    assert f["missing_customer"].sum() == 2
    assert f.loc[raw["stock_code"] == "POST", "non_product_code"].all()
    assert f["non_positive_price"].sum() == 2  # zero price and the adjustment
    assert f.loc[(raw["stock_code"] == "10004") & (raw["price"] == 500.0), "price_outlier"].all()


def test_clean_sales_keeps_only_sales(raw):
    s = data.clean_sales(raw)
    assert not s["invoice"].str.startswith(("C", "A")).any()
    assert (s["quantity"] > 0).all() and (s["price"] > 0).all()
    assert set(s["stock_code"]) == {"10001", "10002", "10003", "10004"}
    assert len(s) == 6  # 2 + 1 (no customer) + 3
    assert (s["revenue"] == s["quantity"] * s["price"]).all()
    # the line without a customer stays: the imputation step needs it
    assert s["customer_id"].isna().sum() == 1


def test_quality_report_counts_add_up(raw):
    q = data.quality_report(raw).set_index("rule")
    assert q.loc["all lines", "lines"] == len(raw)
    assert q.loc["kept as sale lines", "lines"] == len(data.clean_sales(raw))


def test_descriptions_pick_the_most_frequent(raw):
    s = data.clean_sales(raw)
    s.loc[s.index[-1], "description"] = "OTHER NAME"
    d = data.descriptions(s)
    assert d["10004"] == "ITEM 10004"
