import json

import pandas as pd
import pytest

from b2bretail import lake


def test_delta_tables_round_trip_and_keep_versions(tmp_path):
    sales = pd.DataFrame({"invoice": ["1", "2"], "stock_code": ["A", "B"], "customer_id": pd.array([1, None], "Int64"),
                          "invoice_date": pd.to_datetime(["2011-01-01", "2011-02-01"]), "revenue": [1.0, 2.0]})
    info = lake._write(sales.assign(month=["2011-01", "2011-02"]), tmp_path / "silver" / "sales", ["month"])
    assert info["rows"] == 2 and info["version"] == 0
    lake._write(sales.head(1).assign(month=["2011-01"]), tmp_path / "silver" / "sales", ["month"])
    assert len(lake.read(tmp_path, "silver/sales")) == 1
    assert len(lake.read(tmp_path, "silver/sales", version=0)) == 2


def test_mlflow_run_logs_metrics(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from b2bretail import tracking

    (tmp_path / "summary.json").write_text(json.dumps({"x": 1}))
    rm = pd.DataFrame([{"mode": "all products", "model": "Buy again", "ndcg@10": 0.3, "recall@10": 0.2}])
    summary = {"tuning": {"best": {"epochs": 3, "als": [32, 10.0]}},
               "split": {"cutoff": "2011-10-09", "train_customers": 10, "train_products": 5}}
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    run_id = tracking.log_run(uri, summary, rm, tmp_path, experiment="test")
    run = mlflow.MlflowClient(uri).get_run(run_id)
    assert run.data.metrics["all_products/buy_again/ndcg_at_10"] == 0.3
    assert run.data.params["als"] == "[32, 10.0]"
