"""Spark and pandas apply the same cleaning rules. Runs where PySpark and Java are available
(the spark job in CI); skipped elsewhere."""
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("pyspark")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spark"))
SAMPLE = ROOT / "data" / "sample_lines.csv"


@pytest.fixture(scope="module")
def spark():
    import etl_als

    s = etl_als.spark_session("parity")
    yield s
    s.stop()


def test_spark_cleaning_matches_pandas(spark):
    import etl_als
    from b2bretail import data

    pdf = data.clean_sales(data.standardise(pd.read_csv(SAMPLE, dtype={"Invoice": str, "StockCode": str})))
    sdf = etl_als.clean_sales(etl_als.read_lines(spark, str(SAMPLE))).toPandas()
    assert len(sdf) == len(pdf)
    assert abs(sdf["revenue"].sum() - pdf["revenue"].sum()) < 1e-6 * pdf["revenue"].sum()
    assert sdf["invoice"].nunique() == pdf["invoice"].nunique()


def test_spark_als_recommends(spark):
    import etl_als

    sales = etl_als.clean_sales(etl_als.read_lines(spark, str(SAMPLE)))
    model, _ = etl_als.fit_als(etl_als.interactions(sales), rank=8, iters=5)
    recs = model.recommendForAllUsers(5).toPandas()
    assert len(recs) > 100
    assert recs["recommendations"].map(len).eq(5).all()
