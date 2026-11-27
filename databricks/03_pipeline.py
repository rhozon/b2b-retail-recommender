"""Databricks task 3: the pandas/PyTorch pipeline (recommenders, segments, assortment,
promotions) on a single node, reading the silver table and logging to the workspace's MLflow."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyspark.sql import SparkSession  # noqa: E402

catalog, schema = sys.argv[1:3]
spark = SparkSession.builder.getOrCreate()
out = Path("/tmp/b2b-retail")
out.mkdir(exist_ok=True)
from b2bretail import cli, data  # noqa: E402

# load_raw reads this cache when it exists; standardise gives it the same types as the local run
data.standardise(spark.table(f"{catalog}.{schema}.bronze_lines").toPandas()).to_parquet(out / "online_retail_II.parquet")

cli.main(["--raw-dir", str(out), "--out", str(out / "results"), "--track", "databricks"])
