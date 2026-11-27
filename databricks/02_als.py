"""Databricks task 2: Spark MLlib implicit ALS on the silver table, logged to the workspace's
MLflow, with the top-10 products per customer written to a gold Delta table."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spark"))
import etl_als  # noqa: E402
import mlflow  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

catalog, schema = sys.argv[1:3]
spark = SparkSession.builder.getOrCreate()
sales = spark.table(f"{catalog}.{schema}.silver_sales")
with mlflow.start_run(run_name="spark-als"):
    mlflow.log_params({"rank": 32, "iters": 15})
    model, idx = etl_als.fit_als(etl_als.interactions(sales), rank=32, iters=15)
    model.recommendForAllUsers(10).write.format("delta").mode("overwrite").saveAsTable(f"{catalog}.{schema}.gold_als_top10")
