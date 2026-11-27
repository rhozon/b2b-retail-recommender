"""Databricks task 1: raw lines to bronze and silver Delta tables in Unity Catalog.
Uses the same cleaning functions as spark/etl_als.py, which CI checks against pandas."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spark"))
import etl_als  # noqa: E402
from pyspark.sql import SparkSession, functions as F  # noqa: E402

raw_path, catalog, schema = sys.argv[1:4]
spark = SparkSession.builder.getOrCreate()
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
lines = etl_als.read_lines(spark, raw_path)
lines.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{catalog}.{schema}.bronze_lines")
sales = etl_als.clean_sales(lines).withColumn("month", F.date_format("invoice_date", "yyyy-MM"))
(sales.write.format("delta").mode("overwrite").option("overwriteSchema", "true").partitionBy("month")
 .saveAsTable(f"{catalog}.{schema}.silver_sales"))
