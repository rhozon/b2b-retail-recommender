"""The cleaning rules and an implicit ALS in PySpark, for running the same pipeline on a cluster.

The pandas version in b2bretail/ is the one used for the results. This job applies the same
rules with Spark DataFrames and trains Spark MLlib's implicit ALS. CI runs it on
data/sample_lines.csv and checks that its sale lines match the pandas cleaning exactly.

    python spark/etl_als.py data/sample_lines.csv
"""
from __future__ import annotations

import sys

from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

NAMES = {"Invoice": "invoice", "StockCode": "stock_code", "Description": "description", "Quantity": "quantity",
         "InvoiceDate": "invoice_date", "Price": "price", "Customer ID": "customer_id", "Country": "country"}


def spark_session(app: str = "b2b-retail") -> SparkSession:
    return (SparkSession.builder.appName(app).master("local[2]")
            .config("spark.sql.shuffle.partitions", "4").config("spark.ui.enabled", "false").getOrCreate())


def read_lines(spark: SparkSession, path: str) -> DataFrame:
    df = spark.read.csv(path, header=True, inferSchema=False)
    for old, new in NAMES.items():
        df = df.withColumnRenamed(old, new)
    return (df.withColumn("invoice", F.trim("invoice"))
              .withColumn("stock_code", F.upper(F.trim("stock_code")))
              .withColumn("quantity", F.col("quantity").cast("long"))
              .withColumn("price", F.col("price").cast("double"))
              .withColumn("customer_id", F.col("customer_id").cast("double").cast("long"))
              .withColumn("invoice_date", F.to_timestamp("invoice_date")))


def clean_sales(lines: DataFrame) -> DataFrame:
    """Same rules as b2bretail.data.clean_sales: drop exact duplicates, cancellations (C),
    adjustments (A), non-positive quantity or price and non-product stock codes."""
    return (lines.dropDuplicates()
                 .filter(~F.col("invoice").startswith("C") & ~F.col("invoice").startswith("A"))
                 .filter((F.col("quantity") > 0) & (F.col("price") > 0))
                 .filter(F.col("stock_code").rlike(r"^\d{5}"))
                 .withColumn("revenue", F.col("quantity") * F.col("price")))


def interactions(sales: DataFrame) -> DataFrame:
    """Customer x product: number of distinct invoices."""
    return (sales.filter(F.col("customer_id").isNotNull())
                 .groupBy("customer_id", "stock_code").agg(F.countDistinct("invoice").alias("invoices")))


def fit_als(inter: DataFrame, rank: int = 16, iters: int = 10):
    idx = StringIndexer(inputCol="stock_code", outputCol="item").fit(inter)
    data = idx.transform(inter).withColumn("user", F.col("customer_id").cast("int"))
    als = ALS(rank=rank, maxIter=iters, regParam=0.1, alpha=10.0, implicitPrefs=True, userCol="user",
              itemCol="item", ratingCol="invoices", coldStartStrategy="drop", seed=0)
    return als.fit(data), idx


def main(path: str) -> None:
    spark = spark_session()
    lines = read_lines(spark, path)
    sales = clean_sales(lines)
    inter = interactions(sales)
    model, idx = fit_als(inter)
    recs = model.recommendForAllUsers(10)
    print(f"raw lines {lines.count()}, sale lines {sales.count()}, customer-product pairs {inter.count()}, "
          f"customers with recommendations {recs.count()}")
    spark.stop()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample_lines.csv")
