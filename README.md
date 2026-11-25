# B2B retail recommender

What should a wholesaler suggest to each retailer that orders from it, which products should
lead each group of retailers' catalogue page, who should get a promotion, and what can be done
with the orders that arrive without an account number? This repository answers the five
questions on the public transaction log of a UK online wholesaler, with every model tested on
a period it never saw.

Case study page: <https://rhozon.github.io/portfolio/b2b-retail-recommender.html>

## Data

[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (Chen, 2012,
UCI Machine Learning Repository, doi:10.24432/C5CG6D, CC BY 4.0): every invoice line of a UK
online retailer of gift-ware between 1 December 2009 and 9 December 2011. Many of its customers
are wholesalers. The code downloads nothing by itself: put `online_retail_II.xlsx` in
`~/data/online-retail-ii` (or set `ORII_DIR`). `data/sample_lines.csv` holds every line of
1,500 random invoices from the same file, for the tests and the Spark job.

## What the code does

| Step | Module | Method |
|---|---|---|
| Data quality | `b2bretail/data.py` | One flag per rule: cancellations, adjustments, non-positive quantity or price, non-product codes, exact duplicates, missing customer, price outliers |
| Recommendations | `recs.py`, `seqmodel.py` | Popularity, buy again, item-to-item co-purchase, implicit ALS (Hu, Koren and Volinsky, 2008) and a GRU next-basket model in PyTorch |
| Evaluation | `split.py`, `metrics.py` | Train before 9 October 2011, test on the last 61 days; settings picked on the 61 days before that; Recall@10 and NDCG@10 with bootstrap intervals over customers |
| Segments | `segment.py` | RFM and basket features, k-means with k chosen by silhouette, named from the medians |
| Missing customer ids | `impute.py` | TF-IDF purchase profiles and cosine similarity within the country, evaluated by hiding known ids |
| Assortment | `assortment.py` | Mixed-integer program: 30 products per segment, at most 4 per product family |
| Promotions | `uplift.py` | Semi-synthetic randomised voucher on real customers; T-, X- and DR-learners against simple rules |
| Spark | `spark/etl_als.py` | Same cleaning rules in PySpark and Spark MLlib implicit ALS; CI checks parity with pandas |

## Results

Numbers from `results/`, written by `python -m b2bretail.cli`. Test window: 9 October to 9
December 2011.

**Data quality.** 1,067,371 raw lines; 19,494 cancellations, 34,335 exact duplicates (22,844
from the 1 to 9 December 2010 overlap between the two sheets of the workbook), 6,093
non-product codes, 6,207 zero or negative prices, 7,881 price outliers (flagged, kept).
1,003,214 sale lines remain; 22.6% of them (13.1% of revenue) have no customer number.

**Recommendations**, NDCG@10 with 95% bootstrap intervals over customers:

| Model | All products (2,041 customers) | New to the customer (1,917 customers) |
|---|---:|---:|
| Buy again | 0.341 (0.329 to 0.353) | not applicable |
| Sequential GRU (PyTorch) | 0.260 (0.249 to 0.271) | 0.071 (0.066 to 0.077) |
| Implicit ALS | 0.196 (0.187 to 0.205) | 0.070 (0.064 to 0.075) |
| Item-to-item (co-purchase) | 0.163 (0.154 to 0.172) | 0.059 (0.054 to 0.065) |
| Popularity (last 90 days) | 0.102 (0.096 to 0.109) | 0.045 (0.041 to 0.049) |

Reorders dominate the next order; for new products the GRU and ALS tie (difference 0.002, 95%
interval -0.004 to 0.007) and both beat co-purchase and popularity.

**Segments** (k = 3 by silhouette): top and active, 33.3% of customers and 83.5% of revenue;
mid-value and lapsed, 46.5% and 12.9%; low-value and lapsed, 20.2% and 3.5%.

**Missing customer ids.** With known ids hidden on 4,790 test-window orders, profile matching
finds the customer first 22.5% of the time and in the top ten 46.3% of the time (customers
seen before), against 4.4% and 11.1% for the busiest customers in the country. The most
confident quarter is right 24.1% of the time, so matches are a review queue, not an automatic
fill. Orders without an id are larger (median 20 lines against 15) and priced higher (median
unit price 3.36 against 1.97), which points to another channel.

**Assortment.** In the top segment, the 30-product shelf with at most 4 per family covers
12.6% of the segment's test-window revenue and reaches 76.8% of its customers, against 11.6%
and 72.6% for the segment's top sellers and 11.9% and 72.3% for the overall top sellers.

**Promotions (semi-synthetic).** Area under the gain curve relative to the oracle, mean of 10
replications on the 5,472 customers with history: X-learner 0.225, DR-learner 0.205,
T-learner 0.158, X-learner on the unscaled outcome 0.132, top spenders first 0.122, most
lapsed first 0.449. At a budget of 20% of customers the X-learner brings 39,590 of true
incremental revenue against 21,940 at random. The rule "most lapsed first" matches the
simulated mechanism; with the effect scaled by 2 and 4 the X-learner reaches 0.411 and 0.605
(`results/uplift_sensitivity.csv`).

## Run it

Python 3.11 or later:

```bash
pip install -e ".[dev]"
pytest -v
python -m b2bretail.cli --out results --site path/to/assets/case10
```

The full run takes about 20 minutes on a laptop CPU. The Spark job needs Java 17 and
`pip install -e ".[spark]"`:

```bash
python spark/etl_als.py data/sample_lines.csv
```

CI runs the test suite and, in a second job with Java 17, the Spark parity tests and the Spark
job on the sample.

## License

Code under the MIT license. The data belongs to its authors and is used under CC BY 4.0.
