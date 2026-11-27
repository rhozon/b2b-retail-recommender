"""MLflow tracking: one run per pipeline execution, with the tuned settings, every model's
test metrics and the result files as artifacts. Locally the store is SQLite; on Databricks the
same code logs to the workspace's managed MLflow (tracking URI "databricks")."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


def _key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-/ ]", "", s).strip().replace(" ", "_").lower()


def log_run(uri: str, summary: dict, recs_metrics: pd.DataFrame, out: Path, experiment: str = "b2b-retail") -> str:
    import mlflow

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name="pipeline") as run:
        best = summary.get("tuning", {}).get("best", {})
        mlflow.log_params({k: (str(v) if isinstance(v, (list, tuple)) else v) for k, v in best.items()})
        mlflow.log_params({"cutoff": summary["split"]["cutoff"], "customers": summary["split"]["train_customers"],
                           "products": summary["split"]["train_products"]})
        for _, r in recs_metrics.iterrows():
            base = f"{_key(r['mode'])}/{_key(r['model'])}"
            mlflow.log_metrics({f"{base}/ndcg_at_10": float(r["ndcg@10"]), f"{base}/recall_at_10": float(r["recall@10"])})
        for f in ("summary.json", "recs_metrics.csv", "segment_profile.csv", "assortment.csv", "promos.json"):
            if (out / f).exists():
                mlflow.log_artifact(str(out / f))
        return run.info.run_id
