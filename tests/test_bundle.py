"""The Databricks bundle cannot be deployed in CI (no workspace), so CI checks what it can: the
file parses, every task points to a file in the repository, dependencies name existing tasks,
and each task script compiles."""
import py_compile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_bundle_tasks_point_to_files_and_dependencies_exist():
    b = yaml.safe_load((ROOT / "databricks.yml").read_text())
    tasks = b["resources"]["jobs"]["b2b_retail_pipeline"]["tasks"]
    keys = {t["task_key"] for t in tasks}
    for t in tasks:
        f = ROOT / t["spark_python_task"]["python_file"]
        assert f.exists(), f
        py_compile.compile(str(f), doraise=True)
        for d in t.get("depends_on", []):
            assert d["task_key"] in keys
