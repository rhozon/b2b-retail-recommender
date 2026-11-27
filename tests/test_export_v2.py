import json
from pathlib import Path

import pandas as pd
import pytest

from b2bretail import export_v2

RESULTS = Path(__file__).resolve().parents[1] / "results"


@pytest.mark.skipif(not (RESULTS / "promos.json").exists(), reason="needs the committed results")
def test_v2_data_is_strict_and_matches_results(tmp_path):
    out = tmp_path / "d.json"
    export_v2.main([str(RESULTS), str(out)])
    d = json.loads(out.read_text(encoding="utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    rm = pd.read_csv(RESULTS / "recs_metrics.csv")
    assert len(d["recs"]) == len(rm)
    assert {"SASRec, product by product", "SASRec, basket by basket"} <= set(rm["model"])
    assert [c["campaign_id"] for c in d["promos"]] == [8, 13, 18]
    assert all("path" not in v for v in (d["lake"] or {}).values())
