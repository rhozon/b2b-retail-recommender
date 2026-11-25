"""Write data/sample_lines.csv: every raw line of 1,500 random invoices, cancellations and
adjustments included, in the original column names. Used by the tests and by the Spark job
in CI, so neither needs the full 44 MB workbook."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2bretail import data  # noqa: E402

raw = data.load_raw(sys.argv[1] if len(sys.argv) > 1 else data.RAW_DIR)
rng = np.random.default_rng(2026)
inv = rng.choice(raw["invoice"].unique(), size=1500, replace=False)
s = raw[raw["invoice"].isin(inv)].copy()
s = s.rename(columns={v: k for k, v in data.COLUMNS.items()})
s["InvoiceDate"] = s["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")
out = Path(__file__).resolve().parents[1] / "data" / "sample_lines.csv"
s.to_csv(out, index=False)
print(out, len(s))
