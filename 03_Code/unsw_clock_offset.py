"""
Step 3.10: how UNSW-NB15 Stime relates to the NF-v3 start time of the same
flow. Uses conversations with exactly one record in each arm.
Writes 04_Results/phase3/unsw_clock_offset.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import releases as R
from manifest import write_manifest

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "04_Results" / "instrument" / "UNSW-NB15"
OUT = ROOT / "04_Results" / "phase3" / "unsw_clock_offset.csv"
Q = [0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]

rows, ins = [], []
for day in ("2015-01-22", "2015-02-18"):
    d = FROZEN / day
    ins += [d / "records.parquet", d / "conversations.parquet"]
    r = pd.read_parquet(d / "records.parquet", columns=["arm", "conv", "t0"])
    t = pd.read_parquet(d / "conversations.parquet",
                        columns=["conv", "n_unsw_original", "n_nf_unsw_v3"])
    one = t.loc[(t["n_unsw_original"] == 1) & (t["n_nf_unsw_v3"] == 1), "conv"]
    x = r[r["conv"].isin(one)]
    o = x[x["arm"] == "unsw_original"].set_index("conv")["t0"]
    # records hold the start of the widened interval; Stime is half a second later
    o = o + R.RESOLUTION["unsw_original"] / 2.0
    n = x[x["arm"] == "nf_unsw_v3"].set_index("conv")["t0"]
    dt = (n - o.reindex(n.index)).dropna()
    row = {"day": day, "flows": len(dt), "share_in_minus_half_to_half":
           round(float(((dt >= -0.5) & (dt <= 0.5)).mean()), 6),
           "mean": round(float(dt.mean()), 6)}
    for q in Q:
        row[f"q{q}"] = round(float(dt.quantile(q)), 6)
    rows.append(row)
pd.DataFrame(rows).to_csv(OUT, index=False)
write_manifest("unsw_clock_offset", inputs=ins, outputs=[OUT])
print(pd.DataFrame(rows).T)
