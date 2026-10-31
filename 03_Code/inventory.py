"""
Step 2.9: release inventory.

Writes, for every release, file and UTC day of flow start:
    04_Results/release_inventory.csv     rows, columns, five tuple, clock, labels, SHA256
    04_Results/release_label_counts.csv  rows, first and last UTC start per
                                         release, file, day and label
and a run manifest.

Run from the workspace root:
    .venv/bin/python 03_Code/inventory.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import sha256, write_manifest  # noqa: E402
from releases import RELEASES, files, load_keys  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "04_Results" / "release_inventory.csv"
OUT_LABELS = ROOT / "04_Results" / "release_label_counts.csv"
FIVE = ["src", "sport", "dst", "dport", "proto"]


def n_columns(path, release):
    if release == "unsw_original":
        with open(path, encoding="latin-1") as f:
            return len(next(csv.reader(f)))
    with open(path, encoding="latin-1") as f:
        return len(f.readline().split(","))


def resolution(t):
    """Finest unit the timestamps actually use."""
    t = t.dropna().to_numpy()
    if t.size == 0:
        return "none"
    us = np.round(t * 1e6).astype(np.int64)
    for unit, step in (("minute", 60_000_000), ("second", 1_000_000),
                       ("millisecond", 1_000), ("microsecond", 1)):
        if np.all(us % step == 0):
            return unit
    return "sub-microsecond"


def utc(x):
    return pd.Timestamp(x, unit="s").strftime("%Y-%m-%d %H:%M:%S") if np.isfinite(x) else ""


def main():
    rows, label_rows, inputs = [], [], []
    for rel, meta in RELEASES.items():
        for path in files(rel):
            print(f"{rel}: {path.name}", flush=True)
            inputs.append(path)
            digest = sha256(path)
            ncol = n_columns(path, rel)
            df = load_keys(rel, path)
            has_five = bool(df[FIVE].notna().all(axis=None)) and len(df) > 0
            res = resolution(df["t_utc"])
            day = pd.to_datetime(df["t_utc"], unit="s").dt.strftime("%Y-%m-%d")
            df["day_utc"] = day.fillna("no timestamp")
            for d, g in df.groupby("day_utc", sort=True):
                counts = g["label"].value_counts()
                rows.append({
                    "release": rel, "benchmark": meta["benchmark"],
                    "exporter": meta["exporter"], "file": path.name,
                    "day_utc": d, "rows": len(g), "columns": ncol,
                    "five_tuple": has_five,
                    "unparsed_timestamps": int(g["t_utc"].isna().sum()),
                    "timestamp_example": str(g["t_raw"].iloc[0]),
                    "clock_as_shipped": meta["tz"],
                    "utc_offset_hours": "" if meta["utc_offset"] is None
                    else meta["utc_offset"] / 3600.0,
                    "resolution": res,
                    "first_utc": utc(g["t_utc"].min()), "last_utc": utc(g["t_utc"].max()),
                    "n_labels": len(counts),
                    "labels": "; ".join(f"{k}={v}" for k, v in counts.items()),
                    "sha256": digest, "path": str(path), "note": meta["note"],
                })
                span = g.groupby("label")["t_utc"].agg(["min", "max"])
                for k, v in counts.items():
                    label_rows.append({"release": rel, "file": path.name,
                                       "day_utc": d, "label": k, "rows": int(v),
                                       "first_utc": utc(span.loc[k, "min"]),
                                       "last_utc": utc(span.loc[k, "max"])})
            del df
    pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(label_rows).to_csv(OUT_LABELS, index=False)
    m = write_manifest("release_inventory", inputs=inputs, outputs=[OUT, OUT_LABELS])
    print(f"wrote {OUT}, {OUT_LABELS}, {m}")


if __name__ == "__main__":
    main()
