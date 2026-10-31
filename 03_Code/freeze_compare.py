"""
Step 9.4: compare the clean-clone freeze with the working copy, file by file.

Usage, from the working copy root:
    .venv/bin/python 03_Code/freeze_compare.py <path of the clone>

For every result file (csv, md, tex tables, numbers) the script reports whether the
two copies are byte identical and, for csv files that differ, the largest absolute
difference over numeric cells and the number of differing text cells. Files only in
one copy are listed. Manifests, logs and run bookkeeping are skipped (they carry
times and paths). Writes 04_Results/freeze/compare.csv and prints a summary.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SKIP_PARTS = {"manifests", "logs", "freeze", "steps", "__pycache__", "cells", "preds", "repro",
              "retrain"}
GLOBS = ["04_Results/**/*.csv", "04_Results/**/*.md", "05_Paper/tables/*.tex"]


def files(base):
    out = set()
    for g in GLOBS:
        for p in base.glob(g):
            rel = p.relative_to(base)
            if not SKIP_PARTS & set(rel.parts[:-1]):
                out.add(rel)
    return out


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def csv_diff(a, b):
    x, y = pd.read_csv(a, low_memory=False), pd.read_csv(b, low_memory=False)
    if x.shape != y.shape or list(x.columns) != list(y.columns):
        return f"shape {x.shape} vs {y.shape}", np.nan, np.nan
    num, txt = 0.0, 0
    for c in x.columns:
        u, v = x[c], y[c]
        if pd.api.types.is_numeric_dtype(u) and pd.api.types.is_numeric_dtype(v):
            d = (u.astype(float) - v.astype(float)).abs()
            both_nan = u.isna() & v.isna()
            one_nan = u.isna() ^ v.isna()
            txt += int(one_nan.sum())
            if (~both_nan & ~one_nan).any():
                num = max(num, float(d[~both_nan & ~one_nan].max()))
        else:
            txt += int((u.astype(str) != v.astype(str)).sum())
    return "same shape", num, txt


def main():
    clone = Path(sys.argv[1]).resolve()
    fa, fb = files(ROOT), files(clone)
    rows = []
    for rel in sorted(fa | fb):
        a, b = ROOT / rel, clone / rel
        if rel not in fb:
            rows.append(dict(file=str(rel), status="only in working copy"))
            continue
        if rel not in fa:
            rows.append(dict(file=str(rel), status="only in clone"))
            continue
        if md5(a) == md5(b):
            rows.append(dict(file=str(rel), status="identical"))
            continue
        r = dict(file=str(rel), status="differs")
        if rel.suffix == ".csv":
            r["detail"], r["max_abs_num_diff"], r["text_cells_differ"] = csv_diff(a, b)
        rows.append(r)
    out = pd.DataFrame(rows)
    p = ROOT / "04_Results" / "freeze" / "compare.csv"
    out.to_csv(p, index=False)
    print(out["status"].value_counts().to_string())
    print(out[out["status"] != "identical"].to_string())
    print("wrote", p)


if __name__ == "__main__":
    main()
