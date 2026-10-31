"""
Step 3.1: sweep the conversation gap T and report, per class, how many
conversations each value produces and how many endpoint keys it splits.

Run from the workspace root:
    .venv/bin/python 03_Code/gap_sweep.py CICIDS2017 Tuesday
    .venv/bin/python 03_Code/gap_sweep.py UNSW-NB15 2015-01-22
Writes 04_Results/phase3/gap_sweep_<bench>_<day>.csv and a manifest.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instrument as I
from manifest import write_manifest

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "04_Results" / "phase3"
T_GRID = [5, 15, 30, 60, 120, 300]


def sweep(bench, day, grid=T_GRID, use_cache=True):
    cfg = I.BENCHMARKS[bench]
    rec = I.load_day(bench, day, use_cache)
    gt = I.load_unsw_gt() if bench == "UNSW-NB15" else None
    rows = []
    for T in grid:
        conv, table, stats = I.build_conversations(rec, T, cfg["defining"])
        if bench == "CICIDS2017":
            lab = I.label_cic(table, day)[0]
        else:
            lab = I.label_unsw(table, day, gt)[0]
        table["sched"] = lab
        nper = table.groupby("key_str").size()
        table["key_split"] = table["key_str"].map(nper) > 1
        for cls, g in table.groupby("sched"):
            rows.append({"benchmark": bench, "day": day, "T": T, "class": cls,
                         "conversations": len(g), "keys": g["key_str"].nunique(),
                         "keys_split": int(g.loc[g["key_split"], "key_str"].nunique()),
                         "records": int(g[[f"n_{a}" for a in cfg["arms"]]].to_numpy().sum()),
                         "nondef_mapped": stats["n_mapped"],
                         "nondef_unmapped": stats["n_unmapped"],
                         "nondef_ambiguous": stats["n_ambiguous"]})
        rows.append({"benchmark": bench, "day": day, "T": T, "class": "ALL",
                     "conversations": len(table), "keys": table["key_str"].nunique(),
                     "keys_split": int(table.loc[table["key_split"], "key_str"].nunique()),
                     "records": len(rec), "nondef_mapped": stats["n_mapped"],
                     "nondef_unmapped": stats["n_unmapped"],
                     "nondef_ambiguous": stats["n_ambiguous"]})
        print(bench, day, "T", T, "conversations", len(table), flush=True)
    df = pd.DataFrame(rows)
    # change against the next larger T, per class
    df = df.sort_values(["class", "T"], kind="mergesort")
    nxt = df.groupby("class")["conversations"].shift(-1)
    df["rel_change_to_next_T"] = ((df["conversations"] - nxt) / df["conversations"]).round(6)
    return df.sort_values(["T", "class"], kind="mergesort")


if __name__ == "__main__":
    bench, day = sys.argv[1], sys.argv[2]
    OUT.mkdir(parents=True, exist_ok=True)
    df = sweep(bench, day)
    out = OUT / f"gap_sweep_{bench}_{day}.csv"
    df.to_csv(out, index=False)
    write_manifest(f"gap_sweep_{bench}_{day}", inputs=[], outputs=[out],
                   params={"benchmark": bench, "day": day, "T_grid": T_GRID})
    print(out)
