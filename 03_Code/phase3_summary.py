"""
Step 3.10: combine the per-day Phase 3 summaries into one file per table,
04_Results/phase3/all_<table>.csv, plus all_gap_sweep.csv and
phase3_overview.csv (one row per benchmark day).
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instrument as I
from manifest import write_manifest

ROOT = Path(__file__).resolve().parent.parent
P3 = ROOT / "04_Results" / "phase3"
FROZEN = ROOT / "04_Results" / "instrument"
TABLES = ["arm_facts", "label_agreement", "label_confusion", "window_rule_sensitivity",
          "cardinality", "presence", "attempted", "mixed", "episodes", "leakage", "pair_stats"]

outs, ins = [], []
for t in TABLES:
    parts = []
    for f in sorted((P3 / t).glob("*.csv")):
        bench, day = f.stem.split("_", 1)
        if f.stat().st_size < 2:
            continue
        df = pd.read_csv(f)
        df.insert(0, "day", day)
        df.insert(0, "benchmark", bench)
        parts.append(df)
        ins.append(f)
    if parts:
        p = P3 / f"all_{t}.csv"
        pd.concat(parts, ignore_index=True).to_csv(p, index=False)
        outs.append(p)
sw = sorted(P3.glob("gap_sweep_*.csv"))
if sw:
    p = P3 / "all_gap_sweep.csv"
    pd.concat([pd.read_csv(f) for f in sw], ignore_index=True).to_csv(p, index=False)
    outs.append(p)
    ins += sw

rows = []
for bench, cfg in I.BENCHMARKS.items():
    for day in cfg["days"]:
        d = FROZEN / bench / day
        if not (d / "conversations.parquet").is_file():
            continue
        t = pd.read_parquet(d / "conversations.parquet")
        atk = ~t["sched"].isin([I.BENIGN, I.EXCLUDED])
        r = {"benchmark": bench, "day": day, "conversations": len(t),
             "attack_conversations": int(atk.sum()),
             "excluded_conversations": int((t["sched"] == I.EXCLUDED).sum()),
             "in_all_arms": int(t["in_all"].sum()),
             "attack_in_all_arms": int((t["in_all"] & atk).sum()),
             "attack_classes": ";".join(sorted(t.loc[atk, "sched"].unique()))}
        for a in cfg["arms"]:
            r[f"records_{a}"] = int(t[f"n_{a}"].sum())
        rows.append(r)
if rows:
    p = P3 / "phase3_overview.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    outs.append(p)
write_manifest("phase3_summary", inputs=ins, outputs=outs)
print("\n".join(str(o) for o in outs))
