"""Phase 7 helpers: the Phase 5 arm table with the phase7_extras columns."""
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instrument as I  # noqa: E402
import phase5_data as D  # noqa: E402
import phase5_grid as G  # noqa: E402
import phase7_extras as X  # noqa: E402

_load_arm = G.load_arm
EXTRA_COLS = ["cT5", "cT30", "cT60", "cT120", "own", "sched_fold", "episode_unit"]
META = ["gconv", "day", "sched", "in_all", "bytes", "file", "row"] + \
    [f"g{s}" for s in G.SEEDS] + [f"r{s}" for s in G.SEEDS]


def extras(bench, src_arm):
    """Extras of one source arm only (read with a row filter, to save memory)."""
    days = I.BENCHMARKS[bench]["days"]
    return pd.concat([pd.read_parquet(X.CACHE / bench / f"extras_{d}.parquet",
                                      filters=[("arm", "==", src_arm)]) for d in days],
                     ignore_index=True)


def load_arm_ext(bench, arm, features=True):
    """Phase 5 arm table (same rows, same order) plus the extras columns.
    features=False reads only the metadata columns (same rows as G.load_arm)."""
    if features:
        t, fcols = _load_arm(bench, arm)
    else:
        p = D.CACHE / "main" / bench / f"{G.source_arm(bench, arm)}.parquet"
        names = pq.ParquetFile(p).schema.names
        cols = META + [c for c in ("dup", "near") if c in names]
        t = pd.read_parquet(p, columns=cols)
        if arm == "unsw_dedup":
            t = t[~t["dup"]]
        elif arm == "unsw_collapsed":
            t = t[~t["dup"] & ~t["near"]]
        t, fcols = t.reset_index(drop=True), []
    ex = extras(bench, G.source_arm(bench, arm)).drop(columns=["arm"])
    m = t[["day", "file", "row"]].astype({"day": str, "file": str}).merge(
        ex, on=["day", "file", "row"], how="left", validate="one_to_one")
    if m["own"].isna().any():
        raise AssertionError(f"{bench} {arm}: {int(m['own'].isna().sum())} records without extras")
    for c in EXTRA_COLS:
        t[c] = m[c].to_numpy()
    return t, fcols
