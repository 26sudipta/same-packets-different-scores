"""
Phase 11, review Major comment 5: the grid runs on a subsample that keeps 0.2 of
benign conversations and at most 20,000 conversations per attack class, which
raises attack prevalence. This script trains one model and one seed on the full
data of every primary arm, as a check of the precision-driven results.

    build   full-data arm tables (every scored conversation), written by
            phase5_data.build with benign fraction 1 and no attack cap, to
            Datasets/_cache_phase5/full/<bench>/<arm>.parquet
    run     random forest, seed 0, grouped split, the Phase 5 tuned setting of the
            arm, scored like a Phase 5 cell

The split columns are the instrument's, so the full-data test set holds every
test conversation, of which the subsample's test set is a part.

Usage, from the workspace root, NIDS_CACHE_ROOT set (phase11_common):
    .venv/bin/python 03_Code/phase11_full.py build CICIDS2017
    .venv/bin/python 03_Code/phase11_full.py run CICIDS2017 [--arms ...] [--models rf]

Outputs: 04_Results/phase11/full/{preds,cells}/<bench>/<arm>/rf_grouped_s0.*
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase11_common as C  # noqa: E402
import phase5_data as D  # noqa: E402
import phase5_grid as G  # noqa: E402

OUT = C.OUT / "full"
TAG = "full"
NO_CAP = 10 ** 12


def build(bench):
    # one call over all days, so that string features are coded on one level list.
    # Arms whose table is already written are skipped: the first run on the laptop (14 GB)
    # was killed while building BCCC after the other six arms were written, so a restart
    # builds only what is missing. String coding is per arm, so this gives the same tables.
    cfg = D.I.BENCHMARKS[bench]
    out_dir = D.CACHE / TAG / bench
    arms = list(cfg["arms"])
    missing = [a for a in arms if not (out_dir / f"{a}.parquet").is_file()]
    print(f"arms to build: {missing}", flush=True)
    if not missing:
        return
    if bench == "UNSW-NB15":
        cfg["arms"] = missing
        try:
            D.build(bench, None, TAG, benign_fraction=1.0, attack_cap=NO_CAP)
        finally:
            cfg["arms"] = arms
        return
    # CICIDS2017: phase5_data.build holds the records of all seven arms at once (15.2 M rows),
    # which the laptop cannot do for BCCC on top. With no subsample every scored conversation
    # is kept, so each arm can be built from its own records alone; the steps below are those
    # of phase5_data.build for one arm, in the same order, so the table is the same.
    for arm in missing:
        build_one(bench, arm, out_dir)


def build_one(bench, arm, out_dir):
    import time
    import pandas as pd
    I, F, R = D.I, D.F, D.R
    t_start = time.time()
    cfg = I.BENCHMARKS[bench]
    convs, recs = [], []
    for di, day in enumerate(cfg["days"]):
        d = D.FROZEN / bench / day
        t = pd.read_parquet(d / "conversations.parquet", columns=["conv", "sched", "in_all"])
        r = pd.read_parquet(d / "records.parquet", columns=D.REC_COLS)
        r["arm"] = r["arm"].astype(str)
        r = r[r["arm"] == arm].reset_index(drop=True)
        r["file"] = r["file"].astype(str)
        t = t[t["sched"].to_numpy() != I.EXCLUDED].reset_index(drop=True)
        t["gconv"] = di * 10 ** 8 + t["conv"].to_numpy()
        r = r[r["sched"] != I.EXCLUDED]
        r["gconv"] = di * 10 ** 8 + r["conv"].to_numpy()
        r["day"] = day
        convs.append(t[["gconv", "in_all"]])
        recs.append(r)
        print(f"{arm} {day}: {len(t):,} scored conversations, {len(r):,} records", flush=True)
    table = pd.concat(convs, ignore_index=True)
    a = pd.concat(recs, ignore_index=True)
    del recs
    a = a[a["gconv"].isin(set(table["gconv"].tolist()))].reset_index(drop=True)
    a["in_all"] = table.set_index("gconv")["in_all"].reindex(a["gconv"]).to_numpy()
    parts = []
    for f, g in a.groupby("file", sort=True):
        path = next(p for p in R.files(arm) if p.name == f)
        x, _ = F.read_features(arm, path, g["row"].to_numpy(),
                               g["src"].to_numpy(dtype=object),
                               g["dst"].to_numpy(dtype=object))
        x.index = g.index
        parts.append(x)
    x = pd.concat(parts).sort_index()
    del parts
    x, _ = F.code_strings(x, arm)
    meta_cols = ["gconv", "day", "sched", "in_all", "bytes", "file", "row"] + \
        [f"g{s_}" for s_ in D.SEEDS] + [f"r{s_}" for s_ in D.SEEDS]
    m = a[meta_cols].copy()
    fx = pd.DataFrame(F.to_matrix(x), columns=["f:" + c for c in x.columns])
    out = pd.concat([m, fx], axis=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / f"{arm}.parquet", index=False)
    print(f"{arm}: {len(out):,} records, {x.shape[1]} features "
          f"({time.time() - t_start:.0f} s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["build", "run"])
    ap.add_argument("bench")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--models", nargs="*", default=["rf"])
    a = ap.parse_args()
    if a.what == "build":
        build(a.bench)
    else:
        print(G.run(a.bench, a.arms or G.PRIMARY_ARMS[a.bench], a.models, ["grouped"], [0],
                    TAG, OUT).to_string())
