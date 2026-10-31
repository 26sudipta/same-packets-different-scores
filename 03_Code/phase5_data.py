"""
Steps 5.2 and 5.3: the Phase 5 training tables, one per arm, from the frozen
Phase 3 instrument.

Usage, from the workspace root:
    .venv/bin/python 03_Code/phase5_data.py CICIDS2017 [--days Tuesday] [--tag trial]
    .venv/bin/python 03_Code/phase5_data.py UNSW-NB15

What it does, per benchmark:

1. Reads conversations.parquet and records.parquet of every day (or the given
   days). Conversation ids become global: day position * 10^8 + conv.
2. Drops `excluded` conversations and draws the subsample on conversations,
   by a hash keyed on SUBSAMPLE_SEED (independent of the split seeds 0 to 4):
   benign conversations whose hash falls in the lowest BENIGN_FRACTION, and
   for each attack class the ATTACK_CAP conversations with the smallest hash
   (all of them when the class is smaller). The same conversations are kept
   in every arm, so the arms still see the same traffic.
3. For UNSW-NB15 marks, per day and before subsampling, the exact duplicate
   lines of the original (all 49 fields, as in Phase 4) and the near repeats
   left after them (same five tuple, start, end and bytes).
4. Reads each arm's features through features.read_features, checking the
   addresses of every row against the instrument, and writes one parquet per
   arm to Datasets/_cache_phase5/<tag>/<benchmark>/<arm>.parquet: meta
   columns, then the features prefixed "f:".
5. Writes 04_Results/phase5/feature_table.csv (step 5.2) and the subsample
   counts per class.

The settings live in 04_Results/phase5/settings.json, written once in step
5.3 from the time trial.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import instrument as I  # noqa: E402
import releases as R  # noqa: E402
from manifest import write_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "04_Results" / "instrument"
OUT = ROOT / "04_Results" / "phase5"
CACHE = R.CACHE_ROOT / "_cache_phase5"
SETTINGS = OUT / "settings.json"
SUBSAMPLE_SEED = 1000
SEEDS = [0, 1, 2, 3, 4]
REC_COLS = ["arm", "file", "row", "conv", "src", "sport", "dst", "dport", "proto", "t0", "t1",
            "bytes", "sched"] + [f"g{s}" for s in SEEDS] + [f"r{s}" for s in SEEDS]


def load_settings(path=SETTINGS):
    with open(path) as f:
        return json.load(f)


def subsample(table, bench, day_tables, benign_fraction, attack_cap):
    """Keep flag per conversation (table has one row per scored conversation)."""
    h = np.concatenate([I.unit_hash(I.conv_uid(bench, d, t), SUBSAMPLE_SEED)
                        for d, t in day_tables])
    u = h.astype(np.float64) / 2.0 ** 64
    sched = table["sched"].to_numpy()
    keep = np.zeros(len(table), dtype=bool)
    ben = sched == I.BENIGN
    keep[ben] = u[ben] < benign_fraction
    df = pd.DataFrame({"s": sched, "h": h, "i": np.arange(len(table))})
    df = df[~ben].sort_values(["s", "h", "i"], kind="mergesort")
    rank = df.groupby("s").cumcount().to_numpy()
    keep[df["i"].to_numpy()[rank < attack_cap]] = True
    return keep


def unsw_flags(rec):
    """Exact duplicate and near repeat flags for the UNSW-NB15 original, one day."""
    import agreement as A
    o = (rec["arm"] == "unsw_original").to_numpy()
    lh = A.unsw_line_hashes()
    sub = rec.loc[o, ["file", "row"]]
    h = np.empty(len(sub), dtype=np.uint64)
    fs = sub["file"].astype(str).to_numpy()
    for f in np.unique(fs):
        m = fs == f
        h[m] = lh[f][sub["row"].to_numpy()[m]]
    dup = np.zeros(len(rec), dtype=bool)
    dup[o] = pd.Series(h).duplicated(keep="first").to_numpy()
    near = np.zeros(len(rec), dtype=bool)
    left = o & ~dup
    k = rec.loc[left, ["src", "sport", "dst", "dport", "proto", "t0", "t1", "bytes"]]
    near[left] = k.duplicated(keep="first").to_numpy()
    return dup, near


def build(bench, days=None, tag="main", benign_fraction=None, attack_cap=None):
    t_start = time.time()
    cfg = I.BENCHMARKS[bench]
    days = days or cfg["days"]
    s = load_settings() if SETTINGS.is_file() else {}
    bf = benign_fraction if benign_fraction is not None else s["benign_fraction"]
    cap = attack_cap if attack_cap is not None else s["attack_cap"]
    convs, recs, day_tables = [], [], []
    for di, day in enumerate(cfg["days"]):
        if day not in days:
            continue
        d = FROZEN / bench / day
        t = pd.read_parquet(d / "conversations.parquet",
                            columns=["conv", "key_str", "cs", "sched", "in_all"])
        r = pd.read_parquet(d / "records.parquet", columns=REC_COLS)
        r["arm"] = r["arm"].astype(str)
        r["file"] = r["file"].astype(str)
        if bench == "UNSW-NB15":
            r["dup"], r["near"] = unsw_flags(r)
        scored = t["sched"].to_numpy() != I.EXCLUDED
        t = t[scored].reset_index(drop=True)
        day_tables.append((day, t))
        t["gconv"] = di * 10 ** 8 + t["conv"].to_numpy()
        t["day"] = day
        r = r[r["sched"] != I.EXCLUDED]
        r["gconv"] = di * 10 ** 8 + r["conv"].to_numpy()
        r["day"] = day
        convs.append(t)
        recs.append(r)
        print(f"{day}: {len(t):,} scored conversations, {len(r):,} records", flush=True)
    table = pd.concat(convs, ignore_index=True)
    keep = subsample(table, bench, day_tables, bf, cap)
    table["keep"] = keep
    kept = set(table.loc[keep, "gconv"].tolist())
    rec = pd.concat(recs, ignore_index=True)
    rec = rec[rec["gconv"].isin(kept)].reset_index(drop=True)
    in_all = table.set_index("gconv")["in_all"]
    rec["in_all"] = in_all.reindex(rec["gconv"]).to_numpy()
    print(f"kept {keep.sum():,} of {len(table):,} conversations, {len(rec):,} records",
          flush=True)

    out_dir = CACHE / tag / bench
    out_dir.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ftab, outputs = [], []
    for arm in cfg["arms"]:
        a = rec[rec["arm"] == arm].reset_index(drop=True)
        parts, dropped = [], None
        for f, g in a.groupby("file", sort=True):
            path = next(p for p in R.files(arm) if p.name == f)
            x, dropped = F.read_features(arm, path, g["row"].to_numpy(),
                                         g["src"].to_numpy(dtype=object),
                                         g["dst"].to_numpy(dtype=object))
            x.index = g.index
            parts.append(x)
        x = pd.concat(parts).sort_index()
        x, levels = F.code_strings(x, arm)
        meta_cols = ["gconv", "day", "sched", "in_all", "bytes", "file", "row"] + \
            [f"g{s_}" for s_ in SEEDS] + [f"r{s_}" for s_ in SEEDS]
        if bench == "UNSW-NB15":
            meta_cols += ["dup", "near"]
        m = a[meta_cols].copy()
        fx = pd.DataFrame(F.to_matrix(x), columns=["f:" + c for c in x.columns])
        out = pd.concat([m, fx], axis=1)
        p = out_dir / f"{arm}.parquet"
        out.to_parquet(p, index=False)
        outputs.append(p)
        ftab.append({"benchmark": bench, "arm": arm, "n_features": x.shape[1],
                     "features": ";".join(x.columns),
                     "dropped": ";".join(f"{c} ({F.why_dropped(c)})" for c in dropped),
                     "coded_strings": ";".join(levels), "records": len(out)})
        print(f"{arm}: {len(out):,} records, {x.shape[1]} features "
              f"({time.time() - t_start:.0f} s)", flush=True)
    ft = pd.DataFrame(ftab)
    counts = table.groupby("sched").agg(conversations=("keep", "size"), kept=("keep", "sum"))
    counts = counts.reset_index()
    counts.insert(0, "benchmark", bench)
    if tag == "main":
        fp = OUT / "feature_table.csv"
        old = pd.read_csv(fp) if fp.is_file() else pd.DataFrame()
        if len(old):
            old = old[old["benchmark"] != bench]
        pd.concat([old, ft], ignore_index=True).to_csv(fp, index=False)
        cp = OUT / "subsample_counts.csv"
        old = pd.read_csv(cp) if cp.is_file() else pd.DataFrame()
        if len(old):
            old = old[old["benchmark"] != bench]
        pd.concat([old, counts], ignore_index=True).to_csv(cp, index=False)
        write_manifest(f"phase5_data_{bench}", inputs=[SETTINGS], outputs=[fp, cp] + outputs,
                       params={"benchmark": bench, "days": days, "benign_fraction": bf,
                               "attack_cap": cap, "subsample_seed": SUBSAMPLE_SEED,
                               "seconds": round(time.time() - t_start)})
    return ft, counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("--days", nargs="*")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--benign-fraction", type=float)
    ap.add_argument("--attack-cap", type=int)
    a = ap.parse_args()
    ft, counts = build(a.bench, a.days, a.tag, a.benign_fraction, a.attack_cap)
    print(ft[["arm", "n_features", "records"]].to_string())
    print(counts.to_string())
