"""
Step 9.2: numbers the manuscript quotes that came from one-off checks during Phases 3 and 4
and were not written to a file. Each is recomputed here from the raw releases or the frozen
instrument, so that make_numbers.py can read it.

    .venv/bin/python 03_Code/phase9_oneoffs.py

Writes
    04_Results/phase3/unsw_protocol_names.csv   UNSW-NB15 GT rows whose Argus protocol name the
                                                system protocol table cannot resolve
                                                (Threats note 21)
    04_Results/phase3/unsw_trunc_window.csv     share of paired flows that a truncated reading
                                                of Stime would put outside its one-second window
                                                (Threats note 14)
    04_Results/phase4/unsw_near_repeats.csv     near repeats as the unsw_collapsed arm defines
                                                them, both days, and a field comparison on a
                                                sample of 18 February (Threats note 23)

Memory: the UNSW-NB15 original is read one file at a time; peak about 3 GB.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instrument as I  # noqa: E402
import releases as R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
P3 = ROOT / "04_Results" / "phase3"
P4 = ROOT / "04_Results" / "phase4"
FROZEN = ROOT / "04_Results" / "instrument" / "UNSW-NB15"


def protocol_names():
    f = R.DATA / "UNSW_NB15_Original" / "NUSW-NB15_GT.csv"
    g = pd.read_csv(f, usecols=["Protocol"], dtype=str, encoding="latin-1")
    num = g["Protocol"].map(I.proto_number)
    bad = g.loc[num == -1, "Protocol"].str.strip().str.lower()
    top = bad.value_counts()
    t = pd.DataFrame([dict(gt_rows=len(g), unresolved_rows=int((num == -1).sum()),
                           unresolved_names=int(top.size), most_common=top.index[0],
                           most_common_rows=int(top.iloc[0]))])
    t.to_csv(P3 / "unsw_protocol_names.csv", index=False)
    print(t.to_string())


def trunc_window():
    """Same pairing as 03_Code/unsw_clock_offset.py (conversations with one record in each arm).
    Rounded Stime puts the NF-v3 start in [Stime - 0.5, Stime + 0.5]; a truncated Stime would
    need it in [Stime, Stime + 1). The share outside the truncated window is the share with a
    negative offset or an offset of 1 s or more."""
    rows = []
    for day in I.UNSW_DAYS:
        d = FROZEN / day
        r = pd.read_parquet(d / "records.parquet", columns=["arm", "conv", "t0"])
        t = pd.read_parquet(d / "conversations.parquet", columns=["conv", "n_unsw_original", "n_nf_unsw_v3"])
        one = t.loc[(t["n_unsw_original"] == 1) & (t["n_nf_unsw_v3"] == 1), "conv"]
        x = r[r["conv"].isin(one)]
        o = x[x["arm"] == "unsw_original"].set_index("conv")["t0"] + R.RESOLUTION["unsw_original"] / 2.0
        nf = x[x["arm"] == "nf_unsw_v3"].set_index("conv")["t0"]
        dt = (nf - o.reindex(nf.index)).dropna()
        rows.append(dict(day=day, flows=len(dt),
                         share_in_rounded_window=float(((dt >= -0.5) & (dt <= 0.5)).mean()),
                         share_outside_truncated_window=float(((dt < 0) | (dt >= 1)).mean())))
    t = pd.DataFrame(rows)
    t.to_csv(P3 / "unsw_trunc_window.csv", index=False)
    print(t.to_string())


def near_repeats(sample=3000):
    """Near repeats of the UNSW-NB15 original as the unsw_collapsed sensitivity arm defines them
    (03_Code/phase5_data.py, unsw_flags): after exact duplicate lines are dropped, a record with
    the same five tuple, start, end and bytes as an earlier record of its day. Counted on both
    days. For the first `sample` near repeats of 18 February (file and row order), the raw lines
    of the record and of its earlier match are compared field by field.

    The Phase 4 one-off check that first reported 24,016 pairs kept no code; its definition
    ("same start and duration") could not be rebuilt exactly (nine variants give 23,721 to
    32,252 on 18 February), so the paper reports this definition, which is the one the
    sensitivity arm removes."""
    import phase5_data as D
    names = R.unsw_names()
    rows, raw = [], {}
    key = ["src", "sport", "dst", "dport", "proto", "t0", "t1", "bytes"]
    for day in I.UNSW_DAYS:
        r = pd.read_parquet(D.FROZEN / "UNSW-NB15" / day / "records.parquet", columns=D.REC_COLS)
        r["arm"] = r["arm"].astype(str)
        r["file"] = r["file"].astype(str)
        dup, near = D.unsw_flags(r)
        o = (r["arm"] == "unsw_original").to_numpy()
        rows.append((f"records_after_exact_dups_{day}", int((o & ~dup).sum())))
        rows.append((f"near_repeats_{day}", int(near.sum())))
        if day != I.UNSW_DAYS[-1]:
            continue
        left = r.loc[o & ~dup, ["file", "row"] + key].copy()
        left["near"] = near[o & ~dup]
        left["gid"] = left.groupby(key, sort=False, observed=True).ngroup()
        firsts = left[~left["near"]].set_index("gid")[["file", "row"]]
        s = left[left["near"]].sort_values(["file", "row"], kind="mergesort").head(sample)
        base = firsts.loc[s["gid"].to_numpy()].reset_index(drop=True)
        s = s[["file", "row"]].reset_index(drop=True)
        for f in sorted(set(s["file"]) | set(base["file"])):
            path = next(p for p in R.files("unsw_original") if p.name == f)
            df = pd.read_csv(path, header=None, names=names, dtype=str, encoding="latin-1",
                             low_memory=False)
            if R.unsw_repeats_previous_last_line(path):
                df = df.iloc[1:]
            raw[f] = df.reset_index(drop=True)
        def lines(t):
            return pd.concat([raw[f].iloc[g["row"].to_numpy()].assign(_i=g.index.to_numpy())
                              for f, g in t.groupby("file")]).sort_values("_i").drop(columns="_i")
        a = lines(s).reset_index(drop=True)
        b = lines(base).reset_index(drop=True)
        diff = pd.DataFrame({c: a[c].fillna("").str.strip() != b[c].fillna("").str.strip()
                             for c in names})
        assert not diff.drop(columns=[c for c in names if c not in
                                      ("srcip", "dstip", "Stime", "Ltime")]).any().any()
        ct = [c for c in names if c.startswith("ct_")]
        only_cat = (diff["attack_cat"] & ~diff.drop(columns=["attack_cat"]).any(axis=1)).sum()
        only_ct = (diff[ct].any(axis=1) & ~diff.drop(columns=ct).any(axis=1)).sum()
        rows += [("sample", len(s)), ("differ_only_attack_cat", int(only_cat)),
                 ("differ_only_ct_features", int(only_ct))]
    t = pd.DataFrame(rows, columns=["quantity", "value"])
    t.to_csv(P4 / "unsw_near_repeats.csv", index=False)
    print(t.to_string())


if __name__ == "__main__":
    protocol_names()
    trunc_window()
    near_repeats()
