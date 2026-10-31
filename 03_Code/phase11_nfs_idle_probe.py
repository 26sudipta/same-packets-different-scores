"""
Phase 11, review minor comment 8: what are the NFS-2023 idle splits of slowloris and
slowhttptest? Takes the frozen instrument records of both NFS-2023 releases on
Wednesday, keeps the conversations the schedule labels slowloris or slowhttptest,
finds the records that Phase 4 calls idle splits (agreement.split_counts: not a
repeat, tail, cut or overlap, and starting more than 1 s plus the clock resolution
after the previous record of the conversation ended), and reads what NFStream wrote
about each such record and the one before it from the release CSV.

NFStream 6.5.4 (the version Pekar and Jozsa name) ends a flow in three ways only
(meter.py, lib_engine.c flow_expiration_handler): idle (no packet for idle_timeout,
60 s here, on a clock forced to be monotonic), active (packet time minus first-seen
time reaches active_timeout, 120 s here) and a plugin's custom expiry (TE releases:
a FIN or RST packet).

Usage: .venv/bin/python 03_Code/phase11_nfs_idle_probe.py
Output: 04_Results/phase11/nfs_idle_probe.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agreement as AG  # noqa: E402
import phase11_common as C  # noqa: E402
import releases as R  # noqa: E402

DAY = "Wednesday"
CLASSES = ["dos_slowloris", "dos_slowhttptest"]
COLS = ["src_ip", "src_port", "dst_ip", "dst_port", "bidirectional_first_seen_ms",
        "bidirectional_last_seen_ms", "bidirectional_duration_ms", "bidirectional_packets",
        "src2dst_packets", "dst2src_packets", "bidirectional_syn_packets",
        "bidirectional_fin_packets", "bidirectional_rst_packets", "bidirectional_max_ps",
        "bidirectional_psh_packets", "label"]


def main():
    rec = pd.read_parquet(C.ROOT / "04_Results" / "instrument" / "CICIDS2017" / DAY /
                          "records.parquet",
                          columns=["arm", "file", "row", "conv", "rev", "t0", "t1", "sched"])
    rows = []
    for arm in ("nfs2023_nte", "nfs2023_te"):
        r = R.RESOLUTION[arm]
        g = rec[(rec["arm"] == arm) & rec["sched"].isin(CLASSES)].sort_values(
            ["conv", "t0", "t1"], kind="mergesort").reset_index(drop=True)
        conv, t0 = g["conv"].to_numpy(), g["t0"].to_numpy()
        dur = g["t1"].to_numpy() - t0 - r
        i = np.nonzero(np.r_[False, conv[1:] == conv[:-1]])[0]
        gap = t0[i] - (t0[i - 1] + dur[i - 1])
        since = t0[i] - t0[i - 1]
        idle = (gap > AG.GAP_SLACK + r) & (since < 120 - 1.0) & \
            ~((dur[i] <= AG.TAIL_DUR) & (gap <= AG.GAP_SLACK + r))
        files = g["file"].unique()
        assert len(files) == 1
        csv = pd.read_csv(R.DATA / R.RELEASES[arm]["folder"] / files[0],
                          usecols=lambda c: c in COLS)
        for j in i[idle]:
            a, b = csv.iloc[int(g.at[j - 1, "row"])], csv.iloc[int(g.at[j, "row"])]
            rows.append({"arm": arm, "cls": g.at[j, "sched"],
                         "gap_s": t0[j] - t0[j - 1] - dur[j - 1],
                         "since_prev_start_s": t0[j] - t0[j - 1],
                         "same_direction": (a["src_ip"], a["src_port"]) == (b["src_ip"], b["src_port"]),
                         **{"prev_" + c: a[c] for c in COLS}, **{"cur_" + c: b[c] for c in COLS}})
    df = pd.DataFrame(rows)
    C.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.OUT / "nfs_idle_probe.csv", index=False)
    pd.set_option("display.width", 250)
    for (a, c), d in df.groupby(["arm", "cls"]):
        print(f"\n== {a} {c}: {len(d)} idle splits")
        print("same direction as previous record:", int(d["same_direction"].sum()))
        for side in ("prev", "cur"):
            print(f"  {side}: SYN>0 {int((d[side + '_bidirectional_syn_packets'] > 0).sum())}"
                  f" FIN>0 {int((d[side + '_bidirectional_fin_packets'] > 0).sum())}"
                  f" RST>0 {int((d[side + '_bidirectional_rst_packets'] > 0).sum())}"
                  f" packets median {d[side + '_bidirectional_packets'].median()}"
                  f" src2dst median {d[side + '_src2dst_packets'].median()}"
                  f" dst2src median {d[side + '_dst2src_packets'].median()}"
                  f" duration s median {d[side + '_bidirectional_duration_ms'].median() / 1e3:.2f}")
        print("  gap s quantiles", d["gap_s"].quantile([0, .25, .5, .75, 1]).round(2).tolist())
        print("  since previous start s quantiles",
              d["since_prev_start_s"].quantile([0, .25, .5, .75, 1]).round(2).tolist())
        print(d[["gap_s", "prev_src_ip", "prev_src_port", "prev_bidirectional_packets",
                 "prev_bidirectional_fin_packets", "prev_bidirectional_rst_packets",
                 "cur_src_ip", "cur_src_port", "cur_bidirectional_packets",
                 "cur_bidirectional_syn_packets", "cur_bidirectional_fin_packets",
                 "cur_bidirectional_rst_packets", "cur_bidirectional_duration_ms"]].head(6).to_string())


if __name__ == "__main__":
    main()
