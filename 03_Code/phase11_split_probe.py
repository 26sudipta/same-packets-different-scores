"""
Phase 11, review minor comment 8: the two other split behaviours Phase 4 left
unexplained. For the conversations the schedule labels with a class, takes every
record after the first in its conversation (frozen instrument, conversations in every
arm only is not required) and reads from the release CSV what ended the record
before it and what the record itself holds.

  BCCC-CIC-IDS2017 (NTLFlowLyzer): DoS Hulk against DDoS and GoldenEye
  BCCC-CIC-IDS2017 and CIC original: FTP-Patator (three records per conversation)

NTLFlowLyzer (network_flow_capturer.py, flow_is_ended) closes a flow when the next
packet of its five-tuple arrives and the flow already holds a RST packet, or a FIN in
each direction, or has been idle longer than activity_timeout, or is older than
max_flow_duration; that next packet opens a new flow.

Usage: .venv/bin/python 03_Code/phase11_split_probe.py
Output: 04_Results/phase11/split_probe.csv (one row per class, arm and case)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase11_common as C  # noqa: E402
import releases as R  # noqa: E402

CASES = [("bccc2017", "Wednesday", ["dos_hulk", "dos_goldeneye"]),
         ("bccc2017", "Friday", ["ddos"]),
         ("bccc2017", "Tuesday", ["ftp_patator"]),
         ("cic_original", "Tuesday", ["ftp_patator"])]
FIELDS = {"bccc2017": dict(rst="rst_flag_counts", ffin="fwd_fin_flag_counts",
                           bfin="bwd_fin_flag_counts", pk="packets_count",
                           pay="total_payload_bytes", syn="syn_flag_counts", sport="src_port"),
          "cic_original": dict(rst="RST Flag Count", fin="FIN Flag Count",
                               pk=["Total Fwd Packets", "Total Backward Packets"],
                               pay=["Total Length of Fwd Packets", "Total Length of Bwd Packets"],
                               syn="SYN Flag Count", sport="Source Port")}


def read_rows(arm, file, rows):
    path = R.DATA / R.RELEASES[arm]["folder"] / file
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df


def col(df, f):
    if isinstance(f, list):
        return sum(pd.to_numeric(df[c], errors="coerce") for c in f)
    return pd.to_numeric(df[f], errors="coerce")


def main():
    out = []
    for arm, day, classes in CASES:
        rec = pd.read_parquet(C.ROOT / "04_Results" / "instrument" / "CICIDS2017" / day /
                              "records.parquet",
                              columns=["arm", "file", "row", "conv", "t0", "t1", "sched", "sport"])
        g = rec[(rec["arm"] == arm) & rec["sched"].isin(classes)].sort_values(
            ["conv", "t0", "t1"], kind="mergesort").reset_index(drop=True)
        F = FIELDS[arm]
        feats = []
        for file, gg in g.groupby("file"):
            df = read_rows(arm, file, gg["row"])
            sub = df.iloc[gg["row"].to_numpy()]
            if not (pd.to_numeric(sub[F["sport"]]).to_numpy() == gg["sport"].to_numpy()).all():
                raise AssertionError(f"{arm} {file}: rows do not match the instrument")
            e = pd.DataFrame(index=gg.index)
            e["rst"] = col(sub, F["rst"]).to_numpy()
            if "ffin" in F:
                e["fin_both"] = ((col(sub, F["ffin"]) > 0) & (col(sub, F["bfin"]) > 0)).to_numpy()
                e["fin_any"] = ((col(sub, F["ffin"]) + col(sub, F["bfin"])) > 0).to_numpy()
            else:
                e["fin_both"] = np.nan
                e["fin_any"] = (col(sub, F["fin"]) > 0).to_numpy()
            e["pk"] = col(sub, F["pk"]).to_numpy()
            e["pay"] = col(sub, F["pay"]).to_numpy()
            e["syn"] = col(sub, F["syn"]).to_numpy()
            feats.append(e)
        g = g.join(pd.concat(feats))
        n_per = g.groupby("conv").size()
        g["k"] = g.groupby("conv").cumcount()
        prev = g.shift()
        succ = g[g["k"] > 0]
        pv = prev.loc[succ.index]
        for cls in classes:
            m = (succ["sched"] == cls).to_numpy()
            s, p = succ[m], pv[m]
            convs = g.loc[g["sched"] == cls, "conv"].unique()
            npc = n_per.reindex(convs)
            out.append({"arm": arm, "cls": cls, "conversations": len(convs),
                        "records_per_conv_mean": npc.mean(),
                        "share_1": (npc == 1).mean(), "share_2": (npc == 2).mean(),
                        "share_3": (npc == 3).mean(), "share_4plus": (npc >= 4).mean(),
                        "extra_records": len(s),
                        "prev_has_rst": int((p["rst"] > 0).sum()),
                        "prev_fin_both": int((p["fin_both"] == True).sum()),  # noqa: E712
                        "prev_fin_any": int((p["fin_any"] == True).sum()),  # noqa: E712
                        "cur_no_payload": int((s["pay"] == 0).sum()),
                        "cur_packets_median": s["pk"].median(),
                        "cur_has_syn": int((s["syn"] > 0).sum()),
                        "cur_has_rst": int((s["rst"] > 0).sum()),
                        "gap_s_median": float(np.median(s["t0"].to_numpy() - p["t1"].to_numpy()))})
    df = pd.DataFrame(out)
    C.OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.OUT / "split_probe.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.T.to_string())


if __name__ == "__main__":
    main()
