"""
Phase 11, review Major comment 1: the CICIDS2017 grid under a payload-aware
ground truth, to separate the exporter's part of the RQ3 and RQ4 results from
the part made by propagating the schedule label to payload-free records.

Rule. A record keeps its conversation's schedule label when it carries payload
and becomes benign when it carries none. PortScan is exempt, because a probe
carries no payload by design and the rule would remove the class. A conversation
is an attack of class k when at least one of its records still carries k
(phase11_common.truth_any). "Carries no payload", per release, from the
release's own fields:
    cic_original, wtmc2021, crisis2022_rpd   Total Length of Fwd + Bwd Packet(s) = 0
                                             (CICFlowMeter counts payload bytes)
    lycos                                    fwd + bwd non-empty packet count = 0
    bccc2017                                 total_payload_bytes = 0
    nfs2023_nte, nfs2023_te                  TCP, largest frame <= 74 bytes and no
                                             packet with the PSH flag
NFStream reports frame sizes, not payload. 74 bytes is an Ethernet, IPv4 and TCP
header with the largest option set of a SYN (60 a padded bare header, 66 a header
with the timestamp option). Frame size alone cannot tell a bare SYN from a data
packet of a few bytes, so a record also needs no PSH packet. Checked on the
subsample (30 October 2026): on every attack class the NFStream rule turns the
same number of conversations benign as the payload-byte rule of WTMC-2021 and
CRiSIS-2022 (dos_slowhttptest 2,808, dos_slowloris 1,662 or 1,661, web brute
force 1,210, web XSS 640, GoldenEye 80, Hulk 56, none for the other classes),
while a 66-byte bound misses the attempted connections, whose SYN is 74 bytes.

Usage, from the workspace root, NIDS_CACHE_ROOT set (phase11_common):
    .venv/bin/python 03_Code/phase11_payload.py check
    .venv/bin/python 03_Code/phase11_payload.py run [--arms ...] [--models ...] [--seeds ...]

Same arm tables, features, models, Phase 5 tuned settings and grouped split as
Phase 5; only the labels differ. Checkpoints:
04_Results/phase11/payload/{preds,cells}/CICIDS2017/<arm>/<model>_grouped_s<seed>.*
The cell score files hold per-class F1 at both units and both populations,
scored against the payload-aware labels.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase11_common as C  # noqa: E402
import phase5_grid as G  # noqa: E402
from manifest import write_manifest  # noqa: E402

BENCH = "CICIDS2017"
OUT = C.OUT / "payload"
EXEMPT = {"portscan"}
FRAME_MAX = 74


def _sum(t, *cols):
    return sum(t["f:" + c].to_numpy(dtype=float) for c in cols)


def payload_free(t, arm):
    if arm == "cic_original":
        return _sum(t, "Total Length of Fwd Packets", "Total Length of Bwd Packets") == 0
    if arm in ("wtmc2021", "crisis2022_rpd"):
        return _sum(t, "Total Length of Fwd Packet", "Total Length of Bwd Packet") == 0
    if arm == "lycos":
        return _sum(t, "fwd_non_empty_pkt_cnt", "bwd_non_empty_pkt_cnt") == 0
    if arm == "bccc2017":
        return _sum(t, "total_payload_bytes") == 0
    if arm in ("nfs2023_nte", "nfs2023_te"):
        return ((t["f:protocol"].to_numpy() == 6) &
                (t["f:bidirectional_max_ps"].to_numpy(dtype=float) <= FRAME_MAX) &
                (t["f:bidirectional_psh_packets"].to_numpy(dtype=float) == 0))
    raise KeyError(arm)


def payload_labels(t, arm):
    y = t["sched"].to_numpy(dtype=object).copy()
    pf = payload_free(t, arm)
    att = (y != C.BENIGN) & ~np.isin(y, list(EXEMPT))
    y[pf & att] = C.BENIGN
    return y, pf


def check():
    rows = []
    for arm in G.PRIMARY_ARMS[BENCH]:
        t, _ = G.load_arm(BENCH, arm)
        y, pf = payload_labels(t, arm)
        conv_att_before = t.loc[t["sched"] != C.BENIGN].groupby("sched")["gconv"].nunique()
        truth = C.truth_any(t["gconv"].to_numpy(), y)
        sched_conv = t.drop_duplicates("gconv").set_index("gconv")["sched"]
        for cls in sorted(t["sched"].unique()):
            m = (t["sched"] == cls).to_numpy()
            convs = sched_conv[sched_conv == cls].index
            rows.append({"benchmark": BENCH, "arm": arm, "cls": cls,
                         "records": int(m.sum()), "payload_free": int((pf & m).sum()),
                         "relabelled": int((m & (y != t["sched"].to_numpy(dtype=object))).sum()),
                         "conversations": len(convs),
                         "conversations_now_benign": int((truth.reindex(convs) == C.BENIGN).sum())
                         if cls != C.BENIGN else 0})
        print(arm, "done", flush=True)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "payload_check.csv", index=False)
    print(df.to_string())
    write_manifest("phase11_payload_check", inputs=[], outputs=[OUT / "payload_check.csv"],
                   params={"frame_max": FRAME_MAX, "exempt": sorted(EXEMPT)})


def run(arms, models, seeds):
    C.use_truth_any()
    load = G.load_arm

    def load_arm(bench, arm, tag="main"):
        t, fcols = load(bench, arm, tag)
        t["sched_schedule"] = t["sched"]
        t["sched"], t["payload_free"] = payload_labels(t, arm)
        return t, fcols

    G.load_arm = load_arm
    tm = G.run(BENCH, arms or G.PRIMARY_ARMS[BENCH], models, ["grouped"], seeds, "main", OUT)
    print(tm.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check", "run"])
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    a = ap.parse_args()
    if a.what == "check":
        check()
    else:
        run(a.arms, a.models, a.seeds)
