"""
Registry of the flow releases used in the journal study, with one loader per
release that returns the five tuple, a UTC start time and the release's own
label.

Every later script reads releases through load_keys() so that column names,
timestamp formats and clock offsets are decided in one place. The clock
offsets below were checked in step 2.11: 03_Code/clock_check.py lines up
known attack windows across all arms and writes 04_Results/release_clock.csv.

Returned frame columns:
    src, sport, dst, dport, proto   five tuple as the release reports it
    t_utc                           flow start, UTC epoch seconds (float)
    t_raw                           the timestamp exactly as shipped (str)
    label                           the release's own label (str)
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("NIDS_DATA", "/home/sudipta/Research/AI-Driven Cyber Threat "
                           "Detection and Log Analysis/Datasets"))
# Where the phase caches (_cache_phase3, 5, 7, 8) live. Defaults to the Datasets folder;
# the freeze run (step 9.4, 03_Code/run_all.sh) points it at an empty folder so that
# nothing is read from an earlier run.
CACHE_ROOT = Path(os.environ.get("NIDS_CACHE_ROOT", DATA))

HOUR = 3600.0
ADT = -3 * HOUR    # Atlantic Daylight Time, where CICIDS2017 was captured
CEST = 2 * HOUR    # Central European Summer Time
EDT = -4 * HOUR    # BCCC timestamps run one hour behind ADT (found in step 2.11)
AST = -4 * HOUR    # Atlantic Standard Time: the CIC-UNSW-NB15 clock (UTC-4 in January and
                   # February; checked against the UNSW-NB15 Stime of the same flows)

# protocol names used by BCCC, mapped to IANA numbers
PROTO_NUM = {"TCP": 6, "UDP": 17, "ICMP": 1}

RELEASES = {
    "cic_original": dict(
        benchmark="CICIDS2017", exporter="CICFlowMeter-V3 (CIC 2017 release)",
        folder="CICIDS2017_Original_LabelledFlows", pattern="*.csv",
        tz="local ADT (UTC-3), 12 hour clock without AM or PM",
        utc_offset=ADT, note="GeneratedLabelledFlows CSVs; the ML CSVs carry no five tuple"),
    "cic_original_ml": dict(
        benchmark="CICIDS2017", exporter="CICFlowMeter-V3 (CIC 2017 release)",
        folder="CICIDS2017_Original_ML_CSVs", pattern="*.csv",
        tz="none (no timestamp column)", utc_offset=None,
        note="MachineLearningCVE CSVs; features only, no five tuple, no timestamp"),
    "wtmc2021": dict(
        benchmark="CICIDS2017", exporter="CICFlowMeter, corrected (Engelen et al. 2021)",
        folder="CICIDS2017_Corrected_WTMC2021", pattern="*.csv",
        tz="CEST (UTC+2), 12 hour clock with AM or PM", utc_offset=CEST,
        note="includes '- Attempted' labels"),
    "crisis2022_rpd": dict(
        benchmark="CICIDS2017", exporter="CICFlowMeter, corrected (Lanvin et al. 2022)",
        folder="CRiSIS2022_RPD", pattern="*.csv",
        tz="local ADT (UTC-3), 24 hour clock", utc_offset=ADT,
        note="Labels/Exp_RPD of gitlab.inria.fr/mlanvin/crisis2022: reordered, "
             "duplicated packets kept (D; Exp_R and Exp_RP are the versions without "
             "duplication), port scan relabelled"),
    "lycos": dict(
        benchmark="CICIDS2017", exporter="LycoSTand (Rosay et al. 2021)",
        folder="LYCOS-IDS2017_GitHubMirror/lycos-ids2017", pattern="*.csv",
        tz="UTC epoch microseconds", utc_offset=0.0,
        note="labelled on 26 Oct 2026 with the authors' labelling.py (attack schedule)"),
    "nfs2023_nte": dict(
        benchmark="CICIDS2017", exporter="NFStream, no TCP flag expiry (Pekar and Jozsa 2024)",
        folder="NFS2023_nTE", pattern="*.csv",
        tz="UTC epoch milliseconds", utc_offset=0.0,
        note="FlowFrontiers/CyberML-DataQuality commit 473574a"),
    "nfs2023_te": dict(
        benchmark="CICIDS2017", exporter="NFStream, TCP flag expiry (Pekar and Jozsa 2024)",
        folder="NFS2023_TE", pattern="*.csv",
        tz="UTC epoch milliseconds", utc_offset=0.0,
        note="FlowFrontiers/CyberML-DataQuality commit 473574a"),
    "bccc2017": dict(
        benchmark="CICIDS2017", exporter="NTLFlowLyzer (Shafi et al.)",
        folder="BCCC_CIC_IDS2017", pattern="*.csv",
        tz="UTC-4, microseconds (one hour behind the capture's ADT)", utc_offset=EDT,
        note="Kaggle bcccdatasets/intrusion-detection-datasets-bccc-cic-ids-2017 v1; "
             "one file per class, not per day"),
    "unsw_original": dict(
        benchmark="UNSW-NB15", exporter="Argus and Bro-IDS (Moustafa and Slay 2015)",
        folder="UNSW_NB15_Original", pattern="UNSW-NB15_[1-4].csv",
        tz="UTC epoch seconds", utc_offset=0.0,
        note="the four full CSVs, no header; names from NUSW-NB15_features.csv; "
             "the repeated first line of files 2 to 4 is dropped"),
    "nf_unsw_v3": dict(
        benchmark="UNSW-NB15", exporter="nProbe NetFlow v3 (Luay et al. 2025)",
        folder="NF_UNSW_NB15_v3", pattern="*/data/NF-UNSW-NB15-v3.csv",
        tz="UTC epoch milliseconds", utc_offset=0.0,
        note="UQ RDM dataset abd2f5d8-e268-4ff0-84fb-f2f7b3ca3e8f, bag SHA1 verified"),
    "cic_unsw": dict(
        benchmark="UNSW-NB15", exporter="CICFlowMeter (Mohammadian et al. 2024)",
        folder="CIC_UNSW_NB15", pattern="CICFlowMeter_out.csv",
        tz="AST (UTC-4), 12 hour clock with AM or PM", utc_offset=AST,
        note="CIC-UNSW-NB15 augmented dataset (CIC/UNB 2024), cicresearch.ca/CICDataset/"
             "CIC-UNSW; CICFlowMeter_out.csv, every labelled flow (Data.csv is the 80:20 "
             "benign subsample without five tuple or timestamp); flows whose label the "
             "authors could not decide were dropped by them"),
}


def files(release):
    r = RELEASES[release]
    return sorted((DATA / r["folder"]).glob(r["pattern"]))


def _read(path, cols, **kw):
    """Read only the wanted columns; header names are matched after strip()."""
    want = set(cols)
    df = pd.read_csv(path, usecols=lambda c: c.strip() in want, dtype=str,
                     encoding="latin-1", low_memory=False, **kw)
    df.columns = [c.strip() for c in df.columns]
    return df


def _cic_original_time(s):
    """d/m/Y H:M[:S], hours 1 to 7 are afternoon (no AM or PM in the file)."""
    s = s.str.strip()
    t = pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    miss = t.isna()
    t[miss] = pd.to_datetime(s[miss], format="%d/%m/%Y %H:%M", errors="coerce")
    pm = t.dt.hour < 8
    t[pm] = t[pm] + pd.Timedelta(hours=12)
    return t


def _naive_to_utc(t, offset):
    sec = (t - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
    return sec - offset


def unsw_names():
    f = DATA / "UNSW_NB15_Original" / "NUSW-NB15_features.csv"
    feat = pd.read_csv(f, encoding="latin-1")
    feat.columns = [c.strip() for c in feat.columns]
    return [n.strip() for n in feat["Name"]]


def unsw_repeats_previous_last_line(path):
    """UNSW-NB15_2 to _4 each start with the last line of the file before.

    Found in step 2.10: the four CSVs hold 2,540,047 lines against the
    2,540,044 records the dataset page states, and the three extra lines are
    exact repeats across the file boundaries. The loader drops the repeat.
    """
    path = Path(path)
    stem = path.stem  # UNSW-NB15_k
    k = int(stem.rsplit("_", 1)[1])
    if k == 1:
        return False
    prev = path.with_name(f"UNSW-NB15_{k - 1}.csv")
    with open(prev, "rb") as f:
        f.seek(-4096, 2)
        last = f.read().rstrip(b"\r\n").rsplit(b"\n", 1)[-1].rstrip(b"\r")
    with open(path, "rb") as f:
        first = f.readline().rstrip(b"\r\n").lstrip(b"\xef\xbb\xbf")
    return first == last.lstrip(b"\xef\xbb\xbf")


# Extra columns for step 3.1: flow duration and bytes, with the unit each
# release uses. Units were checked on 26 October 2026 (Phase 3) from the
# column ranges: CICFlowMeter and LycoSTand report microseconds, NFStream and
# nProbe milliseconds, NTLFlowLyzer and Argus seconds.
EXTRA = {
    "cic_original": (["Flow Duration"], 1e-6,
                     ["Total Length of Fwd Packets", "Total Length of Bwd Packets"]),
    "wtmc2021": (["Flow Duration"], 1e-6,
                 ["Total Length of Fwd Packet", "Total Length of Bwd Packet"]),
    "crisis2022_rpd": (["Flow Duration"], 1e-6,
                       ["Total Length of Fwd Packet", "Total Length of Bwd Packet"]),
    "lycos": (["flow_duration"], 1e-6, ["fwd_pkt_len_tot", "bwd_pkt_len_tot"]),
    "nfs2023_nte": (["bidirectional_duration_ms"], 1e-3, ["bidirectional_bytes"]),
    "nfs2023_te": (["bidirectional_duration_ms"], 1e-3, ["bidirectional_bytes"]),
    "bccc2017": (["duration"], 1.0, ["total_payload_bytes"]),
    "unsw_original": (["dur"], 1.0, ["sbytes", "dbytes"]),
    "nf_unsw_v3": (["FLOW_DURATION_MILLISECONDS"], 1e-3, ["IN_BYTES", "OUT_BYTES"]),
    "cic_unsw": (["Flow Duration"], 1e-6,
                 ["Total Length of Fwd Packet", "Total Length of Bwd Packet"]),
}

# Timestamp resolution in seconds, from the inventory (step 2.9). Used in
# step 3.1 to widen each record's interval by what the clock cannot resolve.
RESOLUTION = {"cic_original": 60.0, "wtmc2021": 1.0, "crisis2022_rpd": 1.0,
              "lycos": 1e-6, "nfs2023_nte": 1e-3, "nfs2023_te": 1e-3,
              "bccc2017": 1e-6, "unsw_original": 1.0, "nf_unsw_v3": 1e-3, "cic_unsw": 1.0}

# How each clock lost its precision. "floor": the true start lies in
# [t, t + r). "round": it lies in [t - r/2, t + r/2). UNSW-NB15 Stime is
# rounded to the nearest second: on 2015-01-22 the NF-v3 start minus the Stime
# of the same flow is spread evenly over -0.5 to +0.5 seconds (step 3.10,
# 04_Results/phase3/unsw_clock_offset.csv). Every other release truncates.
ROUNDING = {"unsw_original": "round"}


def _add_extra(out, df, release):
    dcols, unit, bcols = EXTRA[release]
    dur = pd.to_numeric(df[dcols[0]], errors="coerce").to_numpy() * unit
    out["dur_s"] = dur
    b = np.zeros(len(df))
    for c in bcols:
        b = b + pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy()
    out["bytes"] = b
    return out


def load_keys(release, path, extra=False):
    """Five tuple, UTC start and label for one file of one release.

    With extra=True the frame also carries dur_s (flow duration, seconds)
    and bytes (both directions), read in the same pass.
    """
    off = RELEASES[release]["utc_offset"]
    more = []
    if extra and release in EXTRA:
        more = EXTRA[release][0] + EXTRA[release][2]
    if release == "cic_original":
        df = _read(path, ["Source IP", "Source Port", "Destination IP",
                          "Destination Port", "Protocol", "Timestamp", "Label"] + more)
        df = df.dropna(how="all")
        out = pd.DataFrame({"src": df["Source IP"], "sport": df["Source Port"],
                            "dst": df["Destination IP"], "dport": df["Destination Port"],
                            "proto": df["Protocol"], "t_raw": df["Timestamp"],
                            "label": df["Label"]})
        out["t_utc"] = _naive_to_utc(_cic_original_time(df["Timestamp"]), off)
    elif release == "cic_original_ml":
        df = _read(path, ["Destination Port", "Label"])
        out = pd.DataFrame({"src": None, "sport": None, "dst": None,
                            "dport": df["Destination Port"], "proto": None,
                            "t_raw": None, "label": df["Label"]})
        out["t_utc"] = np.nan
    elif release in ("wtmc2021", "crisis2022_rpd", "cic_unsw"):
        df = _read(path, ["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol",
                          "Timestamp", "Label"] + more)
        fmt = "%d/%m/%Y %H:%M:%S" if release == "crisis2022_rpd" else "%d/%m/%Y %I:%M:%S %p"
        out = pd.DataFrame({"src": df["Src IP"], "sport": df["Src Port"],
                            "dst": df["Dst IP"], "dport": df["Dst Port"],
                            "proto": df["Protocol"], "t_raw": df["Timestamp"],
                            "label": df["Label"]})
        t = pd.to_datetime(df["Timestamp"].str.strip(), format=fmt, errors="coerce")
        out["t_utc"] = _naive_to_utc(t, off)
    elif release == "lycos":
        df = _read(path, ["src_addr", "src_port", "dst_addr", "dst_port", "ip_prot",
                          "timestamp", "label"] + more)
        out = pd.DataFrame({"src": df["src_addr"], "sport": df["src_port"],
                            "dst": df["dst_addr"], "dport": df["dst_port"],
                            "proto": df["ip_prot"], "t_raw": df["timestamp"],
                            "label": df["label"]})
        out["t_utc"] = pd.to_numeric(df["timestamp"]) / 1e6 - off
    elif release in ("nfs2023_nte", "nfs2023_te"):
        df = _read(path, ["src_ip", "src_port", "dst_ip", "dst_port", "protocol",
                          "bidirectional_first_seen_ms", "label"] + more)
        out = pd.DataFrame({"src": df["src_ip"], "sport": df["src_port"],
                            "dst": df["dst_ip"], "dport": df["dst_port"],
                            "proto": df["protocol"],
                            "t_raw": df["bidirectional_first_seen_ms"],
                            "label": df["label"]})
        out["t_utc"] = pd.to_numeric(df["bidirectional_first_seen_ms"]) / 1e3 - off
    elif release == "bccc2017":
        df = _read(path, ["src_ip", "src_port", "dst_ip", "dst_port", "protocol",
                          "timestamp", "label"] + more)
        out = pd.DataFrame({"src": df["src_ip"], "sport": df["src_port"],
                            "dst": df["dst_ip"], "dport": df["dst_port"],
                            "proto": df["protocol"].map(lambda p: PROTO_NUM.get(p, p)),
                            "t_raw": df["timestamp"], "label": df["label"]})
        t = pd.to_datetime(df["timestamp"].str.strip(), format="ISO8601", errors="coerce")
        out["t_utc"] = _naive_to_utc(t, off)
    elif release == "unsw_original":
        names = unsw_names()
        df = pd.read_csv(path, header=None, names=names, dtype=str,
                         encoding="latin-1", low_memory=False)
        if unsw_repeats_previous_last_line(path):
            df = df.iloc[1:]
        lab = df["attack_cat"].fillna("").str.strip()
        lab = lab.where(lab != "", "Normal")
        out = pd.DataFrame({"src": df["srcip"], "sport": df["sport"],
                            "dst": df["dstip"], "dport": df["dsport"],
                            "proto": df["proto"], "t_raw": df["Stime"],
                            "label": lab})
        out["t_utc"] = pd.to_numeric(df["Stime"], errors="coerce") - off
    elif release == "nf_unsw_v3":
        df = _read(path, ["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR",
                          "L4_DST_PORT", "PROTOCOL", "FLOW_START_MILLISECONDS",
                          "Attack"] + more)
        out = pd.DataFrame({"src": df["IPV4_SRC_ADDR"], "sport": df["L4_SRC_PORT"],
                            "dst": df["IPV4_DST_ADDR"], "dport": df["L4_DST_PORT"],
                            "proto": df["PROTOCOL"],
                            "t_raw": df["FLOW_START_MILLISECONDS"], "label": df["Attack"]})
        out["t_utc"] = pd.to_numeric(df["FLOW_START_MILLISECONDS"]) / 1e3 - off
    else:
        raise KeyError(release)
    if more:
        out = _add_extra(out, df, release)
    out["label"] = out["label"].astype(str).str.strip()
    return out.reset_index(drop=True)
