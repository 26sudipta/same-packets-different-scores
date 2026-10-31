"""
Step 5.2: the features of each arm, row aligned with the Phase 3 instrument.

The instrument's records carry (file, row), where row is the position of the
record in releases.load_keys(arm, file). read_features() reads the same file
with every column, applies the same row filters as load_keys (blank lines in
the CIC original, the repeated first line of UNSW-NB15 files 2 to 4), and
returns the feature columns of the requested rows.

Each arm keeps its own features. What is dropped is listed in DROP, per arm:
identifiers (addresses, ports, MAC and OUI, flow IDs, VLAN IDs), timestamps,
exporter bookkeeping, random protocol identifiers and labels. Protocol stays.
String columns that remain are coded by sorted level (code_strings).
"""
import numpy as np
import pandas as pd

import releases as R

DROP = {
    "cic_original": ["Flow ID", "Source IP", "Source Port", "Destination IP",
                     "Destination Port", "Timestamp", "Label"],
    "wtmc2021": ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp", "Label"],
    "crisis2022_rpd": ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp",
                       "Label"],
    "lycos": ["flow_id", "src_addr", "src_port", "dst_addr", "dst_port", "timestamp", "label"],
    "nfs2023_nte": ["id", "expiration_id", "src_ip", "src_mac", "src_oui", "src_port",
                    "dst_ip", "dst_mac", "dst_oui", "dst_port", "ip_version", "vlan_id",
                    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
                    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
                    "dst2src_first_seen_ms", "dst2src_last_seen_ms", "label"],
    "bccc2017": ["flow_id", "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "label"],
    "unsw_original": ["srcip", "sport", "dstip", "dsport", "Stime", "Ltime", "stcpb", "dtcpb",
                      "attack_cat", "Label"],
    "nf_unsw_v3": ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR",
                   "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "DNS_QUERY_ID", "Label",
                   "Attack"],
    "cic_unsw": ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp", "Label"],
}
DROP["nfs2023_te"] = DROP["nfs2023_nte"]

WHY = {
    "address": "identifier", "port": "identifier", "mac": "identifier", "oui": "identifier",
    "flow id": "identifier", "flow_id": "identifier", "vlan": "identifier",
    "time": "timestamp", "seen_ms": "timestamp", "milliseconds": "timestamp",
    "label": "label", "attack": "label", "expiration": "exporter bookkeeping",
    "ip_version": "exporter bookkeeping", "tcpb": "random protocol identifier",
    "dns_query_id": "random protocol identifier",
}

# Columns that are categorical strings in some release and are kept as codes.
STRING_FEATURES = {"bccc2017": ["protocol"], "unsw_original": ["proto", "state", "service"]}

# The column subset load_keys reads for cic_original (with extra=True); its
# dropna(how="all") over that subset decides which lines count as rows.
_CIC_KEYCOLS = (["Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol",
                 "Timestamp", "Label"] + R.EXTRA["cic_original"][0] + R.EXTRA["cic_original"][2])

# Address columns used to check that the rows line up with the instrument.
ADDR = {"cic_original": ("Source IP", "Destination IP"),
        "wtmc2021": ("Src IP", "Dst IP"), "crisis2022_rpd": ("Src IP", "Dst IP"),
        "lycos": ("src_addr", "dst_addr"), "nfs2023_nte": ("src_ip", "dst_ip"),
        "nfs2023_te": ("src_ip", "dst_ip"), "bccc2017": ("src_ip", "dst_ip"),
        "unsw_original": ("srcip", "dstip"), "nf_unsw_v3": ("IPV4_SRC_ADDR", "IPV4_DST_ADDR"),
        "cic_unsw": ("Src IP", "Dst IP")}


def why_dropped(col):
    c = col.lower()
    for k, v in WHY.items():
        if k in c:
            return v
    return "identifier"


def read_full(arm, path):
    """Every column of one release file, rows numbered as in load_keys."""
    if arm == "unsw_original":
        df = pd.read_csv(path, header=None, names=R.unsw_names(), encoding="latin-1",
                         low_memory=False)
        if R.unsw_repeats_previous_last_line(path):
            df = df.iloc[1:]
    else:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        if arm == "cic_original":
            df = df.dropna(how="all", subset=_CIC_KEYCOLS)
    return df.reset_index(drop=True)


def read_features(arm, path, rows, src=None, dst=None):
    """Feature frame for the given rows of one file (in the given order).

    With src and dst (the instrument's record addresses for those rows) every
    row is checked against the file; a mismatch raises.
    """
    df = read_full(arm, path)
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) and rows.max() >= len(df):
        raise ValueError(f"{arm} {path.name}: row {rows.max()} beyond {len(df)} rows")
    sub = df.iloc[rows].reset_index(drop=True)
    if src is not None:
        a, b = ADDR[arm]
        ok = ((sub[a].astype(str).str.strip().to_numpy() == np.asarray(src, dtype=object)) &
              (sub[b].astype(str).str.strip().to_numpy() == np.asarray(dst, dtype=object)))
        if not ok.all():
            raise ValueError(f"{arm} {path.name}: {int((~ok).sum())} rows do not line up")
    keep = [c for c in sub.columns if c not in DROP[arm]]
    out = sub[keep].copy()
    strings = STRING_FEATURES.get(arm, [])
    for c in keep:
        if c in strings:
            out[c] = out[c].astype(str).str.strip()
        elif not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out, [c for c in df.columns if c in DROP[arm]]


def code_strings(df, arm):
    """Code the kept string columns by sorted level (the same codes for every
    split and seed of an arm). Returns the frame and the levels per column."""
    levels = {}
    for c in STRING_FEATURES.get(arm, []):
        if c in df.columns:
            lv = sorted(pd.unique(df[c].astype(str)))
            levels[c] = lv
            df[c] = pd.Categorical(df[c].astype(str), categories=lv).codes.astype(np.float32)
    return df, levels


def to_matrix(df):
    """float32 matrix: infinite values become missing, missing becomes -1."""
    x = df.to_numpy(dtype=np.float64, na_value=np.nan)
    x[~np.isfinite(x)] = np.nan
    x = np.nan_to_num(x, nan=-1.0)
    return np.clip(x, -3.0e38, 3.0e38).astype(np.float32)
