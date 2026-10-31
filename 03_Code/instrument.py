"""
Phase 3 measurement instrument, steps 3.1 to 3.7 of 00_Plan/Implementation_Plan.md.

One code path turns the flow releases of one benchmark day into three tables:

    records        every flow record of every arm, with the conversation it
                   belongs to, its own label, and its split assignments
    conversations  one row per conversation: key, initiator, interval, the
                   schedule label (labels held fixed across arms), episode,
                   per-arm record counts, grouped split assignments
    pairs          record to record pairing between every two arms

Definitions, written down here so that the code and the methods notes say the
same thing (00_Plan/Phase3_Instrument_Notes.md has the reasons):

Conversation (3.1). Records are blocked on the unordered endpoint pair plus
protocol: (ip:port, ip:port, proto), ports set to 0 for anything but TCP and
UDP. Inside a block, records are ordered by start time and a new conversation
starts when a record begins more than T seconds after every earlier record of
the block has ended. Each record covers [start, start + duration + r], where r
is its release's clock resolution; for a clock that rounds rather than
truncates (UNSW-NB15 Stime) the interval starts r/2 earlier. The conversation is defined on the pooled
records of the "defining" arms, so it is a property of the traffic, not of one
exporter. The minute-resolution CIC original arm does not define
conversations; its records are mapped onto them afterwards. The initiator is
the source endpoint of the earliest record; a record whose source is the
responder is marked reversed.

Ground truth (3.2). Conversations, not flows, are labelled. CICIDS2017: the
attack schedule encoded in LYCOS labelling.py (attacker, victim, port,
protocol, window), applied to the conversation's endpoints in either order
with the conversation start inside the window. The LYCOS drop rules become an
"excluded" label. UNSW-NB15: NUSW-NB15_GT.csv, matched on the same key with
interval overlap.

Pairing (3.3). Inside a conversation, records of two arms pair one to one,
greedily, among pairs whose intervals overlap, ranked by start agreement,
then duration agreement, then direction, then overlap (see PAIR_TOL below).
"""
import hashlib
import socket
from pathlib import Path

import numpy as np
import pandas as pd

import releases as R

CACHE = R.CACHE_ROOT / "_cache_phase3"
CACHE_VERSION = "v2"

CIC_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
CIC_DATE = {"Monday": "2017-07-03", "Tuesday": "2017-07-04", "Wednesday": "2017-07-05",
            "Thursday": "2017-07-06", "Friday": "2017-07-07"}
UNSW_DAYS = ["2015-01-22", "2015-02-18"]
UNSW_SPLIT_EPOCH = 1422000000  # 2015-01-23 08:00 UTC, between the two capture sessions

BENCHMARKS = {
    "CICIDS2017": dict(
        arms=["cic_original", "wtmc2021", "crisis2022_rpd", "lycos",
              "nfs2023_nte", "nfs2023_te", "bccc2017"],
        defining=["wtmc2021", "crisis2022_rpd", "lycos", "nfs2023_nte",
                  "nfs2023_te", "bccc2017"],
        days=CIC_DAYS),
    "UNSW-NB15": dict(
        arms=["unsw_original", "nf_unsw_v3", "cic_unsw"],
        defining=["unsw_original", "nf_unsw_v3", "cic_unsw"],
        days=UNSW_DAYS),
}

CIC_ORIG_FILES = {
    "Monday": ["Monday-WorkingHours.pcap_ISCX.csv"],
    "Tuesday": ["Tuesday-WorkingHours.pcap_ISCX.csv"],
    "Wednesday": ["Wednesday-workingHours.pcap_ISCX.csv"],
    "Thursday": ["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                 "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"],
    "Friday": ["Friday-WorkingHours-Morning.pcap_ISCX.csv",
               "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
               "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"],
}

BENIGN = "benign"
EXCLUDED = "excluded"

# Conversation gap in seconds, chosen in step 3.1 (00_Plan/Phase3_Instrument_Notes.md)
DEFAULT_T = 15
MAP_K = 4  # candidate conversations examined per minute-clock record


# ---------------------------------------------------------------------------
# loading

def files_for(arm, day):
    """Raw files that hold this arm's records for this day. None means: all
    files of the release, with the day taken from the timestamps."""
    folder = R.DATA / R.RELEASES[arm]["folder"]
    if arm == "cic_original":
        return [folder / f for f in CIC_ORIG_FILES[day]]
    if arm in ("wtmc2021", "crisis2022_rpd"):
        return [folder / f"{day}-WorkingHours.csv"]
    if arm == "lycos":
        return [folder / f"{day}-WorkingHours.pcap_lycos.csv"]
    if arm in ("nfs2023_nte", "nfs2023_te"):
        return [folder / f"{day}.csv"]
    return None


def day_of(arm, t_utc):
    """Day label from UTC start time, for the releases not split by day."""
    if arm in ("unsw_original", "nf_unsw_v3", "cic_unsw"):
        return np.where(t_utc < UNSW_SPLIT_EPOCH, UNSW_DAYS[0], UNSW_DAYS[1])
    date = pd.to_datetime(t_utc, unit="s").strftime("%Y-%m-%d")
    inv = {v: k for k, v in CIC_DATE.items()}
    return pd.Series(date).map(inv).fillna("other").to_numpy()


_PROTO_NAMES = {}


def proto_number(name):
    """IANA number for an Argus protocol name; -1 for names that are not IP
    protocols (arp) or that the system protocol table does not know."""
    name = str(name).strip().lower()
    if name not in _PROTO_NAMES:
        try:
            _PROTO_NAMES[name] = socket.getprotobyname(name)
        except OSError:
            _PROTO_NAMES[name] = -1
    return _PROTO_NAMES[name]


KEY_PROTOS = (1, 6, 17)


def key_proto(proto):
    """Protocol as it enters the conversation key: ICMP, TCP and UDP keep their
    number, everything else is -1 ("other"). Argus names many protocols that
    have no number (unas, any) or that the system table lacks (sun-nd, swipe),
    while nProbe writes numbers, so only these three can be matched across
    arms. Ports are already 0 outside TCP and UDP (hand check, UNSW case 17)."""
    proto = pd.Series(proto)
    return proto.where(proto.isin(KEY_PROTOS), -1).astype(np.int64)


def _port(s):
    s = s.astype(str).str.strip()
    hexa = s.str.startswith("0x")
    out = pd.to_numeric(s.where(~hexa), errors="coerce")
    if hexa.any():
        out[hexa] = s[hexa].map(lambda x: int(x, 16))
    return out.fillna(0).astype(np.int64)


def normalise(df, arm):
    """Typed five tuple, interval and record identity for one loaded file."""
    if arm == "unsw_original":
        proto = df["proto"].map(proto_number).astype(np.int64)
    else:
        proto = pd.to_numeric(df["proto"], errors="coerce").fillna(-1).astype(np.int64)
    sport, dport = _port(df["sport"]), _port(df["dport"])
    l4 = proto.isin([6, 17])
    sport, dport = sport.where(l4, 0), dport.where(l4, 0)
    dur = pd.Series(df["dur_s"]).fillna(0).clip(lower=0).to_numpy()
    t0 = df["t_utc"].to_numpy(dtype=float)
    if R.ROUNDING.get(arm) == "round":
        t0 = t0 - R.RESOLUTION[arm] / 2.0
    return pd.DataFrame({
        "src": df["src"].astype(str).str.strip(), "sport": sport.to_numpy(),
        "dst": df["dst"].astype(str).str.strip(), "dport": dport.to_numpy(),
        "proto": proto.to_numpy(), "t0": t0,
        "t1": t0 + dur + R.RESOLUTION[arm],
        "bytes": df["bytes"].to_numpy(dtype=float), "label": df["label"].to_numpy()})


def _cache_path(arm, day):
    return CACHE / CACHE_VERSION / arm / f"{day}.parquet"


def load_arm_day(arm, day, use_cache=True):
    """All records of one arm on one day, with file and row identity.

    row is the position of the record in releases.load_keys(arm, file), so
    later phases can fetch the record's features through the same loader.
    """
    p = _cache_path(arm, day)
    if use_cache and p.is_file():
        return pd.read_parquet(p)
    paths = files_for(arm, day)
    whole = paths is None
    if whole:
        paths = R.files(arm)
    parts = []
    for f in paths:
        raw = R.load_keys(arm, f, extra=True)
        df = normalise(raw, arm)
        df.insert(0, "row", np.arange(len(df), dtype=np.int64))
        df.insert(0, "file", f.name)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "arm", arm)
    if whole:
        # a release not split by day is read once and cached for every day
        days = day_of(arm, out["t0"].to_numpy())
        if use_cache:
            for d in pd.unique(days):
                q = _cache_path(arm, d)
                q.parent.mkdir(parents=True, exist_ok=True)
                out[days == d].reset_index(drop=True).to_parquet(q, index=False)
        return out[days == day].reset_index(drop=True)
    if use_cache:
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(p, index=False)
    return out


def load_day(bench, day, use_cache=True):
    arms = BENCHMARKS[bench]["arms"]
    rec = pd.concat([load_arm_day(a, day, use_cache) for a in arms], ignore_index=True)
    return add_keys(rec)


def add_keys(rec):
    """Unordered endpoint key and per-record endpoint strings."""
    ea = rec["src"] + ":" + rec["sport"].astype(str)
    eb = rec["dst"] + ":" + rec["dport"].astype(str)
    lo = ea.where(ea <= eb, eb)
    hi = eb.where(ea <= eb, ea)
    rec = rec.copy()
    rec["kproto"] = key_proto(rec["proto"]).to_numpy()
    key = lo + "|" + hi + "|" + rec["kproto"].astype(str)
    rec["key"] = pd.factorize(key, sort=True)[0].astype(np.int64)
    rec["key_str"] = pd.Categorical(key)
    rec["src_ep"] = pd.Categorical(ea)
    rec["arm"] = pd.Categorical(rec["arm"])
    rec["file"] = pd.Categorical(rec["file"])
    rec["idx"] = np.arange(len(rec), dtype=np.int64)
    return rec


# ---------------------------------------------------------------------------
# 3.1 conversations

def _gap_split(key, t0, t1, T):
    """Conversation number per record (input order) for one pooled set."""
    order = np.lexsort((t1, t0, key))
    k, s, e = key[order], t0[order], t1[order]
    cm = pd.Series(e).groupby(k).cummax().to_numpy()
    prev = np.r_[-np.inf, cm[:-1]]
    newkey = np.r_[True, k[1:] != k[:-1]]
    new = newkey | (s > prev + T)
    cid_sorted = np.cumsum(new) - 1
    cid = np.empty_like(cid_sorted)
    cid[order] = cid_sorted
    return cid


def build_conversations(rec, T, defining):
    """Assign every record to a conversation. Returns (conv_of_record, conv table, stats)."""
    arms = list(rec["arm"].cat.categories)
    isdef = rec["arm"].isin(defining).to_numpy()
    conv = np.full(len(rec), -1, dtype=np.int64)
    d = rec[isdef]
    conv[isdef] = _gap_split(d["key"].to_numpy(), d["t0"].to_numpy(), d["t1"].to_numpy(), T)
    ncd = conv[isdef].max() + 1 if isdef.any() else 0
    stats = {"n_mapped": 0, "n_unmapped": 0, "n_ambiguous": 0}
    nd = rec[~isdef]
    if len(nd):
        c = pd.DataFrame({"key": d["key"].to_numpy(), "conv": conv[isdef],
                          "t0": d["t0"].to_numpy(), "t1": d["t1"].to_numpy()})
        ct = c.groupby("conv").agg(key=("key", "first"), cs=("t0", "min"), ce=("t1", "max"))
        ct = ct.reset_index().sort_values(["cs", "conv"], kind="mergesort")
        left = pd.DataFrame({"pos": np.flatnonzero(~isdef), "key": nd["key"].to_numpy(),
                             "t0": nd["t0"].to_numpy(), "t1": nd["t1"].to_numpy()})
        # latest possible true start of a minute-truncated record is t0 + resolution
        reso = np.array([R.RESOLUTION[a] for a in nd["arm"].astype(str)])
        left["t_late"] = left["t0"] + reso
        left = left.sort_values(["t_late", "pos"], kind="mergesort")
        # candidates: the latest conversation of the key starting before the
        # record's latest possible start, and up to MAP_K - 1 before it. A
        # candidate is reachable when the record starts within T of its end.
        # Among reachable ones the record joins the conversation whose span is
        # closest to the record's duration (the minute clock cannot place the
        # start, the duration is exact), ties to the later one.
        ct = ct.sort_values(["key", "cs", "conv"], kind="mergesort").reset_index(drop=True)
        same = ct["key"].to_numpy()
        prev = np.where(np.r_[False, same[1:] == same[:-1]], np.arange(len(ct)) - 1, -1)
        ct["row"] = np.arange(len(ct))
        m = pd.merge_asof(left, ct[["key", "cs", "row"]].sort_values("cs", kind="mergesort"),
                          left_on="t_late", right_on="cs", by="key", direction="backward")
        row = m["row"].fillna(-1).astype(np.int64).to_numpy()
        dur = m["t1"].to_numpy() - m["t_late"].to_numpy()  # t1 = t0 + duration + r
        cs_arr, ce_arr, conv_arr = ct["cs"].to_numpy(), ct["ce"].to_numpy(), ct["conv"].to_numpy()
        best = np.full(len(m), -1, dtype=np.int64)
        best_score = np.full(len(m), np.inf)
        n_reach = np.zeros(len(m), dtype=np.int64)
        cur = row.copy()
        for _ in range(MAP_K):
            valid = cur >= 0
            reach = valid.copy()
            reach[valid] = m["t0"].to_numpy()[valid] <= ce_arr[cur[valid]] + T
            score = np.full(len(m), np.inf)
            score[reach] = np.abs(dur[reach] - (ce_arr[cur[reach]] - cs_arr[cur[reach]]))
            better = reach & (score < best_score - 1e-9)
            best[better] = conv_arr[cur[better]]
            best_score[better] = score[better]
            n_reach += reach
            cur = np.where(valid, prev[np.maximum(cur, 0)], -1)
        ok = pd.Series(best >= 0)
        m["conv"] = best
        conv[m.loc[ok.to_numpy(), "pos"].to_numpy()] = best[ok.to_numpy()]
        stats.update(n_mapped=int(ok.sum()), n_unmapped=int((~ok).sum()),
                     n_ambiguous=int((n_reach > 1).sum()))
        ok = ok.to_numpy()
        un = m.loc[~ok, "pos"].to_numpy()
        if len(un):
            u = rec.iloc[un]
            cu = _gap_split(u["key"].to_numpy(), u["t0"].to_numpy(), u["t1"].to_numpy(), T)
            conv[un] = ncd + cu
    # renumber conversations deterministically by (key, start)
    tmp = pd.DataFrame({"conv": conv, "key": rec["key"].to_numpy(), "t0": rec["t0"].to_numpy(),
                        "def": isdef})
    first = tmp.groupby("conv").agg(key=("key", "first"), cs=("t0", "min"))
    first = first.sort_values(["key", "cs"], kind="mergesort")
    remap = pd.Series(np.arange(len(first), dtype=np.int64), index=first.index)
    conv = remap.loc[conv].to_numpy()
    table = conversation_table(rec, conv, defining, arms)
    return conv, table, stats


def conversation_table(rec, conv, defining, arms):
    r = pd.DataFrame({"conv": conv, "key_str": rec["key_str"].astype(str).to_numpy(),
                      "proto": rec["kproto"].to_numpy(), "src": rec["src"].to_numpy(),
                      "sport": rec["sport"].to_numpy(), "dst": rec["dst"].to_numpy(),
                      "dport": rec["dport"].to_numpy(), "t0": rec["t0"].to_numpy(),
                      "t1": rec["t1"].to_numpy(), "arm": rec["arm"].astype(str).to_numpy(),
                      "def": rec["arm"].isin(defining).to_numpy()})
    arm_rank = {a: i for i, a in enumerate(BENCH_ORDER(arms, defining))}
    r["arm_rank"] = r["arm"].map(arm_rank)
    # defining arms first, then earliest start, then arm order: the initiator record
    r["nondef"] = ~r["def"]
    head = r.sort_values(["conv", "nondef", "t0", "arm_rank"], kind="mergesort") \
        .drop_duplicates("conv")
    t = head[["conv", "key_str", "proto", "src", "sport", "dst", "dport"]].rename(
        columns={"src": "a_ip", "sport": "a_port", "dst": "b_ip", "dport": "b_port"})
    span_def = r[r["def"]].groupby("conv").agg(cs=("t0", "min"), ce=("t1", "max"))
    span_all = r.groupby("conv").agg(cs_all=("t0", "min"), ce_all=("t1", "max"))
    t = t.set_index("conv").join(span_def).join(span_all)
    t["cs"] = t["cs"].fillna(t["cs_all"])
    t["ce"] = t["ce"].fillna(t["ce_all"])
    t = t.drop(columns=["cs_all", "ce_all"])
    n = pd.crosstab(r["conv"], r["arm"])
    for a in arms:
        t[f"n_{a}"] = n[a] if a in n.columns else 0
        t[f"n_{a}"] = t[f"n_{a}"].fillna(0).astype(np.int32)
    t["n_arms"] = sum((t[f"n_{a}"] > 0).astype(int) for a in arms)
    t["in_all"] = t["n_arms"] == len(arms)
    t["in_all_defining"] = sum((t[f"n_{a}"] > 0).astype(int) for a in defining) == len(defining)
    return t.reset_index()


def BENCH_ORDER(arms, defining):
    """Tie-break order for the initiator record: microsecond clocks first."""
    pref = ["lycos", "bccc2017", "nfs2023_nte", "nfs2023_te", "nf_unsw_v3",
            "wtmc2021", "crisis2022_rpd", "unsw_original", "cic_unsw", "cic_original"]
    return [a for a in pref if a in arms] + [a for a in arms if a not in pref]


def reversed_flag(rec, conv, table):
    ini = (table["a_ip"].astype(str) + ":" + table["a_port"].astype(str)).to_numpy()
    return rec["src_ep"].astype(str).to_numpy() != ini[conv]


# ---------------------------------------------------------------------------
# 3.2 ground truth

# (day, label, ip_a, ip_b, port_a, port_b, proto, windows) from LYCOS
# labelling.py. ip_a is the record source side there, ip_b the destination
# side; at conversation level the pair may appear in either order. Windows are
# UTC epoch seconds, inclusive, as in the script. Later rules overwrite earlier
# ones, as in the script.
CIC_SCHEDULE = [
    ("Tuesday", "ftp_patator", "172.16.0.1", "192.168.10.50", None, None, None,
     [(1499170620, 1499175000)]),
    ("Tuesday", "ssh_patator", "172.16.0.1", "192.168.10.50", None, None, None,
     [(1499188140, 1499191860)]),
    ("Wednesday", "heartbleed", "172.16.0.1", "192.168.10.51", 45022, 444, None, None),
    ("Wednesday", "dos_slowloris", "172.16.0.1", "192.168.10.50", None, 80, None,
     [(1499256060, 1499260260), (1499275440, 1499275500)]),
    ("Wednesday", "dos_slowhttptest", "172.16.0.1", "192.168.10.50", None, 80, None,
     [(1499260500, 1499261820)]),
    ("Wednesday", "dos_hulk", "172.16.0.1", "192.168.10.50", None, 80, None,
     [(1499262180, 1499263620)]),
    ("Wednesday", "dos_goldeneye", "172.16.0.1", "192.168.10.50", None, 80, None,
     [(1499263800, 1499264340)]),
    ("Thursday", "web_bruteforce", "172.16.0.1", "192.168.10.50", None, None, 6,
     [(1499343300, 1499346000)]),
    ("Thursday", "web_xss", "172.16.0.1", "192.168.10.50", None, None, 6,
     [(1499346935, 1499348100)]),
    ("Thursday", "web_sqli", "172.16.0.1", "192.168.10.50", None, None, 6,
     [(1499348400, 1499348576)]),
    ("Friday", "bot", None, "205.174.165.73", None, None, None,
     [(1499430840, 1499436122)]),
    ("Friday", "ddos", "172.16.0.1", "192.168.10.50", None, None, 6,
     [(1499453791, 1499454973)]),
    ("Friday", "portscan", "172.16.0.1", "192.168.10.50", None, None, 6,
     [(1499443530, 1499451842)]),
]
# LYCOS drop rules, kept as an "excluded" label so the conversations stay
# visible but are not scored.
CIC_EXCLUDE = [
    ("Thursday", "afternoon (infiltration) dropped by LYCOS", None, None, 1499353200, None),
    ("Friday", "205.174.165.73 after 1499436193 (LYCOS: portscan wrongly labelled)",
     "205.174.165.73", None, 1499436193, None),
    ("Friday", "52.6.13.28 (LYCOS: labelled Bot by CIC, not on the schedule)",
     "52.6.13.28", None, None, None),
    ("Friday", "52.7.235.158 (LYCOS: labelled Bot by CIC, not on the schedule)",
     "52.7.235.158", None, None, None),
]


def _rule_match(t, ip_a, ip_b, port_a, port_b, proto):
    """Endpoint test in both orders: side x plays ip_a, side y plays ip_b."""
    def side(xi, xp, yi, yp):
        m = np.ones(len(t), dtype=bool)
        if ip_a is not None:
            m &= (t[xi] == ip_a).to_numpy()
        if port_a is not None:
            m &= (t[xp] == port_a).to_numpy()
        if ip_b is not None:
            m &= (t[yi] == ip_b).to_numpy()
        if port_b is not None:
            m &= (t[yp] == port_b).to_numpy()
        return m
    m = side("a_ip", "a_port", "b_ip", "b_port") | side("b_ip", "b_port", "a_ip", "a_port")
    if proto is not None:
        m &= (t["proto"] == proto).to_numpy()
    return m


def label_cic(table, day, window_rule="start"):
    """Schedule label per conversation. window_rule 'start': conversation
    start inside the window (the LYCOS per-record rule); 'overlap': the
    conversation interval meets the window (sensitivity)."""
    lab = np.full(len(table), BENIGN, dtype=object)
    rule_id = np.full(len(table), -1)
    win_id = np.full(len(table), -1)
    cs, ce = table["cs"].to_numpy(), table["ce"].to_numpy()
    for i, (d, name, ia, ib, pa, pb, pr, wins) in enumerate(CIC_SCHEDULE):
        if d != day:
            continue
        m = _rule_match(table, ia, ib, pa, pb, pr)
        if wins is None:
            lab[m] = name
            rule_id[m] = i
            win_id[m] = 0
            continue
        for w, (ws, we) in enumerate(wins):
            inw = (cs >= ws) & (cs <= we) if window_rule == "start" else (cs <= we) & (ce >= ws)
            mm = m & inw
            lab[mm] = name
            rule_id[mm] = i
            win_id[mm] = w
    for d, why, ip, _, after, _ in CIC_EXCLUDE:
        if d != day:
            continue
        m = np.ones(len(table), dtype=bool)
        if ip is not None:
            m &= ((table["a_ip"] == ip) | (table["b_ip"] == ip)).to_numpy()
        if after is not None:
            m &= cs > after if ip is not None else cs >= after
        lab[m] = EXCLUDED
        rule_id[m] = -1
        win_id[m] = -1
    return lab, rule_id, win_id


def load_unsw_gt():
    f = R.DATA / "UNSW_NB15_Original" / "NUSW-NB15_GT.csv"
    g = pd.read_csv(f, usecols=["Start time", "Last time", "Attack category", "Protocol",
                                "Source IP", "Source Port", "Destination IP",
                                "Destination Port"], dtype=str, encoding="latin-1")
    proto = g["Protocol"].map(proto_number).astype(np.int64)
    sp, dp = _port(g["Source Port"]), _port(g["Destination Port"])
    l4 = proto.isin([6, 17])
    out = pd.DataFrame({"src": g["Source IP"].str.strip(), "sport": sp.where(l4, 0),
                        "dst": g["Destination IP"].str.strip(), "dport": dp.where(l4, 0),
                        "proto": proto,
                        "t0": pd.to_numeric(g["Start time"], errors="coerce"),
                        "label": g["Attack category"].map(canonical_unsw)})
    out["t1"] = pd.to_numeric(g["Last time"], errors="coerce") + 1.0
    ea = out["src"] + ":" + out["sport"].astype(str)
    eb = out["dst"] + ":" + out["dport"].astype(str)
    lo, hi = ea.where(ea <= eb, eb), eb.where(ea <= eb, ea)
    out["proto"] = key_proto(out["proto"]).to_numpy()
    out["key_str"] = lo + "|" + hi + "|" + out["proto"].astype(str)
    out["day"] = np.where(out["t0"] < UNSW_SPLIT_EPOCH, UNSW_DAYS[0], UNSW_DAYS[1])
    return out


def label_unsw(table, day, gt=None, tol=1.0):
    """Ground truth category per conversation from NUSW-NB15_GT.csv: GT rows
    on the same key whose interval meets the conversation's (tolerance tol
    seconds, the GT's resolution). Several categories: the most frequent,
    ties broken alphabetically; the count is returned."""
    gt = load_unsw_gt() if gt is None else gt
    g = gt[gt["day"] == day][["key_str", "t0", "t1", "label", "src", "dst"]]
    c = table[["conv", "key_str", "cs", "ce"]]
    m = c.merge(g, on="key_str", how="inner")
    m = m[(m["t0"] <= m["ce"] + tol) & (m["t1"] >= m["cs"] - tol)]
    cnt = m.groupby(["conv", "label"]).size().rename("n").reset_index()
    cnt = cnt.sort_values(["conv", "n", "label"], ascending=[True, False, True],
                          kind="mergesort")
    top = cnt.drop_duplicates("conv").set_index("conv")["label"]
    ncat = cnt.groupby("conv").size()
    att = m.sort_values(["conv", "t0"], kind="mergesort").drop_duplicates("conv") \
        .set_index("conv")[["src", "dst"]]
    lab = table["conv"].map(top).fillna(BENIGN).to_numpy(dtype=object)
    n_cat = table["conv"].map(ncat).fillna(0).astype(int).to_numpy()
    attacker = table["conv"].map(att["src"]).to_numpy(dtype=object)
    victim = table["conv"].map(att["dst"]).to_numpy(dtype=object)
    return lab, n_cat, attacker, victim


# ---------------------------------------------------------------------------
# canonical class names

def canonical_cic(label):
    """(class, attempted) for any CICIDS2017 release label."""
    s = str(label).strip().lower()
    att = "attempted" in s
    s = s.replace("attempted", "")
    s = "".join(ch if ch.isalnum() else " " for ch in s)
    w = s.split()
    j = "".join(w)
    if j in ("benign", "normal"):
        c = BENIGN
    elif "ftp" in j:
        c = "ftp_patator"
    elif "ssh" in j:
        c = "ssh_patator"
    elif "slowloris" in j:
        c = "dos_slowloris"
    elif "slowhttptest" in j:
        c = "dos_slowhttptest"
    elif "hulk" in j:
        c = "dos_hulk"
    elif "goldeneye" in j:
        c = "dos_goldeneye"
    elif "heartbleed" in j:
        c = "heartbleed"
    elif "brute" in j:
        c = "web_bruteforce"
    elif "xss" in j:
        c = "web_xss"
    elif "sql" in j:
        c = "web_sqli"
    elif "infilt" in j:
        c = "infiltration"
    elif "ddos" in j:
        c = "ddos"
    elif "bot" in j:
        c = "bot"
    elif "portscan" in j or ("port" in j and "scan" in j):
        c = "portscan"
    else:
        c = "unknown:" + str(label).strip()
    return c, att


def canonical_unsw(label):
    s = str(label).strip().lower()
    if s in ("", "normal", "benign", "nan"):
        return BENIGN
    if s.startswith("backdoor"):
        return "backdoor"
    return s.replace(" ", "_")


def canonical(bench, labels):
    u = pd.unique(pd.Series(labels, dtype=object))
    if bench == "CICIDS2017":
        m = {x: canonical_cic(x) for x in u}
        return (np.array([m[x][0] for x in labels], dtype=object),
                np.array([m[x][1] for x in labels], dtype=bool))
    m = {x: canonical_unsw(x) for x in u}
    return np.array([m[x] for x in labels], dtype=object), np.zeros(len(labels), dtype=bool)


# ---------------------------------------------------------------------------
# 3.3 pairing

# Pairing rule, revised after the step 3.4 hand check (00_Plan/Phase3_Hand_Check.md).
# A candidate pair needs the two widened intervals to overlap. Candidates are
# then ranked, best first, by
#   1. start disagreement beyond what the coarser clock can resolve,
#      the gap between the start windows [t0_a, t0_a + r_a) and [t0_b, t0_b + r_b),
#      rounded to the millisecond (until 27 Oct 2026: |start_a - start_b| - max(r_a, r_b),
#      which let a minute clock match a record that starts in the next minute)
#      (two records that begin at the same packet describe the same flow start)
#   2. duration disagreement |dur_a - dur_b|, rounded to the millisecond
#      (durations are measured to the microsecond even where the start time
#      is only known to the minute)
#   3. same direction before reversed
#   4. larger overlap, then smaller raw start difference, then row order
# and assigned greedily, one to one. The first version ranked by overlap alone
# with a 0.5 second tolerance; the hand check found that it paired records by
# row order when a minute-resolution clock made overlaps tie, and that it
# paired tail records that do not overlap at all.
PAIR_TOL = 0.0
CAND_CAP = 250_000  # above this many candidate pairs in one conversation, pair in time order


def _greedy(ca, ia, ib):
    """One to one assignment over candidates already sorted best first within conv."""
    out_a, out_b = [], []
    cur = None
    used_a, used_b = set(), set()
    for c, a, b in zip(ca.tolist(), ia.tolist(), ib.tolist()):
        if c != cur:
            cur, used_a, used_b = c, set(), set()
        if a in used_a or b in used_b:
            continue
        used_a.add(a)
        used_b.add(b)
        out_a.append(a)
        out_b.append(b)
    return out_a, out_b


def _ordered(ra, rb):
    """Time-ordered pairing for very large conversations."""
    out_a, out_b = [], []
    for c, ga in ra.groupby("conv", sort=True):
        gb = rb[rb["conv"] == c]
        A = ga.sort_values(["t0", "idx"])[["idx", "t0", "t1"]].to_numpy()
        B = gb.sort_values(["t0", "idx"])[["idx", "t0", "t1"]].to_numpy()
        i = j = 0
        while i < len(A) and j < len(B):
            ov = min(A[i, 2], B[j, 2]) - max(A[i, 1], B[j, 1])
            if ov > PAIR_TOL:
                out_a.append(int(A[i, 0]))
                out_b.append(int(B[j, 0]))
                i += 1
                j += 1
            elif A[i, 2] <= B[j, 1]:
                i += 1
            else:
                j += 1
    return out_a, out_b


def _rank(cand, res_a, res_b):
    cand["ov"] = np.minimum(cand["t1_a"], cand["t1_b"]) - np.maximum(cand["t0_a"], cand["t0_b"])
    cand = cand[cand["ov"] > PAIR_TOL].copy()
    dt = (cand["t0_a"] - cand["t0_b"]).abs()
    # each true start lies in [t0, t0 + r) (truncated clocks; the rounded UNSW
    # clock is shifted by r/2 on load), so the disagreement is the gap between
    # the two start windows, not |dt| - r (changed 27 Oct 2026, hand check case 5)
    gap = np.maximum(cand["t0_b"] - (cand["t0_a"] + res_a), cand["t0_a"] - (cand["t0_b"] + res_b))
    cand["k_start"] = np.round(np.maximum(0.0, gap), 3)
    dur_a = cand["t1_a"] - cand["t0_a"] - res_a
    dur_b = cand["t1_b"] - cand["t0_b"] - res_b
    cand["k_dur"] = np.round((dur_a - dur_b).abs(), 3)
    cand["k_dir"] = (cand["rev_a"] != cand["rev_b"]).astype(np.int8)
    cand["k_ov"] = -cand["ov"]
    cand["k_dt"] = dt
    return cand.sort_values(["conv", "k_start", "k_dur", "k_dir", "k_ov", "k_dt",
                             "idx_a", "idx_b"], kind="mergesort")


def pair_arms(rec, conv, a, b, rev=None):
    """Record pairs between arms a and b. Returns DataFrame arm_a, arm_b, ia, ib, ov."""
    if rev is None:
        rev = rec["rev"].to_numpy() if "rev" in rec else np.zeros(len(rec), dtype=bool)
    base = pd.DataFrame({"idx": rec["idx"].to_numpy(), "conv": conv,
                         "t0": rec["t0"].to_numpy(), "t1": rec["t1"].to_numpy(),
                         "rev": np.asarray(rev, dtype=bool)})
    arm = rec["arm"].astype(str).to_numpy()
    ra, rb = base[arm == a], base[arm == b]
    na, nb = ra["conv"].value_counts(), rb["conv"].value_counts()
    both = na.index.intersection(nb.index)
    na, nb = na.loc[both], nb.loc[both]
    big = both[(na * nb > CAND_CAP).to_numpy()]
    small = both.difference(big)
    parts = []
    am, bm = ra[ra["conv"].isin(small)], rb[rb["conv"].isin(small)]
    if len(am):
        cand = am.merge(bm, on="conv", suffixes=("_a", "_b"))
        cand = _rank(cand, R.RESOLUTION[a], R.RESOLUTION[b])
        oa, ob = _greedy(cand["conv"].to_numpy(), cand["idx_a"].to_numpy(),
                         cand["idx_b"].to_numpy())
        parts.append(pd.DataFrame({"ia": oa, "ib": ob}))
    if len(big):
        oa, ob = _ordered(ra[ra["conv"].isin(big)], rb[rb["conv"].isin(big)])
        parts.append(pd.DataFrame({"ia": oa, "ib": ob}))
    p = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame({"ia": [], "ib": []})
    p["ia"] = p["ia"].astype(np.int64)
    p["ib"] = p["ib"].astype(np.int64)
    t0, t1 = rec["t0"].to_numpy(), rec["t1"].to_numpy()
    p["ov"] = np.minimum(t1[p["ia"]], t1[p["ib"]]) - np.maximum(t0[p["ia"]], t0[p["ib"]])
    p = p.sort_values(["ia", "ib"], kind="mergesort").reset_index(drop=True)
    p.insert(0, "arm_b", b)
    p.insert(0, "arm_a", a)
    n1 = int(((na == 1) & (nb == 1)).sum())
    return p, {"n_conv_1to1": n1, "n_conv_multi": int(len(small) - n1),
               "n_conv_big": int(len(big))}


def cardinality(table, a, b):
    """Per conversation cardinality class between arms a and b."""
    na, nb = table[f"n_{a}"].to_numpy(), table[f"n_{b}"].to_numpy()
    out = np.full(len(table), "none", dtype=object)
    out[(na == 1) & (nb == 1)] = "1:1"
    out[(na > 1) & (nb == 1)] = "n:1"
    out[(na == 1) & (nb > 1)] = "1:n"
    out[(na > 1) & (nb > 1)] = "n:m"
    out[(na > 0) & (nb == 0)] = "a_only"
    out[(na == 0) & (nb > 0)] = "b_only"
    return out


# ---------------------------------------------------------------------------
# 3.7 splits

TEST_FRACTION = 0.3
SEEDS = [0, 1, 2, 3, 4]


def unit_hash(strings, seed):
    """Deterministic uint64 per string: pandas' SipHash of the string with a
    16 byte key derived from the seed (pandas.util.hash_pandas_object)."""
    key = f"seed{int(seed):012d}"
    return pd.util.hash_pandas_object(pd.Series(strings, dtype=object), index=False,
                                      hash_key=key, categorize=False).to_numpy()


def stratified_hash_split(units, strata, seed, frac=TEST_FRACTION):
    """1 test, 0 train. Inside each stratum, the round(frac * n) units with the
    smallest hash go to test (round half up). Units are strings."""
    h = unit_hash(units, seed)
    df = pd.DataFrame({"s": strata, "h": h, "i": np.arange(len(units))})
    df = df.sort_values(["s", "h", "i"], kind="mergesort")
    rank = df.groupby("s").cumcount().to_numpy()
    n = df.groupby("s")["h"].transform("size").to_numpy()
    ntest = np.floor(frac * n + 0.5)
    out = np.zeros(len(units), dtype=np.int8)
    out[df["i"].to_numpy()] = (rank < ntest).astype(np.int8)
    return out


def conv_uid(bench, day, table):
    return (bench + "|" + day + "|" + table["key_str"].astype(str) + "|" +
            table["cs"].map(lambda x: f"{x:.6f}")).to_numpy()
