"""
Step 9.3, the sanity gate: three results of the paper, drawn at random, re-derived
from the raw release CSVs with code written for this check only.

Usage, from the workspace root (after the Phase 3 to 6 outputs exist):
    .venv/bin/python 03_Code/phase9_sanity.py

The draw. Candidates are the RQ macros of 04_Results/freeze/numbers.csv that the
manuscript (manuscript.tex and the threats section) uses and whose value holds a
digit, sorted by name; numpy default_rng(20261028) picks three without
replacement. On 28 Oct 2026 there were 170 candidates and the draw gave
RQoneAlphaCicLo, RQthreeLeakTeMin and RQoneFtpThreeBccc. The script repeats the
draw and stops if it no longer gives these three.

What is taken from the frozen pipeline and what is not. The conversation of
each record (records.parquet, conv) and the CICIDS2017 exclusion rule
(conversations.parquet, sched) are taken as given; they were checked by hand
in step 3.4. Everything else is rebuilt here: every record's label and five
tuple are read again from the raw CSV row it points to (file, row) and compared
with the instrument; labels are mapped to classes by the explicit table RAW_CLASS
below, not by instrument.canonical_cic; the per-conversation label, the random
split, the leakage and the counts are computed with new code; alpha comes from
the `krippendorff` package with its own bootstrap.

Tolerances, fixed before the run:
    (a) alpha point estimate within 1e-6 of alpha.csv; bootstrap bounds within
        2e-4 of alpha.csv (Monte Carlo error of a 2.5 or 97.5 percentile over
        1,000 resamples, interval width about 3e-4); and the rounded value of
        the macro unchanged.
    (b) and (c) exact: they involve no resampling.
The first full run (28 Oct 2026, freeze log) failed one check of (b): the class list
lacked the Phase 6 eligibility rule (at least 20 common conversations in both arms), which
the check then applied; the leakage minimum itself matched.

Outputs:
    04_Results/freeze/sanity_9_3.csv   one row per check: expected, re-derived, pass
    04_Results/freeze/manifests/phase9_sanity_*.json
"""
import sys
import time
from pathlib import Path

import krippendorff
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import releases as R  # noqa: E402  (paths of the raw files only)
from manifest import write_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "04_Results" / "instrument" / "CICIDS2017"
OUT = ROOT / "04_Results" / "freeze"
DRAW_SEED = 20261028
DRAWN = ["RQoneAlphaCicLo", "RQthreeLeakTeMin", "RQoneFtpThreeBccc"]
DRAW_CANDIDATES = 170   # RQ macros quoted by the manuscript at the 9.3 draw, 28 Oct 2026
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
ARMS = ["cic_original", "wtmc2021", "crisis2022_rpd", "lycos", "nfs2023_nte", "nfs2023_te",
        "bccc2017"]
BOOT = 1000
SEEDS = [0, 1, 2, 3, 4]
TEST_FRACTION = 0.3
MIN_CONV = 20   # Phase 6 protocol: classes with at least 20 conversations in both arms
T0 = time.time()

# raw header names: source ip, source port, destination ip, destination port, protocol, label
RAW_COLS = {
    "cic_original": ["Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol",
                     "Label"],
    "wtmc2021": ["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Label"],
    "crisis2022_rpd": ["Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Label"],
    "lycos": ["src_addr", "src_port", "dst_addr", "dst_port", "ip_prot", "label"],
    "nfs2023_nte": ["src_ip", "src_port", "dst_ip", "dst_port", "protocol", "label"],
    "nfs2023_te": ["src_ip", "src_port", "dst_ip", "dst_port", "protocol", "label"],
    "bccc2017": ["src_ip", "src_port", "dst_ip", "dst_port", "protocol", "label", "timestamp"],
}

# Every raw label string of the seven CICIDS2017 arms (as listed from records.parquet on
# 28 Oct 2026) and its class, written out by hand. "Attempted" labels keep their class
# (the multiclass variant of the paper). The CIC original web labels carry a byte 0x96
# (an en dash in cp1252) between "Web Attack" and the attack name; they are matched on
# the text around it.
RAW_CLASS = {
    "BENIGN": "benign", "Benign": "benign", "benign": "benign",
    "Bot": "bot", "Botnet_ARES": "bot", "bot": "bot", "Bot - Attempted": "bot",
    "DDoS": "ddos", "DDoS_LOIT": "ddos", "ddos": "ddos",
    "PortScan": "portscan", "Port_Scan": "portscan", "portscan": "portscan",
    "Port_scan": "portscan",
    "FTP-Patator": "ftp_patator", "ftp_patator": "ftp_patator",
    "FTP-Patator - Attempted": "ftp_patator",
    "SSH-Patator": "ssh_patator", "ssh_patator": "ssh_patator",
    "SSH-Patator - Attempted": "ssh_patator",
    "DoS GoldenEye": "dos_goldeneye", "DoS_GoldenEye": "dos_goldeneye",
    "dos_goldeneye": "dos_goldeneye", "DoS GoldenEye - Attempted": "dos_goldeneye",
    "DoS Hulk": "dos_hulk", "DoS_Hulk": "dos_hulk", "dos_hulk": "dos_hulk",
    "DoS Hulk - Attempted": "dos_hulk",
    "DoS Slowhttptest": "dos_slowhttptest", "DoS_Slowhttptest": "dos_slowhttptest",
    "dos_slowhttptest": "dos_slowhttptest", "DoS Slowhttptest - Attempted": "dos_slowhttptest",
    "DoS slowloris": "dos_slowloris", "DoS_Slowloris": "dos_slowloris",
    "dos_slowloris": "dos_slowloris", "DoS slowloris - Attempted": "dos_slowloris",
    "Heartbleed": "heartbleed", "heartbleed": "heartbleed",
    "Infiltration": "infiltration", "Infiltration - Attempted": "infiltration",
    "Web Attack - Brute Force": "web_bruteforce", "Web_Brute_Force": "web_bruteforce",
    "webattack_bruteforce": "web_bruteforce",
    "Web Attack - Brute Force - Attempted": "web_bruteforce",
    "Web Attack - Sql Injection": "web_sqli", "Web_SQL_Injection": "web_sqli",
    "webattack_sql_injection": "web_sqli",
    "Web Attack - XSS": "web_xss", "Web_XSS": "web_xss", "webattack_xss": "web_xss",
    "Web Attack - XSS - Attempted": "web_xss",
}
CIC_WEB = {"Brute Force": "web_bruteforce", "Sql Injection": "web_sqli", "XSS": "web_xss"}


def raw_class(label):
    s = str(label).strip()
    if s in RAW_CLASS:
        return RAW_CLASS[s]
    rest = s[len("Web Attack"):].strip(" \x96") if s.startswith("Web Attack") else None
    if rest in CIC_WEB:   # "Web Attack \x96 XSS"
        return CIC_WEB[rest]
    raise KeyError(f"raw label not in the table: {s!r}")


def log(*a):
    print(f"[{time.time() - T0:6.0f} s]", *a, flush=True)


def redo_draw():
    m = pd.read_csv(OUT / "numbers.csv")
    tex = (ROOT / "05_Paper" / "springer" / "manuscript.tex").read_text() + \
        (ROOT / "05_Paper" / "sections" / "11_threats.tex").read_text()
    import re
    used = set(re.findall(r"\\([A-Za-z]+)", tex))
    c = m[m["macro"].isin(used) & m["macro"].str.startswith("RQ")]
    c = c[c["value"].astype(str).str.contains(r"\d")].sort_values("macro").reset_index(drop=True)
    pick = list(c.loc[np.random.default_rng(DRAW_SEED).choice(len(c), 3, replace=False), "macro"])
    log(f"draw: {len(c)} candidates, picked {pick}")
    if len(c) == DRAW_CANDIDATES:
        assert pick == DRAWN, pick
    else:
        # the manuscript has quoted more macros since the 9.3 draw, so the same seed picks others;
        # the checks below are written for the 9.3 draw, which must still be quoted
        assert set(DRAWN) <= set(c["macro"]), set(DRAWN) - set(c["macro"])
        log(f"candidates changed since the 9.3 draw ({DRAW_CANDIDATES}); checking the 9.3 draw {DRAWN}")
    return len(c), m.set_index("macro")["value"]


def port_int(s):
    s = pd.Series(s, dtype=object).astype(str).str.strip()
    out = np.zeros(len(s), dtype=np.int64)
    for i, x in enumerate(s.to_numpy()):
        if x.startswith("0x"):
            out[i] = int(x, 16)
        else:
            try:
                out[i] = int(float(x))
            except ValueError:
                out[i] = 0
    return out


def proto_int(s):
    names = {"TCP": 6, "UDP": 17, "ICMP": 1}
    out = []
    for x in pd.Series(s, dtype=object).astype(str).str.strip():
        try:
            out.append(int(float(x)))
        except ValueError:
            out.append(names.get(x.upper(), -1))
    return np.array(out, dtype=np.int64)


def read_raw(arm, fname):
    path = next(p for p in R.files(arm) if p.name == fname)
    want = RAW_COLS[arm]
    df = pd.read_csv(path, usecols=lambda c: c.strip() in want, dtype=str, encoding="latin-1",
                     low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if arm == "cic_original":
        df = df.dropna(how="all")   # the release has empty lines; row counts after them
    return df[want].reset_index(drop=True)


def arm_records(arm, cols):
    parts = []
    for di, day in enumerate(DAYS):
        r = pd.read_parquet(FROZEN / day / "records.parquet", columns=["arm"] + cols,
                            filters=[("arm", "==", arm)])
        r["arm"] = r["arm"].astype(str)
        r = r[r["arm"] == arm]
        r.insert(0, "day", day)
        r.insert(1, "gconv", di * 10 ** 8 + r["conv"].to_numpy())
        parts.append(r)
    return pd.concat(parts, ignore_index=True)


def check_against_raw(arm, rec):
    """Read every record's raw row again. Returns the raw class per record (rec order)
    and the counts of five tuple and label mismatches."""
    cls = np.empty(len(rec), dtype=object)
    bad_tuple = bad_label = 0
    files = rec["file"].astype(str).to_numpy()
    for f in np.unique(files):
        idx = np.flatnonzero(files == f)
        raw = read_raw(arm, f)
        rows = rec["row"].to_numpy()[idx]
        assert rows.max() < len(raw), (arm, f, rows.max(), len(raw))
        x = raw.iloc[rows]
        c = RAW_COLS[arm]
        src, dst = x[c[0]].astype(str).str.strip().to_numpy(), x[c[2]].astype(str).str.strip().to_numpy()
        pr = proto_int(x[c[4]])
        l4 = np.isin(pr, [6, 17])
        sp, dp = np.where(l4, port_int(x[c[1]]), 0), np.where(l4, port_int(x[c[3]]), 0)
        r = rec.iloc[idx]
        ok = (src == r["src"].astype(str).to_numpy()) & (dst == r["dst"].astype(str).to_numpy()) \
            & (sp == r["sport"].to_numpy()) & (dp == r["dport"].to_numpy()) \
            & (pr == r["proto"].to_numpy())
        bad_tuple += int((~ok).sum())
        lab = x[c[5]].astype(str).str.strip().to_numpy()
        bad_label += int((lab != r["label"].astype(str).str.strip().to_numpy()).sum())
        cls[idx] = [raw_class(v) for v in lab]
    return cls, bad_tuple, bad_label


# ---------------------------------------------------------------------------
# (a) Krippendorff's alpha, CICIDS2017, all days, own labels, units present in 2+ arms

def conv_label(gconv, cls):
    """Most frequent attack class among the records of each conversation, ties to the
    alphabetically first; benign when it has no attack record."""
    d = pd.DataFrame({"g": gconv, "c": cls})
    att = d[d["c"] != "benign"]
    if len(att):
        n = att.value_counts(["g", "c"]).rename("n").reset_index()
        n = n.sort_values(["g", "n", "c"], ascending=[True, False, True]).groupby("g").head(1)
        top = n.set_index("g")["c"]
    else:
        top = pd.Series(dtype=object)
    allg = pd.Index(pd.unique(gconv))
    out = pd.Series("benign", index=allg, dtype=object)
    out.loc[top.index] = top.to_numpy()
    return out


def check_alpha(expected):
    scored = []
    for di, day in enumerate(DAYS):
        t = pd.read_parquet(FROZEN / day / "conversations.parquet", columns=["conv", "sched"])
        t = t[t["sched"].astype(str) != "excluded"]
        scored.append(di * 10 ** 8 + t["conv"].to_numpy())
    scored = pd.Index(np.concatenate(scored))
    labels, checks = {}, []
    for arm in ARMS:
        rec = arm_records(arm, ["file", "row", "conv", "src", "sport", "dst", "dport", "proto",
                                "label"])
        cls, bt, bl = check_against_raw(arm, rec)
        checks.append(dict(arm=arm, records=len(rec), tuple_mismatch=bt, label_mismatch=bl))
        keep = rec["gconv"].isin(scored).to_numpy()
        labels[arm] = conv_label(rec["gconv"].to_numpy()[keep], cls[keep])
        log(f"(a) {arm}: {len(rec):,} records re-read, {bt} five tuple and {bl} label "
            f"mismatches, {len(labels[arm]):,} scored conversations")
        del rec
    cats = sorted(set().union(*[set(v.unique()) for v in labels.values()]))
    code = {c: i for i, c in enumerate(cats)}
    M = np.full((len(ARMS), len(scored)), np.nan)
    pos = pd.Series(np.arange(len(scored)), index=scored)
    for j, arm in enumerate(ARMS):
        s = labels[arm]
        M[j, pos.loc[s.index].to_numpy()] = s.map(code).to_numpy(dtype=float)
    M = M[:, (~np.isnan(M)).sum(0) >= 2]
    units = M.shape[1]
    a = krippendorff.alpha(reliability_data=M, level_of_measurement="nominal")
    log(f"(a) {units:,} units, alpha {a:.6f}; bootstrap of {BOOT} resamples")
    rng = np.random.default_rng(DRAW_SEED)
    boots = np.empty(BOOT)
    for b in range(BOOT):
        boots[b] = krippendorff.alpha(reliability_data=M[:, rng.integers(0, units, units)],
                                      level_of_measurement="nominal")
        if (b + 1) % 100 == 0:
            log(f"(a) bootstrap {b + 1} of {BOOT}")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    e = expected
    rows = [
        dict(check="a: units", expected=e["units"], rederived=units, tol=0,
             passed=units == e["units"]),
        dict(check="a: alpha", expected=e["alpha"], rederived=round(a, 6), tol=1e-6,
             passed=abs(a - e["alpha"]) <= 1e-6),
        dict(check="a: alpha_lo", expected=e["alpha_lo"], rederived=round(lo, 6), tol=2e-4,
             passed=abs(lo - e["alpha_lo"]) <= 2e-4),
        dict(check="a: alpha_hi", expected=e["alpha_hi"], rederived=round(hi, 6), tol=2e-4,
             passed=abs(hi - e["alpha_hi"]) <= 2e-4),
        dict(check="a: macro RQoneAlphaCicLo", expected=e["macro"], rederived=f"{lo:.3f}", tol=0,
             passed=f"{lo:.3f}" == str(e["macro"])),
    ]
    for c in checks:
        rows.append(dict(check=f"a: raw rows {c['arm']}", expected="0 mismatches",
                         rederived=f"{c['tuple_mismatch']} tuple, {c['label_mismatch']} label "
                                   f"of {c['records']:,}",
                         tol=0, passed=c["tuple_mismatch"] == 0 and c["label_mismatch"] == 0))
    return rows


# ---------------------------------------------------------------------------
# (b) random split leakage, NFS-2023 TE, common population, classes with fpc >= 1.95

def hash_split(units, strata, seed):
    """The rule of 00_Plan (step 3.5): inside each stratum the round half up of
    0.3 n units with the smallest keyed SipHash go to test; ties by position."""
    h = pd.util.hash_pandas_object(pd.Series(units, dtype=object), index=False,
                                   hash_key=f"seed{seed:012d}", categorize=False).to_numpy()
    out = np.zeros(len(units), dtype=np.int8)
    strata = np.asarray(strata, dtype=object)
    for s in np.unique(strata):
        i = np.flatnonzero(strata == s)
        o = i[np.lexsort((i, h[i]))]
        out[o[:int(np.floor(TEST_FRACTION * len(i) + 0.5))]] = 1
    return out


def check_leak(expected):
    arm = "nfs2023_te"
    rec = arm_records(arm, ["file", "row", "conv", "src", "sport", "dst", "dport", "proto", "label",
                            "sched"] + [f"r{s}" for s in SEEDS])
    rec["sched"] = rec["sched"].astype(str)
    rec = rec[rec["sched"] != "excluded"].reset_index(drop=True)
    # the split, rebuilt per day from the rule and compared with the frozen one
    split_bad = 0
    mine = {s: np.zeros(len(rec), dtype=np.int8) for s in SEEDS}
    for day in DAYS:
        m = (rec["day"] == day).to_numpy()
        uid = (arm + "|" + rec.loc[m, "file"].astype(str) + "|" + rec.loc[m, "row"].astype(str))
        for s in SEEDS:
            x = hash_split(uid.to_numpy(), rec.loc[m, "sched"].to_numpy(), s)
            mine[s][m] = x
            split_bad += int((x != rec.loc[m, f"r{s}"].to_numpy()).sum())
    log(f"(b) split rebuilt for {len(rec):,} TE records, {split_bad} differ from the frozen split")
    # the Phase 5 subsample: the conversations of the TE training table
    tab = pd.read_parquet(R.CACHE_ROOT / "_cache_phase5" / "main" / "CICIDS2017" / f"{arm}.parquet",
                          columns=["gconv", "file", "row", "in_all", "sched"])
    tab["file"] = tab["file"].astype(str)
    key = rec["file"].astype(str) + "|" + rec["row"].astype(str)
    tkey = tab["file"] + "|" + tab["row"].astype(str)
    inrec = key.isin(set(tkey)).to_numpy()
    sub = rec[inrec].reset_index(drop=True)
    assert len(sub) == len(tab), (len(sub), len(tab))
    # five tuple and label of the records in the table, from the raw CSVs
    cls, bt, bl = check_against_raw(arm, sub)
    log(f"(b) {len(sub):,} records in the Phase 5 table, {bt} five tuple and {bl} label mismatches")
    common = set(tab.loc[tab["in_all"].astype(bool), "gconv"])
    sub["common"] = sub["gconv"].isin(common)
    leaks = []
    for s in SEEDS:
        sp = mine[s][inrec]
        train = pd.Index(pd.unique(sub.loc[sp == 0, "gconv"]))
        te = sub[(sp == 1) & sub["common"].to_numpy()]
        lk = te["gconv"].isin(train).groupby(te["sched"]).mean()
        leaks.append(lk.rename(s))
    leak = pd.concat(leaks, axis=1).mean(axis=1)
    c = sub[sub["common"]]
    fpc = c.groupby("sched").size() / c.groupby("sched")["gconv"].nunique()
    # eligibility (Phase 6 protocol): at least MIN_CONV conversations on the common
    # population in both arms, counted from both Phase 5 tables
    nte = pd.read_parquet(R.CACHE_ROOT / "_cache_phase5" / "main" / "CICIDS2017" / "nfs2023_nte.parquet",
                          columns=["gconv", "in_all", "sched"])
    ncv = pd.DataFrame({
        "te": c.groupby("sched")["gconv"].nunique(),
        "nte": nte[nte["in_all"].astype(bool)].groupby("sched")["gconv"].nunique()}).fillna(0)
    elig = ncv.index[(ncv["te"] >= MIN_CONV) & (ncv["nte"] >= MIN_CONV)]
    log("(b) classes below", MIN_CONV, "conversations in an arm:",
        ", ".join(f"{k} ({int(r.te)}, {int(r.nte)})" for k, r in ncv.drop(elig).iterrows()))
    att = fpc.drop("benign").reindex(elig.drop("benign", errors="ignore"))
    att = att[att >= 1.95]
    got = leak.loc[att.index]
    log("(b) classes with fpc >= 1.95:", ", ".join(f"{k} {v:.4f}" for k, v in got.items()))
    lo = float(got.min())
    e = expected
    return [
        dict(check="b: split rebuilt from the rule", expected=0, rederived=split_bad, tol=0,
             passed=split_bad == 0),
        dict(check="b: raw rows nfs2023_te (table)", expected="0 mismatches",
             rederived=f"{bt} tuple, {bl} label of {len(sub):,}", tol=0, passed=bt == 0 and bl == 0),
        dict(check="b: classes kept", expected=e["classes"], rederived=";".join(sorted(att.index)),
             tol=0, passed=";".join(sorted(att.index)) == e["classes"]),
        dict(check="b: min leakage (class)", expected=f"{e['leak']:.6f} ({e['cls']})",
             rederived=f"{lo:.6f} ({got.idxmin()})", tol=1e-9,
             passed=abs(lo - e["leak"]) <= 1e-9 and got.idxmin() == e["cls"]),
        dict(check="b: macro RQthreeLeakTeMin", expected=e["macro"], rederived=f"{lo:.2f}", tol=0,
             passed=f"{lo:.2f}" == str(e["macro"])),
    ]


# ---------------------------------------------------------------------------
# (c) FTP-Patator conversations present in every arm that BCCC writes as three records

def check_ftp(expected):
    inall = []
    for di, day in enumerate(DAYS):
        t = pd.read_parquet(FROZEN / day / "conversations.parquet",
                            columns=["conv", "sched", "key_str", "in_all"])
        t = t[(t["sched"].astype(str) == "ftp_patator") & t["in_all"].astype(bool)]
        inall.append(pd.DataFrame({"day": day, "gconv": di * 10 ** 8 + t["conv"].to_numpy(),
                                   "key": t["key_str"].astype(str).to_numpy()}))
    conv = pd.concat(inall, ignore_index=True)
    # 1. count BCCC records per conversation from the record table
    rec = arm_records("bccc2017", ["file", "row", "conv", "src", "sport", "dst", "dport", "proto",
                                   "label", "t0", "t1"])
    r = rec[rec["gconv"].isin(set(conv["gconv"]))].reset_index(drop=True)
    n_rec = r.groupby("gconv").size().reindex(conv["gconv"]).fillna(0).astype(int)
    cls, bt, bl = check_against_raw("bccc2017", r)
    # 2. count from the raw CSVs alone: rows of every BCCC file whose unordered endpoint
    # key equals the conversation's and whose start falls inside the conversation's span
    # across all arms, widened by T = 15 s
    span = []
    for di, day in enumerate(DAYS):
        x = pd.read_parquet(FROZEN / day / "records.parquet", columns=["conv", "t0", "t1"])
        x["gconv"] = di * 10 ** 8 + x["conv"].to_numpy()
        x = x[x["gconv"].isin(set(conv["gconv"]))]
        span.append(x.groupby("gconv").agg(cs=("t0", "min"), ce=("t1", "max")))
    span = pd.concat(span)
    conv = conv.join(span, on="gconv")
    raws = []
    for p in R.files("bccc2017"):
        d = read_raw("bccc2017", p.name)
        c = RAW_COLS["bccc2017"]
        pr = proto_int(d[c[4]])
        l4 = np.isin(pr, [6, 17])
        ea = d[c[0]].astype(str).str.strip() + ":" + pd.Series(np.where(l4, port_int(d[c[1]]), 0)).astype(str)
        eb = d[c[2]].astype(str).str.strip() + ":" + pd.Series(np.where(l4, port_int(d[c[3]]), 0)).astype(str)
        kp = np.where(np.isin(pr, [1, 6, 17]), pr, -1)
        key = np.where(ea <= eb, ea + "|" + eb, eb + "|" + ea) + "|" + kp.astype(str)
        # BCCC clock: naive local time one hour behind ADT, so UTC = naive + 4 h
        ts = pd.to_datetime(d[c[6]].str.strip(), format="ISO8601", errors="coerce")
        t_utc = (ts - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy() + 4 * 3600.0
        m = pd.Series(key).isin(set(conv["key"])).to_numpy()
        raws.append(pd.DataFrame({"key": key[m], "t": t_utc[m]}))
    raw = pd.concat(raws, ignore_index=True)
    j = conv.merge(raw, on="key", how="left")
    j["inside"] = (j["t"] >= j["cs"] - 15.0) & (j["t"] <= j["ce"] + 15.0)
    n_raw = j.groupby("gconv")["inside"].sum().reindex(conv["gconv"]).astype(int)
    same = int((n_raw.to_numpy() == n_rec.to_numpy()).sum())
    share_rec = float((n_rec == 3).mean())
    share_raw = float((n_raw == 3).mean())
    log(f"(c) {len(conv):,} conversations; three records: {share_rec:.6f} (record table), "
        f"{share_raw:.6f} (raw CSV count); counts equal for {same:,}")
    e = expected
    return [
        dict(check="c: conversations", expected=e["convs"], rederived=len(conv), tol=0,
             passed=len(conv) == e["convs"]),
        dict(check="c: raw rows bccc2017", expected="0 mismatches",
             rederived=f"{bt} tuple, {bl} label of {len(r):,}; classes "
                       + ";".join(f"{k} {v}" for k, v in pd.Series(cls).value_counts().items()),
             tol=0, passed=bt == 0 and bl == 0),
        dict(check="c: share with 3 records (record table)", expected=e["share"],
             rederived=round(share_rec, 6), tol=5e-7, passed=abs(share_rec - e["share"]) <= 5e-7),
        dict(check="c: share with 3 records (raw CSV count)", expected=e["share"],
             rederived=round(share_raw, 6), tol=5e-7, passed=abs(share_raw - e["share"]) <= 5e-7),
        dict(check="c: per conversation counts equal", expected=len(conv), rederived=same, tol=0,
             passed=same == len(conv)),
        dict(check="c: macro RQoneFtpThreeBccc", expected=e["macro"],
             rederived=f"{100 * share_raw:.0f}", tol=0,
             passed=f"{100 * share_raw:.0f}" == str(e["macro"])),
    ]


def main():
    n_cand, macros = redo_draw()
    P = ROOT / "04_Results"
    a = pd.read_csv(P / "phase4" / "alpha.csv")
    a = a[(a["variant"] == "all") & (a["benchmark"] == "CICIDS2017") & (a["day"] == "ALL")
          & (a["question"] == "labels") & (a["label_variant"] == "multiclass")
          & (a["population"] == "present_in_2plus") & (a["class"] == "ALL")
          & (a["raters"] == "all")]
    assert len(a) == 1
    a = a.iloc[0]
    t = pd.read_csv(P / "phase6" / "table_6_1.csv").set_index("cls").drop("benign")
    t = t[t["fpc_te"] >= 1.95]
    f = pd.read_csv(P / "phase4" / "flows_per_conv.csv")
    f = f[(f["variant"] == "all") & (f["benchmark"] == "CICIDS2017") & (f["day"] == "ALL")
          & (f["population"] == "common") & (f["class"] == "ftp_patator")
          & (f["arm"] == "bccc2017")].iloc[0]
    rows = []
    rows += check_ftp(dict(convs=int(f["conversations"]), share=float(f["share_3"]),
                           macro=macros["RQoneFtpThreeBccc"]))
    rows += check_leak(dict(classes=";".join(sorted(t.index)), leak=float(t["leak_random_te"].min()),
                            cls=t["leak_random_te"].idxmin(), macro=macros["RQthreeLeakTeMin"]))
    rows += check_alpha(dict(units=int(a["units"]), alpha=float(a["alpha"]),
                             alpha_lo=float(a["alpha_lo"]), alpha_hi=float(a["alpha_hi"]),
                             macro=macros["RQoneAlphaCicLo"]))
    out = pd.DataFrame(rows)
    p = OUT / "sanity_9_3.csv"
    out.to_csv(p, index=False)
    print(out.to_string(), flush=True)
    write_manifest("phase9_sanity", inputs=[OUT / "numbers.csv", P / "phase4" / "alpha.csv",
                                            P / "phase4" / "flows_per_conv.csv",
                                            P / "phase6" / "table_6_1.csv"],
                   outputs=[p], params=dict(draw_seed=DRAW_SEED, candidates=n_cand, drawn=DRAWN,
                                            boot=BOOT, krippendorff="package"),
                   out_dir=OUT / "manifests")
    log("all passed" if out["passed"].all() else "SOME CHECKS FAILED")
    sys.exit(0 if out["passed"].all() else 1)


if __name__ == "__main__":
    main()
