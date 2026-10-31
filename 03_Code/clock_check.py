"""
Step 2.11: put every arm on one clock and check it.

For each benchmark, take attack windows that can be found without any
release's labels (attacker address, victim address, victim port, day) and
compare their first and last flow start across arms after the UTC offsets in
releases.py are applied. Tolerance is the coarsest timestamp resolution of the
benchmark's arms (60 s for CICIDS2017, where the original release has minute
timestamps; 1 s for UNSW-NB15). A window can miss flows at one end in one arm
(a coverage difference, itself a result for RQ1); a clock error moves every
window. The per-arm verdict in release_clock_summary.csv is therefore the
median start deviation over windows.

For CICIDS2017 the windows are also compared with the attack schedule encoded
in the LYCOS labelling.py (epoch microseconds, UTC), which is the schedule
reference used in this study.

Writes 04_Results/release_clock.csv, release_clock_summary.csv and a manifest.
    .venv/bin/python 03_Code/clock_check.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import write_manifest  # noqa: E402
from releases import RELEASES, files, load_keys  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "04_Results" / "release_clock.csv"
OUT_SUMMARY = ROOT / "04_Results" / "release_clock_summary.csv"

ATT, VIC = "172.16.0.1", "192.168.10.50"
# name, UTC day, victim, victim port (None = any), LYCOS labeller window (UTC epoch s)
CIC_WINDOWS = [
    ("Tue FTP-Patator", "2017-07-04", VIC, "21", (1499170620, 1499175000)),
    ("Tue SSH-Patator", "2017-07-04", VIC, "22", (1499188140, 1499191860)),
    ("Wed DoS, port 80", "2017-07-05", VIC, "80", (1499256060, 1499264340)),
    ("Wed Heartbleed", "2017-07-05", "192.168.10.51", "444", None),
    ("Thu web attacks, port 80", "2017-07-06", VIC, "80", (1499343300, 1499348576)),
    ("Fri DDoS and port scan", "2017-07-07", VIC, None, (1499443530, 1499454973)),
]
UNSW_ATTACKER_NET = "175.45.176."
TOL = {"CICIDS2017": 60.0, "UNSW-NB15": 1.0}


def attacker_rows(rel):
    parts = []
    for p in files(rel):
        d = load_keys(rel, p)
        if RELEASES[rel]["benchmark"] == "CICIDS2017":
            d = d[d["src"].astype(str).str.strip() == ATT]
        else:
            d = d[d["src"].astype(str).str.startswith(UNSW_ATTACKER_NET)]
        parts.append(d[["dst", "dport", "t_utc"]])
    d = pd.concat(parts, ignore_index=True)
    d["dst"] = d["dst"].astype(str).str.strip()
    d["dport"] = d["dport"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    d["day"] = pd.to_datetime(d["t_utc"], unit="s").dt.strftime("%Y-%m-%d")
    return d


def main():
    rows, inputs = [], []
    for bench in ("CICIDS2017", "UNSW-NB15"):
        arms = [r for r, m in RELEASES.items()
                if m["benchmark"] == bench and m["utc_offset"] is not None]
        data = {}
        for r in arms:
            print("loading", r, flush=True)
            inputs += files(r)
            data[r] = attacker_rows(r)
        if bench == "CICIDS2017":
            windows = CIC_WINDOWS
        else:
            days = sorted(set().union(*[set(d["day"]) for d in data.values()]))
            windows = [(f"attacker subnet, {day}", day, None, None, None) for day in days]
        for name, day, vic, port, sched in windows:
            res = {}
            for r, d in data.items():
                w = d[d["day"] == day]
                if vic is not None:
                    w = w[w["dst"] == vic]
                if port is not None:
                    w = w[w["dport"] == port]
                res[r] = (w["t_utc"].min(), w["t_utc"].max(), len(w))
            firsts = pd.Series({r: v[0] for r, v in res.items()}).dropna()
            lasts = pd.Series({r: v[1] for r, v in res.items()}).dropna()
            ref_first, ref_last = firsts.median(), lasts.median()
            for r, (t0, t1, n) in res.items():
                d0, d1 = t0 - ref_first, t1 - ref_last
                rows.append({
                    "benchmark": bench, "window": name, "release": r, "flows": n,
                    "utc_offset_hours": RELEASES[r]["utc_offset"] / 3600.0,
                    "first_utc": pd.Timestamp(t0, unit="s") if n else pd.NaT,
                    "last_utc": pd.Timestamp(t1, unit="s") if n else pd.NaT,
                    "first_minus_median_s": round(d0, 3) if n else None,
                    "last_minus_median_s": round(d1, 3) if n else None,
                    "schedule_start_utc": pd.Timestamp(sched[0], unit="s") if sched else pd.NaT,
                    "schedule_end_utc": pd.Timestamp(sched[1], unit="s") if sched else pd.NaT,
                    "start_within_tolerance": bool(n) and abs(d0) <= TOL[bench],
                    "end_within_tolerance": bool(n) and abs(d1) <= TOL[bench],
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    # A wrong offset moves every window by the same whole number of hours. A
    # window that is missing flows at one end (coverage) moves only that end.
    # So the verdict uses the median start deviation over the windows.
    summ = (out[out["flows"] > 0]
            .groupby(["benchmark", "release"], sort=False)
            .agg(utc_offset_hours=("utc_offset_hours", "first"),
                 windows=("window", "size"),
                 windows_start_aligned=("start_within_tolerance", "sum"),
                 windows_both_ends_aligned=("end_within_tolerance",
                                            lambda e: int((e & out.loc[e.index, "start_within_tolerance"]).sum())),
                 median_start_dev_s=("first_minus_median_s", "median"),
                 max_abs_start_dev_s=("first_minus_median_s", lambda x: x.abs().max()))
            .reset_index())
    summ["tolerance_s"] = summ["benchmark"].map(TOL)
    summ["clock_verified"] = summ["median_start_dev_s"].abs() <= summ["tolerance_s"]
    summ.to_csv(OUT_SUMMARY, index=False)
    m = write_manifest("release_clock", inputs=inputs, outputs=[OUT, OUT_SUMMARY],
                       params={"tolerance_s": TOL})
    print(out.to_string())
    print(summ.to_string())
    print(f"wrote {OUT}, {OUT_SUMMARY}, {m}")


if __name__ == "__main__":
    main()
