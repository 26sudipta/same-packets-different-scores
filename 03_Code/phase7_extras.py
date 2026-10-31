"""
Phase 7, step 7.3 inputs: per record unit ids and labels for the sensitivity
variants, aligned to the Phase 5 arm tables by (day, arm, file, row).

Usage, from the workspace root:
    .venv/bin/python 03_Code/phase7_extras.py [--benchmarks CICIDS2017 UNSW-NB15]

For every day the records of all arms are loaded as in Phase 3
(instrument.load_day) and the conversations are rebuilt with
instrument.build_conversations at T = 15 s (checked against the frozen
instrument, must be identical) and at T = 5, 30, 60 and 120 s. From the frozen
instrument it also takes the arm's own label, the Attempted-folded schedule
label and the episode.

Unit ids are global within a benchmark: day index * 10**8 + conversation, the
Phase 5 `gconv` scheme. Episode unit: an attack conversation's episode
(EP_BASE + day index * 10**8 + episode); a benign conversation has no episode
and stays its own unit.

Output: Datasets/_cache_phase7/<bench>/extras_<day>.parquet, and
04_Results/phase7/t_rebuild_check.csv (T = 15 identity check, conversations
per T, rebuilt conversations whose records carry two schedule classes).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instrument as I  # noqa: E402
import releases as R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "04_Results" / "instrument"
CACHE = R.CACHE_ROOT / "_cache_phase7"
OUT = ROOT / "04_Results" / "phase7"
TS = [5, 30, 60, 120]
EP_BASE = 9 * 10 ** 9


def day_extras(bench, di, day, check):
    t0 = time.time()
    cfg = I.BENCHMARKS[bench]
    fr = pd.read_parquet(FROZEN / bench / day / "records.parquet",
                         columns=["idx", "arm", "file", "row", "conv", "own", "sched"])
    ct = pd.read_parquet(FROZEN / bench / day / "conversations.parquet",
                         columns=["conv", "sched", "sched_attempted_fold", "episode"])
    if not (ct["conv"].to_numpy() == np.arange(len(ct))).all():
        raise AssertionError(f"{bench} {day}: conversation table not indexed by conv")
    rec = I.load_day(bench, day, use_cache=True)
    same = (len(rec) == len(fr) and (rec["arm"].astype(str).to_numpy() == fr["arm"].astype(str).to_numpy()).all()
            and (rec["file"].astype(str).to_numpy() == fr["file"].astype(str).to_numpy()).all()
            and (rec["row"].to_numpy() == fr["row"].to_numpy()).all())
    if not same:
        raise AssertionError(f"{bench} {day}: loaded records not in frozen order")
    out = pd.DataFrame({"day": day, "arm": fr["arm"].astype(str).to_numpy(),
                        "file": fr["file"].astype(str).to_numpy(), "row": fr["row"].to_numpy()})
    base = di * 10 ** 8
    sched = fr["sched"].to_numpy(dtype=object)
    for T in [15] + TS:
        conv, _, _ = I.build_conversations(rec, T, cfg["defining"])
        if T == 15:
            ident = bool((conv == fr["conv"].to_numpy()).all())
            if not ident:
                raise AssertionError(f"{bench} {day}: T = 15 rebuild differs from the frozen "
                                     f"instrument in {(conv != fr['conv'].to_numpy()).sum()} records")
        else:
            out[f"cT{T}"] = base + conv
        sc = sched != I.EXCLUDED
        g = pd.DataFrame({"c": conv[sc], "s": sched[sc]})
        n_lab = g.groupby("c")["s"].nunique()
        check.append({"benchmark": bench, "day": day, "T": T, "conversations": int(len(n_lab)),
                      "two_or_more_classes": int((n_lab > 1).sum()),
                      "identical_to_frozen": ident if T == 15 else np.nan})
        print(f"{bench} {day} T={T}: {len(n_lab):,} conversations, "
              f"{int((n_lab > 1).sum())} with two classes ({time.time() - t0:.0f} s)", flush=True)
    c = fr["conv"].to_numpy()
    out["own"] = fr["own"].astype(str).to_numpy()
    out["sched_fold"] = ct["sched_attempted_fold"].to_numpy(dtype=object)[c]
    ep = ct["episode"].to_numpy()[c]
    out["episode_unit"] = np.where(ep >= 0, EP_BASE + base + ep, base + c)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", nargs="*", default=["CICIDS2017", "UNSW-NB15"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    check = []
    for bench in a.benchmarks:
        d = CACHE / bench
        d.mkdir(parents=True, exist_ok=True)
        for di, day in enumerate(I.BENCHMARKS[bench]["days"]):
            p = d / f"extras_{day}.parquet"
            if p.is_file():
                print(f"{bench} {day}: cached", flush=True)
                continue
            day_extras(bench, di, day, check).to_parquet(p, index=False)
    if check:
        # rows of the days rebuilt now replace their earlier rows (before 29 Oct 2026 they
        # were appended, so a rerun on a checkout doubled the file)
        cp = OUT / "t_rebuild_check.csv"
        new = pd.DataFrame(check)
        key = ["benchmark", "day", "T"]
        old = pd.read_csv(cp) if cp.is_file() else new.iloc[:0]
        old = old[~old.set_index(key).index.isin(new.set_index(key).index)]
        pd.concat([old, new], ignore_index=True).to_csv(cp, index=False)
