"""
Phase 8: attach the records of CRiSIS-2022 Exp_RP (not an instrument arm) to the frozen
instrument conversations, so that RP can be resampled and scored by conversation
(protocol 8.3 step 4 and 8.4). Exp_R has the same rows as Exp_RP (phase8_flow.py R3rp
checks it), so the map serves both.

Rule: the instrument's own rule for records outside the defining arms
(instrument.build_conversations): a record joins a conversation of the same unordered
key (endpoints and protocol) that starts no later than the record's latest possible
start (t0 + clock resolution, 1 s as for Exp_RPD) and ends no more than T = 15 s before
the record starts; of up to MAP_K such conversations, the one whose span is closest to
the record's duration. This is the protocol's "span widened by T contains its start"
with the instrument's tie-break. Records with no such conversation are counted and left
out (unit of their own in phase8_decomp.py).

Usage:  .venv/bin/python 03_Code/phase8_rp_map.py [--check 20]
Writes  Datasets/_cache_phase8/convmap/crisis2022_rp.parquet (file, row, gconv, sched)
        04_Results/phase8/sets/rp_map_stats.csv
        04_Results/phase8/sets/rp_map_handcheck.md (twenty records for standing gate 3)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instrument as I  # noqa: E402
import phase8_common as P  # noqa: E402
import releases as R  # noqa: E402

BENCH = "CICIDS2017"
LIKE = "crisis2022_rpd"   # same CSV layout and clock


def rp_records(day):
    path = R.DATA / P.EXTRA_RELEASES["crisis2022_rp"]["folder"] / f"{day}-WorkingHours.csv"
    raw = R.load_keys(LIKE, path, extra=True)
    rec = I.normalise(raw, LIKE)
    rec.insert(0, "row", np.arange(len(rec), dtype=np.int64))
    rec.insert(0, "file", path.name)
    rec.insert(0, "arm", "crisis2022_rp")
    return I.add_keys(rec)


def map_day(day, T=I.DEFAULT_T):
    d = P.FROZEN / BENCH / day
    ct = pd.read_parquet(d / "conversations.parquet",
                         columns=["conv", "key_str", "cs", "ce", "sched"])
    rec = rp_records(day)
    allk = pd.concat([ct["key_str"].astype(str), rec["key_str"].astype(str)], ignore_index=True)
    codes = pd.factorize(allk)[0]
    ct["key"], kr = codes[:len(ct)], codes[len(ct):]
    left = pd.DataFrame({"pos": np.arange(len(rec)), "key": kr, "t0": rec["t0"].to_numpy(),
                         "t1": rec["t1"].to_numpy()})
    left["t_late"] = left["t0"] + R.RESOLUTION[LIKE]
    left = left.sort_values(["t_late", "pos"], kind="mergesort")
    ct = ct.sort_values(["key", "cs", "conv"], kind="mergesort").reset_index(drop=True)
    same = ct["key"].to_numpy()
    prev = np.where(np.r_[False, same[1:] == same[:-1]], np.arange(len(ct)) - 1, -1)
    ct["row"] = np.arange(len(ct))
    m = pd.merge_asof(left, ct[["key", "cs", "row"]].sort_values("cs", kind="mergesort"),
                      left_on="t_late", right_on="cs", by="key", direction="backward")
    row = m["row"].fillna(-1).astype(np.int64).to_numpy()
    dur = m["t1"].to_numpy() - m["t_late"].to_numpy()
    cs_arr, ce_arr, conv_arr = ct["cs"].to_numpy(), ct["ce"].to_numpy(), ct["conv"].to_numpy()
    best = np.full(len(m), -1, dtype=np.int64)
    best_score = np.full(len(m), np.inf)
    n_reach = np.zeros(len(m), dtype=np.int64)
    cur = row.copy()
    t0 = m["t0"].to_numpy()
    for _ in range(I.MAP_K):
        valid = cur >= 0
        reach = valid.copy()
        reach[valid] = t0[valid] <= ce_arr[cur[valid]] + T
        score = np.full(len(m), np.inf)
        score[reach] = np.abs(dur[reach] - (ce_arr[cur[reach]] - cs_arr[cur[reach]]))
        better = reach & (score < best_score - 1e-9)
        best[better] = conv_arr[cur[better]]
        best_score[better] = score[better]
        n_reach += reach
        cur = np.where(valid, prev[np.maximum(cur, 0)], -1)
    conv = np.full(len(rec), -1, dtype=np.int64)
    conv[m["pos"].to_numpy()] = best
    amb = np.zeros(len(rec), dtype=np.int64)
    amb[m["pos"].to_numpy()] = n_reach
    rec["conv"] = conv
    rec["n_candidates"] = amb
    sched = ct.set_index("conv")["sched"]
    rec["sched"] = sched.reindex(conv).to_numpy()
    return rec, ct


def main(n_check):
    days = I.BENCHMARKS[BENCH]["days"]
    out, stats, samples = [], [], []
    rng = np.random.default_rng(0)
    for di, day in enumerate(days):
        rec, ct = map_day(day)
        ok = rec["conv"].to_numpy() >= 0
        canon, _ = I.canonical(BENCH, rec["label"].to_numpy())
        stats.append({"day": day, "records": len(rec), "mapped": int(ok.sum()),
                      "unmapped": int((~ok).sum()),
                      "ambiguous": int((rec["n_candidates"] > 1).sum()),
                      "unmapped_attack": int(((~ok) & (canon != P.BENIGN)).sum())})
        print(stats[-1], flush=True)
        o = rec.loc[ok, ["file", "row", "conv", "sched"]].copy()
        o["gconv"] = di * 10 ** 8 + o["conv"].to_numpy()
        out.append(o[["file", "row", "gconv", "sched"]])
        # hand-check sample: per day, attack and benign records, one ambiguous if any
        pick = []
        for mask in (ok & (canon != P.BENIGN), ok & (canon == P.BENIGN),
                     ok & (rec["n_candidates"].to_numpy() > 1), ~ok):
            idx = np.flatnonzero(mask)
            if len(idx):
                pick.append(int(rng.choice(idx)))
        samples.append((day, rec, ct, pick, ok, canon))
    # with no ambiguous or unattached records and no attacks on Monday the picks above give
    # fewer than n_check cases; fill up to n_check with further mapped records, attack first,
    # one day at a time
    total = sum(len(x[3]) for x in samples)
    while total < n_check:
        grew = False
        for day, rec, ct, pick, ok, canon in samples:
            if total >= n_check:
                break
            for mask in (ok & (canon != P.BENIGN), ok & (canon == P.BENIGN)):
                idx = np.setdiff1d(np.flatnonzero(mask), pick)
                if len(idx):
                    pick.append(int(rng.choice(idx)))
                    total += 1
                    grew = True
                    break
        if not grew:
            break
    samples = [x[:4] for x in samples]
    res = pd.concat(out, ignore_index=True)
    res["file"] = res["file"].astype(str)
    res["sched"] = res["sched"].astype(str)
    p = P.CACHE / "convmap" / "crisis2022_rp.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(p, index=False)
    (P.OUT / "sets").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stats).to_csv(P.OUT / "sets" / "rp_map_stats.csv", index=False)
    write_handcheck(samples, n_check)


def write_handcheck(samples, n_check):
    """For each picked RP record: the record, its conversation, and the RPD records
    of that conversation (from the frozen records table)."""
    lines = ["# Hand check: CRiSIS-2022 Exp_RP records attached to instrument conversations",
             "", "Generated by 03_Code/phase8_rp_map.py (standing gate 3). Times are UTC "
             "seconds; t1 = start + duration + 1 s clock resolution.", ""]
    k = 0
    for day, rec, ct, pick in samples:
        recs = pd.read_parquet(P.FROZEN / BENCH / day / "records.parquet",
                               columns=["arm", "row", "conv", "src", "sport", "dst", "dport",
                                        "proto", "t0", "t1", "label"],
                               filters=[("arm", "==", "crisis2022_rpd")])
        for i in pick:
            if k >= n_check:
                break
            k += 1
            r = rec.iloc[i]
            lines.append(f"## Case {k}: {day}, RP row {int(r['row'])}")
            lines.append("")
            lines.append(f"RP record: {r['src']}:{r['sport']} -> {r['dst']}:{r['dport']} "
                         f"proto {r['proto']}, t0 {r['t0']:.0f}, t1 {r['t1']:.3f}, "
                         f"label {r['label']!s}, candidates {int(r['n_candidates'])}")
            if r["conv"] < 0:
                same = ct[ct["key_str"].astype(str) == str(r["key_str"])]
                lines.append(f"Not attached. Conversations with this key: {len(same)}; "
                             "nearest spans: " + "; ".join(
                                 f"[{a:.0f}, {b:.0f}]" for a, b in
                                 same.assign(d=(same['cs'] - r['t0']).abs())
                                 .nsmallest(3, 'd')[['cs', 'ce']].to_numpy()))
            else:
                c = ct[ct["conv"] == r["conv"]].iloc[0]
                lines.append(f"Conversation {int(r['conv'])}: key {c['key_str']}, span "
                             f"[{c['cs']:.0f}, {c['ce']:.0f}], schedule class {c['sched']}")
                q = recs[recs["conv"] == r["conv"]].sort_values("t0").head(6)
                lines.append("RPD records of that conversation (first six):")
                for _, x in q.iterrows():
                    lines.append(f"- row {int(x['row'])}: {x['src']}:{x['sport']} -> "
                                 f"{x['dst']}:{x['dport']}, t0 {x['t0']:.0f}, t1 {x['t1']:.3f}, "
                                 f"label {x['label']}")
            lines += ["", "Verdict: ", ""]
    path = P.OUT / "sets" / "rp_map_handcheck.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"hand check: {k} cases in {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, default=20)
    main(ap.parse_args().check)
