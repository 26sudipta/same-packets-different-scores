"""
Step 3.4: pick twenty conversations per benchmark for a hand check of the
conversation key (3.1), the schedule label (3.2) and the pairing (3.3).

    .venv/bin/python 03_Code/hand_check.py CICIDS2017 Tuesday
    .venv/bin/python 03_Code/hand_check.py UNSW-NB15 2015-02-18
    .venv/bin/python 03_Code/hand_check.py UNSW-NB15 2015-01-22 --focus cic_unsw

Prints every raw record of each chosen conversation (time, interval,
direction, bytes, own label) and the pairing the code produced, to
04_Results/phase3/hand_check/<bench>_<day>_cases.txt, and writes the records
as a fixture for the unit tests (03_Code/tests/fixtures/). The decisions made
by hand are recorded in 00_Plan/Phase3_Hand_Check.md.

With --focus ARM the twenty cases are drawn around one arm that joined after
the first hand check (the CIC-UNSW-NB15 release, 29 October 2026), and the
files carry the arm in their name, so the first cases stay as they were.
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instrument as I

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "04_Results" / "instrument"
OUT = ROOT / "04_Results" / "phase3" / "hand_check"
FIX = Path(__file__).resolve().parent / "tests" / "fixtures"


def pick(table, arms, rng):
    n = table[[f"n_{a}" for a in arms]].to_numpy()
    atk = ~table["sched"].isin([I.BENIGN, I.EXCLUDED]).to_numpy()
    nz = (n > 0).sum(1)
    chosen, why = [], []

    def take(mask, reason, k=1):
        idx = np.flatnonzero(mask & ~np.isin(np.arange(len(table)), chosen))
        if len(idx) == 0:
            return
        for i in rng.choice(idx, size=min(k, len(idx)), replace=False):
            chosen.append(int(i))
            why.append(reason)

    classes = [c for c in table.loc[atk, "sched"].unique()]
    for c in sorted(classes)[:4]:
        cm = (table["sched"] == c).to_numpy()
        take(cm & (n.max(1) > 1), f"attack {c}, some arm has several records")
        take(cm & (n.max(1) == 1) & (nz == len(arms)), f"attack {c}, one record in every arm")
    take((n == 1).all(1), "one record in every arm")
    if len(arms) > 2:
        take((n[:, 0] > 1) & (n[:, 1:] == 1).all(1), f"{arms[0]} splits, every other arm one record")
        ia, ib = arms.index("nfs2023_nte"), arms.index("nfs2023_te")
        take((n[:, ia] == 1) & (n[:, ib] > 1), "TE splits what nTE keeps whole")
        take((n[:, ia] > 1) & (n[:, ib] > 1) & (n[:, ia] != n[:, ib]), "nTE and TE both split, n:m")
        for a in (arms[0], "bccc2017", "lycos"):
            j = arms.index(a)
            take((n[:, j] > 0) & (nz == 1), f"present in {a} only")
        take(table["proto"].to_numpy() == 17, "UDP")
        take(table["proto"].to_numpy() == 1, "ICMP")
    else:
        take((n[:, 0] > 1) & (n[:, 1] == 1), f"n:1, {arms[0]} splits")
        take((n[:, 0] == 1) & (n[:, 1] > 1), f"1:n, {arms[1]} splits")
        take((n[:, 0] > 1) & (n[:, 1] > 1), "n:m")
        take((n[:, 0] > 0) & (n[:, 1] == 0), f"{arms[0]} only")
        take((n[:, 0] == 0) & (n[:, 1] > 0), f"{arms[1]} only")
        take((table["gt_categories"] > 1).to_numpy(), "several ground truth categories")
        take(table["proto"].to_numpy() == 17, "UDP")
        take(~table["proto"].isin([6, 17]).to_numpy(), "neither TCP nor UDP")
    keycount = table.groupby("key_str")["conv"].transform("size").to_numpy()
    take(keycount > 1, "same endpoint key, two conversations (gap rule)")
    big = n.sum(1)
    take((big > 5) & (big < 40), "long conversation, many records")
    while len(chosen) < 20:
        take(np.ones(len(table), dtype=bool), "random")
    return chosen[:20], why[:20]


def pick_focus(table, arms, focus, rng):
    """Twenty cases that test how the records of one arm join the conversations
    and pair with the other arms."""
    n = table[[f"n_{a}" for a in arms]].to_numpy()
    j = arms.index(focus)
    f, rest = n[:, j], np.delete(n, j, axis=1)
    atk = ~table["sched"].isin([I.BENIGN, I.EXCLUDED]).to_numpy()
    chosen, why = [], []

    def take(mask, reason, k=1):
        idx = np.flatnonzero(mask & ~np.isin(np.arange(len(table)), chosen))
        if len(idx) == 0:
            return
        for i in rng.choice(idx, size=min(k, len(idx)), replace=False):
            chosen.append(int(i))
            why.append(reason)

    for c in sorted(table.loc[atk, "sched"].unique())[:4]:
        cm = (table["sched"] == c).to_numpy()
        take(cm & (f > 1), f"attack {c}, {focus} has several records")
        take(cm & (f == 1) & (rest == 1).all(1), f"attack {c}, one record in every arm")
    take(atk & (f == 0) & (rest > 0).any(1), f"attack, no {focus} record")
    take((f == 1) & (rest == 1).all(1), "one record in every arm")
    take((f > 1) & (rest == 1).all(1), f"{focus} splits, every other arm one record")
    take((f == 1) & (rest > 1).any(1), f"{focus} one record, another arm several")
    take((f > 0) & (rest == 0).all(1), f"present in {focus} only")
    take((f == 0) & (rest > 0).all(1), f"in every other arm, not in {focus}")
    take(table["proto"].to_numpy() == 17, "UDP")
    take(~table["proto"].isin([6, 17]).to_numpy() & (f > 0), f"neither TCP nor UDP, {focus} present")
    keycount = table.groupby("key_str")["conv"].transform("size").to_numpy()
    take((keycount > 1) & (f > 0), "same endpoint key, two conversations (gap rule)")
    big = n.sum(1)
    take((big > 5) & (big < 40) & (f > 0), "long conversation, many records")
    while len(chosen) < 20:
        take(f > 0, "random")
    return chosen[:20], why[:20]


def main(bench, day, seed=3, focus=None):
    cfg = I.BENCHMARKS[bench]
    arms = cfg["arms"]
    d = FROZEN / bench / day
    rec = pd.read_parquet(d / "records.parquet")
    table = pd.read_parquet(d / "conversations.parquet")
    pairs = pd.read_parquet(d / "pairs.parquet")
    rng = np.random.default_rng(seed)
    tag = f"{bench}_{day}" + (f"_{focus}" if focus else "")
    keep = FIX / f"handcheck_{tag}_cases.csv"
    if focus and not keep.is_file():
        chosen, why = pick_focus(table, arms, focus, rng)
    elif keep.is_file():
        # the cases checked by hand stay the same when the tables are rebuilt:
        # the conversation on each recorded key that starts closest to the recorded start
        prev = pd.read_csv(keep)
        # keys written before the key protocol rule (instrument.key_proto)
        kp = prev["key_str"].str.rsplit("|", n=1)
        prev["key_str"] = kp.str[0] + "|" + I.key_proto(kp.str[1].astype(int)).astype(str)
        chosen, why = [], []
        for p in prev.itertuples():
            cand = table.index[table["key_str"] == p.key_str]
            if len(cand) == 0:
                continue
            i = cand[np.argmin(np.abs(table.loc[cand, "cs"].to_numpy() - p.cs))]
            chosen.append(int(i))
            why.append(p.reason)
    else:
        chosen, why = pick(table, arms, rng)
    sel = table.iloc[chosen]
    # the conversations' keys, so the fixture also holds neighbours on the same key
    keys = set(sel["key_str"])
    fx = rec[rec["conv"].isin(table.loc[table["key_str"].isin(keys), "conv"])]
    OUT.mkdir(parents=True, exist_ok=True)
    FIX.mkdir(parents=True, exist_ok=True)
    lines = [f"Hand check cases, {bench} {day}, seed {seed}", ""]
    covered = set()
    for k, (i, reason) in enumerate(zip(chosen, why), 1):
        t = table.iloc[i]
        r = rec[rec["conv"] == t["conv"]].sort_values(["t0", "arm"])
        present = [a for a in arms if t[f"n_{a}"] > 0]
        covered |= set(itertools.combinations(sorted(present, key=arms.index), 2))
        lines.append(f"== case {k}: conv {t['conv']}  ({reason})")
        lines.append(f"   key {t['key_str']}  initiator {t['a_ip']}:{t['a_port']}  "
                     f"cs {t['cs']:.6f} ce {t['ce']:.6f}  sched {t['sched']}  "
                     f"episode {t['episode']}")
        lines.append("   counts " + ", ".join(f"{a}={t[f'n_{a}']}" for a in arms))
        others = table[(table["key_str"] == t["key_str"]) & (table["conv"] != t["conv"])]
        if len(others):
            lines.append("   other conversations on this key: " + "; ".join(
                f"conv {o.conv} [{o.cs:.3f}, {o.ce:.3f}]" for o in others.itertuples()))
        for x in r.itertuples():
            lines.append(f"   idx {x.idx:>8} {x.arm:15s} t0 {x.t0:.6f} t1 {x.t1:.6f} "
                         f"dur {x.t1 - x.t0:10.6f} {x.src}:{x.sport}->{x.dst}:{x.dport} "
                         f"rev {int(x.rev)} bytes {x.bytes:.0f} own {x.label}")
        ids = set(r["idx"])
        p = pairs[pairs["ia"].isin(ids)]
        for x in p.itertuples():
            lines.append(f"   pair {x.arm_a}:{x.ia} ~ {x.arm_b}:{x.ib} overlap {x.ov:.6f}")
        lines.append("")
    allpairs = set(itertools.combinations(arms, 2))
    lines.append(f"arm pairs covered: {len(covered & allpairs)} of {len(allpairs)}; "
                 f"missing: {sorted(allpairs - covered)}")
    txt = OUT / f"{tag}_cases.txt"
    txt.write_text("\n".join(lines) + "\n")
    fx.to_csv(FIX / f"handcheck_{tag}.csv", index=False)
    sel[["conv", "key_str", "cs", "sched"]].assign(reason=why).to_csv(
        FIX / f"handcheck_{tag}_cases.csv", index=False)
    print(txt)


if __name__ == "__main__":
    args = sys.argv[1:]
    focus = None
    if "--focus" in args:
        k = args.index("--focus")
        focus = args[k + 1]
        del args[k:k + 2]
    main(args[0], args[1], focus=focus)
