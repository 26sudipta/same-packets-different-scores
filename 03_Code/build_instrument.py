"""
Steps 3.2 to 3.7 and 3.9: build and freeze the instrument for one benchmark day.

Run from the workspace root:
    .venv/bin/python 03_Code/build_instrument.py CICIDS2017 Tuesday [--no-cache] [--out DIR]
    .venv/bin/python 03_Code/build_instrument.py UNSW-NB15 2015-01-22

Frozen tables (parquet, not in git, hashed in the manifest):
    04_Results/instrument/<bench>/<day>/records.parquet
    04_Results/instrument/<bench>/<day>/conversations.parquet
    04_Results/instrument/<bench>/<day>/pairs.parquet
Summaries (CSV, in git): 04_Results/phase3/<table>/<bench>_<day>.csv

Settings recorded in the manifest: T, window rule, Attempted setting, mixed
conversation rule, split seeds and test fraction, pairing tolerance.
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instrument as I
import releases as R
from manifest import write_manifest

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "04_Results" / "instrument"
SUMMARY = ROOT / "04_Results" / "phase3"

SETTINGS = {
    "T_seconds": I.DEFAULT_T,     # chosen in step 3.1, see 00_Plan/Phase3_Instrument_Notes.md
    "window_rule": "start",       # schedule window test on the conversation start
    "attempted": "keep",          # primary: Attempted conversations keep their schedule class
    "mixed_rule": "any",          # own-label variant: any attack record makes an attack conversation
    "episode_gap_seconds_unsw": 300,
    "test_fraction": I.TEST_FRACTION,
    "seeds": I.SEEDS,
    "pair_tolerance_seconds": I.PAIR_TOL,
}


def own_conversation_labels(conv, own, rule):
    """Own-label variant at conversation level: any, majority or all."""
    df = pd.DataFrame({"conv": conv, "own": own})
    df["atk"] = df["own"] != I.BENIGN
    if rule == "any":
        a = df[df["atk"]].groupby(["conv", "own"]).size().rename("n").reset_index()
        a = a.sort_values(["conv", "n", "own"], ascending=[True, False, True], kind="mergesort")
        top = a.drop_duplicates("conv").set_index("conv")["own"]
        return pd.Series(top.reindex(np.unique(conv)).fillna(I.BENIGN).to_numpy(),
                         index=np.unique(conv))
    if rule == "majority":
        a = df.groupby(["conv", "own"]).size().rename("n").reset_index()
        a["b"] = a["own"] == I.BENIGN
        a = a.sort_values(["conv", "n", "b", "own"], ascending=[True, False, True, True],
                          kind="mergesort")
        return a.drop_duplicates("conv").set_index("conv")["own"]
    if rule == "all":
        g = df.groupby("conv")
        allatk = g["atk"].all()
        a = df[df["atk"]].groupby(["conv", "own"]).size().rename("n").reset_index()
        a = a.sort_values(["conv", "n", "own"], ascending=[True, False, True], kind="mergesort")
        top = a.drop_duplicates("conv").set_index("conv")["own"].reindex(allatk.index)
        return pd.Series(np.where(allatk, top, I.BENIGN), index=allatk.index)
    raise ValueError(rule)


def episodes(table, bench, gap):
    """Episode id per attack conversation (-1 for benign and excluded)."""
    atk = ~table["sched"].isin([I.BENIGN, I.EXCLUDED])
    t = table[atk].copy()
    lo = np.where(t["a_ip"] <= t["b_ip"], t["a_ip"], t["b_ip"])
    hi = np.where(t["a_ip"] <= t["b_ip"], t["b_ip"], t["a_ip"])
    t["pair"] = pd.Series(lo, index=t.index) + "~" + pd.Series(hi, index=t.index)
    if bench == "CICIDS2017":
        t["w"] = t["sched_window"].astype(int)
    else:
        t = t.sort_values(["sched", "pair", "cs", "conv"], kind="mergesort")
        cm = t.groupby(["sched", "pair"])["ce"].cummax()
        prev = cm.groupby([t["sched"], t["pair"]]).shift(1)
        t["w"] = (t["cs"] > prev + gap).groupby([t["sched"], t["pair"]]).cumsum()
    t = t.sort_values(["sched", "pair", "w", "conv"], kind="mergesort")
    eid = t.groupby(["sched", "pair", "w"], sort=True).ngroup()
    out = pd.Series(-1, index=table.index, dtype=np.int64)
    out.loc[eid.index] = eid.to_numpy()
    return out.to_numpy()


def build(bench, day, use_cache=True, frozen=FROZEN, summary=SUMMARY, settings=SETTINGS,
          pairs=True):
    t_start = time.time()
    cfg = I.BENCHMARKS[bench]
    arms = cfg["arms"]
    rec = I.load_day(bench, day, use_cache)
    T = settings["T_seconds"]
    conv, table, stats = I.build_conversations(rec, T, cfg["defining"])
    rec["conv"] = conv
    rec["rev"] = I.reversed_flag(rec, conv, table)

    # 3.2 schedule ground truth
    if bench == "CICIDS2017":
        lab, rid, wid = I.label_cic(table, day, settings["window_rule"])
        alt = I.label_cic(table, day, "overlap")[0]
        table["sched"], table["sched_rule"], table["sched_window"] = lab, rid, wid
        table["sched_alt_window"] = alt
        table["gt_categories"] = 0
    else:
        gt = I.load_unsw_gt()
        lab, ncat, attacker, victim = I.label_unsw(table, day, gt)
        table["sched"], table["sched_rule"], table["sched_window"] = lab, -1, -1
        table["sched_alt_window"] = lab
        table["gt_categories"] = ncat
    own, own_att = I.canonical(bench, rec["label"].to_numpy(dtype=object))
    rec["own"], rec["own_att"] = own, own_att
    rec["sched"] = table["sched"].to_numpy()[conv]

    # 3.5 Attempted switch (from WTMC-2021, the only arm that marks attempts)
    attack = ~table["sched"].isin([I.BENIGN, I.EXCLUDED])
    if "wtmc2021" in arms:
        w = rec[rec["arm"] == "wtmc2021"]
        watk = w["own"] != I.BENIGN
        n_att = (watk & w["own_att"]).groupby(w["conv"]).sum()
        n_full = (watk & ~w["own_att"]).groupby(w["conv"]).sum()
        only = (n_att > 0) & (n_full == 0)
        table["wtmc_attempted_only"] = table["conv"].map(only).fillna(False).astype(bool)
    else:
        table["wtmc_attempted_only"] = False
    fold = attack & table["wtmc_attempted_only"]
    table["sched_attempted_fold"] = np.where(fold, I.BENIGN, table["sched"])
    table["sched_attempted_separate"] = np.where(fold, table["sched"] + "_attempted",
                                                 table["sched"])

    # 3.6 episodes
    table["episode"] = episodes(table, bench, settings["episode_gap_seconds_unsw"])

    # 3.7 splits
    uid = I.conv_uid(bench, day, table)
    scored = table["sched"].to_numpy() != I.EXCLUDED
    for s in settings["seeds"]:
        g = np.full(len(table), -1, dtype=np.int8)
        g[scored] = I.stratified_hash_split(uid[scored], table["sched"].to_numpy()[scored], s,
                                            settings["test_fraction"])
        table[f"g{s}"] = g
        rec[f"g{s}"] = g[conv]
    rid_str = (rec["arm"].astype(str) + "|" + rec["file"].astype(str) + "|" +
               rec["row"].astype(str)).to_numpy()
    rsc = rec["sched"].to_numpy() != I.EXCLUDED
    arm_arr = rec["arm"].astype(str).to_numpy()
    for s in settings["seeds"]:
        r = np.full(len(rec), -1, dtype=np.int8)
        for a in arms:
            m = rsc & (arm_arr == a)
            r[m] = I.stratified_hash_split(rid_str[m], rec["sched"].to_numpy()[m], s,
                                           settings["test_fraction"])
        rec[f"r{s}"] = r

    # summaries -----------------------------------------------------------
    out = {}
    # 3.1 conversation and direction facts
    rows = []
    for a in arms:
        m = arm_arr == a
        rows.append({"arm": a, "records": int(m.sum()),
                     "conversations_present": int((table[f"n_{a}"] > 0).sum()),
                     "reversed_records": int(rec["rev"].to_numpy()[m].sum()),
                     "reversed_share": float(rec["rev"].to_numpy()[m].mean()) if m.any() else np.nan})
    d = pd.DataFrame(rows)
    d["nondef_mapped"], d["nondef_unmapped"], d["nondef_ambiguous"] = \
        stats["n_mapped"], stats["n_unmapped"], stats["n_ambiguous"]
    out["arm_facts"] = d

    # 3.2 own label against schedule label
    conf = rec.groupby(["arm", "sched", "own"], observed=True).size().rename("records").reset_index()
    out["label_confusion"] = conf
    agg = rec.assign(agree=rec["own"] == rec["sched"]) \
        .groupby(["arm", "sched"], observed=True)["agree"].agg(["size", "sum"]).reset_index()
    agg.columns = ["arm", "sched", "records", "own_agrees"]
    agg["agree_share"] = (agg["own_agrees"] / agg["records"]).round(6)
    convs = []
    for a in arms:
        ra = rec[arm_arr == a]
        own_any = own_conversation_labels(ra["conv"].to_numpy(), ra["own"].to_numpy(), "any")
        s_ = table.set_index("conv")["sched"].reindex(own_any.index)
        cc = pd.DataFrame({"sched": s_.to_numpy(), "agree": own_any.to_numpy() == s_.to_numpy()})
        cc = cc.groupby("sched")["agree"].agg(["size", "sum"]).reset_index()
        cc.columns = ["sched", "conversations", "own_any_agrees"]
        cc.insert(0, "arm", a)
        convs.append(cc)
    cc = pd.concat(convs)
    out["label_agreement"] = agg.merge(cc, on=["arm", "sched"], how="outer")
    out["label_agreement"]["conv_agree_share"] = (
        out["label_agreement"]["own_any_agrees"] / out["label_agreement"]["conversations"]).round(6)
    # window rule sensitivity
    out["window_rule_sensitivity"] = pd.crosstab(table["sched"], table["sched_alt_window"]) \
        .stack().rename("conversations").reset_index() \
        .rename(columns={"sched": "start_rule", "sched_alt_window": "overlap_rule"}) \
        .query("conversations > 0")

    # 3.3 cardinality per arm pair and class
    cards = []
    for a, b in itertools.combinations(arms, 2):
        c = I.cardinality(table, a, b)
        x = pd.DataFrame({"sched": table["sched"], "card": c})
        x = x[x["card"] != "none"].groupby(["sched", "card"]).size().rename("conversations").reset_index()
        x.insert(0, "arm_b", b)
        x.insert(0, "arm_a", a)
        cards.append(x)
    out["cardinality"] = pd.concat(cards, ignore_index=True)
    pres = table.groupby("sched").agg(conversations=("conv", "size"), in_all=("in_all", "sum"),
                                      in_all_defining=("in_all_defining", "sum"))
    for a in arms:
        pres[f"present_{a}"] = (table[f"n_{a}"] > 0).groupby(table["sched"]).sum()
        pres[f"records_{a}"] = table[f"n_{a}"].groupby(table["sched"]).sum()
    out["presence"] = pres.reset_index()

    # 3.5 attempted and mixed
    att = table[attack].groupby("sched").agg(
        conversations=("conv", "size"),
        with_wtmc=("n_wtmc2021", lambda s: int((s > 0).sum())) if "wtmc2021" in arms
        else ("conv", lambda s: 0),
        attempted_only=("wtmc_attempted_only", "sum")).reset_index()
    out["attempted"] = att
    mixed = []
    for a in arms:
        ra = rec[arm_arr == a]
        isatk = ra["own"] != I.BENIGN
        g = pd.DataFrame({"conv": ra["conv"], "atk": isatk, "own": ra["own"]})
        n_atk = g.groupby("conv")["atk"].sum()
        n_all = g.groupby("conv").size()
        top = own_conversation_labels(ra["conv"].to_numpy(), ra["own"].to_numpy(), "any")
        df = pd.DataFrame({"own_any": top, "n_atk": n_atk, "n_all": n_all})
        df = df[df["n_atk"] > 0]
        df["mixed"] = df["n_atk"] < df["n_all"]
        maj = own_conversation_labels(ra["conv"].to_numpy(), ra["own"].to_numpy(), "majority")
        alll = own_conversation_labels(ra["conv"].to_numpy(), ra["own"].to_numpy(), "all")
        df["majority_attack"] = maj.reindex(df.index) != I.BENIGN
        df["all_attack"] = alll.reindex(df.index) != I.BENIGN
        x = df.groupby("own_any").agg(attack_conversations=("n_all", "size"),
                                      mixed=("mixed", "sum"),
                                      attack_under_majority=("majority_attack", "sum"),
                                      attack_under_all=("all_attack", "sum")).reset_index()
        x.insert(0, "arm", a)
        mixed.append(x)
    out["mixed"] = pd.concat(mixed, ignore_index=True).rename(columns={"own_any": "own_class"})

    # 3.6 episode counts per class and arm
    ep = table[table["episode"] >= 0]
    rows = []
    for cls, g in ep.groupby("sched"):
        r_ = {"sched": cls, "episodes": g["episode"].nunique(), "conversations": len(g)}
        for a in arms:
            r_[f"episodes_seen_{a}"] = g.loc[g[f"n_{a}"] > 0, "episode"].nunique()
        rows.append(r_)
    out["episodes"] = pd.DataFrame(rows)

    # 3.7 leakage per arm, split kind, seed and class
    rows = []
    for a in arms:
        ra = rec[(arm_arr == a) & rsc]
        for kind in ("g", "r"):
            for s in settings["seeds"]:
                col = ra[f"{kind}{s}"]
                f = pd.DataFrame({"conv": ra["conv"], "sched": ra["sched"],
                                  "test": col == 1, "train": col == 0})
                h = f.groupby("conv").agg(sched=("sched", "first"), test=("test", "any"),
                                          train=("train", "any"),
                                          test_records=("test", "sum"))
                h = h[h["test"]]
                h["leak"] = h["train"]
                for cls, gg in h.groupby("sched"):
                    rows.append({"arm": a, "split": "grouped" if kind == "g" else "random_flow",
                                 "seed": s, "sched": cls, "test_conversations": len(gg),
                                 "test_records": int(gg["test_records"].sum()),
                                 "leaky_conversations": int(gg["leak"].sum()),
                                 "leakage_share": round(float(gg["leak"].mean()), 6)})
    out["leakage"] = pd.DataFrame(rows)

    # 3.3 record pairs
    pair_frames, pair_stats = [], []
    if pairs:
        for a, b in itertools.combinations(arms, 2):
            p, st = I.pair_arms(rec, conv, a, b)
            pair_frames.append(p)
            st.update(arm_a=a, arm_b=b, pairs=len(p))
            pair_stats.append(st)
            print("pairs", a, b, len(p), flush=True)
    pairs_df = pd.concat(pair_frames, ignore_index=True) if pair_frames else \
        pd.DataFrame(columns=["arm_a", "arm_b", "ia", "ib", "ov"])
    pairs_df["arm_a"] = pd.Categorical(pairs_df["arm_a"], categories=arms)
    pairs_df["arm_b"] = pd.Categorical(pairs_df["arm_b"], categories=arms)
    ps = pd.DataFrame(pair_stats)
    if len(ps):
        cnt = rec["arm"].astype(str).value_counts()
        ps["records_a"] = ps["arm_a"].map(cnt)
        ps["records_b"] = ps["arm_b"].map(cnt)
        ps["paired_share_a"] = (ps["pairs"] / ps["records_a"]).round(6)
        ps["paired_share_b"] = (ps["pairs"] / ps["records_b"]).round(6)
    out["pair_stats"] = ps

    # 3.9 freeze ---------------------------------------------------------
    fdir = frozen / bench / day
    fdir.mkdir(parents=True, exist_ok=True)
    keep = ["idx", "arm", "file", "row", "conv", "rev", "src", "sport", "dst", "dport",
            "proto", "t0", "t1", "bytes", "label", "own", "own_att", "sched"] + \
        [f"g{s}" for s in settings["seeds"]] + [f"r{s}" for s in settings["seeds"]]
    r_out = rec[keep].copy()
    for c in ("own", "sched", "label"):
        r_out[c] = r_out[c].astype(str)
    t_out = table.copy()
    t_out["key_str"] = t_out["key_str"].astype(str)
    paths = {"records": fdir / "records.parquet", "conversations": fdir / "conversations.parquet",
             "pairs": fdir / "pairs.parquet"}
    r_out.to_parquet(paths["records"], index=False)
    t_out.to_parquet(paths["conversations"], index=False)
    pairs_df.to_parquet(paths["pairs"], index=False)
    csvs = []
    for name, df in out.items():
        p = summary / name / f"{bench}_{day}.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        csvs.append(p)
    inputs = []
    for a in arms:
        fl = I.files_for(a, day) or R.files(a)
        inputs += fl
    if bench == "UNSW-NB15":
        inputs.append(R.DATA / "UNSW_NB15_Original" / "NUSW-NB15_GT.csv")
    mf = write_manifest(f"instrument_{bench}_{day}", inputs=inputs,
                        outputs=list(paths.values()) + csvs,
                        params={"benchmark": bench, "day": day, "settings": settings,
                                "use_cache": use_cache, "cache_version": I.CACHE_VERSION,
                                "frozen_dir": str(fdir),
                                "seconds": round(time.time() - t_start, 1)},
                        out_dir=summary / "manifests")
    print("records", len(rec), "conversations", len(table), "pairs", len(pairs_df),
          "seconds", round(time.time() - t_start, 1), mf, flush=True)
    return rec, table, pairs_df, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("day")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=None, help="frozen table root (default 04_Results/instrument)")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--no-pairs", action="store_true")
    a = ap.parse_args()
    build(a.bench, a.day, use_cache=not a.no_cache,
          frozen=Path(a.out) if a.out else FROZEN,
          summary=Path(a.summary) if a.summary else SUMMARY, pairs=not a.no_pairs)
