"""
Phase 11, review Major comment 4: how much does the conversation set depend on
which releases define it? The instrument pools the records of the defining arms
(every arm except CIC original, which is mapped afterwards). This rebuilds the
conversations of every day with the frozen code (instrument.build_conversations,
T = 15 s) under these variants:

    drop:<arm>      one defining arm left out of the pool, its records removed
    +cic_original   CICIDS2017 only: CIC original also defines conversations

and compares each with the frozen instrument on the records both contain. Two
partitions of the same records agree on a conversation when it holds exactly the
same records in both. Per benchmark, variant and schedule class (frozen label):

    frozen        frozen conversations with at least one record in the variant
    identical     of those, conversations the variant reproduces record for record
    n_variant     conversations of the variant (classes from the frozen label of
                  their first record)
    jaccard       identical / (frozen + n_variant - identical)
    relabelled    variant conversations whose schedule label differs from the
                  frozen label of their first record

It also writes, for the records of the Phase 5 subsample, the variant's
conversation and label, so that stored predictions can be scored on the
variant's conversations without retraining (phase11_analysis.py).

Usage, from the workspace root, NIDS_CACHE_ROOT set (phase11_common):
    .venv/bin/python 03_Code/phase11_instrument_loo.py CICIDS2017
Outputs: 04_Results/phase11/loo/instrument_loo.csv (one row per benchmark,
variant and class, ALL for the total) and Datasets/_cache_phase11/loo/<bench>/
<variant>.parquet (day, arm, file, row, gconv_v, sched_v).
"""
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instrument as I  # noqa: E402
import phase11_common as C  # noqa: E402
import phase5_data as D  # noqa: E402
import releases as R  # noqa: E402
from manifest import write_manifest  # noqa: E402

FROZEN = C.ROOT / "04_Results" / "instrument"
OUT = C.OUT / "loo"
CACHE = R.CACHE_ROOT / "_cache_phase11" / "loo"


def variants(bench):
    cfg = I.BENCHMARKS[bench]
    v = [(f"drop:{a}", a, [d for d in cfg["defining"] if d != a]) for a in cfg["defining"]]
    if bench == "CICIDS2017":
        v.append(("+cic_original", None, cfg["defining"] + ["cic_original"]))
    return v


def subsample_keys(bench):
    """(day, file, row, arm) of every record in the Phase 5 subsample tables."""
    keys = []
    src = sorted({a if not a.startswith("unsw_") else "unsw_original"
                  for a in I.BENCHMARKS[bench]["arms"]})
    for arm in src:
        p = D.CACHE / "main" / bench / f"{arm}.parquet"
        k = pd.read_parquet(p, columns=["day", "file", "row"])
        k["arm"] = arm
        keys.append(k)
    k = pd.concat(keys, ignore_index=True)
    k["day"], k["file"] = k["day"].astype(str), k["file"].astype(str)
    return k


def label(bench, table, day, gt):
    if bench == "CICIDS2017":
        return I.label_cic(table, day)[0]
    return I.label_unsw(table, day, gt)[0]


def compare(fz_conv, v_conv, fz_lab_of_conv):
    """Agreement of two partitions of the same records."""
    df = pd.DataFrame({"f": fz_conv, "v": v_conv})
    nf_per_v = df.groupby("v")["f"].nunique()
    nv_per_f = df.groupby("f")["v"].nunique()
    one_v = nf_per_v[nf_per_v == 1].index
    one_f = nv_per_f[nv_per_f == 1].index
    pairs = df.drop_duplicates("v").set_index("v")["f"]
    ident_f = set(pairs.reindex(one_v).to_numpy().tolist()) & set(one_f.tolist())
    f_cls = pd.Series(fz_lab_of_conv).reindex(np.unique(fz_conv))
    v_first = df.drop_duplicates("v").set_index("v")["f"]
    v_cls = pd.Series(fz_lab_of_conv).reindex(v_first.to_numpy())
    v_cls.index = v_first.index
    rows = []
    for cls in sorted(set(f_cls.dropna()) | {"ALL"}):
        fm = f_cls.index if cls == "ALL" else f_cls[f_cls == cls].index
        vm = v_cls.index if cls == "ALL" else v_cls[v_cls == cls].index
        idn = len(set(fm.tolist()) & ident_f)
        rows.append({"cls": cls, "frozen": len(fm), "identical": idn, "n_variant": len(vm),
                     "jaccard": idn / (len(fm) + len(vm) - idn) if len(fm) + len(vm) else np.nan})
    return pd.DataFrame(rows)


def main(bench):
    t0 = time.time()
    cfg = I.BENCHMARKS[bench]
    gt = I.load_unsw_gt() if bench == "UNSW-NB15" else None
    sub = subsample_keys(bench)
    rows, maps = [], {name: [] for name, _, _ in variants(bench)}
    for di, day in enumerate(cfg["days"]):
        rec = I.load_day(bench, day, use_cache=True)
        fr = pd.read_parquet(FROZEN / bench / day / "records.parquet",
                             columns=["arm", "file", "row", "conv"])
        ft = pd.read_parquet(FROZEN / bench / day / "conversations.parquet",
                             columns=["conv", "sched"])
        if not ((rec["arm"].astype(str).to_numpy() == fr["arm"].astype(str).to_numpy()).all()
                and (rec["row"].to_numpy() == fr["row"].to_numpy()).all()):
            raise AssertionError(f"{bench} {day}: loaded records not in frozen order")
        fz_lab = ft.set_index("conv")["sched"].to_dict()
        fz_conv = fr["conv"].to_numpy()
        s = sub[sub["day"] == day]
        for name, drop, defining in variants(bench):
            keep = (rec["arm"].astype(str) != drop).to_numpy() if drop else \
                np.ones(len(rec), dtype=bool)
            r = rec[keep].copy()
            r["arm"] = r["arm"].cat.remove_unused_categories()
            conv, table, _ = I.build_conversations(r, I.DEFAULT_T, defining)
            lab = label(bench, table, day, gt)
            vlab = pd.Series(np.asarray(lab, dtype=object), index=table["conv"].to_numpy())
            cmp_ = compare(fz_conv[keep], conv, fz_lab)
            # conversations whose label changed (first record's frozen label against new)
            first = pd.DataFrame({"v": conv, "f": fz_conv[keep]}).drop_duplicates("v")
            changed = (first["f"].map(fz_lab).to_numpy(dtype=object) !=
                       vlab.reindex(first["v"]).to_numpy(dtype=object))
            cmp_["relabelled"] = np.where(cmp_["cls"] == "ALL", int(changed.sum()), np.nan)
            cmp_["benchmark"], cmp_["variant"], cmp_["day"] = bench, name, day
            rows.append(cmp_)
            # variant conversation and label for the subsample's records
            m = pd.DataFrame({"arm": r["arm"].astype(str).to_numpy(),
                              "file": r["file"].astype(str).to_numpy(),
                              "row": r["row"].to_numpy(), "conv": conv})
            m = m.merge(s[["arm", "file", "row"]].drop_duplicates(), on=["arm", "file", "row"])
            m["gconv_v"] = di * 10 ** 8 + m["conv"].to_numpy()
            m["sched_v"] = vlab.reindex(m["conv"]).to_numpy(dtype=object)
            m["day"] = day
            maps[name].append(m.drop(columns=["conv"]))
            print(f"{bench} {day} {name}: {len(table):,} conversations, jaccard "
                  f"{cmp_.loc[cmp_['cls'] == 'ALL', 'jaccard'].iloc[0]:.5f} "
                  f"({time.time() - t0:.0f} s)", flush=True)
            del r, conv, table
            gc.collect()
        del rec, fr
        gc.collect()
    df = pd.concat(rows, ignore_index=True)
    agg = df.groupby(["benchmark", "variant", "cls"], as_index=False)[
        ["frozen", "identical", "n_variant", "relabelled"]].sum(min_count=1)
    agg["jaccard"] = agg["identical"] / (agg["frozen"] + agg["n_variant"] - agg["identical"])
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"instrument_loo_{bench}.csv"
    agg.to_csv(p, index=False)
    df.to_csv(OUT / f"instrument_loo_{bench}_by_day.csv", index=False)
    (CACHE / bench).mkdir(parents=True, exist_ok=True)
    for name, parts in maps.items():
        pd.concat(parts, ignore_index=True).to_parquet(
            CACHE / bench / f"{name.replace(':', '_')}.parquet", index=False)
    write_manifest(f"phase11_instrument_loo_{bench}", inputs=[], outputs=[p],
                   params={"T": I.DEFAULT_T, "variants": [v[0] for v in variants(bench)]})
    print(agg[agg["cls"] == "ALL"].to_string())


if __name__ == "__main__":
    main(sys.argv[1])
