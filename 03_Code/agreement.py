"""
Phase 4 (RQ1, agreement), steps 4.1, 4.2 and the evidence for 4.4. Reads the
frozen instrument and writes the agreement tables.

Run from the workspace root:
    .venv/bin/python 03_Code/agreement.py [--boot 1000]

Inputs: 04_Results/instrument/<bench>/<day>/{conversations,records,pairs}.parquet
Outputs, 04_Results/phase4/:
    agreement_matrix.csv  4.1, per benchmark, day, class and arm pair
    fpc_hist.csv          conversations by number of records, per arm (the raw counts)
    flows_per_conv.csv    4.1 and 4.3, summary of fpc_hist per arm
    alpha.csv             4.2, Krippendorff's alpha over all arms
    alpha_pairs.csv       4.2, the same for every pair of arms
    split_causes.csv      4.4, how each extra record of a conversation follows the one before
    arm_durations.csv     4.4, record duration quantiles per arm (the observed duration cap)

Day "ALL" pools the days of a benchmark. Class "ALL" pools every scored class,
"ATTACK" every attack class; "MACRO_ATTACK" (flows_per_conv only) is the
unweighted mean over attack classes. Conversations labelled excluded (the
CICIDS2017 drop rules) are left out of everything. Variant "unsw_dups_dropped"
repeats every UNSW-NB15 table without the exact duplicate records of the
original CSVs (Phase 3 hand check rule; identical on five tuple, start, end,
bytes and label, which is a subset of the 49 fields).

Definitions are in 00_Plan/Phase4_Agreement_Notes.md.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instrument as I
import releases as R
from manifest import write_manifest

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "04_Results" / "instrument"
OUT = ROOT / "04_Results" / "phase4"
CARDS = ["1:1", "1:n", "n:1", "n:m", "a_only", "b_only"]
HIST_CAP = 4          # flows per conversation shown as 1, 2, 3, 4+
TAIL_DUR = 0.01       # 4.4: a record of at most 10 ms after the previous one ended
GAP_SLACK = 1.0       # 4.4: "right after" means within 1 s plus the clock resolution
CAP_MAX = 130.0      # 4.4: longest record length still read as an active timeout
SEED = 0
REC_COLS = ["idx", "arm", "file", "row", "conv", "rev", "src", "sport", "dst", "dport", "proto",
            "t0", "t1", "bytes", "label", "own", "own_att", "sched"]


# ---------------------------------------------------------------------------
# loading

def load(bench, day):
    d = FROZEN / bench / day
    t = pd.read_parquet(d / "conversations.parquet")
    rec = pd.read_parquet(d / "records.parquet", columns=REC_COLS)
    pairs = pd.read_parquet(d / "pairs.parquet")
    rec["arm"] = rec["arm"].astype(str)
    pairs["arm_a"] = pairs["arm_a"].astype(str)
    pairs["arm_b"] = pairs["arm_b"].astype(str)
    assert (rec["idx"].to_numpy() == np.arange(len(rec))).all()
    return t, rec, pairs


_LINE_HASH = {}


def unsw_line_hashes():
    """Hash of every raw line of the UNSW-NB15 original CSVs, indexed like the
    instrument's `row` (position after the loader drops a repeated first line)."""
    if not _LINE_HASH:
        for f in R.files("unsw_original"):
            with open(f, "rb") as fh:
                lines = fh.read().decode("latin-1").splitlines()
            if R.unsw_repeats_previous_last_line(f):
                lines = lines[1:]
            _LINE_HASH[f.name] = pd.util.hash_pandas_object(
                pd.Series(lines, dtype=object), index=False).to_numpy()
    return _LINE_HASH


def drop_unsw_dups(t, rec, pairs):
    """The same day without the exact duplicate records of the UNSW-NB15 original
    CSVs: a record whose raw line (all 49 fields) equals an earlier record's line."""
    o = (rec["arm"] == "unsw_original").to_numpy()
    lh = unsw_line_hashes()
    sub = rec.loc[o, ["file", "row"]]
    h = np.empty(len(sub), dtype=np.uint64)
    for f, g in sub.groupby(sub["file"].astype(str), observed=True):
        h[np.flatnonzero(sub["file"].astype(str).to_numpy() == f)] = lh[f][g["row"].to_numpy()]
    dup = np.zeros(len(rec), dtype=bool)
    dup[o] = pd.Series(h).duplicated(keep="first").to_numpy()
    rec2 = rec[~dup].copy()
    n = rec2[rec2["arm"] == "unsw_original"].groupby("conv").size()
    t2 = t.copy()
    t2["n_unsw_original"] = n.reindex(t2["conv"]).fillna(0).astype(np.int32).to_numpy()
    pairs2 = pairs[~dup[pairs["ia"].to_numpy()] & ~dup[pairs["ib"].to_numpy()]]
    return t2, rec2, pairs2, int(dup.sum())


def groups(sched):
    """(class name, boolean mask) for each class, then ATTACK and ALL."""
    s = pd.Series(sched, dtype=object).to_numpy()
    atk = sorted(c for c in pd.unique(s) if c not in (I.BENIGN, I.EXCLUDED))
    out = [(c, s == c) for c in [I.BENIGN] + atk]
    out.append(("ATTACK", ~np.isin(s, [I.BENIGN, I.EXCLUDED])))
    out.append(("ALL", s != I.EXCLUDED))
    return out


# ---------------------------------------------------------------------------
# 4.1 agreement matrix (counts per day; ratios after pooling)

def matrix_counts(t, rec, pairs, arms):
    rows = []
    rec_sched = rec["sched"].to_numpy(dtype=object)
    # pairs name records by their id (`idx`); ids are positions only while no record has
    # been dropped, and variant unsw_dups_dropped drops some
    pos = pd.Index(rec["idx"].to_numpy()).get_indexer(pairs["ia"].to_numpy())
    assert (pos >= 0).all(), "a pair names a record that is not in the table"
    p_sched = rec_sched[pos]
    for cls, m in groups(t["sched"]):
        tc = t[m]
        if cls == "ATTACK":
            rm = ~np.isin(rec_sched, [I.BENIGN, I.EXCLUDED])
            pm = ~np.isin(p_sched, [I.BENIGN, I.EXCLUDED])
        elif cls == "ALL":
            rm = rec_sched != I.EXCLUDED
            pm = p_sched != I.EXCLUDED
        else:
            rm = rec_sched == cls
            pm = p_sched == cls
        flows = rec[rm].groupby("arm").size()
        npairs = pairs[pm].groupby(["arm_a", "arm_b"]).size()
        for a, b in itertools.combinations(arms, 2):
            na, nb = tc[f"n_{a}"].to_numpy(), tc[f"n_{b}"].to_numpy()
            c = I.cardinality(tc, a, b)
            r = {"class": cls, "arm_a": a, "arm_b": b,
                 "flows_a": int(flows.get(a, 0)), "flows_b": int(flows.get(b, 0)),
                 "conv_a": int((na > 0).sum()), "conv_b": int((nb > 0).sum()),
                 "conv_both": int(((na > 0) & (nb > 0)).sum()),
                 "conv_union": int(((na > 0) | (nb > 0)).sum()),
                 "flow_pairs": int(npairs.get((a, b), 0))}
            for k in CARDS:
                r[f"card_{k}"] = int((c == k).sum())
            rows.append(r)
    return pd.DataFrame(rows)


def matrix_finish(df):
    df = df.copy()
    df["flow_ratio_a_to_b"] = (df["flows_a"] / df["flows_b"]).round(4)
    df["fpc_a"] = (df["flows_a"] / df["conv_a"]).round(4)
    df["fpc_b"] = (df["flows_b"] / df["conv_b"]).round(4)
    df["conv_jaccard"] = (df["conv_both"] / df["conv_union"]).round(6)
    # pairs are one to one (checked when loading), so this is the Jaccard index of
    # the two record sets with paired records counted as the same object
    df["flow_jaccard"] = (df["flow_pairs"] /
                          (df["flows_a"] + df["flows_b"] - df["flow_pairs"])).round(6)
    for k in CARDS:
        df[f"share_{k}"] = (df[f"card_{k}"] / df["conv_union"]).round(6)
    return df


# ---------------------------------------------------------------------------
# flows per conversation

def fpc_hist(t, arms):
    rows = []
    for cls, m in groups(t["sched"]):
        tc = t[m]
        for pop, pm in (("own", np.ones(len(tc), bool)), ("common", tc["in_all"].to_numpy())):
            for a in arms:
                n = tc[f"n_{a}"].to_numpy()[pm]
                n = n[n > 0]
                vc = pd.Series(n).value_counts()
                for k, c in vc.items():
                    rows.append({"class": cls, "arm": a, "population": pop,
                                 "flows": int(k), "conversations": int(c)})
    return pd.DataFrame(rows)


def fpc_summary(h):
    keys = ["variant", "benchmark", "day", "class", "arm", "population"]
    out = []
    for k, g in h.groupby(keys, sort=False):
        g = g.groupby("flows")["conversations"].sum().sort_index()
        f, c = g.index.to_numpy(), g.to_numpy()
        conv = c.sum()
        cum = np.cumsum(c) / conv
        q = lambda p: int(f[np.searchsorted(cum, p)])
        r = dict(zip(keys, k))
        r.update(conversations=int(conv), flows=int((f * c).sum()),
                 fpc=round(float((f * c).sum() / conv), 4),
                 median=q(0.5), p90=q(0.9), p99=q(0.99), max=int(f.max()),
                 share_gt1=round(float(c[f > 1].sum() / conv), 6))
        for j in range(1, HIST_CAP):
            r[f"share_{j}"] = round(float(c[f == j].sum() / conv), 6)
        r[f"share_{HIST_CAP}plus"] = round(float(c[f >= HIST_CAP].sum() / conv), 6)
        out.append(r)
    s = pd.DataFrame(out)
    atk = s[~s["class"].isin([I.BENIGN, "ATTACK", "ALL"])]
    mk = ["variant", "benchmark", "day", "arm", "population"]
    mac = atk.groupby(mk, sort=False).agg(fpc=("fpc", "mean"), share_gt1=("share_gt1", "mean"),
                                          classes=("class", "size")).reset_index()
    mac["class"] = "MACRO_ATTACK"
    mac["fpc"] = mac["fpc"].round(4)
    mac["share_gt1"] = mac["share_gt1"].round(6)
    return pd.concat([s, mac], ignore_index=True)


# ---------------------------------------------------------------------------
# 4.2 Krippendorff's alpha, nominal, from value patterns

def _unique_rows(N):
    N = np.ascontiguousarray(N)
    v = N.view(np.dtype((np.void, N.dtype.itemsize * N.shape[1]))).ravel()
    _, first, inv = np.unique(v, return_index=True, return_inverse=True)
    return N[first], inv.ravel()


def patterns(codes, K):
    """codes: units x raters, -1 missing. Unique count vectors (units with at least
    two values) and their frequencies."""
    U = codes.shape[0]
    N = np.zeros(U * K, dtype=np.int64)
    rows = np.arange(U, dtype=np.int64) * K
    for j in range(codes.shape[1]):
        c = codes[:, j].astype(np.int64)
        ok = c >= 0
        N += np.bincount(rows[ok] + c[ok], minlength=U * K)
    N = N.reshape(U, K).astype(np.int16)
    N = N[N.sum(1) >= 2]
    if len(N) == 0:
        return np.zeros((0, K), np.int16), np.zeros(0, np.int64)
    P, inv = _unique_rows(N)
    return P, np.bincount(inv, minlength=len(P)).astype(np.int64)


def merge_patterns(parts, K):
    parts = [(P, c) for P, c in parts if len(P)]
    if not parts:
        return np.zeros((0, K), np.int16), np.zeros(0, np.int64)
    P = np.concatenate([p for p, _ in parts])
    c = np.concatenate([c for _, c in parts])
    U, inv = _unique_rows(P)
    return U, np.bincount(inv, weights=c, minlength=len(U)).astype(np.int64)


def _coinc(P):
    """Per pattern coincidence matrix, flattened: (P^T P - diag P) / (m - 1)."""
    Pf = P.astype(float)
    m = Pf.sum(1)
    O = np.einsum("uk,ul->ukl", Pf, Pf)
    idx = np.arange(P.shape[1])
    O[:, idx, idx] -= Pf
    return (O / (m - 1)[:, None, None]).reshape(len(P), -1)


def _alpha_from_O(Oflat, K):
    O = Oflat.reshape(-1, K, K)
    nc = O.sum(2)
    n = nc.sum(1)
    tr = np.trace(O, axis1=1, axis2=2)
    de = n * n - (nc ** 2).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = 1.0 - (n - 1.0) * (n - tr) / de
    a[de <= 0] = np.nan
    return a, tr / n, n


def alpha(P, cnt, K, boot=0, rng=None):
    """Nominal alpha, observed agreement, pairable values, units, bootstrap 95% interval.
    The interval resamples units (conversations) with replacement."""
    r = {"units": int(cnt.sum()), "patterns": int(len(P))}
    if len(P) == 0:
        return {**r, "alpha": np.nan, "observed_agreement": np.nan, "pairable_values": 0,
                "alpha_lo": np.nan, "alpha_hi": np.nan}
    Of = _coinc(P)
    a, ao, n = _alpha_from_O((cnt[None, :].astype(float) @ Of), K)
    r.update(alpha=float(a[0]), observed_agreement=float(ao[0]), pairable_values=float(n[0]))
    lo = hi = np.nan
    if boot and not np.isnan(a[0]):
        U = int(cnt.sum())
        pr = cnt / U
        res = []
        for b0 in range(0, boot, 100):
            W = rng.multinomial(U, pr, size=min(100, boot - b0)).astype(float)
            res.append(_alpha_from_O(W @ Of, K)[0])
        res = np.concatenate(res)
        lo, hi = np.nanpercentile(res, [2.5, 97.5])
    r.update(alpha_lo=float(lo), alpha_hi=float(hi))
    return r


def own_conv_labels(rec, variant):
    """Arm's own label per (arm, conversation), "any" rule (as build_instrument.py):
    the most frequent attack class among its attack records, benign if none."""
    r = rec[["arm", "conv", "own", "own_att"]].copy()
    own = r["own"].to_numpy(dtype=object).copy()
    if variant == "attempted_benign":
        own[r["own_att"].to_numpy()] = I.BENIGN
    if variant == "binary":
        own = np.where(own == I.BENIGN, I.BENIGN, "attack").astype(object)
    r["own"] = own
    a = r[r["own"] != I.BENIGN].groupby(["arm", "conv", "own"]).size().rename("n").reset_index()
    a = a.sort_values(["arm", "conv", "n", "own"], ascending=[True, True, False, True],
                      kind="mergesort").drop_duplicates(["arm", "conv"])
    base = r[["arm", "conv"]].drop_duplicates()
    base = base.merge(a[["arm", "conv", "own"]], on=["arm", "conv"], how="left")
    base["own"] = base["own"].fillna(I.BENIGN)
    return base


def code_matrix(t, arms, values=None, cats=None):
    """units x raters code matrix in t's row order. values: None for presence,
    else a frame (arm, conv, own)."""
    if values is None:
        return np.stack([(t[f"n_{a}"].to_numpy() > 0).astype(np.int16) for a in arms], 1)
    pos = pd.Series(np.arange(len(t)), index=t["conv"].to_numpy())
    C = np.full((len(t), len(arms)), -1, dtype=np.int16)
    cmap = {c: i for i, c in enumerate(cats)}
    for j, a in enumerate(arms):
        v = values[values["arm"] == a]
        p = pos.reindex(v["conv"].to_numpy()).to_numpy()
        ok = ~np.isnan(p)
        C[p[ok].astype(int), j] = v["own"].map(cmap).to_numpy()[ok]
    return C


LABEL_VARIANTS = ["multiclass", "binary", "attempted_benign"]


def alpha_parts(t, rec, arms, cats):
    """Pattern sets per (question, label variant, population, class, rater set)."""
    t = t[t["sched"] != I.EXCLUDED].reset_index(drop=True)
    rec = rec[rec["sched"] != I.EXCLUDED]
    out = {}
    grp = groups(t["sched"])
    pres = code_matrix(t, arms)
    labs = {v: code_matrix(t, arms, own_conv_labels(rec, v), cats) for v in LABEL_VARIANTS}
    inall = t["in_all"].to_numpy()
    raters = [("all", list(range(len(arms))))] + \
             [(f"{arms[i]}|{arms[j]}", [i, j]) for i, j in itertools.combinations(range(len(arms)), 2)]
    for rname, cols in raters:
        sub = pres[:, cols]
        union = sub.sum(1) > 0
        for cls, m in grp:
            out[("existence", "presence", "all", cls, rname)] = patterns(sub[m & union], 2)
            for v in LABEL_VARIANTS:
                L = labs[v][:, cols]
                out[("labels", v, "present_in_2plus", cls, rname)] = patterns(L[m], len(cats))
                if rname == "all":
                    out[("labels", v, "common", cls, rname)] = patterns(L[m & inall], len(cats))
    return out


# ---------------------------------------------------------------------------
# 4.4 how a conversation is cut

def arm_durations(rec):
    rows = []
    for a, g in rec.groupby("arm"):
        d = (g["t1"] - g["t0"]).to_numpy() - R.RESOLUTION[a]
        rows.append({"arm": a, "records": len(d), "p50": np.quantile(d, 0.5),
                     "p99": np.quantile(d, 0.99), "p999": np.quantile(d, 0.999),
                     "max": d.max()})
    return pd.DataFrame(rows)


def split_counts(rec, caps):
    """For every record after the first in its conversation (per arm, in start order),
    which of six cases describes how it follows the record before it:
      repeat   same start and same duration as the previous record (to 1 microsecond)
      tail     at most TAIL_DUR long and starting within GAP_SLACK + r of the previous end
      cut      starts at least cap - 1 s after the previous record started, the arm's
               duration cap being its active timeout
      overlap  starts before the previous record ended, beyond the clock resolution
      idle     starts more than GAP_SLACK + r after the previous end
      other    none of these"""
    rows = []
    for a, g in rec[rec["sched"] != I.EXCLUDED].groupby("arm"):
        r = R.RESOLUTION[a]
        g = g.sort_values(["conv", "t0", "t1"], kind="mergesort")
        conv = g["conv"].to_numpy()
        t0 = g["t0"].to_numpy()
        dur = (g["t1"].to_numpy() - t0) - r
        end = t0 + dur
        nxt = np.r_[False, conv[1:] == conv[:-1]]       # record has a predecessor
        i = np.nonzero(nxt)[0]
        gap = t0[i] - end[i - 1]
        near = gap <= GAP_SLACK + r
        rep = (np.abs(t0[i] - t0[i - 1]) <= 1e-6) & (np.abs(dur[i] - dur[i - 1]) <= 1e-6)
        tail = ~rep & (dur[i] <= TAIL_DUR) & near & (gap >= -r)
        cap = caps.get(a)
        # the previous record's window reached the cap: this one starts at least
        # cap - 1 s after it (its last packet may have come well before the cap)
        hit = (t0[i] - t0[i - 1] >= cap - 1.0) if cap is not None else np.zeros(len(i), bool)
        cut = ~rep & ~tail & hit & (gap >= -r)
        ovl = ~rep & ~tail & ~cut & (gap < -r)
        idle = ~rep & ~tail & ~cut & ~ovl & (gap > GAP_SLACK + r)
        other = ~(rep | tail | cut | ovl | idle)
        sched = g["sched"].to_numpy(dtype=object)[i]
        rev = g["rev"].to_numpy()[i]
        df = pd.DataFrame({"sched": sched, "repeat": rep, "tail": tail, "cut": cut,
                           "overlap": ovl, "idle": idle, "other": other, "rev": rev})
        for cls, m in groups(df["sched"]):
            sd = df[m]
            row = {"class": cls, "arm": a, "extra_records": len(sd),
                   "split_conversations": int(pd.Series(conv[i][m]).nunique()),
                   "extra_reversed": int(sd["rev"].sum())}
            for k in ["repeat", "tail", "cut", "overlap", "idle", "other"]:
                row[k] = int(sd[k].sum())
            rows.append(row)
    return pd.DataFrame(rows)


def split_finish(df):
    df = df.copy()
    for k in ["repeat", "tail", "cut", "overlap", "idle", "other", "extra_reversed"]:
        df[f"share_{k}"] = (df[k] / df["extra_records"]).round(6)
    return df


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--bench", nargs="*", default=list(I.BENCHMARKS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    inputs, mats, hists, splits, durs, alphas, alpha_pairs, dupinfo = [], [], [], [], [], [], [], []

    for bench in a.bench:
        cfg = I.BENCHMARKS[bench]
        arms, days = cfg["arms"], cfg["days"]
        variants = ["all"] + (["unsw_dups_dropped"] if bench == "UNSW-NB15" else [])
        # first pass: duration caps and the label categories of the benchmark
        drows, cats = [], {I.BENIGN, "attack"}
        for day in days:
            rec = pd.read_parquet(FROZEN / bench / day / "records.parquet",
                                  columns=["arm", "t0", "t1", "own"])
            rec["arm"] = rec["arm"].astype(str)
            drows.append(arm_durations(rec).assign(day=day))
            cats |= set(pd.unique(rec["own"]))
        dd = pd.concat(drows)
        # an arm whose longest record is at most CAP_MAX long is read as cutting at that
        # length (active timeout); an arm with longer records has no visible cap
        caps = {k: (v if v <= CAP_MAX else None)
                for k, v in dd.groupby("arm")["max"].max().to_dict().items()}
        dd.insert(0, "benchmark", bench)
        durs.append(dd)
        cats = sorted(cats)
        K = len(cats)
        pats = {v: {} for v in variants}
        for day in days:
            d = FROZEN / bench / day
            inputs += [d / "conversations.parquet", d / "records.parquet", d / "pairs.parquet"]
            t0_, rec0, pairs0 = load(bench, day)
            for v in variants:
                if v == "unsw_dups_dropped":
                    t, rec, pairs, nd = drop_unsw_dups(t0_, rec0, pairs0)
                    dupinfo.append({"benchmark": bench, "day": day, "dropped_records": nd})
                else:
                    t, rec, pairs = t0_, rec0, pairs0
                tag = dict(variant=v, benchmark=bench, day=day)
                mats.append(matrix_counts(t, rec, pairs, arms).assign(**tag))
                hists.append(fpc_hist(t, arms).assign(**tag))
                splits.append(split_counts(rec, caps).assign(**tag))
                for k, pc in alpha_parts(t, rec, arms, cats).items():
                    pats[v].setdefault(k, []).append((day, pc))
                print(bench, day, v, "done", flush=True)
        for v in variants:
            for k, parts in pats[v].items():
                q, lv, pop, cls, rname = k
                Kq = 2 if q == "existence" else K
                for day, pc in parts + [("ALL", merge_patterns([p for _, p in parts], Kq))]:
                    boot = a.boot if (rname == "all" and cls in ("ALL", "ATTACK", I.BENIGN)) or \
                        (rname == "all" and day == "ALL") else 0
                    r = alpha(pc[0], pc[1], Kq, boot=boot, rng=rng)
                    r.update(variant=v, benchmark=bench, day=day, question=q, label_variant=lv,
                             population=pop, **{"class": cls}, raters=rname)
                    (alphas if rname == "all" else alpha_pairs).append(r)

    lead = ["variant", "benchmark", "day"]

    def pooled(frames, keys, finish):
        df = pd.concat(frames, ignore_index=True)
        num = [c for c in df.columns if c not in lead + keys]
        allday = df.groupby(["variant", "benchmark"] + keys, sort=False)[num].sum().reset_index()
        allday["day"] = "ALL"
        df = pd.concat([df, allday], ignore_index=True)
        return finish(df[lead + keys + num]) if finish else df[lead + keys + num]

    outs = {}
    m = pooled(mats, ["class", "arm_a", "arm_b"], matrix_finish)
    outs["agreement_matrix"] = m
    h = pooled(hists, ["class", "arm", "population", "flows"], None)
    outs["fpc_hist"] = h
    outs["flows_per_conv"] = fpc_summary(h)
    outs["split_causes"] = pooled(splits, ["class", "arm"], split_finish)
    outs["arm_durations"] = pd.concat(durs, ignore_index=True).round(6)
    cols = ["variant", "benchmark", "day", "question", "label_variant", "population", "class",
            "raters", "units", "patterns", "pairable_values", "observed_agreement", "alpha",
            "alpha_lo", "alpha_hi"]
    outs["alpha"] = pd.DataFrame(alphas)[cols].round(6)
    outs["alpha_pairs"] = pd.DataFrame(alpha_pairs)[cols].drop(columns=["alpha_lo", "alpha_hi"]).round(6)
    if dupinfo:
        outs["unsw_dups"] = pd.DataFrame(dupinfo)
    paths = []
    for name, df in outs.items():
        p = OUT / f"{name}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    mf = write_manifest("phase4_agreement", inputs=inputs, outputs=paths,
                        params={"boot": a.boot, "seed": SEED, "tail_dur_s": TAIL_DUR,
                                "gap_slack_s": GAP_SLACK, "cap_max_s": CAP_MAX, "hist_cap": HIST_CAP,
                                "benchmarks": a.bench},
                        out_dir=OUT / "manifests")
    print("wrote", len(paths), "tables; manifest", mf)


if __name__ == "__main__":
    main()
