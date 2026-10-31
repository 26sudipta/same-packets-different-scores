"""
Phase 8, step 8.4: redo each comparison on the defined footing: same conversations,
one ground truth (schedule labels), conversation-grouped split, conversation scoring.
Protocol: 00_Plan/Phase8_Protocol.md, section 8.4.

Usage, from the workspace root:
    .venv/bin/python 03_Code/phase8_conv.py R1|R2|R3|R4|R5 [--variant sched|fold]

Per claim, per pair of arms (A, B):
  population  the Phase 5 subsample conversations (the gconv ids held in
              Datasets/_cache_phase5/main), schedule class not excluded, with at
              least one usable record (finite features) in both A and B
  split       the instrument's grouped split g<seed> (1 = test), seeds 0 to 4
  truth       schedule label of the record's conversation (`sched`, Attempted kept as
              attack); --variant fold uses `sched_attempted_fold` (R2 and R3 only)
  model       the claim's own features, preprocessing and model, trained on the
              training conversations' records of that arm
Each cell writes the test records with their prediction to
04_Results/phase8/conv/preds/<claim>/<variant>/<pair>/<arm>/<model>_s<seed>[_<day>].parquet
(file, row, gconv, stratum, y, pred, bytes); phase8_decomp.py --stage conv scores them
per flow and per conversation (rule "any") with the paired bootstrap.

Deviations from 8.2 that the defined footing forces, stated here and in the notes:
  * no row is removed by its released label (Attempted rows of WTMC and CRiSIS in R5,
    the constant-label training rows of R3, the class quotas of R1): the truth is the
    schedule label, so a filter on the released label has no meaning here;
  * under schedule labels Exp_RD and Exp_RPD are the same records (same rows, only
    labels differ), and so are Exp_R and Exp_RP: R3 trains RPD and RP only, and the
    RD to RPD contrast (Q2) is zero by construction.
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import instrument as I  # noqa: E402
import phase8_common as P  # noqa: E402
import phase8_flow as F8  # noqa: E402
import releases as R  # noqa: E402

OUT = P.OUT / "conv" / "preds"
PHASE5 = R.CACHE_ROOT / "_cache_phase5" / "main"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# population

def conv_table(bench):
    """gconv, sched, sched_attempted_fold, g0..g4 for the Phase 5 subsample."""
    keep = set()
    for p in sorted((PHASE5 / bench).glob("*.parquet")):
        keep |= set(pd.read_parquet(p, columns=["gconv"])["gconv"].tolist())
    parts = []
    for di, day in enumerate(I.BENCHMARKS[bench]["days"]):
        cols = ["conv", "sched"] + [f"g{s}" for s in P.SEEDS]
        if bench == "CICIDS2017":
            cols.append("sched_attempted_fold")
        t = pd.read_parquet(P.FROZEN / bench / day / "conversations.parquet", columns=cols)
        t["gconv"] = di * 10 ** 8 + t["conv"].to_numpy()
        t["day"] = day
        parts.append(t.drop(columns="conv"))
    t = pd.concat(parts, ignore_index=True)
    t = t[t["gconv"].isin(keep) & (t["sched"] != I.EXCLUDED)]
    if "sched_attempted_fold" not in t.columns:
        t["sched_attempted_fold"] = t["sched"]
    log(f"{bench}: {len(t):,} subsample conversations")
    return t.set_index("gconv")


def lookup(arm):
    if arm in ("crisis2022_rd", "crisis2022_rpd"):
        return P.conv_lookup("crisis2022_rpd")
    if arm in ("crisis2022_r", "crisis2022_rp"):
        p = P.CACHE / "convmap" / "crisis2022_rp.parquet"
        if not p.is_file():
            raise SystemExit("run phase8_rp_map.py first")
        return pd.read_parquet(p)
    return P.conv_lookup(arm)


def record_bytes(arm):
    """Bytes of every instrument record of the arm (tie-break of rule "any")."""
    parts = []
    bench = P.bench_of(arm)
    for day in I.BENCHMARKS[bench]["days"]:
        parts.append(pd.read_parquet(P.FROZEN / bench / day / "records.parquet",
                                     columns=["file", "row", "bytes"],
                                     filters=[("arm", "==", arm)]))
    out = pd.concat(parts, ignore_index=True)
    out["file"] = out["file"].astype(str)
    return out


def attach(arm, meta, table):
    """meta: file, row (one per usable record). Adds gconv and the conversation
    columns; drops records outside the population."""
    lk = lookup(arm)
    m = meta.merge(lk[["file", "row", "gconv"]], on=["file", "row"], how="left",
                   validate="many_to_one")
    ok = m["gconv"].notna().to_numpy().copy()
    ok[ok] = np.isin(m.loc[ok, "gconv"].astype(np.int64).to_numpy(), table.index.to_numpy())
    m = m[ok].copy()
    m["gconv"] = m["gconv"].astype(np.int64)
    if arm in ("cic_original", "wtmc2021", "unsw_original", "nf_unsw_v3"):
        m = m.merge(record_bytes(arm), on=["file", "row"], how="left", validate="one_to_one")
    else:
        m["bytes"] = 1.0
    return m, np.flatnonzero(ok)


# ---------------------------------------------------------------------------
# features per claim: (meta with file and row, X, column names) of usable records

def feats_r1(arm, day=None):
    k = F8._module_constants(F8.AUTHORS["lycos"] / "create_datasets.py", {"FEAT_LIST_ISCX"})
    parts = []
    for p in P.files(arm):
        raw = P.read_release(arm, p)
        if arm == "cic_original":
            raw.columns = list(k["FEAT_LIST_ISCX"].keys())
        raw.insert(0, "row", np.arange(len(raw), dtype=np.int64))
        raw.insert(0, "file", p.name)
        parts.append(raw)
    df = pd.concat(parts, ignore_index=True)
    drop = [c for c in F8.R1_DROP[arm] if c in df.columns] + ["file", "row"]
    x = df.drop(columns=drop).apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    ok = np.isfinite(x).all(axis=1)
    return df.loc[ok, ["file", "row"]].reset_index(drop=True), x[ok]


def feats_r2(arm, day=None):
    df = F8.read_arm(arm)
    x, ok = F8.clean_numeric(df, F8.R2_DROP[arm])
    return df.loc[ok, ["file", "row"]].reset_index(drop=True), x[ok]


def feats_r3(arm, day=None):
    cols = F8._lanvin_cols()
    df = F8._lanvin_frame(arm, cols, F8.R3_DAYS)
    meta = df[["file", "row"]]
    x = df.drop(columns=["file", "row"])
    return meta, x   # preprocessing is fitted on the training records (tools.preprocessData)


def feats_r4(arm, day=None):
    df = F8.read_arm(arm)
    x = F8.r4_matrix(arm, df)
    return df[["file", "row"]].reset_index(drop=True), x


def feats_r5(arm, day):
    cols = F8._pekar_columns()
    names = F8.R5_CIC_FILES[day] if arm == "cic_original" else (
        [f"{day}.csv"] if arm.startswith("nfs") else [f"{day}-WorkingHours.csv"])
    df = F8.read_arm(arm, names)
    x = df[cols[arm]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    ok = np.isfinite(x).all(axis=1)
    return df.loc[ok, ["file", "row"]].reset_index(drop=True), x[ok]


# ---------------------------------------------------------------------------
# what is compared: pairs per claim

def pairs(claim):
    """[(pair name, A, B, day or None, models)]"""
    if claim == "R1":
        return [("cic_lycos", "cic_original", "lycos", None, ["lda", "dt", "rf"])]
    if claim == "R2":
        return [("cic_wtmc", "cic_original", "wtmc2021", None, ["rf"])]
    if claim == "R3":
        return [("rpd_rp", "crisis2022_rpd", "crisis2022_rp", None, ["dt", "rf", "nb"])]
    if claim == "R4":
        return [("unsw_v3", "unsw_original", "nf_unsw_v3", None, ["et_bin", "et_multi"])]
    # R5: (a) Tuesday, each refined release against CIC-IDS2017; (b) Thursday, nTE
    # against TE (the nte_te pair), CRiSIS against WTMC, nTE against CIC; (c) TE against
    # nTE on every day (nte_te)
    # (CRiSIS is Exp_RP, protocol amendment 1; the Exp_RPD pairs are the extra rows)
    out = [(f"cic_{arm}", "cic_original", arm, "Tuesday", ["rf"])
           for arm in ("wtmc2021", "crisis2022_rpd", "nfs2023_nte", "nfs2023_te")]
    out.append(("wtmc_crisis", "wtmc2021", "crisis2022_rpd", "Thursday", ["rf"]))
    if P.files("crisis2022_rp") and (P.CACHE / "convmap" / "crisis2022_rp.parquet").is_file():
        out.append(("cic_crisis2022_rp", "cic_original", "crisis2022_rp", "Tuesday", ["rf"]))
        out.append(("wtmc_crisis_rp", "wtmc2021", "crisis2022_rp", "Thursday", ["rf"]))
    out.append(("cic_nfs2023_nte", "cic_original", "nfs2023_nte", "Thursday", ["rf"]))
    out += [("nte_te", "nfs2023_nte", "nfs2023_te", d, ["rf"])
            for d in ("Tuesday", "Wednesday", "Thursday", "Friday")]
    return out


FEATS = {"R1": feats_r1, "R2": feats_r2, "R3": feats_r3, "R4": feats_r4, "R5": feats_r5}


def model_for(claim, name, seed):
    if name == "lda":
        return LinearDiscriminantAnalysis()
    if name == "dt":
        return (DecisionTreeClassifier(criterion="gini", max_depth=15, random_state=seed)
                if claim == "R3" else DecisionTreeClassifier(random_state=seed))
    if name == "rf":
        if claim == "R3":
            return RandomForestClassifier(max_depth=15, random_state=seed, n_jobs=P.N_JOBS)
        return RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=P.N_JOBS)
    if name == "nb":
        return BernoulliNB()
    if name.startswith("et_"):
        return ExtraTreesClassifier(n_estimators=100, random_state=seed, n_jobs=P.N_JOBS)
    raise ValueError(name)


def cell_path(claim, variant, pair, arm, model, seed, day):
    tail = f"_{day}" if day else ""
    return OUT / claim / variant / pair / arm / f"{model}_s{seed}{tail}.parquet"


def fit_predict(claim, name, seed, x_tr, y_tr, x_te):
    """Train on (x_tr, y_tr), predict x_te, with the claim's preprocessing."""
    if claim == "R3":
        tr = x_tr.copy()
        tr["Label"] = y_tr
        a, y, cols, _ = F8._lanvin_preprocess(tr.sample(frac=1, random_state=seed))
        te = x_te.copy()
        te["Label"] = "?"
        b, _, _, kept = F8._lanvin_preprocess(te, columns=cols)
        if len(kept) != len(te):
            raise AssertionError("R3 test records were dropped after the population was fixed")
        sc = StandardScaler().fit(a)
        m = model_for(claim, name, seed).fit(sc.transform(a), y)
        return m.predict(sc.transform(b))
    if claim == "R1":
        sc = StandardScaler().fit(x_tr)
        x_tr, x_te = sc.transform(x_tr), sc.transform(x_te)
    m = model_for(claim, name, seed)
    m.fit(x_tr, y_tr)
    return m.predict(x_te)


def r3_usable(x):
    """Rows tools.preprocessData would keep (finite in every numeric column). Text columns
    are pandas 3 str dtype, not object, so both count as text (as in _lanvin_preprocess)."""
    num = x.drop(columns=[c for c in x.columns
                          if x[c].dtype == object or pd.api.types.is_string_dtype(x[c])])
    v = num.apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    return np.isfinite(v).all(axis=1)


def run(claim, variant):
    bench = "UNSW-NB15" if claim == "R4" else "CICIDS2017"
    table = conv_table(bench)
    truth_col = "sched" if variant == "sched" else "sched_attempted_fold"
    cache = {}
    for pair, a, b, day, models in pairs(claim):
        todo = [(arm, m, s) for arm in (a, b) for m in models for s in P.SEEDS
                if not cell_path(claim, variant, pair, arm, m, s, day).is_file()]
        if not todo:
            continue
        side = {}
        for arm in (a, b):
            key = (arm, day)
            if key not in cache:
                cache.clear() if len(cache) > 2 else None
                meta, x = FEATS[claim](arm, day)
                if claim == "R3":
                    ok = r3_usable(x)
                    meta, x = meta[ok].reset_index(drop=True), x[ok].reset_index(drop=True)
                m, keep = attach(arm, meta, table)
                x = x.iloc[keep].reset_index(drop=True) if claim == "R3" else x[keep]
                cache[key] = (m.reset_index(drop=True), x)
                log(f"{claim} {arm} {day or ''}: {len(m):,} usable records in the population")
                gc.collect()
            side[arm] = cache[key]
        common = np.intersect1d(np.unique(side[a][0]["gconv"]), np.unique(side[b][0]["gconv"]))
        if day is not None:
            common = common[table.loc[common, "day"].to_numpy() == day]
        log(f"{claim} {pair} {day or ''}: {len(common):,} conversations in both arms")
        for arm in (a, b):
            m, x = side[arm]
            inpop = np.isin(m["gconv"].to_numpy(), common)
            mm = m[inpop].reset_index(drop=True)
            xx = x[inpop].reset_index(drop=True) if claim == "R3" else x[inpop]
            ct = table.loc[mm["gconv"].to_numpy()]
            y_all = ct[truth_col].to_numpy(dtype=object)
            for model in models:
                for s in P.SEEDS:
                    p = cell_path(claim, variant, pair, arm, model, s, day)
                    if p.is_file():
                        continue
                    t0 = time.time()
                    g = ct[f"g{s}"].to_numpy()
                    tr, te = np.flatnonzero(g == 0), np.flatnonzero(g == 1)
                    y_tr = y_all[tr]
                    if model == "et_bin" or claim == "R5":
                        y_tr = np.where(y_tr == P.BENIGN, P.BENIGN, "attack").astype(object)
                    x_tr = xx.iloc[tr] if claim == "R3" else xx[tr]
                    x_te = xx.iloc[te] if claim == "R3" else xx[te]
                    pred = fit_predict(claim, model, s, x_tr, y_tr, x_te)
                    out = pd.DataFrame({"file": mm["file"].to_numpy()[te].astype(str),
                                        "row": mm["row"].to_numpy()[te],
                                        "gconv": mm["gconv"].to_numpy()[te],
                                        "stratum": ct["sched"].to_numpy()[te].astype(str),
                                        "y": y_all[te].astype(str),
                                        "pred": np.asarray(pred, dtype=object).astype(str),
                                        "bytes": mm["bytes"].to_numpy(dtype=float)[te]})
                    p.parent.mkdir(parents=True, exist_ok=True)
                    tmp = p.with_suffix(".tmp")
                    out.to_parquet(tmp, index=False)
                    tmp.replace(p)
                    log(f"{claim} {variant} {pair} {arm} {model} seed {s} {day or ''}: "
                        f"{len(tr):,} train, {len(te):,} test, {time.time() - t0:.0f} s")
                    gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("claim", choices=["R1", "R2", "R3", "R4", "R5"])
    ap.add_argument("--variant", choices=["sched", "fold"], default="sched")
    a = ap.parse_args()
    if a.variant == "fold" and a.claim not in ("R2", "R3"):
        raise SystemExit("the Attempted fold sensitivity is for R2 and R3 only")
    t_start = time.time()
    run(a.claim, a.variant)
    log(f"{a.claim} {a.variant} finished in {time.time() - t_start:.0f} s")
