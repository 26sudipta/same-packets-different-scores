"""
Phase 8, step 8.2: rebuild each published comparison per flow, as the authors
framed it (protocol: 00_Plan/Phase8_Protocol.md, section 8.2).

Usage, from the workspace root:
    .venv/bin/python 03_Code/phase8_flow.py R1|R2|R3|R4|R5 [--seeds 0 1 2 3 4]

Every cell (claim, arm, model, seed[, day]) writes its test predictions to
04_Results/phase8/preds/<claim>/<arm>/<model>_s<seed>[_<day>].parquet with the
columns file, row, y (the release's own label), pred, and is skipped when the
file exists, so a restart costs one cell. Set sizes and set checks go to
04_Results/phase8/sets/<claim>_*.csv. Metrics are computed by phase8_decomp.py.

Ports of the authors' code keep their logic line by line; every change is
named where it is made. Paths to their code and data are in AUTHORS below.
"""
import argparse
import ast
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_common as P  # noqa: E402
import releases as R  # noqa: E402

PREDS = P.OUT / "preds"
SETS = P.OUT / "sets"
AUTHORS = {
    "lycos": R.DATA / "LYCOS-IDS2017_GitHubMirror",
    "pekar": R.DATA / "_downloads" / "CyberML-DataQuality",
    "lanvin": R.DATA / "_downloads" / "crisis2022_extra" / "repo" / "Performance_Evaluation",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cell_path(claim, arm, model, seed, day=None):
    tail = f"_{day}" if day else ""
    return PREDS / claim / arm / f"{model}_s{seed}{tail}.parquet"


def save_cell(path, test_meta, y, pred):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"file": test_meta["file"].astype(str).to_numpy(),
                        "row": test_meta["row"].to_numpy(dtype=np.int64),
                        "y": np.asarray(y, dtype=object).astype(str),
                        "pred": np.asarray(pred, dtype=object).astype(str)})
    tmp = path.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)


def read_arm(arm, names=None):
    """All rows of the given files (all files when names is None), in the given
    order, with file and row columns first."""
    paths = P.files(arm)
    if names is not None:
        by = {p.name: p for p in paths}
        paths = [by[n] for n in names]
    parts = []
    for p in paths:
        df = P.read_release(arm, p)
        df.insert(0, "row", np.arange(len(df), dtype=np.int64))
        df.insert(0, "file", p.name)
        parts.append(df)
        log(f"{arm}: read {p.name} ({len(df):,} rows)")
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# R1, Rosay et al. 2021: port of create_datasets.py and analysis.py

def _module_constants(path, names):
    """Literal constants of the authors' module, read with ast (nothing is executed)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id in names:
                out[t.id] = ast.literal_eval(node.value)
    return out


R1_CIC_TRAIN = [9.6941, 18.6674, 37.3679, 32.8573, 34.544, 44.2445, 48.9476, 25.2079, 45.4546,
                50.00, 25.0806, 45.1228, 28.5715, 50.0000, 0]
R1_LYCOS_TRAIN = [15.78565, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 49.63280, 50.0, 50.0,
                  50.0, 49.31930]
R1_DROP = {"cic_original": ["flow_byte_per_s", "flow_pkt_per_s", "bwd_psh_flag", "bwd_urg_flag",
                            "avg_fwd_byte_per_bulk", "avg_fwd_pkt_per_bulk", "avg_fwd_bulk_rate",
                            "avg_bwd_byte_per_bulk", "avg_bwd_pkt_per_bulk", "avg_bwd_bulk_rate",
                            "flow_id", "timestamp", "src_addr", "src_port", "dst_addr", "label"],
           "lycos": ["flag_urg", "fwd_flag_urg", "bwd_flag_urg", "flow_id", "timestamp",
                     "src_addr", "src_port", "dst_addr", "label"]}


def _r1_detect_drop_outliers(df):
    """create_datasets.detect_drop_outliers: Infinity strings to NaN, the two
    rate columns to float, then drop every row with a NaN anywhere."""
    df = df.replace(["+Infinity", "-Infinity", "Infinity"], np.nan)
    for c in ("flow_byte_per_s", "flow_pkt_per_s"):
        if c in df.columns:
            df[c] = np.array(df[[c]]).astype(float)
    if df.isna().any().any():
        df = df.dropna(axis=0)
    return df


def _r1_split(df_list, train_pct, key_list):
    """create_datasets.split_dataset with randomize=True (append -> concat)."""
    df_list = [d.sample(frac=1, replace=False, random_state=0) for d in df_list]
    tr, cv, te, sizes = [], [], [], []
    for idx, d in enumerate(df_list):
        n = d.shape[0]
        n_tr = int(n * train_pct[idx] / 100)
        n_cv = int(n * (train_pct[idx] / 2) / 100)
        n_te = int(n * (train_pct[idx] / 2) / 100)
        tr.append(d[:n_tr])
        cv.append(d[n_tr:n_tr + n_cv])
        te.append(d[n_tr + n_cv:n_tr + n_cv + n_te])
        name = "Infiltration" if idx == 14 else key_list[idx]
        sizes.append({"class": name, "rows": n, "train": n_tr, "crossval": n_cv, "test": n_te})
    out = [pd.concat(x).sample(frac=1) for x in (tr, cv, te)]
    return out[0], out[1], out[2], pd.DataFrame(sizes)


def r1_sets():
    src = AUTHORS["lycos"] / "create_datasets.py"
    k = _module_constants(src, {"ISCX_FILELIST", "LYCOS_FILELIST", "FEAT_LIST_ISCX",
                                "ISCX_LABEL_DICT", "LYCOS_LABEL_DICT"})
    lycos_keys = list(k["LYCOS_LABEL_DICT"].keys())
    np.random.seed(2021)   # create_datasets.py, module level
    sets = {}
    for arm, names, pct in (("cic_original", k["ISCX_FILELIST"], R1_CIC_TRAIN),
                            ("lycos", k["LYCOS_FILELIST"], R1_LYCOS_TRAIN)):
        n_cls = 15 if arm == "cic_original" else 14
        df_list = [pd.DataFrame() for _ in range(n_cls)]
        lab_dict = k["ISCX_LABEL_DICT"] if arm == "cic_original" else k["LYCOS_LABEL_DICT"]
        for name in names:
            path = next(p for p in P.files(arm) if p.name == name)
            raw = P.read_release(arm, path)
            ident = pd.DataFrame({"file": name, "row": np.arange(len(raw), dtype=np.int64)})
            if arm == "cic_original":
                # their positional rename to LycoSTand names; our reader already
                # dropped the blank lines that remove_empty_lines drops. The web
                # attack labels keep byte 0x96 under latin-1, as in their dictionary.
                raw.columns = list(k["FEAT_LIST_ISCX"].keys())
            df = pd.concat([ident, raw], axis=1)
            df = df.replace([""], np.nan).dropna(subset=["flow_id"])
            for t in df["label"].value_counts().index.tolist():
                idx = lab_dict[t] if t in lab_dict else lab_dict[_r1_fix_label(t, lab_dict)]
                part = df.loc[df["label"] == t].reset_index(drop=True)
                df_list[idx] = pd.concat([df_list[idx], part], sort=False)
                df_list[idx] = _r1_detect_drop_outliers(df_list[idx])
            log(f"R1 {arm}: {name} done")
        if arm == "cic_original":
            for key, val in k["ISCX_LABEL_DICT"].items():
                if key != "Infiltration":
                    df_list[val]["label"] = lycos_keys[val]
        tr, cv, te, sizes = _r1_split(df_list, pct, lycos_keys)
        sizes.insert(0, "arm", arm)
        sets[arm] = (tr, te, sizes)
    return sets


def _r1_fix_label(t, lab_dict):
    """The CIC web attack labels carry byte 0x96 in the authors' dictionary; our
    latin-1 reader yields the same character. Fail loudly on anything else."""
    for key in lab_dict:
        if key.replace("\x96", "") == t.replace("\x96", "").replace("�", ""):
            return key
    raise KeyError(t)


def run_r1(seeds):
    sets = r1_sets()
    rows = []
    for arm, (tr, te, sizes) in sets.items():
        rows.append(sizes)
        drop = [c for c in R1_DROP[arm] if c in tr.columns] + ["file", "row"]
        x_tr = tr.drop(columns=drop).apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        x_te = te.drop(columns=drop).apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        sc = StandardScaler().fit(x_tr)   # their z_score_std, fitted on training
        x_tr, x_te = sc.transform(x_tr), sc.transform(x_te)
        y_tr, y_te = tr["label"].astype(str).to_numpy(), te["label"].astype(str).to_numpy()
        for model in ("lda", "dt", "rf"):
            for s in ([0] if model == "lda" else seeds):
                p = cell_path("R1", arm, model, s)
                if p.is_file():
                    continue
                t0 = time.time()
                if model == "lda":
                    m = LinearDiscriminantAnalysis()
                elif model == "dt":
                    m = DecisionTreeClassifier(random_state=s)
                else:
                    m = RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=P.N_JOBS)
                m.fit(x_tr, y_tr)
                save_cell(p, te, y_te, m.predict(x_te))
                log(f"R1 {arm} {model} seed {s}: {time.time() - t0:.0f} s")
    SETS.mkdir(parents=True, exist_ok=True)
    pd.concat(rows).to_csv(SETS / "R1_set_sizes.csv", index=False)


# ---------------------------------------------------------------------------
# R2, Engelen et al. 2021: RF, 75/25, Attempted to BENIGN

R2_DROP = {"cic_original": ["Flow ID", "Source IP", "Source Port", "Destination IP",
                            "Destination Port", "Timestamp", "Label"],
           "wtmc2021": ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Timestamp",
                        "Label"]}


def clean_numeric(df, drop):
    """Feature matrix of the remaining columns; rows with NaN or infinite values
    are dropped (returned mask says which rows stay)."""
    x = df.drop(columns=[c for c in drop if c in df.columns] + ["file", "row"])
    x = x.apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    ok = np.isfinite(x).all(axis=1)
    return x, ok


def run_r2(seeds):
    sizes = []
    for arm in ("cic_original", "wtmc2021"):
        df = read_arm(arm)
        y = P.own_label(arm, df)
        y = y.where(~y.str.contains("Attempted"), "BENIGN").to_numpy()
        x, ok = clean_numeric(df, R2_DROP[arm])
        meta = df.loc[ok, ["file", "row"]].reset_index(drop=True)
        x, y = x[ok].astype(np.float32), y[ok]
        sizes.append({"arm": arm, "rows": int(len(ok)), "dropped_nan_inf": int((~ok).sum()),
                      "kept": int(ok.sum())})
        del df
        gc.collect()
        for s in seeds:
            p = cell_path("R2", arm, "rf", s)
            if p.is_file():
                continue
            t0 = time.time()
            i_tr, i_te = train_test_split(np.arange(len(y)), test_size=0.25, random_state=s,
                                          stratify=y)
            m = RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=P.N_JOBS)
            m.fit(x[i_tr], y[i_tr])
            save_cell(p, meta.iloc[i_te], y[i_te], m.predict(x[i_te]))
            del m
            gc.collect()
            log(f"R2 {arm} rf seed {s}: {time.time() - t0:.0f} s")
    SETS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sizes).to_csv(SETS / "R2_set_sizes.csv", index=False)


# ---------------------------------------------------------------------------
# R3, Lanvin et al. 2022: port of Evaluation_relabelling_CICIDS2017_reordered.ipynb

R3_DAYS = ["Friday", "Monday", "Thursday", "Tuesday", "Wednesday"]   # sorted(lfiles)


def _lanvin_cols():
    nb = json.loads((AUTHORS["lanvin"] / "Evaluation_relabelling_CICIDS2017_reordered.ipynb")
                    .read_text(encoding="utf-8"))
    for c in nb["cells"]:
        s = "".join(c["source"])
        if s.strip().startswith("colsKeep"):
            return ast.literal_eval(s.split("=", 1)[1].strip())
    raise RuntimeError("colsKeep not found")


def _lanvin_preprocess(df, columns=None):
    """tools.preprocessData, unchanged in logic (pandas 3: any(1) -> any(axis=1), and text
    columns turned back into object dtype, which the authors' dtype tests expect)."""
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) and df[c].dtype != object:
            df[c] = df[c].astype(object)
    integer, f = [], []
    for i in df.columns[:-1]:
        if df[i].dtype == "int64":
            integer.append(i)
        elif df[i].dtype == "object":
            pass
        else:
            f.append(i)
    df[integer] = df[integer].astype("int32")
    df[f] = df[f].astype("float32")
    if columns is None:
        one = [i for i in df.columns if df[i].value_counts().nunique() < 2]
        if "Label" in one:
            one.remove("Label")
        df = df.drop(one, axis=1)
        df.columns = df.columns.str.strip()
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
        to_drop = list(df.columns[(df.dtypes == object)])
        if "Label" in to_drop:
            to_drop.remove("Label")
        to_drop.append("Src Port")
        df = df.drop([c for c in to_drop if c in df.columns], axis=1)
        columns = df.columns
        df_out = df.drop(list(df.columns[df.dtypes == object]), axis=1)
    else:
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
        sub = df.loc[:, columns]
        df_out = sub.drop(list(sub.columns[sub.dtypes == object]), axis=1)
    return df_out.values, df["Label"].values, columns, df.index


def _lanvin_frame(arm, cols, days):
    parts = []
    keep = set(cols)
    for d in days:
        path = next(p for p in P.files(arm) if p.name == f"{d}-WorkingHours.csv")
        df = pd.read_csv(path, usecols=lambda c: c.strip() in keep)   # memory only
        df.columns = [e.strip() for e in df.columns]
        df = df[cols].copy()
        df.insert(0, "row", np.arange(len(df), dtype=np.int64))
        df.insert(0, "file", path.name)
        parts.append(df)
    df = pd.concat(parts).reset_index(drop=True)
    df["Label"] = df["Label"].astype(str).str.replace("�", "-").str.lower()
    return df


def run_r3(seeds, pairs):
    """pairs: list of (name, constant_from) where constant_from names the arm
    whose labels define 'constant' rows together with the arm itself; None means
    'rows not labelled port_scan' (the RP case, no R release held)."""
    cols = _lanvin_cols()
    checks = []
    for arm, other in pairs:
        df = _lanvin_frame(arm, cols, R3_DAYS)
        lab = {arm: df["Label"].to_numpy().copy()}
        if other is not None:
            # the two releases share every row and every feature value (RD and RPD files are
            # one git blob except Thursday, where only Port_scan labels differ), so only the
            # other release's labels are held, which keeps two full frames out of memory
            od = _lanvin_frame(other, ["Flow ID", "Label"], R3_DAYS)
            same_rows = (od[["file", "row", "Flow ID"]].to_numpy() ==
                         df[["file", "row", "Flow ID"]].to_numpy()).all()
            lab[other] = od["Label"].to_numpy().copy()
            del od
            gc.collect()
            const = (lab[other] == lab[arm])
            checks.append({"arm": arm, "other": other, "same_rows": bool(same_rows),
                           "rows": len(df), "constant": int(const.sum())})
            if not same_rows:
                raise AssertionError(f"{arm} and {other} rows do not line up")
            df["Label"] = lab[other]   # the notebook trains on the first experiment, RD
            base = df
        else:
            const = (df["Label"].to_numpy() != "port_scan")
            checks.append({"arm": arm, "other": "none (rows not labelled port_scan)",
                           "same_rows": True, "rows": len(df), "constant": int(const.sum())})
            base = df
        for s in seeds:
            done = all(cell_path("R3", a, m, s).is_file()
                       for a in ([arm, other] if other else [arm]) for m in ("dt", "rf", "nb"))
            if done:
                continue
            t0 = time.time()
            rng = np.random.RandomState(s)  # the notebook draws with the global unseeded RNG
            tr_df = base[const]
            attacks = sorted(base["Label"].unique())[::-1]
            idx_train, added = [], 0
            for a in attacks:
                sub = tr_df[tr_df["Label"] == a]
                n = round(0.65 * sub.shape[0]) if a != "benign" else added
                added += n
                pick = rng.choice(np.arange(sub.shape[0]), size=n, replace=False)
                idx_train.append(sub.iloc[pick])
            train = pd.concat(idx_train)
            test_idx = np.setdiff1d(np.arange(len(base)), train.index.to_numpy())
            x_tr, y_tr, cols_kept, _ = _lanvin_preprocess(
                train.drop(columns=["file", "row"]).sample(frac=1, random_state=s))
            scaler = StandardScaler().fit(x_tr)
            x_trn = scaler.transform(x_tr)
            models = {"rf": RandomForestClassifier(max_depth=15, random_state=s, n_jobs=P.N_JOBS),
                      "nb": BernoulliNB(),
                      "dt": DecisionTreeClassifier(criterion="gini", max_depth=15, random_state=s)}
            for m in models.values():
                m.fit(x_trn, y_tr)
            for a in ([other, arm] if other else [arm]):
                # the test rows go through the preprocessing in blocks: it acts row by row
                # once the columns are fixed, so the result is the same with less memory
                metas, ys, preds = [], [], {n: [] for n in models}
                for b in range(0, len(test_idx), 300_000):
                    ix = test_idx[b:b + 300_000]
                    te = df.iloc[ix].copy()
                    te["Label"] = lab[a][ix]
                    x_te, y_te, _, kept = _lanvin_preprocess(
                        te.drop(columns=["file", "row"]), columns=cols_kept)
                    x_te = scaler.transform(x_te)
                    metas.append(te.loc[kept, ["file", "row"]])
                    ys.append(y_te)
                    for name, m in models.items():
                        preds[name].append(m.predict(x_te))
                    del te, x_te
                meta, y_te = pd.concat(metas), np.concatenate(ys)
                for name in models:
                    save_cell(cell_path("R3", a, name, s), meta, y_te,
                              np.concatenate(preds[name]))
                gc.collect()
            log(f"R3 {arm} seed {s}: {time.time() - t0:.0f} s")
    SETS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(checks).to_csv(SETS / f"R3_checks_{'_'.join(a for a, _ in pairs)}.csv",
                                index=False)


# ---------------------------------------------------------------------------
# R4, Sarhan et al. 2022: Extra Trees, 70/30, five splits

R4_DROP = {"unsw_original": ["srcip", "sport", "dstip", "dsport", "Stime", "Ltime", "sttl",
                             "dttl", "ct_state_ttl", "attack_cat", "Label"],
           "nf_unsw_v2": ["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT",
                          "MIN_TTL", "MAX_TTL", "Label", "Attack"],
           "nf_unsw_v3": ["IPV4_SRC_ADDR", "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT",
                          "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "MIN_TTL",
                          "MAX_TTL", "Label", "Attack"]}


def r4_matrix(arm, df):
    x = df.drop(columns=[c for c in R4_DROP[arm] if c in df.columns] + ["file", "row"])
    for c in x.columns:
        if not pd.api.types.is_numeric_dtype(x[c]):
            num = pd.to_numeric(x[c], errors="coerce")
            if num.notna().mean() > 0.99:
                x[c] = num
            else:   # Argus proto, state, service: coded by sorted level
                lv = sorted(pd.unique(x[c].astype(str).str.strip()))
                x[c] = pd.Categorical(x[c].astype(str).str.strip(), categories=lv).codes
    x = x.to_numpy(np.float64)
    x[~np.isfinite(x)] = np.nan
    x = np.nan_to_num(x, nan=0.0)   # blank UNSW-NB15 counters mean "not applicable"
    return MinMaxScaler().fit_transform(x).astype(np.float32)


def run_r4(seeds, arms=("unsw_original", "nf_unsw_v2", "nf_unsw_v3")):
    for arm in arms:
        df = read_arm(arm)
        y = P.own_label(arm, df).to_numpy()
        if arm == "unsw_original":
            # the release spells one attack category two ways, "Backdoor" and "Backdoors";
            # one class for training and the split, as for scoring (instrument.canonical)
            y = np.where(y == "Backdoors", "Backdoor", y).astype(object)
        assert len(set(y)) == len(set(P.canonical(arm, y)[0])), "two spellings of one class"
        x = r4_matrix(arm, df)
        meta = df[["file", "row"]].reset_index(drop=True)
        yb = np.where(P.canonical(arm, y)[0] == P.BENIGN, "benign", "attack")
        del df
        gc.collect()
        for s in seeds:
            i_tr, i_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=s,
                                          stratify=y)
            for task, yy in (("bin", yb), ("multi", y)):
                p = cell_path("R4", arm, f"et_{task}", s)
                if p.is_file():
                    continue
                t0 = time.time()
                m = ExtraTreesClassifier(n_estimators=100, random_state=s, n_jobs=P.N_JOBS)
                m.fit(x[i_tr], yy[i_tr])
                save_cell(p, meta.iloc[i_te], y[i_te], m.predict(x[i_te]))
                del m
                gc.collect()
                log(f"R4 {arm} {task} seed {s}: {time.time() - t0:.0f} s")


# ---------------------------------------------------------------------------
# R5, Pekar and Jozsa 2024: the notebook cell, per day

R5_DAYS = ["Tuesday", "Wednesday", "Thursday", "Friday"]
# crisis2022_rp is the authors' CRiSIS-2022 (protocol amendment 1); crisis2022_rpd is kept
# as an extra row for our instrument arm
R5_ARMS = ["cic_original", "wtmc2021", "crisis2022_rp", "nfs2023_nte", "nfs2023_te",
           "crisis2022_rpd"]
R5_CIC_FILES = {"Tuesday": ["Tuesday-WorkingHours.pcap_ISCX.csv"],
                "Wednesday": ["Wednesday-workingHours.pcap_ISCX.csv"],
                "Thursday": ["Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                             "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv"],
                "Friday": ["Friday-WorkingHours-Morning.pcap_ISCX.csv",
                           "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                           "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"]}


def _pekar_columns():
    out = {}
    for arm, nb in (("cic_original", "1-CICIDS-2017/2-ml-without_feat_sel.ipynb"),
                    ("wtmc2021", "2-WTMC-2021/2-ml-without_feat_sel.ipynb"),
                    ("crisis2022_rpd", "3-CRiSIS-2022/2-ml-without_feat_sel.ipynb"),
                    ("nfs2023_nte", "4-NFS-2023/2-NFS-2023-nTE-2-ml-without_feat_sel.ipynb"),
                    ("nfs2023_te", "4-NFS-2023/3-NFS-2023-TE-2-ml-without_feat_sel.ipynb")):
        cells = json.loads((AUTHORS["pekar"] / nb).read_text(encoding="utf-8"))["cells"]
        for c in cells:
            s = "".join(c["source"])
            if "columns_to_keep" in s and "binary_label" in s:
                body = s.split("columns_to_keep = ", 1)[1]
                out[arm] = ast.literal_eval(body[:body.index("]") + 1])
                break
    out["crisis2022_rp"] = out["crisis2022_rpd"]   # same notebook, same CSV layout
    return out


def run_r5(seeds):
    cols = _pekar_columns()
    for day in R5_DAYS:
        for arm in R5_ARMS:
            todo = [s for s in seeds if not cell_path("R5", arm, "rf", s, day).is_file()]
            if not todo or not P.files(arm):
                continue
            names = R5_CIC_FILES[day] if arm == "cic_original" else (
                [f"{day}.csv"] if arm.startswith("nfs") else [f"{day}-WorkingHours.csv"])
            data = read_arm(arm, names)
            if not arm.startswith("nfs"):
                feat = data.drop(columns=["file", "row"])
                feat = feat.replace([np.inf, -np.inf], np.nan)
                data = data[feat.notna().all(axis=1)]     # data.dropna(inplace=True)
                if arm != "cic_original":
                    data = data[~data["Label"].str.contains("Attempted")]
            lab = data[P.label_column(arm)].astype(str)
            y = (lab != "BENIGN").astype(int).to_numpy()
            x = data[cols[arm]].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
            meta = data[["file", "row"]].reset_index(drop=True)
            i_tr, i_te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=42,
                                          stratify=y)
            for s in todo:
                t0 = time.time()
                m = RandomForestClassifier(random_state=s, n_jobs=P.N_JOBS)
                m.fit(x[i_tr], y[i_tr])
                pred = np.where(m.predict(x[i_te]) == 1, "attack", "benign")
                save_cell(cell_path("R5", arm, "rf", s, day), meta.iloc[i_te],
                          lab.to_numpy()[i_te], pred)
                log(f"R5 {day} {arm} seed {s}: {time.time() - t0:.0f} s")
            del data, x
            gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("claim", choices=["R1", "R2", "R3", "R3rp", "R4", "R5"])
    ap.add_argument("--seeds", nargs="*", type=int, default=P.SEEDS)
    a = ap.parse_args()
    t_start = time.time()
    if a.claim == "R1":
        run_r1(a.seeds)
    elif a.claim == "R2":
        run_r2(a.seeds)
    elif a.claim == "R3":
        run_r3(a.seeds, [("crisis2022_rpd", "crisis2022_rd")])
    elif a.claim == "R3rp":
        run_r3(a.seeds, [("crisis2022_rp", "crisis2022_r")])
    elif a.claim == "R4":
        run_r4(a.seeds)
    else:
        run_r5(a.seeds)
    log(f"{a.claim} finished in {time.time() - t_start:.0f} s")
