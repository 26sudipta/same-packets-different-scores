"""
Steps 5.3 and 5.4: tune, train and score every cell of the RQ2 grid.

Usage, from the workspace root:
    .venv/bin/python 03_Code/phase5_grid.py tune CICIDS2017
    .venv/bin/python 03_Code/phase5_grid.py run CICIDS2017 [--arms ...] [--models ...]
            [--splits grouped random] [--seeds 0 1 2 3 4] [--tag main]

A cell is (benchmark, arm, model, split, seed). Training rows are the arm's
records whose split column is 0, test rows those whose column is 1 (grouped:
g<seed>, random flow: r<seed>, both frozen in Phase 3). Labels are the
schedule classes. The seed is also the model's random state.

Checkpointing: each finished cell writes its predictions to
04_Results/phase5/preds/<bench>/<arm>/<model>_<split>_s<seed>.parquet and its
scores to 04_Results/phase5/cells/<bench>/<arm>/<model>_<split>_s<seed>.csv.
A cell whose score file exists is skipped, so a crash costs one cell.

Scores per cell: per class F1 and macro-F1 (C+) at the flow and the
conversation unit (rule "any", ties by bytes), on the own population (every
test record) and the common population (test records of conversations present
in every arm, `in_all`).

UNSW-NB15 arms: unsw_dedup (original without exact duplicate lines, primary),
unsw_released (as released), unsw_collapsed (also without near repeats) and
nf_unsw_v3. The first three read the same table of the original.
"""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_scoring as CS  # noqa: E402
import instrument as I  # noqa: E402
from metrics_macro_f1 import macro_f1_cplus  # noqa: E402
import phase5_data as D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "04_Results" / "phase5"
SEEDS = [0, 1, 2, 3, 4]
SPLITS = {"grouped": "g", "random": "r"}
MODELS = ["rf", "lgbm", "mlp"]
ARMS = {"CICIDS2017": I.BENCHMARKS["CICIDS2017"]["arms"],
        "UNSW-NB15": ["unsw_dedup", "nf_unsw_v3", "cic_unsw", "unsw_released", "unsw_collapsed"]}
PRIMARY_ARMS = {"CICIDS2017": I.BENCHMARKS["CICIDS2017"]["arms"],
                "UNSW-NB15": ["unsw_dedup", "nf_unsw_v3", "cic_unsw"]}
VAL_SEED = 2000
VAL_FRACTION = 0.2
N_JOBS = 8

GRID = {
    "rf": [dict(max_depth=None, min_samples_leaf=1), dict(max_depth=None, min_samples_leaf=3),
           dict(max_depth=20, min_samples_leaf=1), dict(max_depth=20, min_samples_leaf=3)],
    "lgbm": [dict(num_leaves=31, learning_rate=0.1), dict(num_leaves=31, learning_rate=0.05),
             dict(num_leaves=127, learning_rate=0.1), dict(num_leaves=127, learning_rate=0.05)],
    "mlp": [dict(hidden_layer_sizes=(64,), alpha=1e-4), dict(hidden_layer_sizes=(64,), alpha=1e-3),
            dict(hidden_layer_sizes=(128, 64), alpha=1e-4),
            dict(hidden_layer_sizes=(128, 64), alpha=1e-3)],
}


def signed_log1p(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_model(name, cfg, seed):
    cfg = dict(cfg)
    if "hidden_layer_sizes" in cfg:
        cfg["hidden_layer_sizes"] = tuple(cfg["hidden_layer_sizes"])
    if name == "rf":
        return RandomForestClassifier(n_estimators=100, n_jobs=N_JOBS, random_state=seed, **cfg)
    if name == "lgbm":
        # reg_lambda=1: without an L2 penalty the multiclass fit diverges on these
        # data (validation log loss 0.14 after 10 rounds, 9.7 after 200; macro-F1
        # 0.21 on the CIC original tuning split); with it, 0.016 and 0.85. Found
        # on 27 October 2026 before any grid cell was run (PROGRESS_LOG, step 5.3)
        return LGBMClassifier(n_estimators=200, n_jobs=N_JOBS, random_state=seed, verbose=-1,
                              deterministic=True, force_row_wise=True, reg_lambda=1.0, **cfg)
    if name == "mlp":
        return make_pipeline(FunctionTransformer(signed_log1p), StandardScaler(),
                             MLPClassifier(random_state=seed, early_stopping=True, max_iter=50,
                                           batch_size=512, **cfg))
    raise KeyError(name)


def source_arm(bench, arm):
    return "unsw_original" if arm.startswith("unsw_") else arm


def load_arm(bench, arm, tag="main"):
    p = D.CACHE / tag / bench / f"{source_arm(bench, arm)}.parquet"
    t = pd.read_parquet(p)
    if arm == "unsw_dedup":
        t = t[~t["dup"]]
    elif arm == "unsw_collapsed":
        t = t[~t["dup"] & ~t["near"]]
    t = t.reset_index(drop=True)
    fcols = [c for c in t.columns if c.startswith("f:")]
    return t, fcols


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def scores(t, test_idx, pred, bench, arm, model, split, seed):
    """Long table of per class F1 and macro-F1 for one cell."""
    rows = []
    sub = t.iloc[test_idx]
    y = sub["sched"].to_numpy(dtype=object)
    conv = sub["gconv"].to_numpy()
    w = sub["bytes"].to_numpy(dtype=float)
    common = sub["in_all"].to_numpy(dtype=bool)
    for pop, m in (("own", np.ones(len(sub), dtype=bool)), ("common", common)):
        for unit in ("flow", "conversation"):
            r = CS.score(conv[m], y[m], pred[m], unit=unit, weight=w[m])
            if unit == "flow":
                yt, yp = y[m], pred[m]
            else:
                truth = CS.conversation_truth(conv[m], y[m])
                yp = CS.aggregate(conv[m], pred[m], "any", w[m]).reindex(truth.index).to_numpy(
                    dtype=object)
                yt = truth.to_numpy(dtype=object)
            sup = pd.Series(yt).value_counts()
            base = dict(benchmark=bench, arm=arm, model=model, split=split, seed=seed,
                        unit=unit, population=pop)
            for c, f in r["per_class"].items():
                rows.append({**base, "cls": c, "f1": f, "support": int(sup.get(c, 0))})
            rows.append({**base, "cls": "MACRO_CPLUS", "f1": r["macro_f1"],
                         "support": r["n_cplus"]})
            nh = yt != "heartbleed"
            mh, kh = macro_f1_cplus(yt[nh], yp[nh]) if nh.any() else (np.nan, 0)
            rows.append({**base, "cls": "MACRO_CPLUS_NO_HEARTBLEED", "f1": mh, "support": kh})
    return pd.DataFrame(rows)


# the sensitivity versions of the UNSW-NB15 original reuse its tuned setting
TUNED_AS = {"unsw_released": "unsw_dedup", "unsw_collapsed": "unsw_dedup"}


def chosen_config(bench, arm, model):
    arm = TUNED_AS.get(arm, arm)
    p = OUT / "tuning.csv"
    t = pd.read_csv(p)
    t = t[(t["benchmark"] == bench) & (t["arm"] == arm) & (t["model"] == model) & t["chosen"]]
    if len(t) != 1:
        raise ValueError(f"no tuned setting for {bench} {arm} {model}; run tune first")
    return json.loads(t["config"].iloc[0])


def fit_predict(model, cfg, seed, x_tr, y_tr, x_te):
    classes, y_code = np.unique(y_tr, return_inverse=True)
    m = make_model(model, cfg, seed)
    m.fit(x_tr, y_code)
    return classes[m.predict(x_te)]


def tune(bench, arms=None, models=None, tag="main"):
    """Pick one setting per arm and model on seed 0 of the grouped split."""
    out = OUT / "tuning.csv"
    old = pd.read_csv(out) if out.is_file() else pd.DataFrame()
    for arm in arms or PRIMARY_ARMS[bench]:
        t, fcols = load_arm(bench, arm, tag)
        tr = t[t["g0"] == 0]
        conv = tr[["gconv", "sched"]].drop_duplicates("gconv")
        uid = (bench + "|" + conv["gconv"].astype(str)).to_numpy()
        v = I.stratified_hash_split(uid, conv["sched"].to_numpy(), VAL_SEED, VAL_FRACTION)
        val_convs = set(conv["gconv"].to_numpy()[v == 1].tolist())
        isval = tr["gconv"].isin(val_convs).to_numpy()
        x = tr[fcols].to_numpy()
        y = tr["sched"].to_numpy(dtype=object)
        for model in models or MODELS:
            if len(old) and ((old["benchmark"] == bench) & (old["arm"] == arm) &
                             (old["model"] == model)).any():
                print(f"tune {bench} {arm} {model}: already in {out.name}, not tuned again",
                      flush=True)
                continue
            rows = []
            for k, cfg in enumerate(GRID[model]):
                t0 = time.time()
                p = fit_predict(model, cfg, 0, x[~isval], y[~isval], x[isval])
                f, _ = macro_f1_cplus(y[isval], p)
                rows.append({"benchmark": bench, "arm": arm, "model": model, "k": k,
                             "config": json.dumps(cfg), "val_macro_f1": f,
                             "seconds": round(time.time() - t0, 1)})
                print(f"tune {bench} {arm} {model} {cfg}: {f:.4f} "
                      f"({time.time() - t0:.0f} s)", flush=True)
            r = pd.DataFrame(rows)
            r["chosen"] = False
            r.loc[r["val_macro_f1"].idxmax(), "chosen"] = True
            old = pd.concat([old, r], ignore_index=True)
            old.to_csv(out, index=False)
    return old


def run(bench, arms=None, models=None, splits=None, seeds=None, tag="main", out=OUT,
        cfg_override=None):
    timing = []
    for arm in arms or ARMS[bench]:
        t, fcols = load_arm(bench, arm, tag)
        x = t[fcols].to_numpy()
        y = t["sched"].to_numpy(dtype=object)
        for model in models or MODELS:
            cfg = cfg_override if cfg_override is not None else chosen_config(bench, arm, model)
            for split in splits or list(SPLITS):
                for seed in seeds if seeds is not None else SEEDS:
                    name = f"{model}_{split}_s{seed}"
                    cp = out / "cells" / bench / arm / f"{name}.csv"
                    if cp.is_file():
                        continue
                    col = f"{SPLITS[split]}{seed}"
                    tr = np.flatnonzero(t[col].to_numpy() == 0)
                    te = np.flatnonzero(t[col].to_numpy() == 1)
                    t0 = time.time()
                    pred = fit_predict(model, cfg, seed, x[tr], y[tr], x[te])
                    fit_s = time.time() - t0
                    pp = out / "preds" / bench / arm / f"{name}.parquet"
                    pp.parent.mkdir(parents=True, exist_ok=True)
                    pd.DataFrame({"i": te.astype(np.int32),
                                  "pred": pd.Categorical(pred)}).to_parquet(pp, index=False)
                    sc = scores(t, te, pred, bench, arm, model, split, seed)
                    sc["config"] = json.dumps(cfg)
                    sc["n_train"] = len(tr)
                    sc["n_test"] = len(te)
                    sc["seconds"] = round(fit_s, 1)
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    sc.to_csv(cp, index=False)
                    mac = sc[(sc["cls"] == "MACRO_CPLUS") & (sc["population"] == "own")]
                    timing.append({"benchmark": bench, "arm": arm, "model": model,
                                   "split": split, "seed": seed, "n_train": len(tr),
                                   "n_test": len(te), "seconds": round(fit_s, 1),
                                   "peak_rss_mb": round(peak_mb())})
                    print(f"{bench} {arm} {name}: train {len(tr):,} test {len(te):,} "
                          f"{fit_s:.0f} s, peak {peak_mb():.0f} MB, macro "
                          + " ".join(f"{u}={v:.3f}" for u, v in
                                     zip(mac["unit"], mac["f1"])), flush=True)
    return pd.DataFrame(timing)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tune", "run"])
    ap.add_argument("bench")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--splits", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--config", help="JSON setting, bypasses tuning (time trial only)")
    a = ap.parse_args()
    if a.what == "tune":
        tune(a.bench, a.arms, a.models, a.tag)
    else:
        tm = run(a.bench, a.arms, a.models, a.splits, a.seeds, a.tag, Path(a.out),
                 json.loads(a.config) if a.config else None)
        print(tm.to_string())
