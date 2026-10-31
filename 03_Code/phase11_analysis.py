"""
Phase 11 (review response): analyses on stored predictions, with the Phase 5
cluster bootstrap (conversations resampled within schedule class, one draw shared
by every cell so that arms and models stay paired, B = 1000, seed 0).

Subcommands, each writing to 04_Results/phase11/analysis/:

  core     on the frozen Phase 5 grid (grouped split):
           ms_shares.csv     sums of squares per degree of freedom beside the raw
                             shares, with intervals, and H2 judged on both (M3a)
           h2_subsets.csv    H2 on every arm subset of size k - 1, with intervals (M3c)
           equivalence.csv   Definition 1 tested directly: two one-sided tests per
                             model, class and pair of arms, by 90 percent intervals
                             against margins 0.01, 0.02 and 0.05 (M2)
           bca.csv           BCa intervals (grouped jackknife acceleration) for the
                             H2 contrasts and the exporter shares (m11)
           ht.csv            precision, recall and F1 with each conversation weighted
                             by the inverse of its class's subsample fraction, so that
                             the test set has full-data prevalence (M5)
  ht_h2    H2 with the bootstrap and the point estimate both weighted to full-data
           prevalence (M5): ht_h2.csv, ht_h2_verdict.csv
  payload  the same machinery on the payload-aware grid (M1): rq3.csv, h4.csv,
           shares.csv, residual.csv
  payload_h2  H2 on the payload-aware grid (M1): payload_h2.csv, payload_h2_verdict.csv
  own      RQ3 table on each release's own labels, from the Phase 7 own-label
           retraining (M1): rq3_own.csv
  ttl      H2 on UNSW-NB15 with the TTL fields removed (m13): ttl_h2.csv, ttl_macro.csv
  models   H2 and H4 with five models (Phase 5 three plus lr and knn) (M2):
           models_h2.csv, models_w.csv, models_macro.csv
  loo      stored predictions scored on the leave-one-arm-out conversations (M4):
           loo_scores.csv
  full     the full-data random forest against its subsample cell (M5): full.csv

Usage, from the workspace root, NIDS_CACHE_ROOT set (phase11_common):
    .venv/bin/python 03_Code/phase11_analysis.py core [--boot 1000]
"""
import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_scoring as CS  # noqa: E402
import phase11_common as C  # noqa: E402
import phase5_analysis as A  # noqa: E402
import phase5_data as D  # noqa: E402
import phase5_grid as G  # noqa: E402
from manifest import write_manifest  # noqa: E402

OUT = C.OUT / "analysis"
P5 = C.ROOT / "04_Results" / "phase5"
P7 = C.ROOT / "04_Results" / "phase7"
SEEDS = G.SEEDS
MARGINS = [0.01, 0.02, 0.05]
SMALL_LIMITS = [2500, 5000, 10000]
MIN_TEST = [10, 20, 40]
META = ["gconv", "day", "sched", "in_all", "bytes", "file", "row"] + [f"g{s}" for s in SEEDS]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_meta(bench, arm, extra=(), tag="main"):
    """Metadata of a Phase 5 arm table, same rows and order as phase5_grid.load_arm."""
    p = D.CACHE / tag / bench / f"{G.source_arm(bench, arm)}.parquet"
    names = pq.ParquetFile(p).schema.names
    cols = META + [c for c in ("dup", "near") if c in names] + list(extra)
    t = pd.read_parquet(p, columns=cols)
    if arm == "unsw_dedup":
        t = t[~t["dup"]]
    elif arm == "unsw_collapsed":
        t = t[~t["dup"] & ~t["near"]]
    return t.reset_index(drop=True)


class Cells:
    """Per conversation TP, FP and FN of every (arm, model, seed, unit) cell."""

    def __init__(self, classes):
        self.classes = list(classes)
        self.K = len(self.classes)
        self.cidx = {c: i for i, c in enumerate(self.classes)}
        self.raw, self.mats = {}, {}
        self.sched, self.inall = {}, {}

    def add(self, key, conv, y, pred, bytes_, sched_of_conv, inall_of_conv):
        for unit in ("flow", "conversation"):
            if unit == "flow":
                c_, yt, yp = conv, y, pred
            else:
                truth = C.truth_any(conv, y)
                yp = CS.aggregate(conv, pred, "any", bytes_).reindex(truth.index)
                c_, yt, yp = truth.index.to_numpy(), truth.to_numpy(dtype=object), \
                    yp.to_numpy(dtype=object)
            self.raw[key + (unit,)] = (c_, yt, yp)
        self.sched.update(sched_of_conv)
        self.inall.update(inall_of_conv)

    def finalize(self, ht=None):
        convs = np.unique(np.concatenate([v[0] for v in self.raw.values()]))
        self.universe = convs
        pos = pd.Series(np.arange(len(convs)), index=convs)
        self.u_class = pd.Series(self.sched).reindex(convs).to_numpy(dtype=object)
        self.u_common = pd.Series(self.inall).reindex(convs).to_numpy(dtype=bool)
        self.u_ht = (pd.Series(self.u_class).map(ht).to_numpy(dtype=float) if ht is not None
                     else np.ones(len(convs)))
        K = self.K
        self.support = {}
        for key, (c_, yt, yp) in self.raw.items():
            u = pos.reindex(c_).to_numpy()
            yi = np.array([self.cidx[v] for v in yt])
            pi = np.array([self.cidx.get(v, -1) for v in yp])
            hit = yi == pi
            rows = np.concatenate([u[hit], u[~hit & (pi >= 0)], u[~hit]])
            cols = np.concatenate([yi[hit], K + pi[~hit & (pi >= 0)], 2 * K + yi[~hit]])
            M = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)),
                                  shape=(len(convs), 3 * K)).tocsr()
            nz = np.unique(rows)
            self.mats[key] = (nz, M[nz])
            if key[-1] == "conversation":
                self.support[key] = np.bincount(yi, minlength=K)
        self.raw = {}
        return self

    def counts(self, W):
        out = {}
        for key, (nz, M) in self.mats.items():
            out[key] = np.asarray((M.T @ W[:, nz].T).T)
        return out

    def metrics(self, W, what="f1"):
        K = self.K
        out = {}
        for key, Cm in self.counts(W).items():
            tp, fp, fn = Cm[:, :K], Cm[:, K:2 * K], Cm[:, 2 * K:]
            with np.errstate(invalid="ignore", divide="ignore"):
                if what == "f1":
                    v = np.where(tp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
                elif what == "precision":
                    v = np.where(tp + fn > 0, np.where(tp + fp > 0, tp / (tp + fp), 0.0), np.nan)
                else:
                    v = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
            out[key] = v
        return out


def build(bench, arms, models, root_of, label_fn=None, extra_cols=(), classes=None,
          ht=None, conv_map=None):
    """Cells of the grouped split for these arms and models. root_of(arm, model) is
    the folder holding <model>_grouped_s<seed>.parquet. label_fn(meta, arm) returns
    record labels (default: the schedule label). conv_map(meta, arm) may replace the
    conversation id and schedule label (leave-one-out instruments)."""
    t0 = time.time()
    parts = {}
    for arm in arms:
        meta = load_meta(bench, arm, extra_cols.get(arm, ()) if isinstance(extra_cols, dict)
                         else extra_cols)
        if conv_map is not None:
            meta = conv_map(meta, arm)
        y = label_fn(meta, arm) if label_fn else meta["sched"].to_numpy(dtype=object)
        parts[arm] = (meta, y)
    if classes is None:
        classes = sorted(set().union(*[set(m["sched"].unique()) | set(pd.unique(y))
                                       for m, y in parts.values()]))
    cells = Cells(classes)
    for arm, (meta, y) in parts.items():
        conv_all = meta["gconv"].to_numpy()
        sched = dict(zip(conv_all, meta["sched"].to_numpy()))
        inall = dict(zip(conv_all, meta["in_all"].to_numpy()))
        for model in models:
            for s in SEEDS:
                p = root_of(arm, model) / f"{model}_grouped_s{s}.parquet"
                pr = pd.read_parquet(p)
                i = pr["i"].to_numpy()
                cells.add((arm, model, s), conv_all[i], y[i], pr["pred"].astype(str).to_numpy(
                    dtype=object), meta["bytes"].to_numpy(float)[i], sched, inall)
        log(f"{bench} {arm}: loaded ({time.time() - t0:.0f} s)")
    return cells.finalize(ht)


def stack(fd, arms, models, unit):
    """(B, A, M, S, K) from a dict keyed (arm, model, seed, unit)."""
    return np.stack([np.stack([np.stack([fd[(a, m, s, unit)] for s in SEEDS], axis=-2)
                               for m in models], axis=-3) for a in arms], axis=-4)


def replicate(cells, B, pops=("own", "common"), what="f1", chunk=50, seed=0, jack=0, ht=False):
    """Point estimates and bootstrap replicates; optional grouped jackknife rows. ht=True
    weights every conversation by the inverse of its class's subsample fraction."""
    rng = np.random.default_rng(seed)
    N = len(cells.universe)
    hw = cells.u_ht if ht else np.ones(N)
    point = {p: cells.metrics(hw[None, :] * (cells.u_common if p == "common" else 1.0),
                              what) for p in pops}
    reps = {p: {} for p in pops}
    for b0 in range(0, B, chunk):
        W = A.boot_weights(cells.u_class, min(chunk, B - b0), rng) * hw[None, :]
        for p in pops:
            fd = cells.metrics(W * cells.u_common if p == "common" else W, what)
            for k, v in fd.items():
                reps[p].setdefault(k, []).append(v.astype(np.float32))
    reps = {p: {k: np.concatenate(v) for k, v in d.items()} for p, d in reps.items()}
    jk = None
    if jack:
        g = np.random.default_rng(seed + 1).integers(0, jack, size=N)
        Wj = np.stack([(g != j).astype(float) for j in range(jack)])
        jk = {p: cells.metrics(Wj * cells.u_common if p == "common" else Wj, what) for p in pops}
    return point, reps, jk


# --------------------------------------------------------------------------- statistics

DF_NAMES = ["arm", "model", "arm:model", "seed", "residual"]


def ss_parts(F):
    """Sums of squares (..., 5) and total (...) of a balanced arm x model x seed design."""
    A_, M_, S_ = F.shape[-3:]
    gm = F.mean(axis=(-3, -2, -1), keepdims=True)
    ma = F.mean(axis=(-2, -1), keepdims=True)
    mm = F.mean(axis=(-3, -1), keepdims=True)
    ms = F.mean(axis=(-3, -2), keepdims=True)
    mam = F.mean(axis=-1, keepdims=True)
    ss_a = M_ * S_ * ((ma - gm) ** 2).sum(axis=(-3, -2, -1))
    ss_m = A_ * S_ * ((mm - gm) ** 2).sum(axis=(-3, -2, -1))
    ss_am = S_ * ((mam - ma - mm + gm) ** 2).sum(axis=(-3, -2, -1))
    ss_s = A_ * M_ * ((ms - gm) ** 2).sum(axis=(-3, -2, -1))
    ss_t = ((F - gm) ** 2).sum(axis=(-3, -2, -1))
    out = np.stack([ss_a, ss_m, ss_am, ss_s, ss_t - ss_a - ss_m - ss_am - ss_s], axis=-1)
    dfs = np.array([A_ - 1, M_ - 1, (A_ - 1) * (M_ - 1), S_ - 1,
                    A_ * M_ * S_ - 1 - (A_ - 1) - (M_ - 1) - (A_ - 1) * (M_ - 1) - (S_ - 1)])
    return out, ss_t, dfs


def shares_both(F):
    """Raw shares and per-degree-of-freedom shares (SS_f / df_f / SS_total)."""
    ss, tot, dfs = ss_parts(F)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(tot[..., None] > 0, ss / tot[..., None], np.nan)
        perdf = np.where(tot[..., None] > 0, ss / dfs / tot[..., None], np.nan)
    return raw, perdf


def iv(x, lo=2.5, hi=97.5):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return (np.percentile(x, lo), np.percentile(x, hi)) if len(x) else (np.nan, np.nan)


def bca(theta_hat, boot, jack, alpha=0.05):
    """BCa interval: bias correction from the bootstrap, acceleration from a grouped
    jackknife."""
    boot = np.asarray(boot, float)
    boot = boot[np.isfinite(boot)]
    jack = np.asarray(jack, float)
    if not len(boot) or not np.isfinite(theta_hat) or not np.isfinite(jack).all():
        return np.nan, np.nan
    prop = np.clip((boot < theta_hat).mean() + 0.5 * (boot == theta_hat).mean(), 1e-6, 1 - 1e-6)
    z0 = norm.ppf(prop)
    d = jack.mean() - jack
    den = 6 * (d ** 2).sum() ** 1.5
    a = (d ** 3).sum() / den if den > 0 else 0.0
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = norm.ppf(q)
        adj = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        out.append(np.percentile(boot, 100 * adj))
    return tuple(out)


def eligible(cells, arms, models, small, min_test=20):
    """Small classes with at least min_test test conversations in every cell."""
    sup = np.stack([cells.support[(a, m, s, "conversation")] for a in arms for m in models
                    for s in SEEDS])
    mins = sup.min(axis=0)
    return [c for c in small if c in cells.cidx and mins[cells.cidx[c]] >= min_test], \
        dict(zip(cells.classes, mins))


def fixed_classes(cells, arms, models, min_test=20):
    sup = np.stack([cells.support[(a, m, s, "conversation")] for a in arms for m in models
                    for s in SEEDS])
    return [c for c, v in zip(cells.classes, sup.min(axis=0)) if v >= min_test]


def kendall(P, R, fi):
    """W (seed mean) point and replicates on macro-F1 over the fixed classes."""
    Pm = np.nanmean(P[..., fi], axis=-1)
    Rm = np.nanmean(R[..., fi], axis=-1)
    w = float(np.mean(A.kendall_w(np.moveaxis(Pm, -1, 0))))
    wr = A.kendall_w(np.moveaxis(Rm, -1, 1)).mean(axis=-1)
    return w, wr


def h2_rows(bench, label, P, R, classes, elig, jk=None, perdf=False):
    """arm minus model contrast per eligible class, raw or per degree of freedom."""
    rows = []
    rawP, dfP = shares_both(np.moveaxis(P, -1, 0))
    rawR, dfR = shares_both(np.moveaxis(R, -1, 1))
    for c in elig:
        k = classes.index(c)
        for kind, sP, sR in (("raw", rawP, rawR), ("per_df", dfP, dfR)):
            if kind == "per_df" and not perdf:
                continue
            d, dr = sP[k, 0] - sP[k, 1], sR[:, k, 0] - sR[:, k, 1]
            lo, hi = iv(dr)
            rows.append({"benchmark": bench, "setting": label, "kind": kind, "cls": c,
                         "arm_share": sP[k, 0], "model_share": sP[k, 1],
                         "arm_lo": iv(sR[:, k, 0])[0], "arm_hi": iv(sR[:, k, 0])[1],
                         "model_lo": iv(sR[:, k, 1])[0], "model_hi": iv(sR[:, k, 1])[1],
                         "contrast": d, "lo": lo, "hi": hi,
                         "verdict": "exporter above model" if lo > 0 else
                         "model above exporter" if hi < 0 else "undecided"})
    return rows


def h2_verdict(rows):
    df = pd.DataFrame(rows)
    out = []
    for (b, s, k), g in df.groupby(["benchmark", "setting", "kind"]):
        n = len(g)
        a = int((g["verdict"] == "exporter above model").sum())
        out.append({"benchmark": b, "setting": s, "kind": k, "eligible": n,
                    "exporter_above": a,
                    "model_above": int((g["verdict"] == "model above exporter").sum()),
                    "h2": "holds" if a > n / 2 else "does not hold"})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- subcommands

P5_ROOT = lambda bench: (lambda arm, model: P5 / "preds" / bench / arm)  # noqa: E731


def ht_weights(bench):
    sc = pd.read_csv(P5 / "subsample_counts.csv")
    sc = sc[sc["benchmark"] == bench]
    return dict(zip(sc["sched"], sc["conversations"] / sc["kept"]))


def core(B):
    OUT.mkdir(parents=True, exist_ok=True)
    rows_ms, rows_sub, rows_eq, rows_bca, rows_ht, rows_thr = [], [], [], [], [], []
    for bench in ["CICIDS2017", "UNSW-NB15"]:
        arms, models = G.PRIMARY_ARMS[bench], G.MODELS
        cells = build(bench, arms, models, P5_ROOT(bench), ht=ht_weights(bench))
        cls = cells.classes
        point, reps, jk = replicate(cells, B, jack=50)
        log(f"{bench}: bootstrap done")
        # check against the frozen per-class F1 (conversation unit, own population)
        r = pd.read_csv(P5 / "results.csv")
        r = r[(r["benchmark"] == bench) & (r["split"] == "grouped") & (r["population"] == "own")
              & ~r["cls"].str.startswith("MACRO")]
        bad = 0
        for (a, m, s, u), f in point["own"].items():
            q = r[(r["arm"] == a) & (r["model"] == m) & (r["seed"] == s) & (r["unit"] == u)]
            for c, v in zip(q["cls"], q["f1"]):
                bad += abs(f[0, cls.index(c)] - v) > 1e-9
        if bad:
            raise AssertionError(f"{bench}: {bad} point estimates differ from results.csv")
        small = A.SMALL[bench]
        elig, mins = eligible(cells, arms, models, small)
        # M3a: raw and per-df shares, flow unit, own population (the H2 setting)
        P = stack(point["own"], arms, models, "flow")[0]
        R = stack(reps["own"], arms, models, "flow")
        rows_ms += h2_rows(bench, "grouped flow own", P, R, cls, elig, perdf=True)
        for unit in ("conversation",):
            Pu = stack(point["own"], arms, models, unit)[0]
            Ru = stack(reps["own"], arms, models, unit)
            rows_ms += h2_rows(bench, f"grouped {unit} own", Pu, Ru, cls, elig, perdf=True)
        # m11: BCa for the H2 contrast and the exporter share, flow unit, own population
        J = stack(jk["own"], arms, models, "flow")
        rawP, _ = shares_both(np.moveaxis(P, -1, 0))
        rawR, _ = shares_both(np.moveaxis(R, -1, 1))
        rawJ, _ = shares_both(np.moveaxis(J, -1, 1))
        for c in cls:
            k = cls.index(c)
            for what, th, bt, jj in (
                    ("arm_share", rawP[k, 0], rawR[:, k, 0], rawJ[:, k, 0]),
                    ("arm_minus_model", rawP[k, 0] - rawP[k, 1], rawR[:, k, 0] - rawR[:, k, 1],
                     rawJ[:, k, 0] - rawJ[:, k, 1])):
                plo, phi = iv(bt)
                blo, bhi = bca(th, bt, jj)
                rows_bca.append({"benchmark": bench, "cls": c, "statistic": what, "value": th,
                                 "pct_lo": plo, "pct_hi": phi, "bca_lo": blo, "bca_hi": bhi,
                                 "eligible": c in elig})
        # M3c: every arm subset of size k - 1
        for drop in arms:
            sub = [a for a in arms if a != drop]
            Ps = stack(point["own"], sub, models, "flow")[0]
            Rs = stack(reps["own"], sub, models, "flow")
            rows_sub += h2_rows(bench, f"without {drop}", Ps, Rs, cls, elig)
        # m10: thresholds for small and eligible
        counts = pd.read_csv(P5 / "subsample_counts.csv")
        counts = counts[counts["benchmark"] == bench].set_index("sched")["conversations"]
        con = pd.read_csv(P5 / "share_contrasts.csv")
        con = con[(con["benchmark"] == bench) & (con["arm_set"] == "primary") &
                  (con["split"] == "grouped") & (con["unit"] == "flow") &
                  (con["population"] == "own") & (con["contrast"] == "arm_minus_model")]
        con = con.set_index("cls")
        for lim in SMALL_LIMITS:
            for mt in MIN_TEST:
                sm = [c for c in cls if c != "benign" and counts.get(c, 0) < lim]
                el = [c for c in sm if mins.get(c, 0) >= mt]
                above = [c for c in el if con["lo"].get(c, np.nan) > 0]
                rows_thr.append({"benchmark": bench, "small_below": lim, "min_test": mt,
                                 "eligible": len(el), "exporter_above": len(above),
                                 "classes": ";".join(el), "above": ";".join(above),
                                 "h2": "holds" if len(above) > len(el) / 2 else "does not hold"})
        # M2: equivalence, per model, class and pair of arms, both units, common population
        fixed = fixed_classes(cells, arms, models)
        fi = [cls.index(c) for c in fixed]
        for unit in ("flow", "conversation"):
            Pc = stack(point["common"], arms, models, unit)[0]      # A, M, S, K
            Rc = stack(reps["common"], arms, models, unit)           # B, A, M, S, K
            Pm, Rm = Pc.mean(axis=2), Rc.mean(axis=3)                 # A, M, K / B, A, M, K
            Pmac = np.nanmean(Pc[..., fi], axis=-1).mean(axis=-1)     # A, M
            Rmac = np.nanmean(Rc[..., fi], axis=-1).mean(axis=-1)     # B, A, M
            for (i, a), (j, b) in itertools.combinations(list(enumerate(arms)), 2):
                for mi, m in enumerate(models):
                    items = [(c, Pm[i, mi, cls.index(c)] - Pm[j, mi, cls.index(c)],
                              Rm[:, i, mi, cls.index(c)] - Rm[:, j, mi, cls.index(c)])
                             for c in fixed]
                    items.append(("MACRO_FIXED", Pmac[i, mi] - Pmac[j, mi],
                                  Rmac[:, i, mi] - Rmac[:, j, mi]))
                    for c, d, dr in items:
                        lo, hi = iv(dr, 5, 95)
                        row = {"benchmark": bench, "unit": unit, "model": m, "arm_a": a,
                               "arm_b": b, "cls": c, "diff": d, "lo90": lo, "hi90": hi}
                        for mg in MARGINS:
                            row[f"equiv_{mg}"] = bool(lo > -mg and hi < mg)
                        rows_eq.append(row)
        # M5: full-data prevalence by inverse subsample weights (point estimates)
        N = len(cells.universe)
        for pop in ("own", "common"):
            base = np.ones((1, N)) * (cells.u_common if pop == "common" else 1.0)
            for wname, W in (("subsample", base), ("full_prevalence", base * cells.u_ht)):
                mets = {w: cells.metrics(W, w) for w in ("precision", "recall", "f1")}
                for (a, m, s, u) in cells.mats:
                    for c in cls:
                        k = cls.index(c)
                        rows_ht.append({"benchmark": bench, "arm": a, "model": m, "seed": s,
                                        "unit": u, "population": pop, "weights": wname,
                                        "cls": c,
                                        **{w: mets[w][(a, m, s, u)][0, k] for w in mets}})
        log(f"{bench}: core analyses done")
    outs = {"ms_shares.csv": pd.DataFrame(rows_ms), "h2_subsets.csv": pd.DataFrame(rows_sub),
            "equivalence.csv": pd.DataFrame(rows_eq), "bca.csv": pd.DataFrame(rows_bca),
            "ht.csv": pd.DataFrame(rows_ht), "thresholds.csv": pd.DataFrame(rows_thr)}
    outs["ms_h2.csv"] = h2_verdict(rows_ms)
    outs["h2_subsets_verdict.csv"] = h2_verdict(rows_sub)
    for n, df in outs.items():
        df.to_csv(OUT / n, index=False)
    write_manifest("phase11_analysis_core", inputs=[P5 / "results.csv"],
                   outputs=[OUT / n for n in outs], params={"boot": B, "jackknife_groups": 50,
                                                            "margins": MARGINS})


def ht_h2(B):
    """H2 with every test conversation weighted to full-data class prevalence (M5): the
    same contrast as Phase 5 (flow unit, own population, grouped split), point and
    bootstrap both weighted."""
    rows = []
    for bench in ["CICIDS2017", "UNSW-NB15"]:
        arms, models = G.PRIMARY_ARMS[bench], G.MODELS
        cells = build(bench, arms, models, P5_ROOT(bench), ht=ht_weights(bench))
        elig, _ = eligible(cells, arms, models, A.SMALL[bench])
        for label, h in (("subsample prevalence", False), ("full-data prevalence", True)):
            point, reps, _ = replicate(cells, B, pops=("own",), ht=h)
            for unit in ("flow", "conversation"):
                P = stack(point["own"], arms, models, unit)[0]
                R = stack(reps["own"], arms, models, unit)
                rows += h2_rows(bench, f"{label}, {unit}", P, R, cells.classes, elig)
        log(f"ht_h2 {bench} done")
    pd.DataFrame(rows).to_csv(OUT / "ht_h2.csv", index=False)
    h2_verdict(rows).to_csv(OUT / "ht_h2_verdict.csv", index=False)


# --------------------------------------------------------------------------- variants

def rq4_block(bench, cells, B, label, arms, models, tag):
    """H4, exporter share by unit, and the RQ3 nTE/TE table, for a variant grid."""
    point, reps, _ = replicate(cells, B, pops=("common",))
    cls = cells.classes
    fixed = fixed_classes(cells, arms, models)
    fi = [cls.index(c) for c in fixed]
    rows_w, rows_sh, rows_tab = [], [], []
    W = {}
    for unit in ("flow", "conversation"):
        P = stack(point["common"], arms, models, unit)[0]
        R = stack(reps["common"], arms, models, unit)
        W[unit] = kendall(P, R, fi)
        raw, _ = shares_both(np.moveaxis(P, -1, 0))
        rawR, _ = shares_both(np.moveaxis(R, -1, 1))
        for c in fixed:
            k = cls.index(c)
            lo, hi = iv(rawR[:, k, 0])
            rows_sh.append({"variant": label, "benchmark": bench, "unit": unit, "cls": c,
                            "arm_share": raw[k, 0], "lo": lo, "hi": hi,
                            "mean_f1": float(np.nanmean(P[..., k])),
                            "arm_spread": float(np.nanmax(P[..., k].mean(axis=(1, 2))) -
                                                np.nanmin(P[..., k].mean(axis=(1, 2))))})
        for a in arms:
            Pa = P[arms.index(a)]
            for c in cls:
                rows_tab.append({"variant": label, "benchmark": bench, "unit": unit, "arm": a,
                                 "cls": c, "f1_mean": float(np.nanmean(Pa[..., cls.index(c)]))})
    d = W["conversation"][0] - W["flow"][0]
    dr = W["conversation"][1] - W["flow"][1]
    lo, hi = iv(dr)
    rows_w.append({"variant": label, "benchmark": bench, "models": "+".join(models),
                   "w_flow": W["flow"][0], "w_flow_lo": iv(W["flow"][1])[0],
                   "w_flow_hi": iv(W["flow"][1])[1], "w_conversation": W["conversation"][0],
                   "w_conversation_lo": iv(W["conversation"][1])[0],
                   "w_conversation_hi": iv(W["conversation"][1])[1], "diff": d, "lo": lo,
                   "hi": hi, "verdict": "supported" if lo > 0 else "reversed" if hi < 0 else
                   "not supported (undecided)", "fixed_classes": ";".join(fixed)})
    # exporter share falls or rises from flow to conversation
    sh = pd.DataFrame(rows_sh)
    return pd.DataFrame(rows_w), sh, pd.DataFrame(rows_tab)


def payload(B):
    import phase11_payload as PL
    bench, arms, models = "CICIDS2017", G.PRIMARY_ARMS["CICIDS2017"], G.MODELS
    cells = build(bench, arms, models, lambda a, m: PL.OUT / "preds" / bench / a,
                  label_fn=lambda meta, arm: PL.payload_labels(meta, arm)[0],
                  extra_cols=PL_COLS)
    w, sh, tab = rq4_block(bench, cells, B, "payload", arms, models, "payload")
    w.to_csv(OUT / "payload_h4.csv", index=False)
    sh.to_csv(OUT / "payload_shares.csv", index=False)
    tab.to_csv(OUT / "payload_f1.csv", index=False)
    # the same block on the schedule labels, from the same code, as the reference
    cells = build(bench, arms, models, P5_ROOT(bench))
    w0, sh0, tab0 = rq4_block(bench, cells, B, "schedule", arms, models, "schedule")
    w0.to_csv(OUT / "schedule_h4_check.csv", index=False)
    sh0.to_csv(OUT / "schedule_shares_check.csv", index=False)
    tab0.to_csv(OUT / "schedule_f1_check.csv", index=False)
    write_manifest("phase11_analysis_payload", inputs=[], outputs=[OUT / "payload_h4.csv"],
                   params={"boot": B})


def payload_h2(B):
    """H2 on the payload-aware grid (M1): the Phase 5 contrast (grouped split, own
    population, flow and conversation units) with payload-aware labels."""
    import phase11_payload as PL
    bench, arms, models = "CICIDS2017", G.PRIMARY_ARMS["CICIDS2017"], G.MODELS
    cells = build(bench, arms, models, lambda a, m: PL.OUT / "preds" / bench / a,
                  label_fn=lambda meta, arm: PL.payload_labels(meta, arm)[0],
                  extra_cols=PL_COLS)
    point, reps, _ = replicate(cells, B, pops=("own",))
    elig, _ = eligible(cells, arms, models, A.SMALL[bench])
    rows = []
    for unit in ("flow", "conversation"):
        P = stack(point["own"], arms, models, unit)[0]
        R = stack(reps["own"], arms, models, unit)
        rows += h2_rows(bench, f"payload-aware labels, {unit}", P, R, cells.classes, elig)
    pd.DataFrame(rows).to_csv(OUT / "payload_h2.csv", index=False)
    h2_verdict(rows).to_csv(OUT / "payload_h2_verdict.csv", index=False)


PL_COLS = {"cic_original": ["f:Total Length of Fwd Packets", "f:Total Length of Bwd Packets"],
           "wtmc2021": ["f:Total Length of Fwd Packet", "f:Total Length of Bwd Packet"],
           "crisis2022_rpd": ["f:Total Length of Fwd Packet", "f:Total Length of Bwd Packet"],
           "lycos": ["f:fwd_non_empty_pkt_cnt", "f:bwd_non_empty_pkt_cnt"],
           "bccc2017": ["f:total_payload_bytes"],
           "nfs2023_nte": ["f:protocol", "f:bidirectional_max_ps", "f:bidirectional_psh_packets"],
           "nfs2023_te": ["f:protocol", "f:bidirectional_max_ps", "f:bidirectional_psh_packets"]}


def own(B):
    """RQ3 on each release's own labels, from the Phase 7 own-label retraining."""
    import phase7_common as P7C
    bench, arms, models = "CICIDS2017", ["nfs2023_nte", "nfs2023_te"], G.MODELS
    owns = {}

    def label(meta, arm):
        if arm not in owns:
            t, _ = P7C.load_arm_ext(bench, arm, features=False)
            owns[arm] = t["own"].to_numpy(dtype=object)
        return owns[arm]
    cells = build(bench, arms, models, lambda a, m: P7 / "retrain" / "own" / "preds" / bench / a,
                  label_fn=label)
    point, _, _ = replicate(cells, 1, pops=("common",))
    rows = []
    for unit in ("flow", "conversation"):
        P = stack(point["common"], arms, models, unit)[0]
        for a in arms:
            for c in cells.classes:
                rows.append({"variant": "own labels", "benchmark": bench, "unit": unit,
                             "arm": a, "cls": c,
                             "f1_mean": float(np.nanmean(P[arms.index(a)][..., cells.classes.index(c)]))})
    pd.DataFrame(rows).to_csv(OUT / "own_rq3_f1.csv", index=False)


def ttl(B):
    import phase11_ttl as TT
    bench, arms, models = "UNSW-NB15", G.PRIMARY_ARMS["UNSW-NB15"], G.MODELS
    root = {a: (TT.OUT / "preds" / bench / a if a in TT.DROP else P5 / "preds" / bench / a)
            for a in arms}
    rows, mac = [], []
    for label, rf in (("with TTL", lambda a, m: P5 / "preds" / bench / a),
                      ("without TTL", lambda a, m: root[a])):
        cells = build(bench, arms, models, rf)
        point, reps, _ = replicate(cells, B, pops=("own",))
        elig, _ = eligible(cells, arms, models, A.SMALL[bench])
        P = stack(point["own"], arms, models, "flow")[0]
        R = stack(reps["own"], arms, models, "flow")
        rows += h2_rows(bench, label, P, R, cells.classes, elig)
        fixed = fixed_classes(cells, arms, models)
        fi = [cells.classes.index(c) for c in fixed]
        for unit in ("flow", "conversation"):
            Pu = stack(point["own"], arms, models, unit)[0]
            for a in arms:
                for m in models:
                    mac.append({"setting": label, "unit": unit, "arm": a, "model": m,
                                "macro_fixed": float(np.nanmean(
                                    Pu[arms.index(a), models.index(m)][..., fi]))})
    pd.DataFrame(rows).to_csv(OUT / "ttl_h2.csv", index=False)
    h2_verdict(rows).to_csv(OUT / "ttl_h2_verdict.csv", index=False)
    pd.DataFrame(mac).to_csv(OUT / "ttl_macro.csv", index=False)


def models5(B):
    import phase11_models as PM
    rows_h2, rows_w, mac = [], [], []
    for bench in ["CICIDS2017", "UNSW-NB15"]:
        arms = G.PRIMARY_ARMS[bench]
        allm = G.MODELS + PM.MODELS
        root = lambda a, m: (P5 / "preds" / bench / a if m in G.MODELS  # noqa: E731
                             else PM.OUT / "preds" / bench / a)
        cells = build(bench, arms, allm, root)
        point, reps, _ = replicate(cells, B)
        elig, _ = eligible(cells, arms, G.MODELS, A.SMALL[bench])
        fixed = fixed_classes(cells, arms, G.MODELS)
        fi = [cells.classes.index(c) for c in fixed]
        for label, ms in (("three models", G.MODELS), ("five models", allm)):
            P = stack(point["own"], arms, ms, "flow")[0]
            R = stack(reps["own"], arms, ms, "flow")
            rows_h2 += h2_rows(bench, label, P, R, cells.classes, elig, perdf=True)
            W = {u: kendall(stack(point["common"], arms, ms, u)[0],
                            stack(reps["common"], arms, ms, u), fi)
                 for u in ("flow", "conversation")}
            d = W["conversation"][0] - W["flow"][0]
            lo, hi = iv(W["conversation"][1] - W["flow"][1])
            rows_w.append({"benchmark": bench, "models": label, "w_flow": W["flow"][0],
                           "w_flow_lo": iv(W["flow"][1])[0], "w_flow_hi": iv(W["flow"][1])[1],
                           "w_conversation": W["conversation"][0],
                           "w_conversation_lo": iv(W["conversation"][1])[0],
                           "w_conversation_hi": iv(W["conversation"][1])[1],
                           "diff": d, "lo": lo, "hi": hi,
                           "verdict": "supported" if lo > 0 else "reversed" if hi < 0
                           else "not supported (undecided)"})
        for unit in ("flow", "conversation"):
            Pu = stack(point["common"], arms, allm, unit)[0]
            for a in arms:
                for m in allm:
                    mac.append({"benchmark": bench, "unit": unit, "population": "common",
                                "arm": a, "model": m, "macro_fixed": float(np.nanmean(
                                    Pu[arms.index(a), allm.index(m)][..., fi]))})
    pd.DataFrame(rows_h2).to_csv(OUT / "models_h2.csv", index=False)
    h2_verdict(rows_h2).to_csv(OUT / "models_h2_verdict.csv", index=False)
    pd.DataFrame(rows_w).to_csv(OUT / "models_w.csv", index=False)
    pd.DataFrame(mac).to_csv(OUT / "models_macro.csv", index=False)


def loo(B):
    import phase11_instrument_loo as L
    rows = []
    for bench in ["CICIDS2017", "UNSW-NB15"]:
        arms_all, models = G.PRIMARY_ARMS[bench], G.MODELS
        for name, drop, _ in [("frozen", None, None)] + L.variants(bench):
            if name == "frozen":
                cm = None
                arms = arms_all
            else:
                mp = pd.read_parquet(L.CACHE / bench / f"{name.replace(':', '_')}.parquet")
                mp["day"], mp["file"] = mp["day"].astype(str), mp["file"].astype(str)
                arms = [a for a in arms_all if G.source_arm(bench, a) != drop]

                def cm(meta, arm, mp=mp):
                    src = G.source_arm(bench, arm)
                    k = meta[["day", "file", "row"]].astype({"day": str, "file": str})
                    k["arm"] = src
                    m = k.merge(mp, on=["day", "arm", "file", "row"], how="left",
                                validate="one_to_one")
                    if m["gconv_v"].isna().any():
                        raise AssertionError(f"{name} {arm}: unmapped records")
                    meta = meta.copy()
                    meta["gconv"] = m["gconv_v"].to_numpy(dtype=np.int64)
                    meta["sched"] = m["sched_v"].to_numpy(dtype=object)
                    meta = meta[meta["sched"] != "excluded"]
                    return meta
            cells = build(bench, arms, models, P5_ROOT(bench), conv_map=cm) if cm is None \
                else build_loo(bench, arms, models, cm)
            point, _, _ = replicate(cells, 1, pops=("own",))
            fixed = fixed_classes(cells, arms, models)
            fi = [cells.classes.index(c) for c in fixed]
            P = stack(point["own"], arms, models, "conversation")[0]
            w = float(np.mean(A.kendall_w(np.moveaxis(np.nanmean(P[..., fi], axis=-1), -1, 0))))
            for c in cells.classes:
                for a in arms:
                    rows.append({"benchmark": bench, "variant": name, "arm": a, "cls": c,
                                 "conv_f1": float(np.nanmean(P[arms.index(a)][..., cells.classes.index(c)])),
                                 "w_conversation_own": w})
            log(f"loo {bench} {name}: W {w:.3f}")
    pd.DataFrame(rows).to_csv(OUT / "loo_scores.csv", index=False)


def build_loo(bench, arms, models, cm):
    """As build, but the test records of a cell are those of the frozen arm table whose
    conversation survives the variant (records of excluded conversations dropped)."""
    t0 = time.time()
    cells, parts = None, {}
    for arm in arms:
        meta = load_meta(bench, arm)
        meta["_i"] = np.arange(len(meta))
        parts[arm] = cm(meta, arm)
    classes = sorted(set().union(*[set(m["sched"].unique()) for m in parts.values()]))
    cells = Cells(classes)
    for arm, meta in parts.items():
        pos = pd.Series(np.arange(len(meta)), index=meta["_i"].to_numpy())
        conv_all = meta["gconv"].to_numpy()
        y = meta["sched"].to_numpy(dtype=object)
        sched = dict(zip(conv_all, y))
        inall = dict(zip(conv_all, np.ones(len(conv_all), dtype=bool)))
        for model in models:
            for s in SEEDS:
                pr = pd.read_parquet(P5 / "preds" / bench / arm / f"{model}_grouped_s{s}.parquet")
                j = pos.reindex(pr["i"].to_numpy())
                ok = j.notna().to_numpy()
                i = j[ok].astype(int).to_numpy()
                cells.add((arm, model, s), conv_all[i], y[i],
                          pr["pred"].astype(str).to_numpy(dtype=object)[ok],
                          meta["bytes"].to_numpy(float)[i], sched, inall)
        log(f"{bench} {arm}: loaded ({time.time() - t0:.0f} s)")
    return cells.finalize()


def full(B):
    import phase11_full as FU
    rows = []
    for bench in ["CICIDS2017", "UNSW-NB15"]:
        for arm in G.PRIMARY_ARMS[bench]:
            for label, tag, root in (("subsample", "main", P5 / "preds" / bench / arm),
                                     ("full", FU.TAG, FU.OUT / "preds" / bench / arm)):
                meta = load_meta(bench, arm, tag=tag)
                pr = pd.read_parquet(root / "rf_grouped_s0.parquet")
                i = pr["i"].to_numpy()
                conv, y = meta["gconv"].to_numpy()[i], meta["sched"].to_numpy(dtype=object)[i]
                pred = pr["pred"].astype(str).to_numpy(dtype=object)
                common = meta["in_all"].to_numpy(dtype=bool)[i]
                for pop, m in (("own", np.ones(len(i), bool)), ("common", common)):
                    for unit in ("flow", "conversation"):
                        if unit == "flow":
                            yt, yp = y[m], pred[m]
                        else:
                            tr = CS.conversation_truth(conv[m], y[m])
                            yp = CS.aggregate(conv[m], pred[m], "any",
                                              meta["bytes"].to_numpy(float)[i][m]).reindex(
                                tr.index).to_numpy(dtype=object)
                            yt = tr.to_numpy(dtype=object)
                        for c in np.unique(yt):
                            tp = int(((yt == c) & (yp == c)).sum())
                            fp = int(((yt != c) & (yp == c)).sum())
                            fn = int(((yt == c) & (yp != c)).sum())
                            rows.append({"benchmark": bench, "arm": arm, "data": label,
                                         "population": pop, "unit": unit, "cls": c,
                                         "support": tp + fn, "tp": tp, "fp": fp, "fn": fn,
                                         "precision": tp / (tp + fp) if tp + fp else 0.0,
                                         "recall": tp / (tp + fn) if tp + fn else np.nan,
                                         "f1": 2 * tp / (2 * tp + fp + fn)})
            log(f"full {bench} {arm}")
    pd.DataFrame(rows).to_csv(OUT / "full.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["core", "payload", "own", "ttl", "models", "loo", "full",
                                     "ht_h2", "payload_h2"])
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"core": core, "payload": payload, "own": own, "ttl": ttl, "models": models5, "loo": loo,
     "full": full, "ht_h2": ht_h2,
     "payload_h2": payload_h2}[a.what](a.boot)
