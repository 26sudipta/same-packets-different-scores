"""
Step 5.5: variance decomposition and ranking agreement for RQ2, with cluster
bootstrap intervals over conversations; step 5.7 inputs (H2 judgement).

Usage, from the workspace root (after phase5_grid.py run for both benchmarks):
    .venv/bin/python 03_Code/phase5_analysis.py [--boot 1000]

Inputs: 04_Results/phase5/cells/<bench>/<arm>/*.csv (scores per cell) and
04_Results/phase5/preds/... (predictions), with the Phase 5 arm tables.

Outputs in 04_Results/phase5/:
    results.csv          every cell, unit, population and class (long)
    macro_summary.csv    macro-F1 per arm, model, split, unit, population:
                         mean and range over seeds, with and without Heartbleed
    variance_shares.csv  share of the total sum of squares per factor (arm,
                         model, arm x model, seed, residual), per benchmark,
                         arm set, split, unit, population and class, with
                         95 percent bootstrap intervals
    share_contrasts.csv  exporter share minus model share; exporter share on
                         the own population minus on the common population
    kendall_w.csv        Kendall's W of model rankings across arms
    h2_judgement.csv     the protocol's H2 rule, per class and setting
    boot_check.csv       the bootstrap code's point estimates against the
                         cell scores (must agree)

Decomposition. For one class and setting the design is arm x model x seed,
balanced, one F1 per cell. The sums of squares are the classical balanced
ANOVA ones (equal to statsmodels' type I sums for f1 ~ arm * model + seed,
checked in tests/test_phase5.py); the share of a factor is its sum of squares
over the total.

Bootstrap. Conversations are the resampling unit. Each replicate draws, inside
each schedule class, as many conversations as the class holds, with
replacement, from every conversation that is in the test set of some cell
(stratified so that each class keeps its size, which is what a per class F1
conditions on). The same draw is applied to every cell, so arms and models
stay paired. TP, FP and FN are additive over conversations, so a replicate is
a weighted sum of per conversation counts. Percentile intervals, B = 1000,
generator seed 0.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_scoring as CS  # noqa: E402
import phase5_grid as G  # noqa: E402
from manifest import write_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "04_Results" / "phase5"
FACTORS = ["arm", "model", "arm:model", "seed", "residual"]
SMALL = {"CICIDS2017": ["dos_slowhttptest", "ftp_patator", "dos_slowloris", "ssh_patator",
                        "web_bruteforce", "bot", "web_xss", "web_sqli", "heartbleed"],
         "UNSW-NB15": ["generic", "dos", "shellcode", "analysis", "backdoor", "worms"]}
MIN_TEST_CONVS = 20
ARM_SETS = {"CICIDS2017": {"primary": G.PRIMARY_ARMS["CICIDS2017"]},
            "UNSW-NB15": {"primary": ["unsw_dedup", "nf_unsw_v3", "cic_unsw"],
                          "as_released": ["unsw_released", "nf_unsw_v3", "cic_unsw"],
                          "near_repeats_collapsed": ["unsw_collapsed", "nf_unsw_v3", "cic_unsw"],
                          # the two releases of the nine-release study, on the same
                          # (three-arm) instrument: what the tenth release changes
                          "without_cic_unsw": ["unsw_dedup", "nf_unsw_v3"]}}
UNITS = ["flow", "conversation"]
POPS = ["own", "common"]


def collect():
    parts = sorted((OUT / "cells").glob("*/*/*.csv"))
    r = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    r.to_csv(OUT / "results.csv", index=False)
    return r


def macro_summary(r):
    m = r[r["cls"].isin(["MACRO_CPLUS", "MACRO_CPLUS_NO_HEARTBLEED"])]
    g = m.groupby(["benchmark", "arm", "model", "split", "unit", "population", "cls"])["f1"]
    s = g.agg(["mean", "min", "max", "count"]).reset_index()
    s.to_csv(OUT / "macro_summary.csv", index=False)
    return s


def ss_shares(F):
    """Shares of the total sum of squares for a balanced arm x model x seed design.

    F has shape (..., A, M, S). Returns (..., 5) in FACTORS order. All NaN when
    the total sum of squares is zero (every cell equal)."""
    A, M, S = F.shape[-3:]
    gm = F.mean(axis=(-3, -2, -1), keepdims=True)
    ma = F.mean(axis=(-2, -1), keepdims=True)
    mm = F.mean(axis=(-3, -1), keepdims=True)
    ms = F.mean(axis=(-3, -2), keepdims=True)
    mam = F.mean(axis=-1, keepdims=True)
    ss_a = M * S * ((ma - gm) ** 2).sum(axis=(-3, -2, -1))
    ss_m = A * S * ((mm - gm) ** 2).sum(axis=(-3, -2, -1))
    ss_am = S * ((mam - ma - mm + gm) ** 2).sum(axis=(-3, -2, -1))
    ss_s = A * M * ((ms - gm) ** 2).sum(axis=(-3, -2, -1))
    ss_t = ((F - gm) ** 2).sum(axis=(-3, -2, -1))
    ss_r = ss_t - ss_a - ss_m - ss_am - ss_s
    out = np.stack([ss_a, ss_m, ss_am, ss_s, ss_r], axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(ss_t[..., None] > 0, out / ss_t[..., None], np.nan)


def kendall_w(X):
    """Kendall's W with tie correction. X (..., m raters, n items): higher is
    better; items are ranked within each rater."""
    X = np.asarray(X, dtype=float)
    m, n = X.shape[-2:]
    from scipy.stats import rankdata
    R = rankdata(X, axis=-1)
    Rj = R.sum(axis=-2)
    S = ((Rj - Rj.mean(axis=-1, keepdims=True)) ** 2).sum(axis=-1)
    # ties: sum over raters of sum(t^3 - t) over tie groups
    T = np.zeros(X.shape[:-1])
    srt = np.sort(X, axis=-1)
    for i in range(n):
        eq = (srt == srt[..., i:i + 1]).sum(axis=-1)
        T += (eq ** 2 - 1) / 1.0  # each member of a tie group of size t adds t^2 - 1
    T = T.sum(axis=-1)
    den = m ** 2 * (n ** 3 - n) - m * T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 12 * S / den, np.nan)


class Cells:
    """Per conversation TP, FP, FN counts of every cell, for the bootstrap."""

    def __init__(self, bench, arms, classes):
        self.bench, self.classes = bench, list(classes)
        self.K = len(classes)
        self.cidx = {c: i for i, c in enumerate(classes)}
        self.mats, self.keys = {}, []
        self.universe = None

    def build(self, splits=("grouped", "random")):
        convs, mats = [], {}
        for arm in self.arms_all:
            t, _ = G.load_arm(self.bench, arm)
            meta = t[["gconv", "sched", "in_all", "bytes"]]
            for p in sorted((OUT / "preds" / self.bench / arm).glob("*.parquet")):
                model, split, s = p.stem.split("_")
                if split not in splits:
                    continue
                pr = pd.read_parquet(p)
                sub = meta.iloc[pr["i"].to_numpy()]
                conv = sub["gconv"].to_numpy()
                y = sub["sched"].to_numpy(dtype=object)
                pred = pr["pred"].astype(str).to_numpy(dtype=object)
                for unit in UNITS:
                    if unit == "flow":
                        c_, yt, yp = conv, y, pred
                    else:
                        truth = CS.conversation_truth(conv, y)
                        yp = CS.aggregate(conv, pred, "any", sub["bytes"].to_numpy(float))
                        yp = yp.reindex(truth.index).to_numpy(dtype=object)
                        c_, yt = truth.index.to_numpy(), truth.to_numpy(dtype=object)
                    mats[(arm, model, split, int(s[1:]), unit)] = (c_, yt, yp)
                    convs.append(np.unique(c_))
                self._sched = getattr(self, "_sched", {})
                self._sched.update(dict(zip(sub["gconv"].to_numpy(), sub["sched"].to_numpy())))
                self._inall = getattr(self, "_inall", {})
                self._inall.update(dict(zip(sub["gconv"].to_numpy(), sub["in_all"].to_numpy())))
        self.universe = np.unique(np.concatenate(convs))
        pos = pd.Series(np.arange(len(self.universe)), index=self.universe)
        self.u_class = pd.Series(self._sched).reindex(self.universe).to_numpy(dtype=object)
        self.u_common = pd.Series(self._inall).reindex(self.universe).to_numpy(dtype=bool)
        K = self.K
        for key, (c_, yt, yp) in mats.items():
            u = pos.reindex(c_).to_numpy()
            yi = np.array([self.cidx[v] for v in yt])
            pi = np.array([self.cidx.get(v, -1) for v in yp])
            hit = yi == pi
            rows = np.concatenate([u[hit], u[~hit & (pi >= 0)], u[~hit]])
            cols = np.concatenate([yi[hit], K + pi[~hit & (pi >= 0)], 2 * K + yi[~hit]])
            M = sparse.coo_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)),
                                  shape=(len(self.universe), 3 * K)).tocsr()
            nz = np.unique(rows)
            self.mats[key] = (nz, M[nz])
        self.keys = list(self.mats)
        return self

    def f1(self, W):
        """F1 for every cell under weights W (B, N). Returns dict key -> (B, K)."""
        out = {}
        K = self.K
        for key, (nz, M) in self.mats.items():
            C = (M.T @ W[:, nz].T).T
            tp, fp, fn = C[:, :K], C[:, K:2 * K], C[:, 2 * K:]
            sup = tp + fn
            with np.errstate(invalid="ignore", divide="ignore"):
                f = np.where(sup > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
            out[key] = f
        return out


def boot_weights(u_class, B, rng):
    """(B, N) multiplicities: per class, n draws with replacement from its n members."""
    N = len(u_class)
    W = np.zeros((B, N), dtype=np.float64)
    for c in pd.unique(u_class):
        idx = np.flatnonzero(u_class == c)
        n = len(idx)
        draws = rng.integers(0, n, size=(B, n))
        for b in range(B):
            W[b, idx] = np.bincount(draws[b], minlength=n)
    return W


def analyse(bench, r, B, rng_seed=0, chunk=50):
    t0 = time.time()
    classes = sorted(r.loc[(r["benchmark"] == bench) & ~r["cls"].str.startswith("MACRO"),
                           "cls"].unique())
    sets = ARM_SETS[bench]
    arms_all = sorted({a for v in sets.values() for a in v})
    cells = Cells(bench, arms_all, classes)
    cells.arms_all = arms_all
    cells.build()
    models, seeds = G.MODELS, G.SEEDS
    print(f"{bench}: {len(cells.keys)} cell units, universe {len(cells.universe):,} "
          f"conversations ({time.time() - t0:.0f} s)", flush=True)

    # fixed class set for the bootstrap macro: at least MIN_TEST_CONVS test
    # conversations in every grouped seed (own population, conversation unit)
    rb = r[(r["benchmark"] == bench) & (r["split"] == "grouped") &
           (r["unit"] == "conversation") & (r["population"] == "own") &
           ~r["cls"].str.startswith("MACRO")]
    mins = rb.groupby("cls")["support"].min()
    seeds_with = rb.groupby("cls")["seed"].nunique()
    fixed = sorted(c for c in classes if mins.get(c, 0) >= MIN_TEST_CONVS and
                   seeds_with.get(c, 0) == len(seeds))
    fi = [classes.index(c) for c in fixed]

    def stack(fd, arms, split, unit):
        """(B, A, M, S, K) array for one arm set, split and unit."""
        return np.stack([np.stack([np.stack([fd[(a, m, split, s, unit)] for s in seeds],
                                            axis=-2) for m in models], axis=-3)
                         for a in arms], axis=-4)

    # point estimates (weights one), and a check against the cell scores
    ones = np.ones((1, len(cells.universe)))
    point = {pop: cells.f1(ones * (cells.u_common if pop == "common" else 1.0))
             for pop in POPS}
    chk = []
    for (arm, model, split, seed, unit), f in point["own"].items():
        q = r[(r["benchmark"] == bench) & (r["arm"] == arm) & (r["model"] == model) &
              (r["split"] == split) & (r["seed"] == seed) & (r["unit"] == unit) &
              (r["population"] == "own") & ~r["cls"].str.startswith("MACRO")]
        for c, v in zip(q["cls"], q["f1"]):
            chk.append({"benchmark": bench, "arm": arm, "model": model, "split": split,
                        "seed": seed, "unit": unit, "cls": c, "cell_f1": v,
                        "boot_f1": f[0, classes.index(c)]})
    chk = pd.DataFrame(chk)
    bad = (chk["cell_f1"] - chk["boot_f1"]).abs() > 1e-9
    if bad.any():
        raise AssertionError(f"{int(bad.sum())} bootstrap point estimates differ from the cells")

    # bootstrap replicates
    rng = np.random.default_rng(rng_seed)
    reps = {pop: {} for pop in POPS}
    for b0 in range(0, B, chunk):
        W = boot_weights(cells.u_class, min(chunk, B - b0), rng)
        for pop in POPS:
            Wp = W * cells.u_common if pop == "common" else W
            fd = cells.f1(Wp)
            for k, v in fd.items():
                reps[pop].setdefault(k, []).append(v.astype(np.float32))
        print(f"{bench}: bootstrap {b0 + len(W)}/{B} ({time.time() - t0:.0f} s)", flush=True)
    reps = {pop: {k: np.concatenate(v) for k, v in d.items()} for pop, d in reps.items()}

    shares, contrasts, kw = [], [], []
    for set_name, arms in sets.items():
        for split in ["grouped", "random"]:
            for unit in UNITS:
                sh = {}
                for pop in POPS:
                    P = stack(point[pop], arms, split, unit)[0]       # A, M, S, K
                    R = stack(reps[pop], arms, split, unit)           # B, A, M, S, K
                    Ps = ss_shares(np.moveaxis(P, -1, 0))             # K, 5
                    Rs = ss_shares(np.moveaxis(R, -1, 1))             # B, K, 5
                    sh[pop] = (Ps, Rs)
                    defined = np.isfinite(P).all(axis=(0, 1, 2))
                    for k, c in enumerate(classes):
                        for j, fac in enumerate(FACTORS):
                            v = Rs[:, k, j]
                            ok = np.isfinite(v)
                            shares.append({
                                "benchmark": bench, "arm_set": set_name, "split": split,
                                "unit": unit, "population": pop, "cls": c, "factor": fac,
                                "share": Ps[k, j] if defined[k] else np.nan,
                                "lo": np.percentile(v[ok], 2.5) if ok.sum() else np.nan,
                                "hi": np.percentile(v[ok], 97.5) if ok.sum() else np.nan,
                                "boot_valid": int(ok.sum()),
                                "mean_f1": np.nanmean(P[..., k]) if defined[k] else np.nan})
                    # Kendall's W on the fixed class macro-F1
                    Pm = np.nanmean(P[..., fi], axis=-1)                # A, M, S
                    Rm = np.nanmean(R[..., fi], axis=-1)                # B, A, M, S
                    w_seed = kendall_w(np.moveaxis(Pm, -1, 0))          # S
                    w_mean = kendall_w(Pm.mean(axis=-1))
                    rb_seed = kendall_w(np.moveaxis(Rm, -1, 1)).mean(axis=-1)
                    rb_mean = kendall_w(Rm.mean(axis=-1))
                    kw.append({"benchmark": bench, "arm_set": set_name, "split": split,
                               "unit": unit, "population": pop, "raters": len(arms),
                               "items": len(models), "fixed_classes": ";".join(fixed),
                               "w_seed_mean": float(np.mean(w_seed)),
                               "w_seed_mean_lo": np.nanpercentile(rb_seed, 2.5),
                               "w_seed_mean_hi": np.nanpercentile(rb_seed, 97.5),
                               "w_on_seed_means": float(w_mean),
                               "w_on_seed_means_lo": np.nanpercentile(rb_mean, 2.5),
                               "w_on_seed_means_hi": np.nanpercentile(rb_mean, 97.5),
                               "w_per_seed": ";".join(f"{x:.3f}" for x in w_seed)})
                # contrasts: exporter minus model; own minus common (exporter share)
                for k, c in enumerate(classes):
                    for pop in POPS:
                        Ps, Rs = sh[pop]
                        d, dr = Ps[k, 0] - Ps[k, 1], Rs[:, k, 0] - Rs[:, k, 1]
                        ok = np.isfinite(dr)
                        contrasts.append({
                            "benchmark": bench, "arm_set": set_name, "split": split,
                            "unit": unit, "population": pop, "cls": c,
                            "contrast": "arm_minus_model", "value": d,
                            "lo": np.percentile(dr[ok], 2.5) if ok.sum() else np.nan,
                            "hi": np.percentile(dr[ok], 97.5) if ok.sum() else np.nan})
                    (Po, Ro), (Pc, Rc) = sh["own"], sh["common"]
                    d, dr = Po[k, 0] - Pc[k, 0], Ro[:, k, 0] - Rc[:, k, 0]
                    ok = np.isfinite(dr)
                    contrasts.append({
                        "benchmark": bench, "arm_set": set_name, "split": split, "unit": unit,
                        "population": "own_minus_common", "cls": c,
                        "contrast": "arm_share_own_minus_common", "value": d,
                        "lo": np.percentile(dr[ok], 2.5) if ok.sum() else np.nan,
                        "hi": np.percentile(dr[ok], 97.5) if ok.sum() else np.nan})
    print(f"{bench}: done ({time.time() - t0:.0f} s)", flush=True)
    return pd.DataFrame(shares), pd.DataFrame(contrasts), pd.DataFrame(kw), chk, fixed


def h2(contrasts, r):
    """The protocol's H2 rule on the arm_minus_model contrast."""
    rows = []
    for bench, small in SMALL.items():
        rb = r[(r["benchmark"] == bench) & (r["split"] == "grouped") &
               (r["unit"] == "conversation") & (r["population"] == "own")]
        mins = rb.groupby("cls")["support"].min()
        nseed = rb.groupby("cls")["seed"].nunique()
        c = contrasts[(contrasts["benchmark"] == bench) & (contrasts["arm_set"] == "primary") &
                      (contrasts["contrast"] == "arm_minus_model")]
        for cls in small:
            elig = mins.get(cls, 0) >= MIN_TEST_CONVS and nseed.get(cls, 0) == len(G.SEEDS)
            for _, q in c[c["cls"] == cls].iterrows():
                verdict = ("not eligible" if not elig else
                           "exporter above model" if q["lo"] > 0 else
                           "model above exporter" if q["hi"] < 0 else "undecided")
                rows.append({"benchmark": bench, "cls": cls, "split": q["split"],
                             "unit": q["unit"], "population": q["population"],
                             "min_test_conversations": int(mins.get(cls, 0)),
                             "eligible": elig, "arm_minus_model": q["value"],
                             "lo": q["lo"], "hi": q["hi"], "verdict": verdict,
                             "primary_setting": (q["split"] == "grouped" and q["unit"] == "flow"
                                                 and q["population"] == "own")})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--benchmarks", nargs="*", default=["CICIDS2017", "UNSW-NB15"])
    a = ap.parse_args()
    r = collect()
    macro_summary(r)
    S, C, K, CHK, FIX = [], [], [], [], {}
    for bench in a.benchmarks:
        s, c, k, chk, fixed = analyse(bench, r, a.boot)
        S.append(s), C.append(c), K.append(k), CHK.append(chk)
        FIX[bench] = fixed
    S, C, K, CHK = (pd.concat(x, ignore_index=True) for x in (S, C, K, CHK))
    S.to_csv(OUT / "variance_shares.csv", index=False)
    C.to_csv(OUT / "share_contrasts.csv", index=False)
    K.to_csv(OUT / "kendall_w.csv", index=False)
    CHK.to_csv(OUT / "boot_check.csv", index=False)
    H = h2(C, r)
    H.to_csv(OUT / "h2_judgement.csv", index=False)
    outs = [OUT / f for f in ["results.csv", "macro_summary.csv", "variance_shares.csv",
                              "share_contrasts.csv", "kendall_w.csv", "boot_check.csv",
                              "h2_judgement.csv"]]
    write_manifest("phase5_analysis", inputs=sorted((OUT / "cells").glob("*/*/*.csv")),
                   outputs=outs, params={"boot": a.boot, "rng_seed": 0,
                                         "min_test_conversations": MIN_TEST_CONVS,
                                         "fixed_classes": FIX})
    print(H[H["primary_setting"]].to_string())
