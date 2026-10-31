"""
Phase 7 (RQ4): H4, the residual and the sensitivity table, from the Phase 5
predictions (and the two retrained label variants). Protocol:
00_Plan/Phase7_Protocol.md, written before any Phase 7 number.

Usage, from the workspace root (after phase7_extras.py, and phase7_retrain.py
for the retrained variants):
    .venv/bin/python 03_Code/phase7_rq4.py [--variants ...] [--boot 1000]

A variant fixes four things: the scoring unit (column of the arm table: gconv
is the frozen T = 15 s conversation), the truth column, the aggregation rule,
and where the predictions come from. Everything else is the Phase 5 analysis
(phase5_analysis.py): grouped split, seeds 0 to 4, balanced ANOVA shares,
Kendall's W on the fixed class macro-F1, cluster bootstrap over units
stratified by schedule class, B = 1000, generator seed 0.

Truth of a unit when its records disagree (only possible for T other than 15,
episodes, and own labels): the attack class with the most records, ties
alphabetical, benign only if every record is benign. With schedule labels at
the frozen conversation this is the Phase 5 truth.

Outputs in 04_Results/phase7/ (one row block per variant): h4.csv,
shares.csv, contrasts.csv, h2.csv, residual.csv, loao.csv, units.csv,
baseline_check.csv, and sensitivity_table.csv / sensitivity_classes.csv
built by --table from the others.
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
import phase5_analysis as A  # noqa: E402
import phase5_grid as G  # noqa: E402
import phase7_common as C  # noqa: E402
from manifest import write_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "04_Results" / "phase7"
P5 = ROOT / "04_Results" / "phase5"
RETRAIN = OUT / "retrain"
BENIGN = "benign"
UNITS = ["flow", "conversation"]
POPS = ["own", "common"]
MIN_UNITS = A.MIN_TEST_CONVS
LARGE_SHARE, LARGE_SPREAD = 0.25, 0.05

VARIANTS = {
    "baseline": dict(unit="gconv", truth="sched", rule="any", preds=P5),
    "T5": dict(unit="cT5", truth="sched", rule="any", preds=P5),
    "T30": dict(unit="cT30", truth="sched", rule="any", preds=P5),
    "T60": dict(unit="cT60", truth="sched", rule="any", preds=P5),
    "T120": dict(unit="cT120", truth="sched", rule="any", preds=P5),
    "rule_majority": dict(unit="gconv", truth="sched", rule="majority", preds=P5),
    "rule_bytes": dict(unit="gconv", truth="sched", rule="bytes", preds=P5),
    "episode": dict(unit="episode_unit", truth="sched", rule="any", preds=P5),
    "fold_scoring": dict(unit="gconv", truth="sched_fold", rule="any", preds=P5,
                         benches=["CICIDS2017"]),
    "own_scoring": dict(unit="gconv", truth="own", rule="any", preds=P5),
    "fold_retrained": dict(unit="gconv", truth="sched_fold", rule="any",
                           preds=RETRAIN / "fold", benches=["CICIDS2017"]),
    "own_retrained": dict(unit="gconv", truth="own", rule="any", preds=RETRAIN / "own"),
}


def unit_truth(unit, y):
    """Label per unit: most common attack class (ties alphabetical), else benign."""
    df = pd.DataFrame({"u": np.asarray(unit), "y": np.asarray(y, dtype=object)})
    units = pd.Index(np.unique(df["u"]), name="u")
    att = df[df["y"] != BENIGN]
    s = att.groupby(["u", "y"]).size().rename("n").reset_index()
    s = s.sort_values(["u", "n", "y"], ascending=[True, False, True], kind="mergesort")
    top = s.drop_duplicates("u").set_index("u")["y"]
    return top.reindex(units).fillna(BENIGN).astype(object)


class VCells(A.Cells):
    """phase5_analysis.Cells over a variant's unit, truth, rule and predictions."""

    def build_variant(self, v, tables):
        mats, univ, pairs_s, pairs_c = {}, [], [], []
        for arm in self.arms_all:
            t = tables[arm]
            unit = t[v["unit"]].to_numpy()
            y = t[v["truth"]].to_numpy(dtype=object)
            sched = t["sched"].to_numpy(dtype=object)
            inall = t["in_all"].to_numpy(dtype=bool)
            w = t["bytes"].to_numpy(float)
            d = Path(v["preds"]) / "preds" / self.bench / arm
            for p in sorted(d.glob("*.parquet")):
                model, split, s = p.stem.split("_")
                i = pd.read_parquet(p, columns=["i"])["i"].to_numpy()
                univ.append(np.unique(unit[i]))
                pairs_s.append(pd.DataFrame({"u": unit[i], "y": sched[i]}).drop_duplicates())
                pairs_c.append(pd.DataFrame({"u": unit[i], "c": inall[i]}).drop_duplicates())
                if split != "grouped":
                    continue
                pred = pd.read_parquet(p)["pred"].astype(str).to_numpy(dtype=object)
                u_, yt_ = unit[i], y[i]
                for un in UNITS:
                    if un == "flow":
                        c_, yt, yp = u_, yt_, pred
                    else:
                        tr = unit_truth(u_, yt_)
                        yp = CS.aggregate(u_, pred, v["rule"], w[i]).reindex(tr.index)
                        c_, yt, yp = tr.index.to_numpy(), tr.to_numpy(dtype=object), \
                            yp.to_numpy(dtype=object)
                    mats[(arm, model, split, int(s[1:]), un)] = (c_, yt, yp)
        self.universe = np.unique(np.concatenate(univ))
        ps = pd.concat(pairs_s, ignore_index=True).drop_duplicates()
        self.u_class = unit_truth(ps["u"], ps["y"]).reindex(self.universe).to_numpy(dtype=object)
        pc = pd.concat(pairs_c, ignore_index=True)
        self.u_common = pc.groupby("u")["c"].min().reindex(self.universe).to_numpy(dtype=bool)
        classes = sorted({c for (_, yt, _) in mats.values() for c in np.unique(yt)})
        self.classes, self.K = classes, len(classes)
        self.cidx = {c: i for i, c in enumerate(classes)}
        pos = pd.Series(np.arange(len(self.universe)), index=self.universe)
        K = self.K
        for key, (c_, yt, yp) in mats.items():
            u = pos.reindex(c_).to_numpy()
            yi = np.array([self.cidx[x] for x in yt])
            pi = np.array([self.cidx.get(x, -1) for x in yp])
            hit = yi == pi
            rows = np.concatenate([u[hit], u[~hit & (pi >= 0)], u[~hit]])
            cols = np.concatenate([yi[hit], K + pi[~hit & (pi >= 0)], 2 * K + yi[~hit]])
            M = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)),
                                  shape=(len(self.universe), 3 * K)).tocsr()
            nz = np.unique(rows)
            self.mats[key] = (nz, M[nz])
        self.keys = list(self.mats)
        return self


def iv(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return np.nan, np.nan
    return np.percentile(x, 2.5), np.percentile(x, 97.5)


def verdict(lo, hi, up="above", down="below", none="undecided"):
    if not np.isfinite(lo):
        return "not estimable"
    return up if lo > 0 else down if hi < 0 else none


def analyse(bench, vname, v, B, chunk=50):
    t0 = time.time()
    arms = A.ARM_SETS[bench]["primary"]
    tables = {}
    for arm in arms:
        tables[arm], _ = C.load_arm_ext(bench, arm, features=False)
    cells = VCells(bench, arms, [])
    cells.arms_all = arms
    cells.build_variant(v, tables)
    classes, K = cells.classes, cells.K
    models, seeds = G.MODELS, G.SEEDS
    print(f"{vname} {bench}: {len(cells.keys)} cell units, universe {len(cells.universe):,} "
          f"({time.time() - t0:.0f} s)", flush=True)

    ones = np.ones((1, len(cells.universe)))
    point = {pop: cells.f1(ones * (cells.u_common if pop == "common" else 1.0)) for pop in POPS}
    # supports at the conversation unit, own population: fixed set and H2 eligibility
    sup = {}
    for (arm, model, split, s, un), (nz, M) in cells.mats.items():
        if un != "conversation":
            continue
        cnt = np.asarray(M[:, :K].sum(axis=0) + M[:, 2 * K:].sum(axis=0)).ravel()
        for k, c in enumerate(classes):
            if cnt[k] > 0:
                sup.setdefault(c, []).append((s, cnt[k]))
    mins = {c: min(n for _, n in L) for c, L in sup.items()}
    nseed = {c: len({s for s, _ in L}) for c, L in sup.items()}
    fixed = sorted(c for c in classes if mins.get(c, 0) >= MIN_UNITS and
                   nseed.get(c, 0) == len(seeds))
    fi = [classes.index(c) for c in fixed]

    rng = np.random.default_rng(0)
    reps = {pop: {} for pop in POPS}
    for b0 in range(0, B, chunk):
        W = A.boot_weights(cells.u_class, min(chunk, B - b0), rng)
        for pop in POPS:
            fd = cells.f1(W * cells.u_common if pop == "common" else W)
            for k_, x in fd.items():
                reps[pop].setdefault(k_, []).append(x.astype(np.float32))
    reps = {pop: {k_: np.concatenate(x) for k_, x in d.items()} for pop, d in reps.items()}
    print(f"{vname} {bench}: bootstrap done ({time.time() - t0:.0f} s)", flush=True)

    def stack(fd, unit, arms_=arms):
        return np.stack([np.stack([np.stack([fd[(a, m, "grouped", s, unit)] for s in seeds],
                                            axis=-2) for m in models], axis=-3)
                         for a in arms_], axis=-4)

    base = {"variant": vname, "benchmark": bench}
    h4, shares, contr, h2, resid, loao = [], [], [], [], [], []
    for pop in POPS:
        P = {u: stack(point[pop], u)[0] for u in UNITS}
        R = {u: stack(reps[pop], u) for u in UNITS}
        Ps = {u: A.ss_shares(np.moveaxis(P[u], -1, 0)) for u in UNITS}
        Rs = {u: A.ss_shares(np.moveaxis(R[u], -1, 1)) for u in UNITS}
        # Kendall's W, per seed mean and on seed means, both units, paired difference
        Wp, Wr = {}, {}
        for u in UNITS:
            if fi:
                Pm = np.nanmean(P[u][..., fi], axis=-1)
                Rm = np.nanmean(R[u][..., fi], axis=-1)
                Wp[u] = (float(np.mean(A.kendall_w(np.moveaxis(Pm, -1, 0)))),
                         float(A.kendall_w(Pm.mean(axis=-1))))
                Wr[u] = (A.kendall_w(np.moveaxis(Rm, -1, 1)).mean(axis=-1),
                         A.kendall_w(Rm.mean(axis=-1)))
            else:
                Wp[u] = (np.nan, np.nan)
                Wr[u] = (np.full(B, np.nan), np.full(B, np.nan))
        for j, stat in enumerate(["w_seed_mean", "w_on_seed_means"]):
            d = Wp["conversation"][j] - Wp["flow"][j]
            dr = Wr["conversation"][j] - Wr["flow"][j]
            lo, hi = iv(dr)
            row = {**base, "population": pop, "statistic": stat,
                   "w_flow": Wp["flow"][j], "w_flow_lo": iv(Wr["flow"][j])[0],
                   "w_flow_hi": iv(Wr["flow"][j])[1],
                   "w_conversation": Wp["conversation"][j],
                   "w_conversation_lo": iv(Wr["conversation"][j])[0],
                   "w_conversation_hi": iv(Wr["conversation"][j])[1],
                   "diff": d, "diff_lo": lo, "diff_hi": hi,
                   "verdict": verdict(lo, hi, "supported", "reversed", "not supported (undecided)")
                   if len(arms) > 2 else "not informative (two arms)",
                   "fixed_classes": ";".join(fixed), "raters": len(arms)}
            h4.append(row)
        for k, c in enumerate(classes):
            for u in UNITS:
                defined = np.isfinite(P[u][..., k]).all()
                for jf, fac in enumerate(A.FACTORS):
                    lo, hi = iv(Rs[u][:, k, jf])
                    shares.append({**base, "unit": u, "population": pop, "cls": c,
                                   "factor": fac, "share": Ps[u][k, jf] if defined else np.nan,
                                   "lo": lo, "hi": hi})
                dmm = Ps[u][k, 0] - Ps[u][k, 1] if defined else np.nan
                lo, hi = iv(Rs[u][:, k, 0] - Rs[u][:, k, 1])
                contr.append({**base, "unit": u, "population": pop, "cls": c,
                              "contrast": "arm_minus_model", "value": dmm, "lo": lo, "hi": hi,
                              "in_fixed_set": c in fixed,
                              "verdict": verdict(lo, hi, "exporter above model",
                                                 "model above exporter")
                              if c in fixed else "not estimable"})
            ok = np.isfinite(P["flow"][..., k]).all() and np.isfinite(P["conversation"][..., k]).all()
            ds = Ps["conversation"][k, 0] - Ps["flow"][k, 0] if ok else np.nan
            lo, hi = iv(Rs["conversation"][:, k, 0] - Rs["flow"][:, k, 0])
            contr.append({**base, "unit": "conversation_minus_flow", "population": pop, "cls": c,
                          "contrast": "arm_share_change", "value": ds, "lo": lo, "hi": hi,
                          "in_fixed_set": c in fixed,
                          "verdict": verdict(lo, hi, "rises", "falls") if c in fixed
                          else "not estimable"})
            if pop == "common":
                Pc = P["conversation"][..., k]
                arm_means = np.nanmean(Pc, axis=(1, 2))
                spread = float(np.nanmax(arm_means) - np.nanmin(arm_means)) \
                    if np.isfinite(arm_means).any() else np.nan
                sh = Ps["conversation"][k, 0] if np.isfinite(Pc).all() else np.nan
                resid.append({**base, "cls": c, "in_fixed_set": c in fixed,
                              "arm_share": sh, "arm_share_lo": iv(Rs["conversation"][:, k, 0])[0],
                              "arm_share_hi": iv(Rs["conversation"][:, k, 0])[1],
                              "spread_arm_means": spread,
                              **{f"mean_{a}": arm_means[ia] for ia, a in enumerate(arms)},
                              "large_residual": bool(c in fixed and np.isfinite(sh) and
                                                     sh >= LARGE_SHARE and spread >= LARGE_SPREAD)})
                if len(arms) > 2:
                    for ia, a in enumerate(arms):
                        keep = [x for x in range(len(arms)) if x != ia]
                        Pk = Pc[keep]
                        loao.append({**base, "cls": c, "dropped_arm": a,
                                     "arm_share": A.ss_shares(Pk)[0]
                                     if np.isfinite(Pk).all() else np.nan})
    # H2 rule per benchmark: small classes, eligible by units, primary and conversation setting
    for cls in A.SMALL[bench]:
        elig = mins.get(cls, 0) >= MIN_UNITS and nseed.get(cls, 0) == len(seeds)
        for u in UNITS:
            q = [r_ for r_ in contr if r_["cls"] == cls and r_["unit"] == u and
                 r_["population"] == "own" and r_["contrast"] == "arm_minus_model"]
            if not q:
                h2.append({**base, "cls": cls, "unit": u, "eligible": False,
                           "min_test_units": int(mins.get(cls, 0)), "verdict": "not eligible"})
                continue
            q = q[0]
            h2.append({**base, "cls": cls, "unit": u, "eligible": elig,
                       "min_test_units": int(mins.get(cls, 0)), "value": q["value"],
                       "lo": q["lo"], "hi": q["hi"],
                       "verdict": verdict(q["lo"], q["hi"], "exporter above model",
                                          "model above exporter") if elig else "not eligible"})
    return (pd.DataFrame(h4), pd.DataFrame(shares), pd.DataFrame(contr), pd.DataFrame(h2),
            pd.DataFrame(resid), pd.DataFrame(loao), units_table(bench, vname, v, tables))


def units_table(bench, vname, v, tables):
    """Independent units per class and seed, pooled over arms: train, test, and
    units with records on both sides of the grouped split (straddling)."""
    t = pd.concat([x[[v["unit"], "sched"] + [f"g{s}" for s in G.SEEDS]] for x in tables.values()],
                  ignore_index=True)
    lab = unit_truth(t[v["unit"]], t["sched"])
    rows = []
    for s in G.SEEDS:
        g = t.groupby(v["unit"])[f"g{s}"].agg(["min", "max"])
        g["cls"] = lab.reindex(g.index).to_numpy()
        for c, q in g.groupby("cls"):
            rows.append({"variant": vname, "benchmark": bench, "seed": s, "cls": c,
                         "test_units": int((q["max"] == 1).sum()),
                         "train_units": int((q["min"] == 0).sum()),
                         "straddling_units": int(((q["min"] == 0) & (q["max"] == 1)).sum())})
    return pd.DataFrame(rows)


def baseline_check():
    """Phase 7 baseline point estimates (and intervals) against Phase 5."""
    out = []
    kw = pd.read_csv(P5 / "kendall_w.csv")
    kw = kw[(kw["arm_set"] == "primary") & (kw["split"] == "grouped")]
    h4 = pd.read_csv(OUT / "h4.csv")
    h4 = h4[h4["variant"] == "baseline"]
    for _, q in kw.iterrows():
        for stat, col in [("w_seed_mean", "w_seed_mean"), ("w_on_seed_means", "w_on_seed_means")]:
            m = h4[(h4["benchmark"] == q["benchmark"]) & (h4["population"] == q["population"]) &
                   (h4["statistic"] == stat)].iloc[0]
            u = "flow" if q["unit"] == "flow" else "conversation"
            for suf in ["", "_lo", "_hi"]:
                out.append({"what": f"{stat}{suf}", "benchmark": q["benchmark"], "unit": u,
                            "population": q["population"], "phase5": q[col + suf],
                            "phase7": m[f"w_{u}{suf}"]})
    vs = pd.read_csv(P5 / "variance_shares.csv")
    vs = vs[(vs["arm_set"] == "primary") & (vs["split"] == "grouped")]
    sh = pd.read_csv(OUT / "shares.csv")
    sh = sh[sh["variant"] == "baseline"]
    m = vs.merge(sh, on=["benchmark", "unit", "population", "cls", "factor"],
                 suffixes=("_p5", "_p7"))
    for c in ["share", "lo", "hi"]:
        d = (m[c + "_p5"] - m[c + "_p7"]).abs()
        out.append({"what": f"variance share {c}, max abs diff over {len(m)} rows",
                    "phase5": 0.0, "phase7": float(np.nanmax(d.to_numpy()))})
    hj = pd.read_csv(P5 / "h2_judgement.csv")
    hj = hj[(hj["split"] == "grouped") & (hj["population"] == "own")]
    h2 = pd.read_csv(OUT / "h2.csv")
    h2 = h2[h2["variant"] == "baseline"]
    m = hj.merge(h2, on=["benchmark", "cls", "unit"], suffixes=("_p5", "_p7"))
    out.append({"what": f"H2 verdicts differing, of {len(m)}", "phase5": 0,
                "phase7": int((m["verdict_p5"] != m["verdict_p7"]).sum())})
    r = pd.DataFrame(out)
    r.to_csv(OUT / "baseline_check.csv", index=False)
    return r


def h2_bench(h2):
    """Benchmark verdict of the H2 rule: holds if exporter above model in a
    strict majority of eligible small classes."""
    rows = []
    for (v, b, u), q in h2.groupby(["variant", "benchmark", "unit"]):
        e = q[q["eligible"]]
        n_up = int((e["verdict"] == "exporter above model").sum())
        rows.append({"variant": v, "benchmark": b, "unit": u, "eligible": len(e),
                     "exporter_above": n_up,
                     "holds": "not estimable" if not len(e) else
                     ("holds" if n_up > len(e) / 2 else "does not hold")})
    return pd.DataFrame(rows)


def table():
    h4 = pd.read_csv(OUT / "h4.csv")
    # protocol 7.1: with two raters W is 1 whenever they agree, so H4 is not informative there
    h4.loc[h4["raters"] <= 2, "verdict"] = "not informative (two arms)"
    h4.to_csv(OUT / "h4.csv", index=False)
    h2 = h2_bench(pd.read_csv(OUT / "h2.csv"))
    h2.to_csv(OUT / "h2_benchmark.csv", index=False)
    co = pd.read_csv(OUT / "contrasts.csv")
    rs = pd.read_csv(OUT / "residual.csv")
    k4 = co[(co["unit"] == "conversation") & (co["population"] == "common") &
            (co["contrast"] == "arm_minus_model")]
    k5 = co[(co["unit"] == "conversation_minus_flow") & (co["population"] == "common")]
    b4 = k4[k4["variant"] == "baseline"].set_index(["benchmark", "cls"])["verdict"]
    b5 = k5[k5["variant"] == "baseline"].set_index(["benchmark", "cls"])["verdict"]
    b6 = rs[(rs["variant"] == "baseline") & rs["large_residual"]].groupby("benchmark")["cls"] \
        .apply(set)
    bh4 = h4[(h4["variant"] == "baseline") & (h4["population"] == "common") &
             (h4["statistic"] == "w_seed_mean")].set_index("benchmark")["verdict"]
    bh2 = h2[h2["variant"] == "baseline"].set_index(["benchmark", "unit"])["holds"]
    rows, cls_rows = [], []
    for (v, b), _ in h4.groupby(["variant", "benchmark"], sort=False):
        q = h4[(h4["variant"] == v) & (h4["benchmark"] == b) & (h4["population"] == "common") &
               (h4["statistic"] == "w_seed_mean")].iloc[0]
        row = {"variant": v, "benchmark": b, "H4_diff": q["diff"], "H4_lo": q["diff_lo"],
               "H4_hi": q["diff_hi"], "K1_H4": q["verdict"],
               "K1_changed": q["verdict"] != bh4.get(b)}
        for u, k in [("flow", "K2"), ("conversation", "K3")]:
            hv = h2[(h2["variant"] == v) & (h2["benchmark"] == b) & (h2["unit"] == u)]
            hv = hv["holds"].iloc[0] if len(hv) else "not estimable"
            row[f"{k}_H2_{u}"] = hv
            row[f"{k}_changed"] = (hv != bh2.get((b, u))) and hv != "not estimable"
        for k, kk, bb in [("K4", k4, b4), ("K5", k5, b5)]:
            q = kk[(kk["variant"] == v) & (kk["benchmark"] == b)].set_index("cls")["verdict"]
            base_cls = [c for (bm, c), x in bb.items() if bm == b and x != "not estimable"]
            ch = ne = 0
            for c in base_cls:
                x = q.get(c, "not estimable")
                if x == "not estimable":
                    ne += 1
                elif x != bb[(b, c)]:
                    ch += 1
                    cls_rows.append({"variant": v, "benchmark": b, "conclusion": k, "cls": c,
                                     "baseline": bb[(b, c)], "variant_verdict": x})
            row[f"{k}_changed"] = ch
            row[f"{k}_not_estimable"] = ne
            row[f"{k}_of"] = len(base_cls)
        s6 = set(rs[(rs["variant"] == v) & (rs["benchmark"] == b) &
                    rs["large_residual"]]["cls"])
        s0 = b6.get(b, set())
        row["K6_large_residual"] = ";".join(sorted(s6))
        row["K6_enter"] = ";".join(sorted(s6 - s0))
        row["K6_leave"] = ";".join(sorted(s0 - s6))
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "sensitivity_table.csv", index=False)
    pd.DataFrame(cls_rows).to_csv(OUT / "sensitivity_classes.csv", index=False)
    return t


def append(name, df, vname, bench):
    p = OUT / name
    old = pd.read_csv(p) if p.is_file() else pd.DataFrame()
    if len(old):
        old = old[~((old["variant"] == vname) & (old["benchmark"] == bench))]
    pd.concat([old, df], ignore_index=True).to_csv(p, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--benchmarks", nargs="*", default=["CICIDS2017", "UNSW-NB15"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--table", action="store_true", help="only rebuild the summary tables")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    names = ["h4.csv", "shares.csv", "contrasts.csv", "h2.csv", "residual.csv", "loao.csv",
             "units.csv"]
    if not a.table:
        for vname in a.variants:
            v = VARIANTS[vname]
            for bench in a.benchmarks:
                if bench not in v.get("benches", [bench]):
                    continue
                d = Path(v["preds"]) / "preds" / bench
                need = len(A.ARM_SETS[bench]["primary"]) * len(G.MODELS) * len(G.SEEDS)
                have = sum(1 for p in d.glob("*/*_grouped_*.parquet")) if d.is_dir() else 0
                if have < need:
                    print(f"{vname} {bench}: {have}/{need} grouped prediction files, skipped",
                          flush=True)
                    continue
                res = analyse(bench, vname, v, a.boot)
                for n, df in zip(names, res):
                    append(n, df, vname, bench)
    if (OUT / "h4.csv").is_file():
        h4 = pd.read_csv(OUT / "h4.csv")
        if (h4["variant"] == "baseline").any():
            print(baseline_check().to_string())
        print(table().to_string())
        write_manifest("phase7_rq4", inputs=[P5 / "results.csv"],
                       outputs=[OUT / n for n in names + ["sensitivity_table.csv",
                                                          "sensitivity_classes.csv",
                                                          "h2_benchmark.csv"]],
                       params={"boot": a.boot, "variants": a.variants,
                               "large_share": LARGE_SHARE, "large_spread": LARGE_SPREAD})
