"""
Phase 6 (RQ3): NFS-2023 nTE against NFS-2023 TE, steps 6.1 and 6.2.

Usage, from the workspace root:
    .venv/bin/python 03_Code/phase6_rq3.py

The protocol, written before any number here was computed, is
00_Plan/Phase6_Protocol.md. No model is trained: the script reads the Phase 5
tables of the two arms (same conversation subsample), the Phase 5 cell scores
(04_Results/phase5/results.csv) and predictions (04_Results/phase5/preds/),
and three bookkeeping columns of the raw release files for the mechanism.

Outputs in 04_Results/phase6/:
    structure.csv   records per conversation per class, arm and population
    leakage.csv     share of test records whose conversation is also in
                    training, per class, arm, population, split and seed
    scores.csv      F1 per class, arm, unit, split and population (mean, min,
                    max over 3 models x 5 seeds)
    table_6_1.csv   the step 6.1 table (common population)
    test_6_2.csv    the step 6.2 test (rho, p, verdict), with robustness rows
    deltas_6_2.csv  per class changes nTE to TE used by the test
    mechanism.csv   TE and nTE records by position within the conversation
    false_alarms.csv benign test records predicted as an attack class
    unit_gaps.csv   |F1 TE - F1 nTE| at the flow and the conversation unit
    fig_rq3_leak_vs_inflation.pdf
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_grid as G  # noqa: E402
import releases as R  # noqa: E402
from manifest import write_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
P5 = ROOT / "04_Results" / "phase5"
OUT = ROOT / "04_Results" / "phase6"
BENCH = "CICIDS2017"
ARMS = ["nfs2023_nte", "nfs2023_te"]
SEEDS = [0, 1, 2, 3, 4]
MIN_CONV = 20
N_PERM = 100_000
PERM_SEED = 6
BENIGN = "benign"
RAW_COLS = ["id", "expiration_id", "bidirectional_first_seen_ms"]


def population(t, pop):
    return t if pop == "own" else t[t["in_all"].to_numpy(dtype=bool)]


def structure(tabs):
    rows = []
    for arm, t in tabs.items():
        for pop in ("own", "common"):
            per = population(t, pop).groupby(["sched", "gconv"]).size()
            g = per.groupby(level=0)
            d = pd.DataFrame({"convs": g.size(), "records": g.sum(), "fpc_mean": g.mean(),
                              "share_multi": g.apply(lambda x: float((x >= 2).mean())),
                              "fpc_max": g.max()})
            d["arm"], d["population"] = arm, pop
            rows.append(d.rename_axis("cls").reset_index())
    return pd.concat(rows, ignore_index=True)


def leakage(tabs):
    rows = []
    for arm, t in tabs.items():
        for split, pre in (("random", "r"), ("grouped", "g")):
            for s in SEEDS:
                col = t[f"{pre}{s}"].to_numpy()
                train_convs = np.unique(t["gconv"].to_numpy()[col == 0])
                for pop in ("own", "common"):
                    te = population(t[col == 1], pop)
                    leaked = np.isin(te["gconv"].to_numpy(), train_convs)
                    d = pd.DataFrame({"cls": te["sched"].to_numpy(), "leaked": leaked})
                    g = d.groupby("cls")["leaked"].agg(["mean", "size"])
                    for c, r in g.iterrows():
                        rows.append(dict(arm=arm, population=pop, split=split, seed=s, cls=c,
                                         test_records=int(r["size"]), leak=float(r["mean"])))
    return pd.DataFrame(rows)


def scores():
    r = pd.read_csv(P5 / "results.csv")
    r = r[(r["benchmark"] == BENCH) & r["arm"].isin(ARMS)]
    g = r.groupby(["arm", "unit", "split", "population", "cls"])["f1"]
    out = g.agg(f1_mean="mean", f1_min="min", f1_max="max", cells="size").reset_index()
    per_model = r.groupby(["arm", "model", "unit", "split", "population", "cls"])["f1"] \
        .mean().rename("f1_mean").reset_index()
    return out, per_model


def eligible(struct):
    s = struct[(struct["population"] == "common") & (struct["cls"] != BENIGN)]
    n = s.pivot(index="cls", columns="arm", values="convs").fillna(0)
    return sorted(n.index[(n[ARMS] >= MIN_CONV).all(axis=1)])


def wide(df, value, keys):
    return df.pivot_table(index="cls", columns=keys, values=value)


def perm_p(x, y):
    """One-sided p for Spearman rho > observed under permutation of y."""
    rho = spearmanr(x, y).statistic
    n = len(x)
    if n <= 8:
        perms = itertools.permutations(y)
        vals = np.array([spearmanr(x, np.array(p)).statistic for p in perms])
    else:
        rng = np.random.default_rng(PERM_SEED)
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        vals = np.empty(N_PERM)
        for k in range(N_PERM):
            vals[k] = np.corrcoef(rx, rng.permutation(ry))[0, 1]
    return float(rho), float((np.sum(vals >= rho - 1e-12)) / len(vals))


def verdict(rho, p):
    if not np.isfinite(rho) or rho <= 0:
        return "not supported"
    return "supported" if p < 0.05 else "direction only"


def deltas(leak, sc_model, classes, pop, model=None):
    lk = leak[(leak["split"] == "random") & (leak["population"] == pop)] \
        .groupby(["arm", "cls"])["leak"].mean().unstack(0)
    s = sc_model[(sc_model["unit"] == "flow") & (sc_model["population"] == pop)]
    if model is not None:
        s = s[s["model"] == model]
    f = s.groupby(["arm", "split", "cls"])["f1_mean"].mean().unstack([0, 1])
    d = pd.DataFrame(index=classes)
    for a in ARMS:
        d[f"leak_{a}"] = lk[a].reindex(classes)
        d[f"f1_grouped_{a}"] = f[(a, "grouped")].reindex(classes)
        d[f"f1_random_{a}"] = f[(a, "random")].reindex(classes)
        d[f"infl_{a}"] = d[f"f1_random_{a}"] - d[f"f1_grouped_{a}"]
    d["d_leak"] = d["leak_nfs2023_te"] - d["leak_nfs2023_nte"]
    d["d_infl"] = d["infl_nfs2023_te"] - d["infl_nfs2023_nte"]
    return d.rename_axis("cls").reset_index()


def mechanism(tabs):
    """Order each conversation's records by first seen; describe them by position."""
    rows = []
    for arm, t in tabs.items():
        raw = []
        for f, g in t.groupby("file"):
            path = next(p for p in R.files(arm) if p.name == f)
            df = R._read(path, RAW_COLS)
            sub = df.iloc[g["row"].to_numpy()]
            raw.append(pd.DataFrame({"expiration_id": pd.to_numeric(sub["expiration_id"]).to_numpy(),
                                     "first_ms": pd.to_numeric(sub["bidirectional_first_seen_ms"])
                                     .to_numpy()}, index=g.index))
        raw = pd.concat(raw).reindex(t.index)
        m = t[["gconv", "sched", "in_all"]].join(raw)
        m = m[m["in_all"].to_numpy(dtype=bool)]
        m["n_rec"] = m.groupby("gconv")["gconv"].transform("size")
        m["pos"] = m.sort_values(["gconv", "first_ms"]).groupby("gconv").cumcount() + 1
        m["pos"] = m["pos"].clip(upper=3)
        f = t.loc[m.index]
        m["packets"] = f["f:bidirectional_packets"]
        m["fwd_bytes"] = f["f:src2dst_bytes"]
        m["bwd_packets"] = f["f:dst2src_packets"]
        m["duration_ms"] = f["f:bidirectional_duration_ms"]
        m["syn"] = f["f:bidirectional_syn_packets"]
        m["fin_or_rst"] = (f["f:bidirectional_fin_packets"] + f["f:bidirectional_rst_packets"]) > 0
        m["ack"] = f["f:bidirectional_ack_packets"] > 0
        # detection of each record over the grouped cells (3 models x 5 seeds)
        hit, seen = np.zeros(len(t)), np.zeros(len(t))
        for p in sorted((P5 / "preds" / BENCH / arm).glob("*_grouped_s*.parquet")):
            pr = pd.read_parquet(p)
            i = pr["i"].to_numpy()
            seen[i] += 1
            hit[i] += (pr["pred"].to_numpy() == t["sched"].to_numpy()[i])
        m["recall"] = np.where(seen[m.index] > 0, hit[m.index] / np.maximum(seen[m.index], 1),
                               np.nan)
        m = m[m["n_rec"] >= 2]
        for (c, pos), g in m.groupby(["sched", "pos"]):
            ex = g["expiration_id"].value_counts(normalize=True)
            rows.append(dict(arm=arm, cls=c, pos=("3+" if pos == 3 else str(pos)),
                             records=len(g), packets_median=g["packets"].median(),
                             fwd_bytes_median=g["fwd_bytes"].median(),
                             bwd_packets_median=g["bwd_packets"].median(),
                             duration_ms_median=g["duration_ms"].median(),
                             share_syn=float((g["syn"] > 0).mean()),
                             share_fin_or_rst=float(g["fin_or_rst"].mean()),
                             share_ack=float(g["ack"].mean()),
                             **{f"share_exp_{int(k)}": float(v) for k, v in ex.items()},
                             recall_grouped=float(g["recall"].mean())))
    return pd.DataFrame(rows)


def false_alarms(tabs):
    """Benign test records predicted as each attack class, grouped cells, mean per cell."""
    rows = []
    for arm, t in tabs.items():
        y = t["sched"].to_numpy()
        common = t["in_all"].to_numpy(dtype=bool)
        cells = sorted((P5 / "preds" / BENCH / arm).glob("*_grouped_s*.parquet"))
        acc = {}
        for p in cells:
            pr = pd.read_parquet(p)
            i = pr["i"].to_numpy()
            pred = pr["pred"].to_numpy()
            b = (y[i] == BENIGN) & common[i] & (pred != BENIGN)
            for c, n in pd.Series(pred[b]).value_counts().items():
                acc[c] = acc.get(c, 0) + n
        for c, n in sorted(acc.items()):
            rows.append(dict(arm=arm, predicted=c, benign_records_per_cell=n / len(cells)))
    return pd.DataFrame(rows)


def figure(d, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.axhline(0, color="0.8", lw=0.8)
    ax.axvline(0, color="0.8", lw=0.8)
    ax.scatter(d["d_leak"], d["d_infl"], s=18, color="#2a6f97")
    for _, r in d.iterrows():
        ax.annotate(r["cls"], (r["d_leak"], r["d_infl"]), fontsize=6, xytext=(3, 2),
                    textcoords="offset points")
    ax.set_xlabel("Change in leakage, TE minus nTE")
    ax.set_ylabel("Change in flow F1 inflation, TE minus nTE")
    fig.tight_layout()
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tabs = {a: G.load_arm(BENCH, a)[0] for a in ARMS}
    for a, t in tabs.items():
        for pre in ("g", "r"):
            vals = sorted(pd.unique(t[[f"{pre}{s}" for s in SEEDS]].to_numpy().ravel()).tolist())
            print(a, pre, "split values", vals, flush=True)

    struct = structure(tabs)
    struct.to_csv(OUT / "structure.csv", index=False)
    leak = leakage(tabs)
    leak.to_csv(OUT / "leakage.csv", index=False)
    g_leak = leak[leak["split"] == "grouped"]["leak"].max()
    print("max leakage on the grouped split (must be 0):", g_leak, flush=True)
    assert g_leak == 0
    sc, sc_model = scores()
    sc.to_csv(OUT / "scores.csv", index=False)
    classes = eligible(struct)
    print("eligible classes:", classes, flush=True)

    # 6.1 table, common population
    st = struct[struct["population"] == "common"].set_index(["cls", "arm"])
    lk = leak[(leak["split"] == "random") & (leak["population"] == "common")] \
        .groupby(["cls", "arm"])["leak"].mean()
    s = sc[sc["population"] == "common"].set_index(["cls", "arm", "unit", "split"])["f1_mean"]
    rows = []
    for c in [BENIGN] + classes:
        row = {"cls": c}
        for a in ARMS:
            k = "nte" if a == "nfs2023_nte" else "te"
            row[f"convs_{k}"] = st.loc[(c, a), "convs"]
            row[f"fpc_{k}"] = st.loc[(c, a), "fpc_mean"]
            row[f"multi_{k}"] = st.loc[(c, a), "share_multi"]
            row[f"leak_random_{k}"] = lk.get((c, a), np.nan)
            for u in ("flow", "conversation"):
                for sp in ("grouped", "random"):
                    row[f"f1_{u}_{sp}_{k}"] = s.get((c, a, u, sp), np.nan)
        rows.append(row)
    t61 = pd.DataFrame(rows)
    t61.to_csv(OUT / "table_6_1.csv", index=False)

    # 6.2 test
    tests, dd = [], None
    for pop in ("common", "own"):
        for model in (None, "rf", "lgbm", "mlp"):
            d = deltas(leak, sc_model, classes, pop, model).dropna()
            rho, p = perm_p(d["d_leak"].to_numpy(), d["d_infl"].to_numpy())
            tests.append(dict(population=pop, model=model or "all", n_classes=len(d), rho=rho,
                              p_one_sided=p, verdict=verdict(rho, p),
                              role="primary" if (pop == "common" and model is None)
                              else "robustness"))
            if pop == "common" and model is None:
                dd = d
    pd.DataFrame(tests).to_csv(OUT / "test_6_2.csv", index=False)
    dd.to_csv(OUT / "deltas_6_2.csv", index=False)
    figure(dd, OUT / "fig_rq3_leak_vs_inflation.pdf")

    # second prediction: conversation-unit gap smaller than flow-unit gap (grouped)
    gaps = []
    for c in classes:
        fg = abs(s.get((c, "nfs2023_te", "flow", "grouped"), np.nan) -
                 s.get((c, "nfs2023_nte", "flow", "grouped"), np.nan))
        cg = abs(s.get((c, "nfs2023_te", "conversation", "grouped"), np.nan) -
                 s.get((c, "nfs2023_nte", "conversation", "grouped"), np.nan))
        gaps.append(dict(cls=c, gap_flow=fg, gap_conversation=cg, smaller_at_conv=cg < fg))
    pd.DataFrame(gaps).to_csv(OUT / "unit_gaps.csv", index=False)

    mech = mechanism(tabs)
    mech.to_csv(OUT / "mechanism.csv", index=False)
    false_alarms(tabs).to_csv(OUT / "false_alarms.csv", index=False)
    write_manifest("phase6_rq3",
                   inputs=[P5 / "results.csv"] + sorted((P5 / "preds" / BENCH).glob("nfs2023_*/*")),
                   outputs=sorted(OUT.glob("*.csv")) + sorted(OUT.glob("*.pdf")),
                   params=dict(min_conv=MIN_CONV, n_perm=N_PERM, perm_seed=PERM_SEED))
    print(pd.DataFrame(tests).to_string(), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
