"""Why does every model score F1 near 0.667 on NFS-2023 TE for several classes?
Confusion of truth against prediction for one cell (lgbm, grouped, seed 0)."""
import pandas as pd
import phase5_grid as G

t, _ = G.load_arm("CICIDS2017", "nfs2023_te")
p = pd.read_parquet(G.OUT / "preds" / "CICIDS2017" / "nfs2023_te" / "lgbm_grouped_s0.parquet")
sub = t.iloc[p["i"].to_numpy()].reset_index(drop=True)
sub["pred"] = p["pred"].to_numpy()
for c in ("ssh_patator", "bot", "dos_goldeneye"):
    a = sub[(sub.sched == c) | (sub.pred == c)]
    print(c, pd.crosstab(a.sched, a.pred).to_string(), sep="\n")
    s = sub[sub.sched == c]
    print("records per conversation", s.groupby("gconv").size().describe().round(2).to_dict())
    fc = [x for x in t.columns if x.startswith("f:")]
    print("rows whose features also occur with another label:",
          int(sub[sub.sched == c].merge(sub[sub.sched != c][fc].drop_duplicates(), on=fc).shape[0]))
