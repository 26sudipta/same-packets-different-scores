"""
Step 5.6 (standing gate 4): three result rows drawn at random from the Phase 5
results table and recomputed from the stored predictions by a separate, plain
counting path (no conv_scoring, no scikit-learn).

Usage, from the workspace root:  .venv/bin/python 03_Code/phase5_sanity.py

The draw uses numpy.random.default_rng(20261027) over the per class rows of
04_Results/phase5/results.csv (macro rows excluded), as the protocol states.
Output: 04_Results/phase5/sanity_check.csv with TP, FP, FN counted here, the
F1 they give and the F1 in the results table.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_grid as G  # noqa: E402
from manifest import write_manifest  # noqa: E402

OUT = G.OUT
SEED = 20261027


def any_rule(conv, pred, byts):
    """Predicted class per conversation, written out longhand: benign unless a
    record is predicted as an attack; then the attack class with most records,
    ties by more bytes, then alphabetical."""
    n, b = defaultdict(Counter), defaultdict(Counter)
    for c, p, w in zip(conv, pred, byts):
        n[c]  # every conversation gets an entry
        if p != "benign":
            n[c][p] += 1
            b[c][p] += w
    out = {}
    for c, cnt in n.items():
        if not cnt:
            out[c] = "benign"
            continue
        out[c] = sorted(cnt, key=lambda k: (-cnt[k], -b[c][k], k))[0]
    return out


def recompute(row):
    t, _ = G.load_arm(row["benchmark"], row["arm"])
    name = f"{row['model']}_{row['split']}_s{row['seed']}"
    pr = pd.read_parquet(OUT / "preds" / row["benchmark"] / row["arm"] / f"{name}.parquet")
    sub = t.iloc[pr["i"].to_numpy()]
    y = sub["sched"].to_numpy(dtype=object)
    p = pr["pred"].astype(str).to_numpy(dtype=object)
    conv = sub["gconv"].to_numpy()
    if row["population"] == "common":
        m = sub["in_all"].to_numpy(dtype=bool)
        y, p, conv, byts = y[m], p[m], conv[m], sub["bytes"].to_numpy()[m]
    else:
        byts = sub["bytes"].to_numpy()
    if row["unit"] == "conversation":
        truth = dict(zip(conv, y))
        pc = any_rule(conv, p, byts)
        keys = sorted(truth)
        y = np.array([truth[k] for k in keys], dtype=object)
        p = np.array([pc[k] for k in keys], dtype=object)
    c = row["cls"]
    tp = int(((y == c) & (p == c)).sum())
    fp = int(((y != c) & (p == c)).sum())
    fn = int(((y == c) & (p != c)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return tp, fp, fn, f1, len(y)


if __name__ == "__main__":
    r = pd.read_csv(OUT / "results.csv")
    r = r[~r["cls"].str.startswith("MACRO")].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    pick = rng.choice(len(r), size=3, replace=False)
    rows = []
    for i in pick:
        q = r.iloc[i]
        tp, fp, fn, f1, n = recompute(q)
        rows.append({"row": int(i), "benchmark": q["benchmark"], "arm": q["arm"],
                     "model": q["model"], "split": q["split"], "seed": int(q["seed"]),
                     "unit": q["unit"], "population": q["population"], "cls": q["cls"],
                     "units_scored": n, "tp": tp, "fp": fp, "fn": fn, "f1_recomputed": f1,
                     "f1_table": q["f1"], "support_table": int(q["support"]),
                     "agree": abs(f1 - q["f1"]) < 1e-12 and tp + fn == int(q["support"])})
    s = pd.DataFrame(rows)
    p = OUT / "sanity_check.csv"
    s.to_csv(p, index=False)
    write_manifest("phase5_sanity", inputs=[OUT / "results.csv"], outputs=[p],
                   params={"seed": SEED})
    print(s.to_string())
    if not s["agree"].all():
        raise SystemExit("sanity check failed")
