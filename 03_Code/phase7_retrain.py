"""
Phase 7, step 7.3, the two retrained label variants (protocol:
00_Plan/Phase7_Protocol.md, variants 5 and 6).

Usage, from the workspace root (after phase7_extras.py):
    .venv/bin/python 03_Code/phase7_retrain.py fold CICIDS2017
    .venv/bin/python 03_Code/phase7_retrain.py own CICIDS2017 [--arms ...]

Exactly the Phase 5 grid code (phase5_grid.run: same arm tables, features,
models, Phase 5 tuned settings, grouped split, seeds 0 to 4), with the training
and test labels replaced:
    fold   the schedule label with WTMC-2021 "attempted only" conversations
           folded into benign (instrument column sched_attempted_fold)
    own    the arm's own released label, canonical class names (Phase 3)

Checkpoints: 04_Results/phase7/retrain/<variant>/preds/<bench>/<arm>/*.parquet,
one per cell, skipped when present (a restart costs one cell). The per cell
score file holds flow-unit scores only; the conversation unit and every
analysis are computed by phase7_rq4.py from the predictions.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_scoring as CS  # noqa: E402
import phase5_grid as G  # noqa: E402
import phase7_common as C  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "04_Results" / "phase7" / "retrain"
LABEL_COL = {"fold": "sched_fold", "own": "own"}


def flow_scores(t, test_idx, pred, bench, arm, model, split, seed):
    sub = t.iloc[test_idx]
    r = CS.score(sub["gconv"].to_numpy(), sub["sched"].to_numpy(dtype=object), pred, "flow")
    base = dict(benchmark=bench, arm=arm, model=model, split=split, seed=seed, unit="flow",
                population="own")
    rows = [{**base, "cls": c, "f1": f} for c, f in r["per_class"].items()]
    rows.append({**base, "cls": "MACRO_CPLUS", "f1": r["macro_f1"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", choices=list(LABEL_COL))
    ap.add_argument("bench")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    a = ap.parse_args()
    col = LABEL_COL[a.variant]

    def load_arm(bench, arm, tag="main"):
        t, fcols = C.load_arm_ext(bench, arm)
        t["sched"] = t[col].to_numpy(dtype=object)
        return t, fcols

    G.load_arm = load_arm
    G.scores = flow_scores
    tm = G.run(a.bench, a.arms or G.PRIMARY_ARMS[a.bench], None, ["grouped"], a.seeds,
               "main", OUT / a.variant)
    print(tm.to_string())
