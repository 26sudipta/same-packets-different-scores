"""
Progress of the Phase 5 run: share of work done and an estimate of the time left.

Usage, from the workspace root:  .venv/bin/python 03_Code/phase5_progress.py

Work is counted in model fits: 4 tuning fits per arm and model (9 tuned arms,
3 models) and one fit per grid cell (CICIDS2017 7 arms, UNSW-NB15 4 arms, 3
models, 2 splits, 5 seeds). Time per fit is the mean of the fits finished so
far for that model (tuning fits use 80 percent of the training data, so the
grid estimate is scaled by 1/0.8). This is an estimate, not a promise.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "04_Results" / "phase5"
TUNED = {"CICIDS2017": 7, "UNSW-NB15": 2}
GRID_ARMS = {"CICIDS2017": 7, "UNSW-NB15": 4}
MODELS = ["rf", "lgbm", "mlp"]

tune = pd.read_csv(OUT / "tuning.csv") if (OUT / "tuning.csv").is_file() else pd.DataFrame()
cells = sorted((OUT / "cells").glob("*/*/*.csv"))
done_cells = pd.concat([pd.read_csv(p, usecols=["model", "seconds"]).head(1) for p in cells],
                       ignore_index=True) if cells else pd.DataFrame(columns=["model", "seconds"])
tune_total = sum(TUNED.values()) * len(MODELS) * 4
grid_total = sum(GRID_ARMS.values()) * len(MODELS) * 2 * 5
sec = {m: tune[tune["model"] == m]["seconds"].mean() if len(tune) else 120.0 for m in MODELS}
gsec = {m: (done_cells[done_cells["model"] == m]["seconds"].mean()
            if (done_cells["model"] == m).any() else sec[m] / 0.8) for m in MODELS}
tune_done = len(tune)
left_tune = sum((sum(TUNED.values()) * 4 - (tune["model"] == m).sum() if len(tune) else
                 sum(TUNED.values()) * 4) * sec[m] for m in MODELS)
left_grid = sum((sum(GRID_ARMS.values()) * 10 - (done_cells["model"] == m).sum()) * gsec[m]
                for m in MODELS)
# overhead: loading tables, scoring, about 15 percent
left = (left_tune + left_grid) * 1.15
total = left + (tune["seconds"].sum() if len(tune) else 0) + done_cells["seconds"].sum()
pct = 100 * (1 - left / total) if total else 0
print(f"Tuning fits done: {tune_done} of {tune_total}")
print(f"Grid cells done:  {len(cells)} of {grid_total}")
print(f"Done by estimated time: {pct:.0f} percent")
print(f"Estimated time left: {left / 3600:.1f} hours")
