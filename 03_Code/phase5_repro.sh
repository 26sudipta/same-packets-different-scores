#!/usr/bin/env bash
# Step 5.4: rerun three grid cells into a separate folder and compare with the checkpointed ones.
set -e
cd "$(dirname "$0")/.."
T=04_Results/phase5/repro
mkdir -p "$T"
P=.venv/bin/python
$P 03_Code/phase5_grid.py run UNSW-NB15 --arms unsw_dedup --models rf --splits grouped --seeds 1 --out "$T"
$P 03_Code/phase5_grid.py run CICIDS2017 --arms wtmc2021 --models lgbm --splits random --seeds 2 --out "$T"
$P 03_Code/phase5_grid.py run CICIDS2017 --arms lycos --models mlp --splits grouped --seeds 0 --out "$T"
$P - <<'PY'
import pandas as pd
from pathlib import Path
O = Path("04_Results/phase5"); T = O / "repro"
rows = []
for b, a, n in [("UNSW-NB15", "unsw_dedup", "rf_grouped_s1"),
                ("CICIDS2017", "wtmc2021", "lgbm_random_s2"),
                ("CICIDS2017", "lycos", "mlp_grouped_s0")]:
    x = pd.read_parquet(O / "preds" / b / a / f"{n}.parquet")
    y = pd.read_parquet(T / "preds" / b / a / f"{n}.parquet")
    same = x.equals(y)
    rows.append(dict(benchmark=b, arm=a, cell=n, rows=len(x), predictions_identical=same))
r = pd.DataFrame(rows)
r.to_csv(O / "reproducibility_check.csv", index=False)
print(r.to_string())
raise SystemExit(0 if r.predictions_identical.all() else 1)
PY
