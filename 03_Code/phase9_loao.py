"""
Step 9.2: leave-one-arm-out variance shares for the RQ2 primary setting (grouped split, flow
unit, own population), so that the numbers the RQ2 text quotes come from a file. Point
estimates only, as in Phase 5 (no bootstrap). Uses the same balanced sum-of-squares split as
03_Code/phase5_analysis.py (ss_shares) on the per-class F1 of 04_Results/phase5/results.csv.

Check: with no arm dropped the shares equal the point estimates in variance_shares.csv.

    .venv/bin/python 03_Code/phase9_loao.py      ->  04_Results/phase5/loao_flow.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_analysis import ss_shares  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
P5 = ROOT / "04_Results" / "phase5"
FACTORS = ["arm", "model", "arm:model", "seed", "residual"]
ARMS = ["cic_original", "wtmc2021", "crisis2022_rpd", "lycos", "nfs2023_nte", "nfs2023_te", "bccc2017"]


def shares(r, arms):
    out = []
    for cls, d in r.groupby("cls"):
        piv = d.pivot_table(index=["arm", "model", "seed"], values="f1")["f1"]
        models, seeds = sorted(d["model"].unique()), sorted(d["seed"].unique())
        idx = pd.MultiIndex.from_product([arms, models, seeds], names=["arm", "model", "seed"])
        F = piv.reindex(idx).to_numpy().reshape(len(arms), len(models), len(seeds))
        if np.isnan(F).any():
            continue
        out.append(dict(zip(FACTORS, ss_shares(F)), cls=cls))
    return pd.DataFrame(out)


def main():
    r = pd.read_csv(P5 / "results.csv")
    r = r[(r["benchmark"] == "CICIDS2017") & (r["split"] == "grouped") & (r["unit"] == "flow")
          & (r["population"] == "own") & r["arm"].isin(ARMS) & ~r["cls"].str.startswith("MACRO")]
    full = shares(r, ARMS)
    vs = pd.read_csv(P5 / "variance_shares.csv")
    vs = vs[(vs["benchmark"] == "CICIDS2017") & (vs["arm_set"] == "primary") & (vs["split"] == "grouped")
            & (vs["unit"] == "flow") & (vs["population"] == "own")]
    chk = full.melt(id_vars="cls", var_name="factor", value_name="mine").merge(vs, on=["cls", "factor"])
    dev = (chk["mine"] - chk["share"]).abs().max()
    print(f"full design against variance_shares.csv: largest difference {dev:.2e}")
    assert dev < 1e-9, dev
    rows = [full.assign(dropped_arm="none")]
    for a in ARMS:
        rows.append(shares(r[r["arm"] != a], [x for x in ARMS if x != a]).assign(dropped_arm=a))
    t = pd.concat(rows, ignore_index=True)[["dropped_arm", "cls"] + FACTORS]
    t.to_csv(P5 / "loao_flow.csv", index=False)
    print(t[t["dropped_arm"].isin(["none", "nfs2023_te"])].round(3).to_string())


if __name__ == "__main__":
    main()
