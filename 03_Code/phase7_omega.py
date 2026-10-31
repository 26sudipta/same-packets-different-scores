"""
Extra analysis, labelled as such (not pre-registered; suggested 27 October 2026
after the manuscript check): omega squared beside the Phase 5 sum-of-squares
shares. The shares compare factors with unequal degrees of freedom (arm 6 on
CICIDS2017, 1 on UNSW-NB15; model 2), which favours the factor with more
levels; omega squared corrects for that.

Usage, from the workspace root:
    .venv/bin/python 03_Code/phase7_omega.py

Balanced arm x model x seed design, one F1 per cell (Phase 5 results.csv,
primary arm sets, grouped split). For a factor f:
    omega2_f = (SS_f - df_f * MS_res) / (SS_total + MS_res)
with MS_res = SS_res / df_res, df_res = (A*M - 1)(S - 1). Point estimates
only (no intervals). Negative values are reported as computed.

Output: 04_Results/phase7/omega_squared.csv and omega_h2.csv (the H2 class
rule restated on omega squared, point estimates: exporter above model if
omega2_arm > omega2_model).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5_analysis as A  # noqa: E402
import phase5_grid as G  # noqa: E402
from manifest import write_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "04_Results" / "phase7"


def omega(F):
    """F (A, M, S). Returns dict factor -> (ss share, omega squared, df)."""
    Aa, M, S = F.shape
    sh = A.ss_shares(F)
    gm = F.mean()
    sst = ((F - gm) ** 2).sum()
    ss = sh * sst
    df = {"arm": Aa - 1, "model": M - 1, "arm:model": (Aa - 1) * (M - 1), "seed": S - 1,
          "residual": (Aa * M - 1) * (S - 1)}
    msr = ss[4] / df["residual"] if df["residual"] else np.nan
    out = {}
    for j, f in enumerate(A.FACTORS[:4]):
        out[f] = (sh[j], (ss[j] - df[f] * msr) / (sst + msr) if sst > 0 else np.nan, df[f])
    return out


def main():
    r = pd.read_csv(ROOT / "04_Results" / "phase5" / "results.csv")
    r = r[(r["split"] == "grouped") & ~r["cls"].str.startswith("MACRO")]
    rows = []
    for bench, arms in [(b, A.ARM_SETS[b]["primary"]) for b in A.ARM_SETS]:
        q = r[(r["benchmark"] == bench) & r["arm"].isin(arms)]
        for (unit, pop, cls), g in q.groupby(["unit", "population", "cls"]):
            p = g.pivot_table(index=["arm", "model", "seed"], values="f1")["f1"]
            idx = pd.MultiIndex.from_product([arms, G.MODELS, G.SEEDS])
            F = p.reindex(idx).to_numpy().reshape(len(arms), len(G.MODELS), len(G.SEEDS))
            if not np.isfinite(F).all():
                continue
            for f, (s, w, d) in omega(F).items():
                rows.append({"benchmark": bench, "unit": unit, "population": pop, "cls": cls,
                             "factor": f, "df": d, "ss_share": s, "omega2": w})
    o = pd.DataFrame(rows)
    o.to_csv(OUT / "omega_squared.csv", index=False)
    h = []
    for bench, small in A.SMALL.items():
        e = pd.read_csv(ROOT / "04_Results" / "phase5" / "h2_judgement.csv")
        e = e[(e["benchmark"] == bench) & e["primary_setting"]].set_index("cls")
        for cls in small:
            x = o[(o["benchmark"] == bench) & (o["unit"] == "flow") & (o["population"] == "own") &
                  (o["cls"] == cls)].set_index("factor")
            if not len(x) or not bool(e.loc[cls, "eligible"]):
                continue
            h.append({"benchmark": bench, "cls": cls,
                      "share_arm": x.loc["arm", "ss_share"], "share_model": x.loc["model", "ss_share"],
                      "omega2_arm": x.loc["arm", "omega2"], "omega2_model": x.loc["model", "omega2"],
                      "phase5_verdict": e.loc[cls, "verdict"],
                      "omega_point": "exporter above model"
                      if x.loc["arm", "omega2"] > x.loc["model", "omega2"]
                      else "model above exporter"})
    h = pd.DataFrame(h)
    h.to_csv(OUT / "omega_h2.csv", index=False)
    write_manifest("phase7_omega", outputs=[OUT / "omega_squared.csv", OUT / "omega_h2.csv"])
    return h


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(main().round(3).to_string())
