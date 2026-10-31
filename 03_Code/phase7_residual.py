"""
Phase 7, step 7.2: where the large residual classes lose F1 at the
conversation unit (grouped split, common population, rule "any").

Usage, from the workspace root (after phase7_rq4.py has written residual.csv):
    .venv/bin/python 03_Code/phase7_residual.py

For every large-residual class and every arm, summed over the three models and
five seeds: true positives, false negatives (with what the conversation was
predicted as), false positives (with the conversation's true class), and for
false positive conversations how many records the arm wrote and whether the
conversation is "multi record" (two or more records in this arm). Output:
04_Results/phase7/residual_errors.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_scoring as CS  # noqa: E402
import phase5_analysis as A  # noqa: E402
import phase7_common as C  # noqa: E402
import phase7_rq4 as Q  # noqa: E402
from manifest import write_manifest  # noqa: E402

OUT = Q.OUT


def main():
    rs = pd.read_csv(OUT / "residual.csv")
    rs = rs[(rs["variant"] == "baseline") & rs["large_residual"]]
    rows = []
    for bench, q in rs.groupby("benchmark"):
        classes = set(q["cls"])
        for arm in A.ARM_SETS[bench]["primary"]:
            t, _ = C.load_arm_ext(bench, arm, features=False)
            nrec = t.groupby("gconv").size()
            for p in sorted((Q.P5 / "preds" / bench / arm).glob("*_grouped_*.parquet")):
                pr = pd.read_parquet(p)
                i = pr["i"].to_numpy()
                sub = t.iloc[i]
                m = sub["in_all"].to_numpy(bool)
                conv = sub["gconv"].to_numpy()[m]
                pred = pr["pred"].astype(str).to_numpy(dtype=object)[m]
                truth = CS.conversation_truth(conv, sub["sched"].to_numpy(dtype=object)[m])
                yp = CS.aggregate(conv, pred, "any", sub["bytes"].to_numpy(float)[m]) \
                    .reindex(truth.index)
                # records per conversation in the test set, and how many predicted attack
                natt = pd.Series(pred != "benign").groupby(conv).sum().reindex(truth.index)
                df = pd.DataFrame({"t": truth.to_numpy(), "p": yp.to_numpy(),
                                   "n": nrec.reindex(truth.index).to_numpy(),
                                   "natt": natt.to_numpy()})
                for c in classes:
                    tp = df[(df["t"] == c) & (df["p"] == c)]
                    fn = df[(df["t"] == c) & (df["p"] != c)]
                    fp = df[(df["t"] != c) & (df["p"] == c)]
                    rows.append({"benchmark": bench, "cls": c, "arm": arm, "cell": p.stem,
                                 "tp": len(tp), "fn": len(fn), "fp": len(fp),
                                 "fn_as_benign": int((fn["p"] == "benign").sum()),
                                 "fp_from_benign": int((fp["t"] == "benign").sum()),
                                 "fp_multi_record": int((fp["n"] >= 2).sum()),
                                 "fp_one_attack_record_of_several":
                                     int(((fp["n"] >= 2) & (fp["natt"] == 1)).sum()),
                                 "fn_multi_record": int((fn["n"] >= 2).sum()),
                                 "class_multi_record_share": float((pd.concat([tp, fn])["n"] >= 2)
                                                                   .mean()),
                                 "benign_multi_record_share":
                                     float((df.loc[df["t"] == "benign", "n"] >= 2).mean())})
    r = pd.DataFrame(rows)
    s = r.groupby(["benchmark", "cls", "arm"]).agg(
        {**{c: "sum" for c in ["tp", "fn", "fp", "fn_as_benign", "fp_from_benign",
                               "fp_multi_record", "fp_one_attack_record_of_several",
                               "fn_multi_record"]},
         "class_multi_record_share": "mean", "benign_multi_record_share": "mean"}).reset_index()
    s["precision"] = s["tp"] / (s["tp"] + s["fp"])
    s["recall"] = s["tp"] / (s["tp"] + s["fn"])
    s.to_csv(OUT / "residual_errors.csv", index=False)
    write_manifest("phase7_residual", outputs=[OUT / "residual_errors.csv"])
    return s


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print(main().round(3).to_string())
