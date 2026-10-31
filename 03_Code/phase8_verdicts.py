"""
Phase 8, step 8.4 verdicts: one row per headline contrast, the published gap, the
rebuilt per-flow gap (8.2), the class weight share and the parts C, W and non-shared
(8.3, metric units, C + W + non-shared = G), the gap per flow and per
conversation on the defined footing (8.4), and the verdict of the protocol:

    survives        same sign as published and the 95 % interval excludes 0
    reverses        interval entirely of the opposite sign
    not supported   otherwise (undecided)
    no-difference claims (R3 Q4, R5 (c)): survives if the interval lies inside
    [-0.01, +0.01], contradicted if it excludes 0, undecided otherwise
    a class with fewer than 20 test conversations in any seed: not estimable

Usage:  .venv/bin/python 03_Code/phase8_verdicts.py [--variant sched|fold]
Writes  04_Results/phase8/verdicts_<variant>.csv and .md
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase8_common as P  # noqa: E402
import phase8_decomp as D  # noqa: E402

DEC = P.OUT / "decomp"
MARGIN = 0.01
MIN_UNITS = 20


def verdict(g, lo, hi, pub_gap, no_diff, n_class):
    if n_class is not None and not np.isnan(n_class) and n_class < MIN_UNITS:
        return "not estimable"
    if np.isnan(lo) or np.isnan(hi):
        return "not computed"
    if no_diff:
        if lo >= -MARGIN and hi <= MARGIN:
            return "survives"
        if lo > 0 or hi < 0:
            return "contradicted"
        return "undecided"
    s = np.sign(pub_gap) if pub_gap is not None and not np.isnan(pub_gap) else np.nan
    if np.isnan(s) or s == 0:
        return "no published direction"
    if (s > 0 and lo > 0) or (s < 0 and hi < 0):   # the whole interval on the published side
        return "survives"
    if (s > 0 and hi < 0) or (s < 0 and lo > 0):
        return "reverses"
    return "not supported"


def _pub_gap(d):
    """B minus A from a claim definition's published values (a pair, or a gap)."""
    for v in d.get("published", {}).values():
        if isinstance(v, tuple) and None not in v:
            return v[1] - v[0]
        if isinstance(v, (int, float)):
            return float(v)
    return np.nan


def load(name):
    p = DEC / name
    return pd.read_csv(p) if p.is_file() else None


def main(variant):
    rows = []
    for claim in ("R1", "R2", "R3", "R4", "R5"):
        flow = load(f"{claim}_gaps.csv")
        cf = load(f"{claim}_conv_{variant}_flow_gaps.csv")
        cc = load(f"{claim}_conv_{variant}_conv_gaps.csv")
        if cc is None:
            continue
        for _, r in cc.iterrows():
            if bool(r.get("side_only", False)):
                continue
            key = r["contrast"]
            f = flow[flow["contrast"] == key].iloc[0] if flow is not None and \
                (flow["contrast"] == key).any() else None
            q = cf[cf["contrast"] == key].iloc[0] if cf is not None and \
                (cf["contrast"] == key).any() else None
            pub = r.get("published_gap", np.nan)
            n_cls = r.get("min_test_units_class", np.nan)
            v = verdict(r["G"], r.get("G_lo", np.nan), r.get("G_hi", np.nan), pub,
                        bool(r.get("no_diff", False)), n_cls)
            if str(r.get("share_note", "")).startswith("zero by construction"):
                v = "not supported (zero by construction)" if not r.get("no_diff") else v
            rows.append(dict(
                claim=claim, contrast=key, day=r.get("day", ""), metric=r["metric"],
                a=r["a"], b=r["b"], proxy=r.get("proxy", ""),
                published_gap=pub,
                flow_rebuilt_gap=None if f is None else f["G"],
                flow_rebuilt_lo=None if f is None else f["G_lo"],
                flow_rebuilt_hi=None if f is None else f["G_hi"],
                flow_direction_reproduced=None if f is None else f.get("direction_reproduced"),
                class_weight_share=None if f is None else f.get("share_C"),
                class_weight_share_lo=None if f is None else f.get("share_C_lo"),
                class_weight_share_hi=None if f is None else f.get("share_C_hi"),
                C=None if f is None else f.get("C"), W=None if f is None else f.get("W"),
                nonshared=None if f is None else f.get("nonshared"),
                defined_flow_gap=None if q is None else q["G"],
                defined_flow_lo=None if q is None else q["G_lo"],
                defined_flow_hi=None if q is None else q["G_hi"],
                conv_gap=r["G"], conv_lo=r.get("G_lo"), conv_hi=r.get("G_hi"),
                conv_A=r.get("A"), conv_B=r.get("B"),
                min_test_conv_class=n_cls, verdict=v))
    # headline contrasts of the protocol with no conversation-level result
    # (Exp_RP and Exp_R were not obtained until 30 Oct 2026; once they are on disk, a missing
    # row is a contrast this variant does not compute, e.g. R5 under the fold truth)
    UNOBTAINED = set() if P.files("crisis2022_rp") else {"crisis2022_rp", "crisis2022_r"}
    have = {(r["claim"], r["contrast"]) for r in rows}
    for claim, make in D.CLAIMS.items():
        for d in make():
            if d.get("side_only") or (claim, d["name"]) in have:
                continue
            if not any(v is not None and not (isinstance(v, tuple) and None in v)
                       for v in d.get("published", {}).values()):
                continue   # a proxy or extra row with no published value of its own
            flow = load(f"{claim}_gaps.csv")
            f = flow[flow["contrast"] == d["name"]].iloc[0] if flow is not None and \
                (flow["contrast"] == d["name"]).any() else None
            if claim == "R3" and d["name"] == "Q2":
                # RD and RPD are one arm under schedule labels (protocol 8.4)
                v, cg = "not supported (zero by construction)", 0.0
            elif {d["a"], d["b"]} & UNOBTAINED:
                v, cg = "not examinable, data not obtained", np.nan
            else:
                v, cg = "not computed", np.nan
            rows.append(dict(
                claim=claim, contrast=d["name"], day=d.get("day", ""), metric=d["metric"],
                a=d["a"], b=d["b"], proxy="",
                published_gap=_pub_gap(d) if f is None else f.get("published_gap"),
                flow_rebuilt_gap=None if f is None else f["G"],
                flow_rebuilt_lo=None if f is None else f["G_lo"],
                flow_rebuilt_hi=None if f is None else f["G_hi"],
                flow_direction_reproduced=None if f is None else f.get("direction_reproduced"),
                class_weight_share=None if f is None else f.get("share_C"),
                class_weight_share_lo=None if f is None else f.get("share_C_lo"),
                class_weight_share_hi=None if f is None else f.get("share_C_hi"),
                C=None if f is None else f.get("C"), W=None if f is None else f.get("W"),
                nonshared=None if f is None else f.get("nonshared"),
                defined_flow_gap=cg, defined_flow_lo=np.nan, defined_flow_hi=np.nan,
                conv_gap=cg, conv_lo=np.nan, conv_hi=np.nan, conv_A=np.nan, conv_B=np.nan,
                min_test_conv_class=np.nan, verdict=v))
    order = {c: i for i, c in enumerate(D.CLAIMS)}
    rows.sort(key=lambda r: order[r["claim"]])
    out = pd.DataFrame(rows)
    out.to_csv(P.OUT / f"verdicts_{variant}.csv", index=False)

    def fmt(x, lo=None, hi=None):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return ""
        s = f"{x:+.3f}"
        if lo is not None and not (isinstance(lo, float) and np.isnan(lo)):
            s += f" [{lo:+.3f}, {hi:+.3f}]"
        return s

    lines = [f"# Phase 8 verdicts ({variant} truth)", "",
             "Generated by 03_Code/phase8_verdicts.py from 04_Results/phase8/decomp. Gaps are "
             "B minus A in the claim's metric; brackets are 95 percent paired cluster bootstrap "
             "intervals over conversations.", "",
             "| Claim | Contrast | Metric | Published | Per flow, rebuilt (8.2) | "
             "Class weight share | C, W, non-shared (8.3) | Per flow, defined | "
             "Per conversation, defined | Verdict |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in out.iterrows():
        name = f"{r['contrast']}" + (f" ({r['proxy']})" if isinstance(r["proxy"], str)
                                     and r["proxy"] else "")
        lines.append(" | ".join([
            f"| {r['claim']}", name, r["metric"], fmt(r["published_gap"]),
            fmt(r["flow_rebuilt_gap"], r["flow_rebuilt_lo"], r["flow_rebuilt_hi"]),
            fmt(r["class_weight_share"], r["class_weight_share_lo"], r["class_weight_share_hi"]),
            ", ".join(fmt(r[k]) for k in ("C", "W", "nonshared")) if r["C"] is not None
            and not (isinstance(r["C"], float) and np.isnan(r["C"])) else "",
            fmt(r["defined_flow_gap"], r["defined_flow_lo"], r["defined_flow_hi"]),
            fmt(r["conv_gap"], r["conv_lo"], r["conv_hi"]), f"{r['verdict']} |"]))
    (P.OUT / f"verdicts_{variant}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out[["claim", "contrast", "published_gap", "flow_rebuilt_gap", "conv_gap", "conv_lo",
               "conv_hi", "verdict"]].round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["sched", "fold"], default="sched")
    main(ap.parse_args().variant)
