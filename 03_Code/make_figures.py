"""
Step 9.5: every figure of the paper, from the frozen result files, in one run.

Run from the workspace root, after make_numbers.py:
    .venv/bin/python 03_Code/make_figures.py

Writes to 05_Paper/figures/ (one point per figure, plan step 9.5), PDF and an EPS twin in eps/,
in the style of figstyle.py (Springer artwork guidelines):
    fig1_flows_per_conversation.pdf  RQ1  how many records each release writes for one conversation
                                          (and the two supplementary versions; fig_flows_per_conv.py)
    fig2_leakage.pdf                 RQ1  the leakage of the random flow split follows the records
                                          per conversation, as Eq. leak predicts
    fig3_exporter_vs_model.pdf       RQ2  exporter share minus model share per small class, at the
                                          flow unit (H2) and the conversation unit (RQ4)
    fig4_rq3_te_vs_nte.pdf           RQ3  the expiry rule lowers flow F1, not conversation F1
                                          (except DoS Hulk)
    fig5_rq3_leak_vs_inflation.pdf   RQ3  leakage rises with the rule, inflation does not
    fig6_rq5_claims.pdf              RQ5  each examinable published gap: as published, rebuilt per
                                          flow and per conversation on the same conversations
    fig_same_packets.pdf, fig_instrument.pdf   drawings without measured values (fig_schematics.py)
Tables are not drawn here: make_numbers.py writes them. Every value plotted is read from a
04_Results file named below; nothing is typed. The PDFs carry no timestamp, so a rerun on the
same inputs is byte identical.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S  # noqa: E402  (sets the style: Springer artwork guidelines)
import fig_flows_per_conv as F1  # noqa: E402
import fig_schematics  # noqa: E402
import instrument as I  # noqa: E402
from manifest import write_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "04_Results"
FIG = ROOT / "05_Paper" / "figures"
CLASS_NAME, ARM_NAME = F1.CLASS_NAME, F1.ARM_NAME
CIC, UNSW = "CICIDS2017", "UNSW-NB15"

BLUE, ORANGE = S.BLUE, S.ORANGE
INK, MUTED, GRID = S.INK, S.MUTED, S.GRID
WIDTH = S.FULL  # both columns; fig2 and fig5 are one column (S.COLUMN)


def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    return S.save(fig, FIG / name)


# ---------------------------------------------------------------------------
# Figure 2 (RQ1): leakage of the random flow split against records per conversation

def leak_of_n(n, q):
    """Eq. leak: share of test conversations with a record in training, n records each."""
    return (1 - q ** n - (1 - q) ** n) / (1 - (1 - q) ** n)


def fig2():
    src_l, src_f = R / "phase3" / "all_leakage.csv", R / "phase4" / "flows_per_conv.csv"
    lk = pd.read_csv(src_l)
    lk = lk[(lk["split"] == "random_flow") & (lk["sched"] != "excluded")]
    obs = lk.groupby(["benchmark", "arm", "sched"])[["leaky_conversations", "test_conversations"]].sum()
    obs = (obs["leaky_conversations"] / obs["test_conversations"]).rename("leak")
    # the leakage was measured on the releases as published, so the records per conversation are
    # taken on the same records (variant "all", duplicate lines of UNSW-NB15 kept)
    f = pd.read_csv(src_f)
    f = f[(f["variant"] == "all") & (f["day"] == "ALL") & (f["population"] == "own")]
    f = f.rename(columns={"class": "sched"}).set_index(["benchmark", "arm", "sched"])["fpc"]
    d = pd.concat([obs, f], axis=1, join="inner").reset_index()
    assert len(d) == len(obs), "every measured leakage needs its records per conversation"
    q = I.TEST_FRACTION
    fig, ax = plt.subplots(figsize=(S.COLUMN, 2.7))
    style(ax, "y")
    n = np.arange(1, 61)
    ax.plot(n, leak_of_n(n, q), color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.text(40, leak_of_n(40, q) - 0.06, f"$\\lambda(n)$ if every conversation\nhas n records, q = {q:.1f}",
            ha="center", va="top", fontsize=S.SMALL, color=MUTED)
    for bench, col, mk in ((CIC, BLUE, "o"), (UNSW, ORANGE, "s")):
        x = d[d["benchmark"] == bench]
        ax.scatter(x["fpc"], x["leak"], s=16, color=col, marker=mk, edgecolor="white",
                   linewidth=0.6, zorder=3, label=bench)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 50])
    ax.set_xticklabels(["1", "2", "3", "5", "10", "20", "50"])
    ax.set_xlim(0.9, 70)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Records per conversation, mean (log scale)")
    ax.set_ylabel("Leakage, random flow split")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.45), fontsize=S.SMALL, handletextpad=0.3)
    fig.tight_layout()
    return save(fig, "fig2_leakage.pdf"), [src_l, src_f]


# ---------------------------------------------------------------------------
# Figure 3 (RQ2, RQ4): exporter share minus model share per small class

def fig3():
    src = R / "phase5" / "h2_judgement.csv"
    h = pd.read_csv(src)
    h = h[(h["split"] == "grouped") & (h["population"] == "own") & h["eligible"]]
    benches = [CIC, UNSW]
    rows = {b: h[(h["benchmark"] == b) & (h["unit"] == "flow")].sort_values("arm_minus_model")["cls"].tolist()
            for b in benches}
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), sharex=True,
                             gridspec_kw={"width_ratios": [len(rows[b]) for b in benches]})
    fig.subplots_adjust(wspace=0.6)
    for k, (ax, b) in enumerate(zip(axes, benches)):
        style(ax)
        ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
        for unit, col, mk, off, label in (("flow", BLUE, "o", 0.15, "Flow unit (H2, primary setting)"),
                                          ("conversation", ORANGE, "s", -0.15, "Conversation unit (RQ4)")):
            d = h[(h["benchmark"] == b) & (h["unit"] == unit)].set_index("cls").loc[rows[b]]
            y = np.arange(len(rows[b])) + off
            ax.errorbar(d["arm_minus_model"], y, xerr=[d["arm_minus_model"] - d["lo"],
                        d["hi"] - d["arm_minus_model"]], fmt=mk, ms=4, color=col, ecolor=col,
                        elinewidth=1.0, capsize=0, mec="white", mew=0.5, label=label, zorder=3)
        ax.set_yticks(range(len(rows[b])))
        ax.set_yticklabels([CLASS_NAME[c] for c in rows[b]], color=INK)
        top = len(rows[b]) - 0.4
        ax.set_ylim(-0.6, top + 0.5)
        ax.set_xlim(-1.05, 1.05)
        S.panel_label(ax, k, b, pad=12)
        ax.set_xlabel("Exporter share minus model share")
        ax.text(1.0, top + 0.1, "exporter leads", ha="right", va="bottom", fontsize=S.SMALL, color=MUTED)
        ax.text(-1.0, top + 0.1, "model leads", ha="left", va="bottom", fontsize=S.SMALL, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return save(fig, "fig3_exporter_vs_model.pdf"), [src]


# ---------------------------------------------------------------------------
# Figure 4 (RQ3): F1 of NFS-2023-nTE and NFS-2023-TE at both units

def fig4():
    src = R / "phase6" / "table_6_1.csv"
    t = pd.read_csv(src).set_index("cls").drop("benign")
    order = (t["f1_flow_grouped_te"] - t["f1_flow_grouped_nte"]).sort_values().index.tolist()  # largest drop first
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.1), sharey=True)
    for k, (ax, unit, title) in enumerate(zip(axes, ("flow", "conversation"), ("Flow unit", "Conversation unit"))):
        style(ax)
        y = np.arange(len(order))
        a = t.loc[order, f"f1_{unit}_grouped_nte"].to_numpy()
        b = t.loc[order, f"f1_{unit}_grouped_te"].to_numpy()
        ax.hlines(y, np.minimum(a, b), np.maximum(a, b), color=GRID, lw=2.0, zorder=1)
        # nTE as a ring, so that it stays visible where the two arms coincide
        ax.scatter(a, y, s=40, facecolor="none", edgecolor=BLUE, linewidth=1.2, zorder=4,
                   label="NFS-2023-nTE")
        ax.scatter(b, y, s=14, color=ORANGE, zorder=3, label="NFS-2023-TE (FIN and RST expiry)")
        ax.set_xlim(0, 1.02)
        S.panel_label(ax, k, title)
        ax.set_xlabel("F1, grouped split, common population")
        ax.set_yticks(y)
        ax.set_yticklabels([CLASS_NAME[c] for c in order], color=INK)
        ax.set_ylim(len(order) - 0.5, -0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save(fig, "fig4_rq3_te_vs_nte.pdf"), [src]


# ---------------------------------------------------------------------------
# Figure 5 (RQ3): change in leakage against change in inflation

def fig5():
    src = R / "phase6" / "deltas_6_2.csv"
    d = pd.read_csv(src)
    fig, ax = plt.subplots(figsize=(S.COLUMN, 2.9))
    style(ax, "y")
    ax.axhline(0, color=MUTED, lw=0.6, zorder=1)
    ax.scatter(d["d_leak"], d["d_infl"], s=18, color=BLUE, edgecolor="white", linewidth=0.6, zorder=3)
    # direct labels for the points that stand apart; the cluster near 0.7 is named in the caption
    # (the two points on the zero line near the origin get leader lines)
    lead = dict(arrowstyle="-", lw=0.5, color=MUTED, shrinkA=1, shrinkB=2)
    for c, dx, dy, ha, arrow in (("web_bruteforce", -4, 0, "right", None),
                                 ("bot", -4, 0, "right", None),
                                 ("dos_goldeneye", -4, 0, "right", None),
                                 ("dos_slowhttptest", 5, 0, "left", None),
                                 ("portscan", 2, 16, "left", lead),
                                 ("dos_slowloris", 40, 26, "left", lead)):
        r = d.set_index("cls").loc[c]
        ax.annotate(CLASS_NAME[c], (r["d_leak"], r["d_infl"]), xytext=(dx, dy),
                    textcoords="offset points", ha=ha, va="center", fontsize=S.SMALL, color=INK,
                    arrowprops=arrow)
    lim = max(abs(d["d_infl"]).max() * 1.4, 0.01)
    ax.set_ylim(-lim, lim)
    assert d["d_leak"].between(-0.03, 0.85).all(), "a point falls outside the fixed x range"
    ax.set_xlim(-0.03, 0.85)
    ax.set_xlabel("Change in leakage, TE minus nTE")
    ax.set_ylabel("Change in inflation (F1)")
    fig.tight_layout()
    return save(fig, "fig5_rq3_leak_vs_inflation.pdf"), [src]


# ---------------------------------------------------------------------------
# Figure 6 (RQ5): published, rebuilt and conversation-level gaps per examinable contrast

CLAIM = {"R1": "R1 LYCOS", "R2": "R2 WTMC-2021", "R3": "R3 CRiSIS-2022", "R4": "R4 NetFlow v2",
         "R5": "R5 NFS-2023"}
DAY = {"Tuesday": "Tue", "Wednesday": "Wed", "Thursday": "Thu", "Friday": "Fri"}
SHORT = {"cic_original": "CIC", "wtmc2021": "WTMC", "nfs2023_nte": "nTE", "nfs2023_te": "TE",
         "crisis2022_rd": "RD", "crisis2022_rpd": "RPD", "crisis2022_rp": "RP"}


def contrast_label(r):
    c = r["contrast"]
    if r["claim"] == "R1":
        return f"{CLAIM['R1']}, {c.upper()}, MCC"
    if r["claim"] == "R2":
        return f"{CLAIM['R2']}, {CLASS_NAME.get(c, c.capitalize())} F1"
    if r["claim"] == "R3":
        return f"{CLAIM['R3']}, {c}: {SHORT[r['b']]} vs {SHORT[r['a']]} recall"
    if r["claim"] == "R4":
        m = r["metric"]
        what = "binary F1" if m == "f1" else CLASS_NAME[m.split(":")[1]] + " recall"
        return f"{CLAIM['R4']} (v3 proxy), {what}"
    return f"{CLAIM['R5']}, {DAY[r['day']]}, {SHORT[r['b']]} vs {SHORT[r['a']]}"


def fig6():
    src = R / "phase8" / "verdicts_sched.csv"
    v = pd.read_csv(src)
    v = v[v["flow_rebuilt_gap"].notna()].reset_index(drop=True)
    n = len(v)
    fig, ax = plt.subplots(figsize=(WIDTH, 0.21 * n + 1.0))
    style(ax)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    y = np.arange(n)
    ax.scatter(v["published_gap"], y, s=22, marker="D", facecolor="white", edgecolor=INK,
               linewidth=0.7, zorder=4, label="Published")
    for col, lo, hi, c, mk, off, label in (
            ("flow_rebuilt_gap", "flow_rebuilt_lo", "flow_rebuilt_hi", BLUE, "o", 0.16, "Rebuilt, per flow"),
            ("conv_gap", "conv_lo", "conv_hi", ORANGE, "s", -0.16, "Per conversation, shared conversations")):
        d = v[v[col].notna()]
        err = np.vstack([(d[col] - d[lo]).fillna(0), (d[hi] - d[col]).fillna(0)])
        ax.errorbar(d[col], d.index + off, xerr=err, fmt=mk, ms=3.8, color=c, ecolor=c,
                    elinewidth=1.0, capsize=0, mec="white", mew=0.5, zorder=3, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels([contrast_label(r) for _, r in v.iterrows()], color=INK, fontsize=S.SMALL)
    ax.set_ylim(n - 0.5, -0.5)
    x0, x1 = -0.55, 0.9
    shown = pd.concat([v["published_gap"], v["flow_rebuilt_lo"], v["flow_rebuilt_hi"],
                       v["conv_lo"], v["conv_hi"], v["flow_rebuilt_gap"], v["conv_gap"]]).dropna()
    assert shown.between(x0, x1).all(), "a gap or interval end falls outside the fixed x range"
    ax.set_xlim(x0, x1)
    for yi, verdict in zip(y, v["verdict"]):
        ax.text(1.01, yi, verdict.replace(" (zero by construction)", ", zero by construction"),
                transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=S.SMALL, color=MUTED)
    for k in range(1, n):  # thin separators between claims
        if v.loc[k, "claim"] != v.loc[k - 1, "claim"]:
            ax.axhline(k - 0.5, color=GRID, lw=0.8, zorder=0)
    ax.set_xlabel("Gap, B minus A, in the claim's metric")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    return save(fig, "fig6_rq5_claims.pdf"), [src]


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    F1.main()  # Figure 1 and its two supplementary versions (writes its own manifest)
    outs, ins = fig_schematics.main(), []
    for f in (fig2, fig3, fig4, fig5, fig6):
        path, src = f()
        print("wrote", path)
        outs.append(path)
        ins += src
    write_manifest("make_figures", inputs=sorted(set(ins)), outputs=outs,
                   out_dir=ROOT / "04_Results" / "freeze" / "manifests")


if __name__ == "__main__":
    main()
