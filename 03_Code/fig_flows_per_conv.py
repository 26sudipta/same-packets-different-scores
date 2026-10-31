"""
Step 4.3, Figure 1: how many records each arm writes for one conversation.

Run from the workspace root, after agreement.py:
    .venv/bin/python 03_Code/fig_flows_per_conv.py

Reads 04_Results/phase4/flows_per_conv.csv (all days pooled, conversations present
in every arm, so every bar describes the same traffic). Writes
    05_Paper/figures/fig1_flows_per_conversation.pdf     six CICIDS2017 classes
    05_Paper/figures/figA_flows_per_conversation_cic.pdf every CICIDS2017 class
    05_Paper/figures/figA_flows_per_conversation_unsw.pdf UNSW-NB15, with and
                                                          without duplicate records
Each bar splits an arm's conversations by the number of records it wrote for them
(1, 2, 3, 4 or more); the number at the end of the bar is the mean.

Main figure classes, chosen to show the three behaviours found in step 4.4
(00_Plan/Phase4_Agreement_Notes.md): benign as the reference; FTP-Patator and
SSH-Patator, doubled by tail records; DoS Hulk and DDoS, where the arms that write
tail records disagree with each other; slowloris, cut by timeouts in every arm.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S  # noqa: E402  (sets the style)

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "04_Results" / "phase4" / "flows_per_conv.csv"
FIG = ROOT / "05_Paper" / "figures"

ARM_NAME = {
    "cic_original": "CIC original", "wtmc2021": "WTMC-2021", "crisis2022_rpd": "CRiSIS-2022",
    "lycos": "LYCOS", "nfs2023_nte": "NFS-2023-nTE", "nfs2023_te": "NFS-2023-TE",
    "bccc2017": "BCCC", "unsw_original": "UNSW-NB15 original", "nf_unsw_v3": "NF-UNSW-NB15-v3",
    "cic_unsw": "CIC-UNSW-NB15"}
CIC_ARMS = ["cic_original", "wtmc2021", "crisis2022_rpd", "lycos", "nfs2023_nte",
            "nfs2023_te", "bccc2017"]
UNSW_ARMS = ["unsw_original", "nf_unsw_v3", "cic_unsw"]
CLASS_NAME = {
    "benign": "Benign", "ftp_patator": "FTP-Patator", "ssh_patator": "SSH-Patator",
    "dos_hulk": "DoS Hulk", "ddos": "DDoS", "dos_slowloris": "DoS slowloris",
    "dos_slowhttptest": "DoS slowhttptest", "dos_goldeneye": "DoS GoldenEye",
    "heartbleed": "Heartbleed", "web_bruteforce": "Web brute force", "web_xss": "Web XSS",
    "web_sqli": "Web SQL injection", "bot": "Bot", "portscan": "PortScan",
    "analysis": "Analysis", "backdoor": "Backdoor", "dos": "DoS", "exploits": "Exploits",
    "fuzzers": "Fuzzers", "generic": "Generic", "reconnaissance": "Reconnaissance",
    "shellcode": "Shellcode", "worms": "Worms"}
MAIN = ["benign", "ftp_patator", "ssh_patator", "dos_hulk", "ddos", "dos_slowloris"]

# ordinal blue ramp (dataviz reference palette, steps 250, 400, 550, 700)
SEG = [("share_1", "1", "#86b6ef"), ("share_2", "2", "#3987e5"),
       ("share_3", "3", "#1c5cab"), ("share_4plus", "4+", "#0d366b")]
INK, MUTED, GRID = S.INK, S.MUTED, S.GRID


def panel(ax, d, arms, title, show_arms, k, show_x=True, show_n=True):
    d = d.set_index("arm").reindex(arms)
    y = list(range(len(arms)))[::-1]
    left = pd.Series(0.0, index=d.index)
    for col, _, colour in SEG:
        ax.barh(y, d[col], left=left, height=0.62, color=colour,
                edgecolor="white", linewidth=0.8)
        left = left + d[col]
    for yi, a in zip(y, arms):
        ax.text(1.03, yi, f"{d.loc[a, 'fpc']:.2f}", va="center", ha="left", color=INK,
                fontsize=S.SMALL)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "50%", "100%"] if show_x else [])
    ax.set_yticks(y)
    ax.tick_params(axis="y", length=0)
    ax.set_yticklabels([ARM_NAME[a] for a in arms] if show_arms else [])
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    n = int(d["conversations"].iloc[0])
    S.panel_label(ax, k, f"{title} (n = {n:,})" if show_n else title)


def figure(df, classes, arms, ncol, path, compact=False):
    """classes: (key, title) or (key, title, show_n). compact (the appendix figures) keeps
    the x labels on the lowest panel of each column only, so the page height of 234 mm the
    guidelines allow is not exceeded."""
    nrow = -(-len(classes) // ncol)
    if compact:
        h = 0.135 * len(arms) * nrow + 0.38 * nrow + 0.6
    else:
        h = 0.17 * len(arms) * nrow + 0.62 * nrow + 0.45
    assert h <= 234 * S.MM, f"{path.name} is {h / S.MM:.0f} mm high, more than 234 mm"
    fig, axes = plt.subplots(nrow, ncol, figsize=(S.FULL, h), squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= len(classes):
            ax.axis("off")
            continue
        c, label, show_n = (classes[k] + (True,))[:3]
        panel(ax, df[df["class"] == c], arms, label, k % ncol == 0, k,
              show_x=not compact or k + ncol >= len(classes), show_n=show_n)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for _, _, col in SEG]
    fig.legend(handles, [n for _, n, _ in SEG], title="Records per conversation",
               ncol=4, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / h))
    return S.save(fig, path)  # no timestamp: reruns are byte identical


def main():
    df = pd.read_csv(SRC)
    df = df[(df["day"] == "ALL") & (df["population"] == "common")]
    FIG.mkdir(parents=True, exist_ok=True)
    cic = df[(df["benchmark"] == "CICIDS2017") & (df["variant"] == "all")]
    out = [figure(cic, [(c, CLASS_NAME[c]) for c in MAIN], CIC_ARMS, 3,
                  FIG / "fig1_flows_per_conversation.pdf")]
    rest = [c for c in cic["class"].unique() if c in CLASS_NAME]
    rest = ["benign"] + sorted(c for c in rest if c != "benign")
    out.append(figure(cic, [(c, CLASS_NAME[c]) for c in rest], CIC_ARMS, 3,
                      FIG / "figA_flows_per_conversation_cic.pdf", compact=True))
    un = df[df["benchmark"] == "UNSW-NB15"]
    ucls = ["benign"] + sorted(c for c in un["class"].unique() if c in CLASS_NAME and c != "benign")
    # both panels of a class count the same conversations, so n is given once, on the left
    panels = [(f"{v}|{c}", f"{CLASS_NAME[c]}, {tag}", v == "all") for c in ucls
              for v, tag in (("all", "as released"), ("unsw_dups_dropped", "duplicates dropped"))]
    fig_df = un.assign(**{"class": un["variant"] + "|" + un["class"]})
    out.append(figure(fig_df, panels, UNSW_ARMS, 2,
                      FIG / "figA_flows_per_conversation_unsw.pdf", compact=True))
    for p in out:
        print("wrote", p)
    from manifest import write_manifest
    write_manifest("phase4_fig1", inputs=[SRC], outputs=out,
                   out_dir=ROOT / "04_Results" / "phase4" / "manifests")


if __name__ == "__main__":
    main()
