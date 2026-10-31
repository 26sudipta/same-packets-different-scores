"""
The two drawings of the paper that carry no measured value: what one exporter rule does to one
conversation (Section 3) and the parts of the instrument (Section 5).

Run from the workspace root (make_figures.py calls it):
    .venv/bin/python 03_Code/fig_schematics.py

Writes 05_Paper/figures/fig_same_packets.pdf and fig_instrument.pdf (and EPS twins). The packet
sequence of the first is illustrative: the record boundaries follow the mechanism of Section 6.3
(the first TE record ends on the FIN, the second is the last ACK of the close), not one measured
conversation. Counts of releases are read from the arm lists, nothing else is a number.
"""
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S  # noqa: E402  (sets the style)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

import fig_flows_per_conv as F1  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "05_Paper" / "figures"
FILL = "#e9f1fb"   # light tint of BLUE for boxes
DARK = "#1c5cab"   # flagged records


def canvas(width_mm, height_mm):
    fig = plt.figure(figsize=(width_mm * S.MM, height_mm * S.MM))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, height_mm)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def arrow(ax, a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.8,
                                 color=S.MUTED, shrinkA=0, shrinkB=0, **kw))


# ---------------------------------------------------------------------------
# Same packets, different records

def same_packets():
    W, H = 174, 62
    fig, ax = canvas(W, H)
    x0, x1 = 34, 118                       # time axis of the conversation, mm
    t = lambda s: x0 + (x1 - x0) * s / 10  # noqa: E731  (time 0..10 to mm)
    rows = {"pk": 50, "nte": 32, "te": 16}
    lab = dict(ha="right", va="center", fontsize=S.SIZE, color=S.INK)
    ax.text(x0 - 4, rows["pk"], "Packets", **lab)
    ax.text(x0 - 4, rows["nte"], "NFS-2023-nTE", **lab)
    ax.text(x0 - 4, rows["te"], "NFS-2023-TE", **lab)
    # packets: up = initiator to responder, down = the reverse direction
    ax.plot([t(-0.2), t(10.2)], [rows["pk"]] * 2, color=S.MUTED, lw=0.6)
    pk = [(0.0, 1, "SYN"), (0.3, -1, ""), (0.6, 1, ""), (1.4, 1, ""), (2.0, -1, ""), (2.9, 1, ""),
          (3.6, -1, ""), (4.5, 1, ""), (5.3, -1, ""), (6.2, 1, "FIN"), (6.6, -1, "FIN"),
          (9.6, 1, "last ACK")]
    for s, d, name in pk:
        ax.plot([t(s)] * 2, [rows["pk"], rows["pk"] + 4.5 * d], color=S.INK, lw=0.9)
        if name:
            ax.text(t(s), rows["pk"] + 5.5 * d, name, ha="center", va="bottom" if d > 0 else "top",
                    fontsize=S.SMALL, color=S.INK)

    def record(a, b, y, flagged, text):
        w = max(t(b) - t(a), 2.2)
        ax.add_patch(Rectangle((t(a) if w > 2.2 else t(a) - 1.1, y - 3), w, 6, lw=0.8,
                               edgecolor=DARK, facecolor=DARK if flagged else "white",
                               hatch=None if flagged else "////"))
        ax.text(t(a) + (t(b) - t(a)) / 2, y - 5, text, ha="center", va="top", fontsize=S.SMALL,
                color=S.INK)
    record(0.0, 9.6, rows["nte"], True, "one record, flagged")
    record(0.0, 6.6, rows["te"], True, "record 1 ends on the FIN, flagged")
    record(9.6, 9.6, rows["te"], False, "record 2, missed")
    # outcome for this conversation, right of the drawing
    cx = 138
    ax.text(cx, rows["nte"] + 9, "Flow\nrecall", ha="center", va="bottom", fontsize=S.SIZE, color=S.INK)
    ax.text(cx + 25, rows["nte"] + 9, "Conversation\nrecall (any)", ha="center", va="bottom",
            fontsize=S.SIZE, color=S.INK)
    for y, flow in ((rows["nte"], "1 of 1"), (rows["te"], "1 of 2")):
        ax.text(cx, y, flow, ha="center", va="center", fontsize=S.SIZE, color=S.INK)
        ax.text(cx + 25, y, "1 of 1", ha="center", va="center", fontsize=S.SIZE, color=S.INK)
    ax.plot([cx - 9, cx + 37], [rows["nte"] + 7.5] * 2, color=S.MUTED, lw=0.6)
    # key
    ax.add_patch(Rectangle((x0, 2), 5, 3, facecolor=DARK, edgecolor=DARK, lw=0.8))
    ax.text(x0 + 7, 3.5, "flagged by the detector", va="center", fontsize=S.SMALL, color=S.INK)
    ax.add_patch(Rectangle((x0 + 50, 2), 5, 3, facecolor="white", edgecolor=DARK, hatch="////", lw=0.8))
    ax.text(x0 + 57, 3.5, "missed", va="center", fontsize=S.SMALL, color=S.INK)
    return S.save(fig, FIG / "fig_same_packets.pdf")


# ---------------------------------------------------------------------------
# The instrument

def box(ax, x, y, w, h, title, body, num=None, wrap=22, fill=FILL, tsize=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                facecolor=fill, edgecolor=S.MUTED, lw=0.6))
    head = f"{num} {title}" if num else title
    ax.text(x + 2, y + h - 2, head, ha="left", va="top", fontsize=tsize, fontweight="bold",
            color=S.INK)
    ax.text(x + 2, y + h - 7, "\n".join(textwrap.wrap(body, wrap)), ha="left", va="top",
            fontsize=S.SMALL, color=S.INK, linespacing=1.25)


def instrument():
    W, H = 174, 104
    fig, ax = canvas(W, H)
    n_cic, n_unsw = len(F1.CIC_ARMS), len(F1.UNSW_ARMS)
    # inputs
    box(ax, 0, 84, 67, 19, "Published releases",
        f"{n_cic} releases of CICIDS2017 and {n_unsw} of UNSW-NB15, records as distributed, "
        "each with its own features and labels", wrap=46, fill="white")
    box(ax, 73, 84, 101, 19, "Ground truth sources",
        "CICIDS2017 attack schedule (LYCOS labelling script); UNSW-NB15 ground truth file",
        wrap=70, fill="white")
    # the five parts
    w, gap, y, h = 32.4, 3, 44, 33
    parts = [("One clock", "UTC for every release, with its offset and clock resolution"),
             ("Conversation key", "endpoint pair and protocol; a silence longer than T opens a "
                                  "new conversation; defined on the pooled records"),
             ("One ground truth", "one label per conversation; every record inherits it"),
             ("N-way alignment", "records per conversation in each release; pairing and "
                                 "cardinality for every pair of releases"),
             ("Splits", "conversation grouped, the same in every release; random flow split "
                        "for leakage")]
    for k, (title, body) in enumerate(parts):
        x = k * (w + gap)
        box(ax, x, y, w, h, title, body, num=k + 1, wrap=21, tsize=8.5)
        if k:
            arrow(ax, (x - gap, y + h / 2), (x, y + h / 2))
    arrow(ax, (w / 2, 84), (w / 2, y + h))
    arrow(ax, (2 * (w + gap) + w / 2, 84), (2 * (w + gap) + w / 2, y + h))
    # frozen table
    ft_y = 30
    box(ax, 0, ft_y, 174, 9, "Frozen table per benchmark and day", "", fill="white")
    ax.text(64, ft_y + 7, "with a manifest of input hashes, settings and code version",
            ha="left", va="top", fontsize=S.SMALL, color=S.INK)
    arrow(ax, (87, y), (87, ft_y + 9))
    # what reads it
    outs = [("Agreement (RQ1)", "conversation and flow Jaccard, records per conversation, "
                                "Krippendorff's alpha of the releases' own labels"),
            ("Detection experiment (RQ2 to RQ4)", "models trained per release; scored per flow "
                                                  "and per conversation on the same test "
                                                  "conversations"),
            ("Published claims (RQ5)", "each claim rebuilt per flow, then compared on shared "
                                       "conversations with one ground truth")]
    ow, og = 55.3, 4
    for k, (title, body) in enumerate(outs):
        x = k * (ow + og)
        box(ax, x, 0, ow, 23, title, body, wrap=40, tsize=8.5)
        arrow(ax, (x + ow / 2, ft_y), (x + ow / 2, 23))
    return S.save(fig, FIG / "fig_instrument.pdf")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    return [same_packets(), instrument()]


if __name__ == "__main__":
    for p in main():
        print("wrote", p)
