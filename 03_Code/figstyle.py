"""
One style for every figure of the paper, after Springer's artwork guidelines (IJIS submission
guidelines, read 30 October 2026): sans serif lettering (Helvetica or Arial), 8 to 12 pt at final
size with little variance, lines at least 0.3 pt, figures 84 mm (one column) or 174 mm (both
columns) wide, fonts embedded, no titles or captions inside a figure, parts labelled (a), (b), ...
and information that survives a black-and-white print (marker shapes, not colour alone).

Figures are drawn at their final width, so the svjour3 two-column build places them at 100
percent. Each figure is written as PDF (for pdflatex) and as EPS (the format the guidelines
prefer for vector art) under 05_Paper/figures/eps/.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # no timestamp in PDF or EPS: reruns are identical

MM = 1 / 25.4
FULL = 174 * MM   # inches, both columns
COLUMN = 84 * MM  # inches, one column

SIZE = 9          # pt, lettering at final size
SMALL = 8         # pt, dense tick labels and point labels only

INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#d9d9d9"
# reference palette (dataviz skill, light mode): categorical slots 1 and 2
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif",
    # Liberation Sans has Arial's metrics and is TrueType, so it embeds cleanly as Type 42; the
    # OpenType (CFF) clones Nimbus Sans and TeX Gyre Heros do not
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "TeX Gyre Heros"],
    "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold", "mathtext.cal": "sans",
    "font.size": SIZE, "axes.titlesize": SIZE, "axes.labelsize": SIZE,
    "xtick.labelsize": SMALL, "ytick.labelsize": SIZE, "legend.fontsize": SIZE,
    "legend.title_fontsize": SIZE,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "legend.frameon": False,
})


def panel_label(ax, k, text, **kw):
    """Title of figure part k as '(a) text', left aligned, the form the guidelines ask for."""
    ax.set_title(f"({chr(ord('a') + k)}) {text}", loc="left", fontsize=SIZE, color=INK, **kw)


def save(fig, path):
    """Write path (PDF) and the EPS twin; return the PDF path."""
    path = Path(path)
    fig.savefig(path, metadata={"CreationDate": None})
    eps = path.parent / "eps" / (path.stem + ".eps")
    eps.parent.mkdir(exist_ok=True)
    fig.savefig(eps, metadata={"CreationDate": None})
    plt.close(fig)
    return path
