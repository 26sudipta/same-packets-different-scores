"""
Write the single-file manuscript Springer asks for: every \\input of a generated table, of
numbers.tex and of a sections file is replaced by that file's text, and figures are referenced by
file name only (the submission puts them next to the .tex file).

    python3 05_Paper/flatten.py      # writes 05_Paper/springer/submission/manuscript.tex
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "springer"
OUT = SRC / "submission"


def inline(text, base):
    def rep(m):
        p = (base / m.group(1))
        if p.suffix != ".tex":
            p = p.with_suffix(".tex")
        body = p.read_text()
        return f"%% ---- begin {m.group(1)} (generated or kept outside, inlined by flatten.py)\n" + \
            inline(body, p.parent).rstrip("\n") + f"\n%% ---- end {m.group(1)}"
    return re.sub(r"\\input\{([^}]+)\}", rep, text)


def main():
    OUT.mkdir(exist_ok=True)
    t = inline((SRC / "manuscript.tex").read_text(), SRC)
    t = t.replace(r"\graphicspath{{../figures/}}", "")
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t)
    for f in figs:
        name = Path(f).name if Path(f).suffix else Path(f).name + ".pdf"
        shutil.copy2(ROOT / "figures" / name, OUT / name)
    for f in ("refs.bib", "sn-jnl.cls", "sn-mathphys-num.bst"):
        shutil.copy2(SRC / f, OUT / f)
    assert r"\input{" not in t
    (OUT / "manuscript.tex").write_text(t)
    print(f"wrote {OUT.relative_to(ROOT.parent)}/manuscript.tex with {len(figs)} figures")


if __name__ == "__main__":
    main()
