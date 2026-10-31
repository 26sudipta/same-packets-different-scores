"""
Step 9.2: find numbers typed in the manuscript that no script produced.

    python3 05_Paper/lint_numbers.py [05_Paper/springer/manuscript.tex]

Text inside \\pending and \\tocheck is skipped: those draft markers must all be gone before
submission and are counted by build.sh. A number is allowed when it is inside a macro from numbers.tex or inside \\litnum, \\setnum or
\\docnum, part of a name (CICIDS2017, NF-UNSW-NB15-v3, RQ1, H2, L2, checklist items P1 and E1), a year or a date, a cross
reference, or layout (column widths, rotations, rules). Inside maths the integers 0, 1 and 2 are
allowed, as they appear in formulas. Everything else is printed with its line number, and the
exit status is the number of findings plus the number of \\VERIFYNUM markers. Line numbers count
the manuscript with its 05_Paper/sections files inlined.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEX = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "springer" / "manuscript.tex"

NAMES = [r"NF-UNSW-NB15-v[23]", r"BCCC-CIC-IDS2017", r"CSE-CIC-IDS2018", r"LYCOS-IDS2017", r"CICIDS2017",
         r"UNSW-NB15", r"WTMC-2021", r"CRiSIS-2022", r"NFS-2023(?:-nTE|-TE)?", r"KDD CUP 99",
         r"NetFlow v3", r"RQ[1-5]", r"\bR[1-5]\b", r"\bH[24]\b", r"\bL2\b", r"\bP[1-3]\b", r"\bE[1-4]\b", r"\bF1\b", r"SHA-256", r"IPv[46]"]
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def strip_braced(t, cmd):
    """Remove \\cmd{...} with balanced braces."""
    out, i = [], 0
    pat = re.compile(r"\\" + cmd + r"\*?(\[[^\]]*\])?\{")
    while True:
        m = pat.search(t, i)
        if not m:
            out.append(t[i:])
            return "".join(out)
        out.append(t[i:m.start()])
        depth, j = 1, m.end()
        while depth:
            depth += {"{": 1, "}": -1}.get(t[j], 0)
            j += 1
        out.append(" " * 0 + "\n" * t[m.start():j].count("\n"))
        i = j


def main():
    src = TEX.read_text()
    # prose kept in other files (05_Paper/sections) is checked as part of the manuscript;
    # generated tables (05_Paper/tables) are not, since the script wrote them
    src = re.sub(r"\\input\{(\.\./sections/[^}]+)\}",
                 lambda m: (TEX.parent / m.group(1)).read_text(), src)
    verify = src.count(r"\VERIFYNUM{")
    start = src.find(r"\begin{document}")
    body = src[max(start, 0):]
    body = re.sub(r"(?<!\\)%.*", "", body)                      # comments
    # maths first: allow 0, 1, 2
    def maths(m):
        return re.sub(r"(?<![\d.])[012](?![\d.])", " ", m.group(0))
    body = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", maths, body, flags=re.S)
    body = re.sub(r"\$[^$]*\$", maths, body)
    for cmd in ("pending", "tocheck", "litnum", "setnum", "docnum", "cite", "ref", "eqref", "label", "url", "input",
                "includegraphics", "bibliography", "email", "VERIFYNUM", "cmidrule", "multicolumn",
                "rotatebox", "resizebox", "setlength", "begin", "end", "affil", "author",
                "footnotetext"):
        body = strip_braced(body, cmd)
    body = re.sub(r"\\cmidrule\(lr\)", "", body)
    body = re.sub(r"p\{[\d.]+cm\}|\{\\textwidth\}|\{[\d.]+pt\}", "", body)
    body = re.sub(r"\\[A-Za-z]+", " ", body)                     # macros, including number macros
    for nm in NAMES:
        body = re.sub(nm, " ", body)
    body = re.sub(rf"\b\d{{1,2}} (?:{MONTHS}) (?:19|20)\d\d\b", " ", body)
    body = re.sub(rf"\b(?:{MONTHS}) (?:19|20)\d\d\b", " ", body)
    body = re.sub(r"\b(?:19|20)\d\d\b", " ", body)               # years
    found = []
    for ln, line in enumerate(body.split("\n"), start=src[:max(start, 0)].count("\n") + 1):
        for m in re.finditer(r"\d[\d,.]*", line):
            found.append((ln, m.group(0), line.strip()[:120]))
    for ln, tok, ctx in found:
        print(f"{TEX.name}:{ln}: {tok!r} in: {ctx}")
    print(f"{len(found)} unwrapped numbers, {verify} VERIFYNUM markers")
    sys.exit(min(len(found) + verify, 255))


if __name__ == "__main__":
    main()
