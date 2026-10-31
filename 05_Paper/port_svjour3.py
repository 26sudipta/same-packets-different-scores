"""
Write the svjour3 two-column version of the manuscript from the flattened sn-jnl submission copy,
so that the paper has one source (springer/manuscript.tex and sections/) and two builds.

    python3 05_Paper/port_svjour3.py     # after flatten.py; writes 05_Paper/svjour3/manuscript.tex

The IJIS guidelines name svjour3 with the twocolumn option. Only the class-specific parts change:
preamble, title block, abstract, keywords, acknowledgements and the bibliography style.
"""
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "springer" / "submission"
OUT = ROOT / "svjour3"
JOURNAL = "International Journal of Information Security"


def one(pattern, text, flags=re.S):
    m = list(re.finditer(pattern, text, flags))
    assert len(m) == 1, (pattern, len(m))
    return m[0]


def braced(text, start):
    """Text of the brace group that opens at text[start] == '{', and the index after it."""
    assert text[start] == "{"
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    raise ValueError("unbalanced braces")


def main():
    t = (SRC / "manuscript.tex").read_text()
    # class and packages
    t = t.replace(one(r"\\documentclass\[[^\]]*\]\{sn-jnl\}", t).group(0),
                  "\\documentclass[twocolumn]{svjour3}\n\\smartqed", 1)
    t = t.replace("\\usepackage{amsthm}%\n", "", 1)           # svjour3 defines its own theorems
    t = re.sub(r"\\theoremstyle\{[^}]*\}%?\n\\newtheorem\{[^}]*\}\{[^}]*\}%?\n", "", t)
    t = t.replace("\\usepackage{rotating}%\n",
                  "\\usepackage{rotating}%\n\\usepackage{url}%\n\\usepackage[hidelinks]{hyperref}%\n", 1)
    # a column of the two-column layout is narrow: let TeX stretch spaces before it overfills a line
    t = t.replace("\\begin{document}", "\\setlength{\\emergencystretch}{3em}\n\\begin{document}", 1)
    # the single-column tables of the source are wider than one column here: span both
    n = t.count("\\begin{table}[t]")
    t = t.replace("\\begin{table}[t]", "\\begin{table*}[t]").replace("\\end{table}", "\\end{table*}")
    assert t.count("\\begin{table*}") == t.count("\\end{table*}") and n > 0, n
    # landscape tables take a whole page across both columns
    t = t.replace("\\begin{sidewaystable}", "\\begin{sidewaystable*}").replace("\\end{sidewaystable}", "\\end{sidewaystable*}")
    # title block: from \title to \maketitle
    m = one(r"\\title\[(?P<short>[^\]]*)\]\{(?P<title>[^}]*)\}.*?\\maketitle", t)
    block = m.group(0)
    a = block.index("\\abstract{")
    abstract, _ = braced(block, a + len("\\abstract"))
    k = block.index("\\keywords{")
    keywords, _ = braced(block, k + len("\\keywords"))
    kw = " \\and ".join(w.strip() for w in keywords.split(","))
    authors = re.findall(r"\\author\*?\[\d\]\{\\fnm\{([^}]*)\} \\sur\{([^}]*)\}\}\\email\{([^}]*)\}", block)
    assert len(authors) == 2, authors
    aff = one(r"\\affil\*?\[1\]\{\\orgdiv\{([^}]*)\}, \\orgname\{([^}]*)\}, "
              r"\\orgaddress\{\\city\{([^}]*)\}, \\country\{([^}]*)\}\}\}", block).groups()
    new = (f"\\journalname{{{JOURNAL}}}\n\n"
           f"\\title{{{m.group('title')}}}\n\\titlerunning{{{m.group('short')}}}\n\n"
           f"\\author{{{authors[0][0]} {authors[0][1]} \\and {authors[1][0]} {authors[1][1]}}}\n"
           f"\\authorrunning{{{authors[0][1]} and {authors[1][1]}}}\n\n"
           f"\\institute{{{authors[0][0]} {authors[0][1]} (corresponding author) \\at "
           f"{aff[0]}, {aff[1]}, {aff[2]}, {aff[3]} \\\\ \\email{{{authors[0][2]}}} \\and "
           f"{authors[1][0]} {authors[1][1]} \\at {aff[0]}, {aff[1]}, {aff[2]}, {aff[3]} \\\\ "
           f"\\email{{{authors[1][2]}}}}}\n\n"
           "\\date{Received: date / Accepted: date}\n\n"
           "\\maketitle\n\n"
           f"\\begin{{abstract}}\n{abstract}\n\\keywords{{{kw}}}\n\\end{{abstract}}")
    t = t.replace(block, new, 1)
    # back matter
    t = t.replace("\\backmatter\n", "", 1)
    m = one(r"\\bmhead\{Acknowledgements\}\n(.*?)\n\n", t)
    t = t.replace(m.group(0), "\\begin{acknowledgements}\n" + m.group(1) + "\n\\end{acknowledgements}\n\n", 1)
    t = t.replace("\\bibliography{refs}", "\\bibliographystyle{spmpsci}\n\\bibliography{refs}", 1)
    OUT.mkdir(exist_ok=True)
    for f in SRC.iterdir():
        if f.suffix in (".pdf",) and f.name != "manuscript.pdf":
            shutil.copy2(f, OUT / f.name)
    shutil.copy2(SRC / "refs.bib", OUT / "refs.bib")
    for f in (ROOT / "svjour3" / "class").iterdir():
        if f.suffix in (".cls", ".clo", ".bst"):
            shutil.copy2(f, OUT / f.name)
    (OUT / "manuscript.tex").write_text(t)
    print(f"wrote {OUT.relative_to(ROOT.parent)}/manuscript.tex")


if __name__ == "__main__":
    main()
