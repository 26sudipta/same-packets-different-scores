#!/usr/bin/env bash
# Build the manuscript and check the number rules (plan step 9.2).
#   cd 05_Paper && bash build.sh
# 1. regenerate numbers.tex and the tables from 04_Results (03_Code/make_numbers.py)
# 2. lint: no number typed in the prose, no \VERIFYNUM left (lint_numbers.py)
# 3. compile springer/manuscript.tex (pdflatex, bibtex, pdflatex twice); fail on undefined
#    references or citations
# 4. write the single-file submission copy (flatten.py) and compile it as well
# 5. port the submission copy to svjour3 two-column (port_svjour3.py) and compile that too
# Draft markers (\pending, \tocheck) are counted and reported; they must be zero at submission
# (Gate 10), not at the freeze.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-../.venv/bin/python}"
[ -x "$PY" ] || PY=python3

"$PY" ../03_Code/make_numbers.py > build_numbers.log
echo "make_numbers: $(tail -n 1 build_numbers.log)"

set +e
"$PY" lint_numbers.py springer/manuscript.tex > build_lint.log
lint=$?
set -e
tail -n 1 build_lint.log
if [ "$lint" -ne 0 ]; then
  cat build_lint.log
  echo "FAIL: typed numbers or VERIFYNUM markers left"
  exit 1
fi

compile() {
  local dir=$1
  ( cd "$dir"
    pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex > build.log
    bibtex manuscript >> build.log
    pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex >> build.log
    pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex >> build.log )
  if grep -E "undefined (references|citations)|Reference .* undefined|Citation .* undefined" "$dir/manuscript.log" >/dev/null; then
    grep -E "undefined" "$dir/manuscript.log" | head -20
    echo "FAIL: undefined references or citations in $dir"
    exit 1
  fi
  echo "compiled $dir/manuscript.pdf ($(grep -c "" "$dir/manuscript.log") log lines)"
}

compile springer
"$PY" flatten.py
compile springer/submission
"$PY" port_svjour3.py
compile svjour3
for d in springer/submission svjour3; do
  echo "overfull boxes in $d: $(grep -c '^Overfull' "$d/manuscript.log" || true)"
done

pending=$( (grep -o '\\pending{' springer/manuscript.tex sections/*.tex || true) | wc -l)
tocheck=$( (grep -o '\\tocheck{' springer/manuscript.tex sections/*.tex || true) | wc -l)
verify=$( (grep -o '\\VERIFYNUM{' springer/manuscript.tex sections/*.tex || true) | wc -l)
echo "VERIFYNUM markers: $verify"
echo "draft markers: $pending pending, $tocheck tocheck (zero required at Gate 10)"
