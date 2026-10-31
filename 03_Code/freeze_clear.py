"""
Step 9.4, clean start of the freeze run (03_Code/run_all.sh calls it at the first start).

A checkout holds the results of the earlier runs, and a step that finds its output already
there may keep it: phase5_grid.py tune keeps every arm and model already in tuning.csv, which
is how the first freeze (28 Oct 2026) reused the committed tuning instead of tuning again. So
that every committed result is written again by the run and none is read back, the committed
files under 04_Results and the paper's tables and figures are removed before the first step,
except the inputs and records in KEEP, which no step of the run writes. After the last step,
--check confirms that every removed file exists again.

    .venv/bin/python 03_Code/freeze_clear.py           remove; list in 04_Results/freeze/cleared.txt
    .venv/bin/python 03_Code/freeze_clear.py --check   was every removed file written again?

Nothing is removed unless the checkout is clean: no tracked file changed, and no untracked or
ignored file under the cleared folders (a fresh clone has none; the working copy, which holds
the gitignored intermediates, always has some, so it is never cleared). Removed files stay in
git, so `git checkout -- .` brings them back.

Manifests (manifests/*.json) are records of earlier runs with a time stamp in their name; each
step writes a new one. --check asks for a manifest of the same name stem that the commit did not
hold, in any manifests folder under the cleared folders: a script may have moved its record
(make_numbers.py wrote to 04_Results/phase4/manifests until Phase 9 and writes to
04_Results/freeze/manifests since; the first check, 29 Oct 2026, looked only in the same folder
and reported the two Phase 4 records as missing).
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ROOTS = ["04_Results", "05_Paper/tables", "05_Paper/figures"]
LIST = ROOT / "04_Results" / "freeze" / "cleared.txt"
REPORT = ROOT / "04_Results" / "freeze" / "regenerated.csv"
# committed files under ROOTS that no step of run_all.sh writes (prefix match)
KEEP = [
    "04_Results/README.md",
    "04_Results/conference_results/",        # results of the conference paper, another pipeline
    "04_Results/smoke_test/",                # step 0.6 probe
    "04_Results/manifests/step0.6_tuesday_probe_",
    "04_Results/freeze_run1/",               # record of the first freeze run, 28 Oct 2026
    "04_Results/freeze_run2/",               # record of the second freeze run, 29 Oct 2026
    "04_Results/phase5/settings.json",       # benign fraction and attack cap, fixed in step 5.3
    "04_Results/phase5/time_trial.csv",      # the step 5.3 time trial that fixed them
    "04_Results/phase7/run_retrain.sh",      # how the Phase 7 retraining was first run
    "04_Results/phase7/logs/retrain_resumed.txt",
    "04_Results/phase8/logs/start_time",
]
STAMP = re.compile(r"_\d{8}T\d{6}Z\.json$")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout


def kept(path):
    return any(path.startswith(k) for k in KEEP)


def is_manifest(path):
    return "/manifests/" in path and STAMP.search(path) is not None


def clear():
    changed = git("status", "--porcelain", "--untracked-files=no").strip()
    if changed:
        sys.exit(f"tracked files changed; commit first:\n{changed}")
    extra = git("status", "--porcelain", "--untracked-files=all", "--ignored", "--", *ROOTS)
    # the run's own record (run_all.sh has opened its log folder by now) does not count
    lines = [x for x in extra.splitlines() if x and not x[3:].startswith("04_Results/freeze/")]
    if lines:
        sys.exit(f"{len(lines)} untracked or ignored files under {', '.join(ROOTS)}, for example\n"
                 + "\n".join(lines[:5]) + "\nthis is not a fresh clone; nothing removed")
    files = [p for p in git("ls-files", "--", *ROOTS).splitlines() if not kept(p)]
    for p in files:
        (ROOT / p).unlink()
    LIST.parent.mkdir(parents=True, exist_ok=True)
    LIST.write_text("\n".join(files) + "\n", encoding="utf-8")
    n_man = sum(is_manifest(p) for p in files)
    print(f"removed {len(files)} committed files ({n_man} manifests), kept "
          f"{len(git('ls-files', '--', *ROOTS).splitlines()) - len(files)}; list in "
          f"{LIST.relative_to(ROOT)}")


def check():
    files = [p for p in LIST.read_text(encoding="utf-8").splitlines() if p]
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *ROOTS).splitlines()
    # manifests written by this run, by name stem (the commit's own are tracked, not listed)
    by_stem = {}
    for q in untracked:
        if is_manifest(q):
            by_stem.setdefault(STAMP.sub("", Path(q).name), []).append(q)
    rows = []
    for p in files:
        if is_manifest(p):
            new = sorted(by_stem.get(STAMP.sub("", Path(p).name), []),
                         key=lambda q: Path(q).parent != Path(p).parent)   # same folder first
            rows.append(dict(path=p, kind="manifest", written_again=bool(new),
                             new_record=new[0] if new else ""))
        else:
            rows.append(dict(path=p, kind="result", written_again=(ROOT / p).is_file(),
                             new_record=""))
    t = pd.DataFrame(rows)
    t.to_csv(REPORT, index=False)
    moved = t[(t["kind"] == "manifest") & t["written_again"]
              & (t["new_record"].map(lambda q: str(Path(q).parent))
                 != t["path"].map(lambda q: str(Path(q).parent)))]
    new = [p for p in untracked if not is_manifest(p) and not p.startswith("04_Results/freeze/")]
    miss = t[~t["written_again"]]
    print(f"{len(t)} removed files: {int(t['written_again'].sum())} written again, "
          f"{len(miss)} not")
    for _, r in moved.iterrows():
        print(f"  record moved: {r['path']} -> {r['new_record']}")
    for p in miss["path"]:
        print("  not written again:", p)
    print(f"{len(new)} result files that were not in the commit:")
    for p in new:
        print("  new:", p)
    sys.exit(1 if len(miss) else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    check() if ap.parse_args().check else clear()
