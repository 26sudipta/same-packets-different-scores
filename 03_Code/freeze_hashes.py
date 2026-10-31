"""
Step 9.4: SHA-256 of every input file the freeze run reads from the Datasets folder.

Writes 04_Results/freeze/input_hashes.csv (path relative to the Datasets folder, bytes, sha256;
a symbolic link is listed with its target and not hashed twice). Caches (_cache_*) and partial
downloads (*.part) are left out: they are not inputs.

    .venv/bin/python 03_Code/freeze_hashes.py
"""
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import releases as R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "04_Results" / "freeze"


def sha256(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(block):
            h.update(b)
    return h.hexdigest()


def main():
    rows = []
    for dirpath, dirnames, filenames in os.walk(R.DATA):
        d = Path(dirpath)
        dirnames[:] = sorted(x for x in dirnames if not x.startswith("_cache"))
        for fn in sorted(filenames):
            p = d / fn
            if fn.endswith(".part"):
                continue
            rel = p.relative_to(R.DATA).as_posix()
            if p.is_symlink():
                rows.append(dict(path=rel, bytes=None, sha256=None, link_to=os.readlink(p)))
                continue
            rows.append(dict(path=rel, bytes=p.stat().st_size, sha256=sha256(p), link_to=None))
            print(rel, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "input_hashes.csv", index=False)
    print(f"{len(t)} files, {t['bytes'].sum() / 1e9:.2f} GB hashed")


if __name__ == "__main__":
    main()
