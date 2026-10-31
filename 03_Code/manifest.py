"""
Run manifest helper.

Every script that produces a result calls write_manifest() at the end. The
manifest records what went in, how it was run and what came out, so any number
in the paper can be traced to one run and checked again later.

Usage from another script:

    from manifest import write_manifest
    write_manifest("e1_agreement", inputs=[csv_a, csv_b], outputs=[out_csv],
                   params={"day": "Tuesday", "seed": 42})

Usage from the command line, for a quick check:

    python 03_Code/manifest.py --name test --inputs a.csv --outputs b.csv --param seed=42

Manifests go to 04_Results/manifests/<name>_<UTC timestamp>.json unless
out_dir is given.
"""
import argparse
import datetime as dt
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "04_Results" / "manifests"
PACKAGES = ["pandas", "numpy", "scipy", "scikit-learn", "lightgbm",
            "statsmodels", "pyarrow", "matplotlib", "krippendorff"]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_record(path):
    p = Path(path).resolve()
    if not p.is_file():
        return {"path": str(p), "missing": True}
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}


def git_state():
    """Commit hash, plus whether tracked files had uncommitted changes."""
    def run(*args):
        r = subprocess.run(["git", "-C", str(ROOT), *args],
                           capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status)}


def package_versions(names=PACKAGES):
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def write_manifest(name, inputs=(), outputs=(), params=None, out_dir=None):
    """Write one JSON manifest and return its path."""
    now = dt.datetime.now(dt.timezone.utc)
    record = {
        "name": name,
        "created_utc": now.isoformat(timespec="seconds"),
        "script": str(Path(sys.argv[0]).resolve()) if sys.argv[0] else None,
        "argv": sys.argv[1:],
        "params": params or {},
        "git": git_state(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "inputs": [file_record(p) for p in inputs],
        "outputs": [file_record(p) for p in outputs],
    }
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}_{now.strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def _parse_param(text):
    key, _, value = text.partition("=")
    return key, value


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Write a run manifest.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--inputs", nargs="*", default=[])
    ap.add_argument("--outputs", nargs="*", default=[])
    ap.add_argument("--param", nargs="*", default=[], help="key=value pairs")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    print(write_manifest(a.name, a.inputs, a.outputs,
                         dict(_parse_param(p) for p in a.param), a.out_dir))
