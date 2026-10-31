"""
Phase 11, review minor comment 13: is the UNSW-NB15 exporter share a time-to-live
shortcut? Earlier work flags the TTL fields of UNSW-NB15 as near proxies of the
label. This retrains the two UNSW-NB15 arms that carry TTL fields without them:
    unsw_dedup   drops sttl, dttl and ct_state_ttl (a count built on the TTLs)
    nf_unsw_v3   drops MIN_TTL and MAX_TTL
CIC-UNSW-NB15 has no TTL field and keeps its Phase 5 cells.

Same arm tables, models, Phase 5 tuned settings, grouped split and seeds as
Phase 5; only the feature columns differ.

Usage, from the workspace root, NIDS_CACHE_ROOT set (phase11_common):
    .venv/bin/python 03_Code/phase11_ttl.py [--models ...] [--seeds ...]

Outputs: 04_Results/phase11/ttl/{preds,cells}/UNSW-NB15/<arm>/*, Phase 5 format.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase11_common as C  # noqa: E402
import phase5_grid as G  # noqa: E402

BENCH = "UNSW-NB15"
OUT = C.OUT / "ttl"
DROP = {"unsw_dedup": ["sttl", "dttl", "ct_state_ttl"], "nf_unsw_v3": ["MIN_TTL", "MAX_TTL"]}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    a = ap.parse_args()
    load = G.load_arm

    def load_arm(bench, arm, tag="main"):
        t, fcols = load(bench, arm, tag)
        drop = ["f:" + c for c in DROP[arm]]
        missing = [c for c in drop if c not in fcols]
        if missing:
            raise KeyError(f"{arm}: no column {missing}")
        return t, [c for c in fcols if c not in drop]

    G.load_arm = load_arm
    print(G.run(BENCH, list(DROP), a.models, ["grouped"], a.seeds, "main", OUT).to_string())
