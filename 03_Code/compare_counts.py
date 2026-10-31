"""
Step 2.10: check the release counts against published figures.

Reads 04_Results/release_label_counts.csv (written by inventory.py) and
compares flow and attack counts with the figures each source prints. Writes
04_Results/count_check.csv and a manifest. A row with match = False needs a
written explanation in PROGRESS_LOG.md before Phase 3.

    .venv/bin/python 03_Code/compare_counts.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manifest import write_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "04_Results" / "release_label_counts.csv"
OUT = ROOT / "04_Results" / "count_check.csv"

PJ = "Pekar and Jozsa 2024, Table 2"
CF = "01_Context/Carried_Forward_Facts.md section 4 or 5"
DAYS = {"Mon": "2017-07-03", "Tue": "2017-07-04", "Wed": "2017-07-05",
        "Thu": "2017-07-06", "Fri": "2017-07-07"}
BENIGN = {"BENIGN", "benign", "Benign", "Normal", "0"}

# release, day (None = all days), flows, attacks (None = not published),
# exclude '- Attempted' rows, source
EXPECTED = [
    ("cic_original", "Tue", 445_909, 13_835, False, PJ),
    ("nfs2023_nte", "Tue", 325_677, 6_972, False, PJ),
    ("nfs2023_te", "Tue", 467_254, 10_943, False, PJ),
    ("wtmc2021", "Tue", 321_984, 6_953, True, PJ + " (attempted excluded)"),
    ("crisis2022_rpd", "Tue", 322_462, 6_953, False, PJ),
    ("nfs2023_nte", "Mon", 376_826, 0, False, PJ),
    ("nfs2023_nte", "Wed", 503_140, 173_873, False, PJ),
    ("nfs2023_nte", "Thu", 358_302, 64_174, False, PJ),
    ("nfs2023_nte", "Fri", 547_186, 253_845, False, PJ),
    ("nfs2023_te", "Mon", 559_775, 0, False, PJ),
    ("nfs2023_te", "Wed", 865_413, 174_398, False, PJ),
    ("nfs2023_te", "Thu", 485_224, 71_480, False, PJ),
    ("nfs2023_te", "Fri", 775_694, 256_733, False, PJ),
    ("cic_original", None, 2_830_743, None, False, CF),
    ("cic_original_ml", None, 2_830_743, None, False, CF),
    ("wtmc2021", None, 2_100_814, None, False, CF),
    ("wtmc2021", None, 2_100_814 - 9_144, None, True, CF + " (minus 9,144 attempted)"),
    ("cic_original", "Tue", 445_909, None, False, CF),
    ("wtmc2021", "Tue", 322_003, None, False, CF),
    ("lycos", "Tue", 319_637, None, False, CF),
    ("unsw_original", None, 2_540_044, None, False,
     "research.unsw.edu.au UNSW-NB15 page (four CSVs)"),
    ("nf_unsw_v3", None, 2_365_424, None, False,
     "staff.itee.uq.edu.au/marius/NIDS_datasets NF-UNSW-NB15-v3"),
    ("cic_unsw", None, 3_540_241, 89_583, False,
     "unb.ca/cic/datasets/cic-unsw-nb15.html, CICFlowMeter column of the class table"),
]


def main():
    lc = pd.read_csv(IN, dtype={"label": str})
    rows = []
    for rel, day, flows, attacks, no_attempted, src in EXPECTED:
        d = lc[lc["release"] == rel]
        if day is not None:
            d = d[d["day_utc"] == DAYS[day]]
        if no_attempted:
            d = d[~d["label"].str.contains("Attempted", na=False)]
        got_flows = int(d["rows"].sum())
        got_att = int(d.loc[~d["label"].isin(BENIGN), "rows"].sum())
        ok = got_flows == flows and (attacks is None or got_att == attacks)
        rows.append({"release": rel, "day": day or "all", "expected_flows": flows,
                     "got_flows": got_flows, "flow_diff": got_flows - flows,
                     "expected_attacks": attacks, "got_attacks": got_att,
                     "attack_diff": None if attacks is None else got_att - attacks,
                     "attempted_excluded": no_attempted, "match": ok, "source": src})
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    m = write_manifest("count_check", inputs=[IN], outputs=[OUT])
    print(out.to_string())
    print(f"wrote {OUT}, {m}")


if __name__ == "__main__":
    main()
