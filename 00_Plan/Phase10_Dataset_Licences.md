# Dataset licences (plan step 10.8)

Checked 29 October 2026 from the pages named below. This is a reading of public pages, not legal
advice; where a page says nothing, the absence is recorded, not interpreted as permission.

| Release | Where the terms were read | What the terms say |
|---|---|---|
| CICIDS2017 (original, GeneratedLabelledFlows) | unb.ca/cic/datasets/ids-2017.html, section "License" | The flows and PCAPs "are publicly available for researchers"; users "should cite" Sharafaldin et al. (ICISSP 2018). No statement on redistribution. |
| WTMC-2021 (Engelen et al.) | intrusion-detection.distrinet-research.be/WTMC2021/ | No licence or terms found on the page. |
| CRiSIS-2022 (Lanvin et al.) | gitlab.inria.fr/mlanvin/crisis2022, GitLab API project record | Repository has no licence (`license: null`). No terms found. |
| LYCOS-IDS2017 (Rosay et al.) | lycos-ids.univ-lemans.fr | "The CSV files, LycoSTand and python source codes are available for researchers"; cite Rosay et al. (WI-IAT 2021). No statement on redistribution. |
| NFS-2023-nTE and NFS-2023-TE (Pekar and Jozsa) | github.com/FlowFrontiers/CyberML-DataQuality, README and GitHub API record | Datasets hosted in the repository; repository has no licence (`license: null`). |
| BCCC-CIC-IDS2017 (Shafi et al.) | yorku.ca/research/bccc/.../intrusion-detection-dataset-bccc-cic-ids2017/ | No licence or terms found on the page. |
| UNSW-NB15 (Moustafa and Slay) | research.unsw.edu.au/projects/unsw-nb15-dataset | "Free use of the UNSW-NB15 dataset for academic research purposes is hereby granted in perpetuity. Use for commercial purposes should be agreed by the authors." Users "have to cite" five listed papers. |
| NF-UNSW-NB15-v3 (Luay et al.) | UQ eSpace record UQ:6e0eda1 (search result excerpt; the page itself could not be fetched) | "Permitted Re-Use with Commercial Use Restriction", with an agreement to acknowledge re-use. Read in full before release. |

## What this means for the code release

Four of the eight releases state no terms we could find, and none states that derived record-level
tables may be redistributed. The frozen instrument tables (`04_Results/instrument/`, gitignored)
and the stored predictions hold one row per record or conversation, with addresses, ports and
timestamps taken from the releases.

**Recommendation (for the authors to decide):** publish the code, the aggregated result files
(`04_Results/phase*/` CSVs of counts, shares, intervals and verdicts, `04_Results/freeze/`
manifests) and the manuscript tables; do not publish the record-level instrument tables or
predictions. Anyone can rebuild them byte for byte from the releases with `03_Code/run_all.sh`, and
the manifests' SHA-256 hashes let them check that they did. If the authors want to publish the
record-level tables, ask the producers of WTMC-2021, CRiSIS-2022, NFS-2023 and BCCC-CIC-IDS2017
first, and keep UNSW-NB15 and NF-UNSW-NB15-v3 derived files to academic use.

Before upload, re-check that no aggregated CSV in `04_Results/` carries record-level rows (the
Phase 8 `sets/` and `preds/` folders, for example).
