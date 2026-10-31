# Same packets, different scores

Code and results for the paper "Same packets, different scores: exporter invariance as a validity
requirement for flow-based intrusion detection evaluation" (Sudipta Das and Nayan Nath, International
Islamic University Chittagong; submitted to the International Journal of Information Security).

The study puts ten published flow releases of two intrusion detection benchmarks (seven of
CICIDS2017, three of UNSW-NB15) on one clock, maps every record to a conversation defined on the
pooled traffic, labels every conversation from the attack schedule, and measures how the exporter
changes agreement, scores, model rankings and published conclusions.

## What is here

| Path | What it is |
|---|---|
| `03_Code/` | Every script. `releases.py` (loaders), `instrument.py` and `build_instrument.py` (clock, key, ground truth, alignment, splits), `phase5_*` (detection grid), `phase6_rq3.py`, `phase7_*` (remedy and sensitivity), `phase8_*` (published claims), `phase11_*` and `review_numbers.py` (robustness analyses added after the primary results, queued by `run_phase11.sh`), `make_numbers.py` and `make_figures.py` (paper numbers, tables, figures), `run_all.sh` (everything in one run), `tests/` (unit tests, including the hand-checked alignment cases) |
| `04_Results/` | Aggregated result files per phase and the manifests of the freeze run (`freeze/`), with SHA-256 hashes of every input and output |
| `05_Paper/` | The manuscript and its build; see `05_Paper/README.md` |
| `00_Plan/` | The written protocols for RQ3 to RQ5, committed before the numbers they govern (`Phase6_Protocol.md` to `Phase8_Protocol.md`), and the dataset terms we found (`Phase10_Dataset_Licences.md`) |

## Relation to the project repository

This repository is a clean copy of the code, results and paper sources of the authors' project
repository at commit `181ef65`. The project repository is private because it also holds working
notes. It records when each protocol was written: the RQ2 protocol with H2 and H4 (commit
`6f6756f`, 27 October 2026) and the RQ3 to RQ5 protocols included here. Its full history is
available to editors and reviewers on request. Some code comments refer to working notes that are
not copied here (`PROGRESS_LOG.md` and other files in `00_Plan/`). The code of the earlier
conference paper and its results (`conference_src/`, `conference_runbooks/`,
`04_Results/conference_results/`) are not included either; none of the scripts here imports them.

## Data

No dataset is redistributed here. Download each release from its producer and accept its terms:
CICIDS2017 (Canadian Institute for Cybersecurity, University of New Brunswick), WTMC-2021 (Engelen
et al.), CRiSIS-2022 (Lanvin et al.), LYCOS-IDS2017 (Rosay et al.), NFS-2023-nTE and NFS-2023-TE
(Pekar and Jozsa), BCCC-CIC-IDS2017 (Behaviour-Centric Cybersecurity Center, York University),
UNSW-NB15 (Moustafa and Slay), NF-UNSW-NB15-v3 (University of Queensland) and CIC-UNSW-NB15
(Mohammadian et al., Canadian Institute for Cybersecurity). The terms we found are
summarised in `00_Plan/Phase10_Dataset_Licences.md`. `03_Code/releases.py` lists the folder and file
names each loader expects and the published counts every release is checked against.

## Reproduce

Python 3.12 and the pinned packages:

```
uv venv .venv --python /usr/bin/python3.12
uv pip install --python .venv/bin/python -r 03_Code/requirements.txt
```

Then, from a fresh clone, with the releases under `NIDS_DATA` and an empty cache folder:

```
NIDS_DATA=/path/to/Datasets NIDS_CACHE_ROOT=$PWD/_freeze_cache bash 03_Code/run_all.sh
watch -n 30 bash 03_Code/freeze_progress.sh      # progress, percent done and time left
```

The run removes the committed results, writes every one of them again from the raw releases, runs
the unit tests and checks that every removed file was rewritten. On a laptop with an Intel
i5-11300H and 14 GB of memory it takes about 12 hours, with a peak of about 8.5 GB. Two full runs
from fresh clones gave identical frozen tables; their predictions and result files differ only in
recorded run times and in the values that a code fix between the runs changed.

To rebuild only the paper from the results:

```
cd 05_Paper && bash build.sh
```

## Licence

Code: MIT licence (`LICENSE`). Result tables, figures and text: CC BY 4.0. The datasets keep
their producers' terms and are not included.

## Citing

See `CITATION.cff`. The version used for the paper is archived on Zenodo: https://doi.org/10.5281/zenodo.23833798 (version 1.0.0).
