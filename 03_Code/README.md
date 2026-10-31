# Code

## What is here

| Path | What it is |
|---|---|
| `reconstruction_agreement.py` | New. Measures agreement between the three CICIDS2017 reconstructions. Produced every number in the gap analysis. Runs in about thirty seconds a day |
| `releases.py` | New, Phase 2. Registry of all nine flow releases and one loader per release: five tuple, UTC start time, own label. Every later script should read data through `load_keys()` |
| `inventory.py` | New, step 2.9. Writes `04_Results/release_inventory.csv` and `release_label_counts.csv` |
| `compare_counts.py` | New, step 2.10. Checks counts against published figures, writes `04_Results/count_check.csv` |
| `clock_check.py` | New, step 2.11. Lines up attack windows across arms, writes `04_Results/release_clock.csv` and `release_clock_summary.csv` |

## Dataset paths

The datasets are not in this folder. They are large and live in a separate mounted folder.

```
<Research folder>/Datasets/
├── CICIDS2017_Original_LabelledFlows/     8 files, has the five tuple and timestamps
├── CICIDS2017_Original_ML_CSVs/           8 files, NO five tuple, features only
├── CICIDS2017_Corrected_WTMC2021/         5 files, one per weekday
└── LYCOS-IDS2017_GitHubMirror/
    ├── pcap_lycos/                        5 files, ships UNLABELLED (label = NeedLabel)
    ├── labelling.py                       assigns labels from the attack schedule
    ├── create_datasets.py                 builds train/crossval/test parquet
    └── cicids2017/csv_files/              zipped copies of the CIC originals
```

Set `DATA` at the top of `reconstruction_agreement.py` to your Datasets path before running.

Phase 2 (26 October 2026) added six folders: `BCCC_CIC_IDS2017`, `CRiSIS2022_RPD`, `NFS2023_nTE`,
`NFS2023_TE`, `UNSW_NB15_Original` and `NF_UNSW_NB15_v3`, and labelled LYCOS into
`LYCOS-IDS2017_GitHubMirror/lycos-ids2017/`. `releases.py` holds every path, and
`Datasets/README.md` holds sources, sizes and quirks.

Run order for Phase 2 outputs, from the workspace root:

```
.venv/bin/python 03_Code/inventory.py
.venv/bin/python 03_Code/compare_counts.py
.venv/bin/python 03_Code/clock_check.py
```

Phase 3 (26 October 2026), the measurement instrument:

| Path | What it is |
|---|---|
| `instrument.py` | Definitions: conversation key and gap rule, ground truth (LYCOS schedule, UNSW GT file), N-way pairing, grouped and random splits |
| `build_instrument.py` | Builds and freezes one benchmark day: `04_Results/instrument/<bench>/<day>/` (parquet, gitignored) and the CSV summaries in `04_Results/phase3/<table>/` |
| `conv_scoring.py` | Flow and conversation level macro-F1 through one code path (rules any, majority, bytes) |
| `metrics_macro_f1.py` | Copy of the conference paper's `macro_f1.py` (not included here) |
| `gap_sweep.py` | Conversation counts per class for T in 5, 15, 30, 60, 120, 300 s |
| `hand_check.py` | Prints the hand check cases and writes the test fixtures |
| `unsw_clock_offset.py` | NF-v3 start minus original `Stime`, the evidence for the rounding rule |
| `phase3_summary.py` | Joins the per day CSVs into `04_Results/phase3/all_*.csv` and `phase3_overview.csv` |
| `tests/test_instrument.py` | Unit tests, including the hand check decisions |

Everything regenerates with one script, from the workspace root (about 25 minutes with the load
cache in `Datasets/_cache_phase3/`, longer without it):

```
bash 03_Code/run_phase3.sh
```

## What the previous pipeline gives you

The conference scripts are worth reading before writing anything new, because most of what the journal
paper needs already exists in some form.

- `preprocessing.py`, `splits.py`, `train_eval.py`: the small shared utilities.
- `run_e1_baseline.py` through `run_e4_group.py`: the four experiments. `run_e4_group.py` holds the
  group by five tuple split logic, which is the direct ancestor of the conversation key this paper
  needs.
- `eval_metrics*.py`, `eval_perclass.py`, `macro_f1.py`: metric computation. Read `macro_f1.py`
  carefully, its header comment still refers to Eq. (2) when the paper renumbered it to Eq. (1).
- `bootstrap_ci_e4.py`: cluster bootstrap over conversation groups, B = 1000.
- `wilcoxon_perclass.py`: the by class signed rank test.
- `run_shap_e4.py`, `shap_analysis.py`: the SHAP work.
- `make_fig_*.py`: figure generation.
- `reconcile_table1.py`: the script that re-derived every table number from the CSVs. The journal paper
  needs its own version of this and it is what makes rule 4 enforceable.

## Known issues carried over

- Three pandas API bugs in the LYCOS scripts were found and fixed during the conference project: a
  missing unzip step, `DataFrame.append` removed in pandas 2.0, and positional `Series[int]` indexing
  removed in pandas 3.0. If you re-clone LYCOS fresh, expect to fix them again.
- The `venv`, the cached `.pkl` and `.npz` intermediates and the raw data were not copied. They total
  about 13 GB and are all regenerable.

## Conventions for new code

- Results go to CSV. Never print a number you intend to put in the paper.
- Fix seeds. Log versions. Write the invocation into a runbook file as you go.
- One script regenerates every number in the manuscript. Build that script early, not at the end.

## Phase 5 (RQ2, invariance)

- `run_phase5.sh data|tune|grid|analysis`: the whole Phase 5 pipeline, one stage at a time.
- `features.py`: model features per arm (identifiers, addresses, ports, timestamps and labels dropped).
- `phase5_data.py`: per arm tables on a conversation subsample shared by all arms.
- `phase5_grid.py`: tuning and the 330 cell grid, checkpointed per cell.
- `phase5_analysis.py`: variance shares, Kendall's W, cluster bootstrap, H2 judgement.
- `phase5_sanity.py`: recount of three random result rows from the predictions.
- `phase5_repro.sh`: reruns three cells and checks the predictions are identical.
- `phase5_progress.py`: progress of a running grid.
- `phase5_te_check.py`: why NFS-2023 TE scores near 0.667 for several classes.
- `tests/test_phase5.py`: unit tests for the shares, Kendall's W and the bootstrap weights.
