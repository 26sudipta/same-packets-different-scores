# Results

Every file here was written by a script in `03_Code/`, and every number in the manuscript is read
from one of them by `03_Code/make_numbers.py` and `03_Code/review_numbers.py`. Nothing is retyped
by hand.

| Path | What it is |
|---|---|
| `phase3/` to `phase8/` | Results of the instrument (RQ1 inputs), agreement (RQ1), the detection grid (RQ2), the natural experiment (RQ3), the remedy (RQ4) and the published claims (RQ5) |
| `phase11/` | Robustness analyses added after the primary results (see the manuscript, Section 5) |
| `freeze/` | Tables and manifests of the freeze run that produced the committed results |
| `freeze_run1/`, `freeze_run2/` | Logs and manifests of the two earlier full runs from fresh clones |
| `manifests/` | One manifest per script run, with SHA-256 hashes of inputs and outputs |
| `release_*.csv`, `count_check.csv` | Release inventory, clock checks and the check of every release against its published counts |

Predictions, arm tables and other large intermediates (`*.parquet`) are not stored; `03_Code/run_all.sh`
regenerates them from the raw releases.
