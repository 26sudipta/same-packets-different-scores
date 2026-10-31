"""
Phase 11, review Major comment 2: two model families unlike the tree ensembles,
so that the ranking agreement of H4 and the model factor of H2 can move.

    lr    multinomial logistic regression (linear)
    knn   k-nearest neighbours (instance based)

Both see the same input as the Phase 5 multilayer perceptron: signed log
transform, then standard scaling fitted on training rows. Grids of four
settings, as for the Phase 5 models:
    lr    C in 0.01, 0.1, 1, 10 (L2 penalty, lbfgs, at most 300 iterations)
    knn   k in 5, 15, uniform or distance weights (Euclidean)

Exactly the Phase 5 procedure otherwise (phase5_grid): same arm tables and
features, one setting per arm and model tuned on seed 0 of the grouped split by
validation macro-F1, then every seed. Grouped split only: H2 is judged on it and
H4 on its common population. These models were added after the protocol; their
results are reported as a sensitivity analysis beside the confirmatory tests.

Usage, from the workspace root, NIDS_CACHE_ROOT set (phase11_common):
    .venv/bin/python 03_Code/phase11_models.py tune CICIDS2017
    .venv/bin/python 03_Code/phase11_models.py run CICIDS2017 [--arms ...] [--models ...]

Outputs: 04_Results/phase11/models/tuning.csv and
{preds,cells}/<bench>/<arm>/<model>_grouped_s<seed>.*, same format as Phase 5.
"""
import argparse
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase11_common as C  # noqa: E402
import phase5_grid as G  # noqa: E402

OUT = C.OUT / "models"
MODELS = ["lr", "knn"]
GRID = {
    "lr": [dict(C=0.01), dict(C=0.1), dict(C=1.0), dict(C=10.0)],
    "knn": [dict(n_neighbors=5, weights="uniform"), dict(n_neighbors=5, weights="distance"),
            dict(n_neighbors=15, weights="uniform"), dict(n_neighbors=15, weights="distance")],
}
_make = G.make_model


def make_model(name, cfg, seed):
    if name == "lr":
        return make_pipeline(FunctionTransformer(G.signed_log1p), StandardScaler(),
                             LogisticRegression(max_iter=300, random_state=seed, **cfg))
    if name == "knn":
        return make_pipeline(FunctionTransformer(G.signed_log1p), StandardScaler(),
                             KNeighborsClassifier(n_jobs=G.N_JOBS, **cfg))
    return _make(name, cfg, seed)


def setup():
    G.GRID.update(GRID)
    G.make_model = make_model
    G.OUT = OUT          # tuning.csv and checkpoints of these models live here
    OUT.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tune", "run"])
    ap.add_argument("bench")
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--seeds", nargs="*", type=int)
    a = ap.parse_args()
    setup()
    arms = a.arms or G.PRIMARY_ARMS[a.bench]
    if a.what == "tune":
        G.tune(a.bench, arms, a.models)
    else:
        print(G.run(a.bench, arms, a.models, ["grouped"], a.seeds, "main", OUT).to_string())
