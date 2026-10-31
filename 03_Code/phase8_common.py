"""
Phase 8 (RQ5) helpers shared by phase8_flow.py, phase8_decomp.py and phase8_conv.py.
Protocol: 00_Plan/Phase8_Protocol.md (committed before any Phase 8 number).

What lives here:
    EXTRA_RELEASES  the three releases Phase 8 adds (CRiSIS-2022 Exp_RD and Exp_RP,
                    NF-UNSW-NB15-v2); they are not instrument arms
    read_release    every column of one file, rows numbered as instrument records
                    (file, row), for instrument arms and the extra releases
    conv_lookup     (file, row) -> global conversation id and schedule class, from
                    the frozen instrument, for the cluster bootstrap
    Confusion       per unit confusion counts (true class x predicted label) kept as
                    a sparse matrix, so that a bootstrap replicate is one product
    metrics         binary and per class metrics from a confusion matrix, with
                    optional class reweighting of its rows (step 8.3)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import instrument as I  # noqa: E402
import releases as R  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "04_Results" / "phase8"
CACHE = R.CACHE_ROOT / "_cache_phase8"
FROZEN = ROOT / "04_Results" / "instrument"
SEEDS = [0, 1, 2, 3, 4]
BENIGN = I.BENIGN
N_JOBS = 8

# Releases added in Phase 8. Same CSV layout as their sibling instrument arm.
EXTRA_RELEASES = {
    # The repository tree (commit in _downloads/crisis2022_extra/commit.txt) gives the
    # same git blob id to Exp_RD and Exp_RPD on Monday, Tuesday, Wednesday and Friday, and
    # to Exp_R and Exp_RP on those days; only Thursday differs. The folders hold the
    # downloaded Thursday file and links to the identical files for the other days.
    "crisis2022_rd": dict(benchmark="CICIDS2017", folder="CRiSIS2022_RD", pattern="*.csv",
                          like="crisis2022_rpd",
                          note="Labels/Exp_RD of gitlab.inria.fr/mlanvin/crisis2022: reordered, "
                               "duplicated packets kept, port scan not relabelled"),
    "crisis2022_r": dict(benchmark="CICIDS2017", folder="CRiSIS2022_R", pattern="*.csv",
                         like="crisis2022_rpd",
                         note="Labels/Exp_R: reordered, duplicated packets removed, port scan "
                              "not relabelled"),
    "crisis2022_rp": dict(benchmark="CICIDS2017", folder="CRiSIS2022_RP", pattern="*.csv",
                          like="crisis2022_rpd",
                          note="Labels/Exp_RP: reordered, duplicated packets removed before "
                               "CICFlowMeter, port scan relabelled"),
    "nf_unsw_v2": dict(benchmark="UNSW-NB15", folder="NF_UNSW_NB15_v2",
                       pattern="NF-UNSW-NB15-v2.csv",
                       like=None,
                       note="NF-UNSW-NB15-v2, UQ RDM 8c6e2a00-ef9c-11ed-827d-e762de186848; "
                            "no timestamps"),
}


def bench_of(arm):
    if arm in EXTRA_RELEASES:
        return EXTRA_RELEASES[arm]["benchmark"]
    return R.RELEASES[arm]["benchmark"]


def files(arm):
    if arm in EXTRA_RELEASES:
        e = EXTRA_RELEASES[arm]
        return sorted((R.DATA / e["folder"]).glob(e["pattern"]))
    return R.files(arm)


def read_release(arm, path):
    """Every column of one file; row i is instrument row i of (arm, file)."""
    if arm in EXTRA_RELEASES:
        df = pd.read_csv(path, encoding="latin-1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        return df.reset_index(drop=True)
    return F.read_full(arm, path)


def label_column(arm):
    return {"cic_original": "Label", "wtmc2021": "Label", "crisis2022_rpd": "Label",
            "crisis2022_rd": "Label", "crisis2022_r": "Label", "crisis2022_rp": "Label", "lycos": "label",
            "nfs2023_nte": "label", "nfs2023_te": "label", "bccc2017": "label",
            "unsw_original": "attack_cat", "nf_unsw_v3": "Attack", "nf_unsw_v2": "Attack"}[arm]


def own_label(arm, df):
    """The release's own label as a stripped string (UNSW-NB15 empty category = Normal)."""
    # fillna first: with the pandas "str" dtype, astype(str) keeps missing values as NaN
    s = df[label_column(arm)].fillna("").astype(str).str.strip()
    if arm == "unsw_original":
        s = s.where(~s.isin(["", "nan"]), "Normal")
    return s


def canonical(arm, labels):
    """(canonical class, attempted flag) of the Phase 3 instrument."""
    return I.canonical(bench_of(arm), np.asarray(labels, dtype=object))


# ---------------------------------------------------------------------------
# conversations of released rows (for the cluster bootstrap)

def conv_lookup(arm):
    """DataFrame file, row, gconv, sched for every instrument record of the arm.
    gconv = day position * 10^8 + conv, as in Phase 5. Cached."""
    p = CACHE / "convmap" / f"{arm}.parquet"
    if p.is_file():
        return pd.read_parquet(p)
    bench = bench_of(arm)
    parts = []
    for di, day in enumerate(I.BENCHMARKS[bench]["days"]):
        d = FROZEN / bench / day
        r = pd.read_parquet(d / "records.parquet", columns=["arm", "file", "row", "conv"],
                            filters=[("arm", "==", arm)])
        t = pd.read_parquet(d / "conversations.parquet", columns=["conv", "sched"])
        r = r.merge(t, on="conv", how="left", validate="many_to_one")
        r["gconv"] = di * 10 ** 8 + r["conv"].to_numpy()
        parts.append(r[["file", "row", "gconv", "sched"]])
    out = pd.concat(parts, ignore_index=True)
    out["file"] = out["file"].astype(str)
    out["sched"] = out["sched"].astype(str)
    p.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(p, index=False)
    return out


def attach_conv(arm, frame):
    """Add gconv and sched to a frame with file and row columns. Rows the
    instrument does not hold (none expected for instrument arms) get a unique
    negative id and stratum 'unmapped'."""
    if arm == "nf_unsw_v2":
        out = frame.copy()
        out["gconv"] = -1 - np.arange(len(out), dtype=np.int64)
        out["sched"] = "unmapped"
        return out
    lk = conv_lookup(arm)
    out = frame.merge(lk, on=["file", "row"], how="left", validate="many_to_one")
    miss = out["gconv"].isna().to_numpy()
    g = out["gconv"].to_numpy(dtype=np.float64)
    g[miss] = -1 - np.arange(int(miss.sum()))
    out["gconv"] = g.astype(np.int64)
    out["sched"] = out["sched"].fillna("unmapped").astype(str)
    return out


# ---------------------------------------------------------------------------
# confusion counts per unit

class Confusion:
    """Sparse (unit x cell) counts. A cell is (true class i, predicted label j)."""

    def __init__(self, units, y_true, y_pred, classes, labels):
        self.classes = list(classes)
        self.labels = list(labels)
        ci = {c: i for i, c in enumerate(self.classes)}
        li = {c: j for j, c in enumerate(self.labels)}
        t = np.array([ci[x] for x in y_true], dtype=np.int64)
        q = np.array([li[x] for x in y_pred], dtype=np.int64)
        self.units, u = np.unique(np.asarray(units), return_inverse=True)
        cell = t * len(self.labels) + q
        self.M = sp.csr_matrix((np.ones(len(cell)), (u, cell)),
                               shape=(len(self.units), len(self.classes) * len(self.labels)))

    def total(self, w=None):
        """(K x L) confusion; w = weights per unit (aligned to self.units) or None."""
        v = np.asarray(self.M.sum(axis=0)).ravel() if w is None else self.M.T @ w
        return v.reshape(len(self.classes), len(self.labels))

    def replicates(self, W):
        """W: (B x units) -> (B x K x L)."""
        V = (self.M.T @ W.T).T
        return V.reshape(W.shape[0], len(self.classes), len(self.labels))


# ---------------------------------------------------------------------------
# metrics from a confusion matrix C (true classes x predicted labels)

def reweight(C, classes, target_share):
    """Scale each true-class row so that row totals follow target_share (dict
    class -> share over the rows kept). Classes without target share get 0."""
    C = np.asarray(C, dtype=float)
    tot = C.sum(axis=-1)
    n = tot.sum(axis=-1, keepdims=True)
    f = np.zeros_like(tot)
    for i, c in enumerate(classes):
        s = target_share.get(c, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            f[..., i] = np.where(tot[..., i] > 0, s * n[..., 0] / tot[..., i], 0.0)
    return C * f[..., None]


def binary_counts(C, classes, labels):
    C = np.asarray(C, dtype=float)
    ta = np.array([c != BENIGN for c in classes])
    pa = np.array([lab != BENIGN for lab in labels])
    tp = C[..., ta, :][..., pa].sum(axis=(-1, -2))
    fn = C[..., ta, :][..., ~pa].sum(axis=(-1, -2))
    fp = C[..., ~ta, :][..., pa].sum(axis=(-1, -2))
    tn = C[..., ~ta, :][..., ~pa].sum(axis=(-1, -2))
    return tp, fp, fn, tn


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def binary_metrics(C, classes, labels):
    tp, fp, fn, tn = binary_counts(C, classes, labels)
    prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {"accuracy": _div(tp + tn, tp + tn + fp + fn), "precision": prec, "recall": rec,
            "f1": _div(2 * tp, 2 * tp + fp + fn), "fpr": _div(fp, fp + tn),
            "mcc": _div(tp * tn - fp * fn, den)}


def class_metrics(C, classes, labels, cls):
    """Per class F1 and recall for one class; predicted labels must use class names."""
    C = np.asarray(C, dtype=float)
    i = classes.index(cls)
    tp = C[..., i, labels.index(cls)] if cls in labels else np.zeros(C.shape[:-2])
    row = C[..., i, :].sum(axis=-1)
    col = C[..., :, labels.index(cls)].sum(axis=-1) if cls in labels else np.zeros(C.shape[:-2])
    return {"f1": _div(2 * tp, row + col), "recall": _div(tp, row)}


def metric(C, classes, labels, name):
    """name: 'mcc', 'accuracy', 'f1', 'recall', 'fpr', 'precision' (binary) or
    'f1:<class>', 'recall:<class>' (per class)."""
    if ":" in name:
        kind, cls = name.split(":", 1)
        if cls not in classes:
            return np.full(np.asarray(C).shape[:-2], np.nan)
        return class_metrics(C, classes, labels, cls)[kind]
    return binary_metrics(C, classes, labels)[name]


def shares(C, classes):
    tot = np.asarray(C, dtype=float).sum(axis=-1)
    n = tot.sum(axis=-1, keepdims=True)
    return tot / n


def boot_weights(strata, B, rng):
    """(B, N) multiplicities, per stratum n draws with replacement from its n units
    (the Phase 5 scheme, phase5_analysis.boot_weights)."""
    strata = np.asarray(strata)
    N = len(strata)
    W = np.zeros((B, N), dtype=np.float64)
    for c in pd.unique(strata):
        idx = np.flatnonzero(strata == c)
        n = len(idx)
        draws = rng.integers(0, n, size=(B, n))
        for b in range(B):
            W[b, idx] = np.bincount(draws[b], minlength=n)
    return W
