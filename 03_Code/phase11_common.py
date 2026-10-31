"""
Phase 11 (review response) helpers shared by the phase11_*.py scripts.

The Phase 11 runs answer a simulated pre-submission review of 30 October 2026;
the manuscript lists them under "Robustness analyses added after the primary
results" (Section 5). They add runs beside the frozen Phase 5
to 8 results and never overwrite them: every output goes to 04_Results/phase11/.

Run every script with NIDS_CACHE_ROOT pointing at the cache of freeze run 3, so
that the arm tables are the ones the frozen results were computed from:

    export NIDS_CACHE_ROOT=".../Datasets/_cache_freeze3"
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_scoring as CS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "04_Results" / "phase11"
BENIGN = "benign"


def truth_any(conv, y_true):
    """True label per conversation when records of one conversation may carry
    different labels (payload-aware and own labels): the conversation is an
    attack of class k when any of its records is labelled k, benign otherwise.
    Raises if one conversation holds two attack classes."""
    df = pd.DataFrame({"conv": np.asarray(conv), "y": np.asarray(y_true, dtype=object)})
    att = df[df["y"] != BENIGN]
    n = att.groupby("conv")["y"].nunique()
    if (n > 1).any():
        raise ValueError(f"{int((n > 1).sum())} conversations carry two attack classes")
    out = pd.Series(BENIGN, index=pd.Index(np.unique(df["conv"]), name="conv"), dtype=object)
    first = att.groupby("conv")["y"].first()
    out.loc[first.index] = first.to_numpy()
    return out


def use_truth_any():
    """Make conv_scoring use truth_any for the conversation unit (in this process)."""
    CS.conversation_truth = truth_any
