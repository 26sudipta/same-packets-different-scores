"""
Step 3.8: score any model per flow and per conversation from one code path.

Both units start from the same inputs: one row per flow record with its
conversation, its true label and the model's prediction. The flow unit scores
those rows directly. The conversation unit first aggregates the predictions
of a conversation's records into one predicted label, then scores one row per
conversation against the conversation's label. Nothing else differs, so a
difference between the two units is a difference of unit only.

Labels are held at the conversation (the schedule label of step 3.2), so the
true label of a flow is the label of its conversation and the true label of
a conversation needs no aggregation.

Aggregation rules for the predicted label of a conversation:

    any       the conversation is flagged if any of its records is predicted
              as an attack; its class is the most common attack class among
              those records (ties: more bytes, then alphabetical). Primary rule,
              because an operator acts on an alert, and a flow-based detector
              raises one alert per flagged record: one flagged record is
              enough for the operator to look at the conversation.
    majority  the most common predicted class (ties: attack before benign,
              then alphabetical).
    bytes     the class with the most bytes (ties: alphabetical).

Macro-F1 is the mean per-class F1 over C+, the classes with test support
(metrics_macro_f1.py, copied from the conference pipeline).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from metrics_macro_f1 import macro_f1_cplus

BENIGN = "benign"
RULES = ("any", "majority", "bytes")
PRIMARY_RULE = "any"


def aggregate(conv, y_pred, rule=PRIMARY_RULE, weight=None, benign=BENIGN):
    """Predicted label per conversation, as a Series indexed by conversation."""
    if rule not in RULES:
        raise ValueError(rule)
    w = np.ones(len(conv)) if weight is None else np.asarray(weight, dtype=float)
    df = pd.DataFrame({"conv": np.asarray(conv), "p": np.asarray(y_pred, dtype=object), "w": w})
    convs = pd.Index(np.unique(df["conv"]), name="conv")
    if rule == "bytes":
        s = df.groupby(["conv", "p"])["w"].sum().rename("score").reset_index()
        s = s.sort_values(["conv", "score", "p"], ascending=[True, False, True],
                          kind="mergesort")
        return s.drop_duplicates("conv").set_index("conv")["p"].reindex(convs)
    if rule == "any":
        att = df[df["p"] != benign]
        s = att.groupby(["conv", "p"]).agg(n=("w", "size"), b=("w", "sum")).reset_index()
        s = s.sort_values(["conv", "n", "b", "p"], ascending=[True, False, False, True],
                          kind="mergesort")
        top = s.drop_duplicates("conv").set_index("conv")["p"]
        return top.reindex(convs).fillna(benign).astype(object)
    s = df.groupby(["conv", "p"]).size().rename("n").reset_index()
    s["is_benign"] = s["p"] == benign
    s = s.sort_values(["conv", "n", "is_benign", "p"], ascending=[True, False, True, True],
                      kind="mergesort")
    return s.drop_duplicates("conv").set_index("conv")["p"].reindex(convs)


def conversation_truth(conv, y_true):
    """True label per conversation. Raises if a conversation carries two labels,
    which would mean the labels were not held at the conversation."""
    df = pd.DataFrame({"conv": np.asarray(conv), "y": np.asarray(y_true, dtype=object)})
    n = df.groupby("conv")["y"].nunique()
    if (n > 1).any():
        raise ValueError(f"{int((n > 1).sum())} conversations carry more than one true label")
    return df.groupby("conv")["y"].first()


def per_class_f1(y_true, y_pred):
    labels = np.unique(np.asarray(y_true, dtype=object))
    f = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return dict(zip(labels.tolist(), f.tolist()))


def score(conv, y_true, y_pred, unit="flow", rule=PRIMARY_RULE, weight=None):
    """Scores at one unit. Returns dict: unit, rule, n, macro_f1, n_cplus, per_class."""
    y_true = np.asarray(y_true, dtype=object)
    y_pred = np.asarray(y_pred, dtype=object)
    if unit == "flow":
        t, p = y_true, y_pred
        rule_used = None
    elif unit == "conversation":
        truth = conversation_truth(conv, y_true)
        pred = aggregate(conv, y_pred, rule, weight).reindex(truth.index)
        t, p = truth.to_numpy(dtype=object), pred.to_numpy(dtype=object)
        rule_used = rule
    else:
        raise ValueError(unit)
    m, k = macro_f1_cplus(t, p)
    return {"unit": unit, "rule": rule_used, "n": int(len(t)), "macro_f1": m,
            "n_cplus": k, "per_class": per_class_f1(t, p)}


def score_both(conv, y_true, y_pred, rule=PRIMARY_RULE, weight=None):
    return (score(conv, y_true, y_pred, "flow"),
            score(conv, y_true, y_pred, "conversation", rule, weight))
