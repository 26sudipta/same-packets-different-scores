"""
Phase 8, steps 8.2 (fidelity table) and 8.3 (decomposition of each gap), from the
per-flow predictions that phase8_flow.py wrote. Protocol: 00_Plan/Phase8_Protocol.md.

Usage, from the workspace root:
    .venv/bin/python 03_Code/phase8_decomp.py R1 [R2 R3 R4 R5]

Outputs in 04_Results/phase8/decomp/:
    <claim>_arms.csv      metric per arm, model and day: mean over seeds, min, max,
                          and the published value where there is one (8.2)
    <claim>_gaps.csv      per contrast: G, G_shared, non-shared part, class weight part C,
                          within-class part W, share C / G_shared, each with its paired
                          cluster bootstrap interval (8.3 step 4), and the published gap
    <claim>_differs.csv   what differs between the two test sets (8.3 step 1)

Bootstrap: units are the instrument conversations of the test records (global id as
in Phase 5); one weight per conversation is shared by every arm, model and seed of the
claim, so the two sides of a gap are paired. Strata are schedule classes. Records the
instrument does not hold (NF-UNSW-NB15-v2, and CRiSIS-2022 RP until it is mapped) are
single units in a stratum of their own class. Replicates are drawn in batches of 20
from one generator with seed 0; replicate 0 of every output is the point estimate.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import conv_scoring as CS  # noqa: E402
import phase8_common as P  # noqa: E402
import releases as R  # noqa: E402

B = 1000
BATCH = 20
RNG_SEED = 0
PREDS = P.OUT / "preds"
DEC = P.OUT / "decomp"
PEKAR = R.DATA / "_downloads" / "CyberML-DataQuality"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# what each claim compares (8.1); published values as (old, new)

def _r1():
    log_vals = {"rf": (0.998103, 0.999201), "dt": (0.997631, 0.998457),
                "lda": (0.778515, 0.933473)}   # python_logs/analysis_info.log, 2021-09-06
    out = []
    for m in ("rf", "dt", "lda"):
        pub = {"mcc": log_vals[m]}
        if m == "rf":
            pub.update({"accuracy": (0.9991, 0.9996)})
        if m == "lda":
            pub.update({"accuracy": (0.8788, 0.9659), "fpr": (0.2261, 0.0635)})
        out.append(dict(name=m, a="cic_original", b="lycos", models=[m], metric="mcc",
                        side=["accuracy", "fpr", "recall", "precision"], published=pub,
                        source="MCC: authors' 2021 log (seed 0); accuracy, FPR: Table 2"))
    return out


def _r2():
    pub = {"bot": (0.60, 0.99), "web_bruteforce": (0.77, 0.95), "web_xss": (0.18, 0.79),
           "infiltration": (0.81, 0.91), "heartbleed": (0.77, 0.93), "web_sqli": (0.11, 0.00)}
    return [dict(name=c, a="cic_original", b="wtmc2021", models=["rf"], metric=f"f1:{c}",
                 side=["macro_f1"], published={f"f1:{c}": v}, source="Table II")
            for c, v in pub.items()]


def _r3():
    models = ["dt", "rf", "nb"]
    src = "section 4.2, Figures 1 and 2 (read from the text: about)"
    out = [dict(name="Q2", a="crisis2022_rd", b="crisis2022_rpd", models=models,
                metric="recall", published={"recall_gap": -0.20}, source=src),
           dict(name="Q3", a="crisis2022_rpd", b="crisis2022_rp", models=models,
                metric="recall", published={"recall": (0.80, 1.00)}, source=src),
           dict(name="Q4", a="crisis2022_rd", b="crisis2022_rp", models=models,
                metric="recall", published={"recall_gap": 0.0}, no_diff=True, source=src)]
    for q in list(out):
        for m in models:
            out.append(dict(q, name=f"{q['name']}_{m}", models=[m], published={},
                            side_only=True))
    return out


def _r4():
    out = [dict(name="bin_v2", a="unsw_original", b="nf_unsw_v2", models=["et_bin"],
                metric="f1", side=["recall", "precision", "fpr"],
                published={"f1": (0.92, 0.97)}, source="Table 8"),
           dict(name="bin_v3", a="unsw_original", b="nf_unsw_v3", models=["et_bin"],
                metric="f1", side=["recall", "precision", "fpr"], published={},
                source="v3 in place of v2 (8.4 proxy)")]
    for c, v in (("fuzzers", (0.5050, 0.8057)), ("worms", (0.0526, 0.8598))):
        out.append(dict(name=f"{c}_v2", a="unsw_original", b="nf_unsw_v2", models=["et_multi"],
                        metric=f"recall:{c}", published={f"recall:{c}": v}, source="Table 9"))
        out.append(dict(name=f"{c}_v3", a="unsw_original", b="nf_unsw_v3", models=["et_multi"],
                        metric=f"recall:{c}", published={}, source="v3 in place of v2"))
    return out


# crisis2022_rp: their CRiSIS-2022 notebook reads Exp_RP (protocol amendment 1)
R5_PEKAR = {"cic_original": "1-CICIDS-2017/datasets", "wtmc2021": "2-WTMC-2021/datasets",
            "crisis2022_rp": "3-CRiSIS-2022/datasets", "nfs2023_te": "4-NFS-2023/NFS-2023-TE",
            "nfs2023_nte": "4-NFS-2023/NFS-2023-nTE"}


def r5_published():
    """F1 per arm and day from the authors' result files (no_feat_sel, binary)."""
    out = {}
    for arm, d in R5_PEKAR.items():
        j = json.loads((PEKAR / d / "no_feat_sel/binary-class/model_performance_metrics.json")
                       .read_text())
        f = j["columns"].index("F1 Score")
        for day, row in zip(j["index"], j["data"]):
            out[(arm, day)] = row[f]
    return out


def _r5():
    pub = r5_published()
    out = []

    def c(name, day, a, b, no_diff=False, extra=False):
        out.append(dict(name=name, a=a, b=b, models=["rf"], metric="f1", day=day,
                        side=["recall", "precision", "fpr"], no_diff=no_diff,
                        published={"f1": (pub.get((a, day)), pub.get((b, day)))},
                        side_only=extra, extra="our instrument arm (Exp_RPD)" if extra else "",
                        source="authors' result files (Table 6)"))

    for arm in ("wtmc2021", "crisis2022_rp", "nfs2023_nte", "nfs2023_te"):
        c(f"a_Tuesday_{arm}_vs_cic", "Tuesday", "cic_original", arm)
    c("b_Thursday_nte_vs_te", "Thursday", "nfs2023_te", "nfs2023_nte")
    c("b_Thursday_crisis_vs_wtmc", "Thursday", "wtmc2021", "crisis2022_rp")
    # protocol amendment 1: the instrument arm (Exp_RPD) beside, not as the claim's test
    c("x_Tuesday_crisis2022_rpd_vs_cic", "Tuesday", "cic_original", "crisis2022_rpd", extra=True)
    c("x_Thursday_rpd_vs_wtmc", "Thursday", "wtmc2021", "crisis2022_rpd", extra=True)
    c("b_Thursday_nte_vs_cic", "Thursday", "cic_original", "nfs2023_nte")
    for day in ("Tuesday", "Wednesday", "Thursday", "Friday"):
        c(f"c_{day}_te_vs_nte", day, "nfs2023_nte", "nfs2023_te", no_diff=True)
    return out


CLAIMS = {"R1": _r1, "R2": _r2, "R3": _r3, "R4": _r4, "R5": _r5}


# ---------------------------------------------------------------------------
# loading

def lookup(arm):
    """file, row -> gconv, sched; None when the release has no conversations."""
    if arm == "crisis2022_rd":   # same rows as RPD (checked by phase8_flow.py R3)
        return P.conv_lookup("crisis2022_rpd")
    if arm == "crisis2022_r":    # same rows as RP (checked by phase8_flow.py R3rp)
        arm = "crisis2022_rp"
    if arm in P.EXTRA_RELEASES:
        p = P.CACHE / "convmap" / f"{arm}.parquet"   # RP: written by phase8_rp_map.py
        return pd.read_parquet(p) if p.is_file() else None
    return P.conv_lookup(arm)


def cell_files(root, arm, model, day):
    tail = f"_{day}" if day else ""
    out = {}
    for s in P.SEEDS:
        p = root / arm / f"{model}_s{s}{tail}.parquet"
        if p.is_file():
            out[s] = p
    return out


# 8.4: the same contrasts on the conversation footing (phase8_conv.py pairs).
# Under schedule labels RD = RPD and R = RP (same records), so R3 Q2 is zero by
# construction and Q4 is Q3; R4 uses NF-UNSW-NB15-v3 for v2 (no timestamps in v2).
CONV_PAIR = {
    "R1": lambda ct: dict(ct, pair="cic_lycos"),
    "R2": lambda ct: dict(ct, pair="cic_wtmc"),
    "R3": lambda ct: (dict(ct, zero=True) if ct["name"].startswith("Q2") else
                      dict(ct, a="crisis2022_rpd", b="crisis2022_rp", pair="rpd_rp")),
    "R4": lambda ct: dict(ct, b="nf_unsw_v3", pair="unsw_v3",
                          proxy="v3 for v2" if ct["b"] == "nf_unsw_v2" else ""),
    "R5": lambda ct: dict(ct, pair=_r5_pair(ct)),
}


def _r5_pair(ct):
    a, b = ct["a"], ct["b"]
    if {a, b} == {"nfs2023_te", "nfs2023_nte"}:
        return "nte_te"
    if {a, b} == {"wtmc2021", "crisis2022_rpd"}:
        return "wtmc_crisis"
    if {a, b} == {"wtmc2021", "crisis2022_rp"}:
        return "wtmc_crisis_rp"
    return f"cic_{b}"


def conv_contrasts(claim):
    out = []
    for ct in CLAIMS[claim]():
        c = CONV_PAIR[claim](ct)
        if claim == "R4" and ct["b"] == "nf_unsw_v3":
            continue   # already the proxy of the v2 contrast
        out.append(c)
    return out


class Units:
    """One index over every bootstrap unit of a claim."""

    def __init__(self):
        self.ids, self.strata = [], []
        self._neg = 0

    def add(self, gconv, strata):
        # one entry per unit and cell, its first occurrence: freeze keeps the first
        # occurrence overall, so this changes nothing but the memory (R3 with RP and R
        # held about 90 million records)
        g = np.asarray(gconv, dtype=np.int64)
        _, first = np.unique(g, return_index=True)
        first.sort()
        self.ids.append(g[first])
        self.strata.append(np.asarray(strata, dtype=object)[first])

    def new_negative(self, n):
        out = -1 - (self._neg + np.arange(n, dtype=np.int64))
        self._neg += n
        return out

    def freeze(self):
        ids = np.concatenate(self.ids)
        st = np.concatenate(self.strata)
        u, first = np.unique(ids, return_index=True)
        self.index = pd.Index(u)
        self.stratum = st[first]
        return self

    def weights(self, rng, b):
        W = np.zeros((b, len(self.index)), dtype=np.float32)
        for c in pd.unique(self.stratum):
            idx = np.flatnonzero(self.stratum == c)
            n = len(idx)
            draws = rng.integers(0, n, size=(b, n), dtype=np.int64)
            for r in range(b):
                W[r, idx] = np.bincount(draws[r], minlength=n)
        return W


def load_cell(claim, arm, path, units, lk_cache, level="flow"):
    """Test records of one cell with canonical truth and prediction and their unit id.
    8.4 cells carry gconv and stratum; with level "conv" they become one row per
    conversation (truth: its schedule label; prediction: rule "any", ties by bytes)."""
    df = pd.read_parquet(path)
    if "gconv" in df.columns:
        if level == "conv":
            truth = CS.conversation_truth(df["gconv"].to_numpy(), df["y"].to_numpy())
            pred = CS.aggregate(df["gconv"].to_numpy(), df["pred"].to_numpy(), "any",
                                df["bytes"].to_numpy()).reindex(truth.index)
            st = df.drop_duplicates("gconv").set_index("gconv")["stratum"].reindex(truth.index)
            df = pd.DataFrame({"gconv": truth.index.to_numpy(), "y": truth.to_numpy(),
                               "pred": pred.to_numpy(), "stratum": st.to_numpy(),
                               "file": "", "row": -1})
        t, att = P.canonical(arm, df["y"].to_numpy())
        q, _ = P.canonical(arm, df["pred"].to_numpy())
        g = df["gconv"].to_numpy(dtype=np.int64)
        units.add(g, df["stratum"].astype(object).to_numpy())
        return dict(true=t, pred=q, unit=g, attempted=int(att.sum()), unmapped=0)
    t, att = P.canonical(arm, df["y"].to_numpy())
    q, _ = P.canonical(arm, df["pred"].to_numpy())
    if arm not in lk_cache:
        lk_cache[arm] = lookup(arm)
    lk = lk_cache[arm]
    if lk is not None:
        m = df[["file", "row"]].merge(lk, on=["file", "row"], how="left", validate="many_to_one")
        g = m["gconv"].to_numpy(dtype=np.float64)
        st = m["sched"].astype(object).to_numpy()
        miss = np.isnan(g)
    else:
        g = np.full(len(df), np.nan)
        st = np.empty(len(df), dtype=object)
        miss = np.ones(len(df), dtype=bool)
    if miss.any():
        key = (arm, str(path.name))
        g[miss] = units.new_negative(int(miss.sum()))
        st[miss] = np.array(["unmapped:" + str(x) for x in t[miss]], dtype=object)
    g = g.astype(np.int64)
    units.add(g, st)
    return dict(true=t, pred=q, unit=g, attempted=int(att.sum()), unmapped=int(miss.sum()))


# ---------------------------------------------------------------------------
# metrics on (n, K, L) confusion stacks

def metric(C, classes, labels, name, shared=None):
    if name == "macro_f1":
        cls = [c for c in classes if shared is None or c in shared]
        return np.nanmean(np.stack([P.class_metrics(C, classes, labels, c)["f1"] for c in cls]),
                          axis=0)
    return P.metric(C, classes, labels, name)


def reweight_to(C, target):
    """Scale each true-class row so the class totals follow target shares (n, K)."""
    tot = C.sum(axis=-1)
    n = tot.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(tot > 0, target * n / np.where(tot > 0, tot, 1), 0.0)
    return C * f[..., None]


def shares(C):
    tot = C.sum(axis=-1)
    return tot / tot.sum(axis=-1, keepdims=True)


def decompose(CA, CB, classes, labels, name, shared):
    """Per replicate: G, G_shared, non-shared, C, W, and the two arm values."""
    mask = np.array([c in shared for c in classes], dtype=float)[None, :, None]
    sA, sB = CA * mask, CB * mask
    mAA = metric(sA, classes, labels, name, shared)
    mBB = metric(sB, classes, labels, name, shared)
    mBA = metric(reweight_to(sB, shares(sA)), classes, labels, name, shared)
    mAB = metric(reweight_to(sA, shares(sB)), classes, labels, name, shared)
    g = metric(CB, classes, labels, name, shared) - metric(CA, classes, labels, name, shared)
    gs = mBB - mAA
    c = 0.5 * ((mBB - mBA) + (mAB - mAA))
    return dict(G=g, G_shared=gs, nonshared=g - gs, C=c, W=gs - c,
                A=metric(CA, classes, labels, name, shared),
                B=metric(CB, classes, labels, name, shared))


# ---------------------------------------------------------------------------

def run(claim, stage="flow", variant="sched", level="flow"):
    """stage "flow": 8.2 and 8.3 on the per-flow rebuilds. stage "conv": 8.4 cells of
    phase8_conv.py, scored per record (level "flow") or per conversation ("conv")."""
    if stage == "flow":
        contrasts, tag = CLAIMS[claim](), claim
    else:
        contrasts, tag = conv_contrasts(claim), f"{claim}_conv_{variant}_{level}"
    zero = [ct for ct in contrasts if ct.get("zero")]
    contrasts = [ct for ct in contrasts if not ct.get("zero")]

    def root(ct):
        if stage == "flow":
            return PREDS / claim
        return P.OUT / "conv" / "preds" / claim / variant / ct["pair"]

    def ckey(ct, arm, m, s):
        return (arm, m, ct.get("day"), s, ct.get("pair"))

    units, lk_cache, cells = Units(), {}, {}
    # every cell any contrast needs
    for ct in contrasts:
        for arm in (ct["a"], ct["b"]):
            for m in ct["models"]:
                for s, p in cell_files(root(ct), arm, m, ct.get("day")).items():
                    key = ckey(ct, arm, m, s)
                    if key not in cells:
                        cells[key] = load_cell(claim, arm, p, units, lk_cache, level)
    if not cells:
        raise SystemExit(f"{tag}: no predictions yet")
    units.freeze()
    log(f"{claim}: {len(cells)} cells, {len(units.index):,} bootstrap units")
    for key, c in cells.items():
        if c["attempted"]:
            log(f"  note: {key} has {c['attempted']} Attempted test records (kept as labelled)")
        c["uidx"] = units.index.get_indexer(c["unit"])

    # class and label lists per contrast; sparse unit x cell matrices per cell
    plans = []
    for ct in contrasts:
        pairs = []
        for m in ct["models"]:
            for s in P.SEEDS:
                ka, kb = ckey(ct, ct["a"], m, s), ckey(ct, ct["b"], m, s)
                if ka in cells and kb in cells:
                    pairs.append((ka, kb))
        if not pairs:
            log(f"  {ct['name']}: no paired cells, skipped")
            continue
        classes = sorted(set().union(*[set(cells[k]["true"]) for p in pairs for k in p]))
        labels = sorted(set(classes).union(*[set(cells[k]["pred"]) for p in pairs for k in p]))
        shared = [sorted(set(cells[ka]["true"]) & set(cells[kb]["true"])) for ka, kb in pairs]
        plans.append(dict(ct=ct, pairs=pairs, classes=classes, labels=labels, shared=shared))

    mats = {}

    def mat(key, classes, labels):
        mk = (key, tuple(classes), tuple(labels))
        if mk not in mats:
            c = cells[key]
            ci = {x: i for i, x in enumerate(classes)}
            li = {x: j for j, x in enumerate(labels)}
            cell = (np.array([ci[x] for x in c["true"]]) * len(labels) +
                    np.array([li[x] for x in c["pred"]]))
            M = sp.csr_matrix((np.ones(len(cell), dtype=np.float32), (c["uidx"], cell)),
                              shape=(len(units.index), len(classes) * len(labels)))
            mats[mk] = M.T.tocsr()
        return mats[mk]

    names = ["G", "G_shared", "nonshared", "C", "W", "A", "B"]
    res = {i: {n: [] for n in names} for i in range(len(plans))}
    side_res = {i: {} for i in range(len(plans))}
    rng = np.random.default_rng(RNG_SEED)
    t0 = time.time()
    done = 0
    while done < B + 1:
        b = min(BATCH, B + 1 - done)
        if done == 0:
            W = np.vstack([np.ones((1, len(units.index)), dtype=np.float32),
                           units.weights(rng, b - 1)])
        else:
            W = units.weights(rng, b)
        WT = np.ascontiguousarray(W.T)
        for i, pl in enumerate(plans):
            K, L = len(pl["classes"]), len(pl["labels"])
            acc = {n: [] for n in names}
            sacc = {}
            for (ka, kb), sh in zip(pl["pairs"], pl["shared"]):
                CA = np.asarray(mat(ka, pl["classes"], pl["labels"]) @ WT).T.reshape(b, K, L)
                CB = np.asarray(mat(kb, pl["classes"], pl["labels"]) @ WT).T.reshape(b, K, L)
                d = decompose(CA.astype(np.float64), CB.astype(np.float64), pl["classes"],
                              pl["labels"], pl["ct"]["metric"], sh)
                for n in names:
                    acc[n].append(d[n])
                for sm in pl["ct"].get("side", []):
                    va = metric(CA, pl["classes"], pl["labels"], sm, sh)
                    vb = metric(CB, pl["classes"], pl["labels"], sm, sh)
                    sacc.setdefault(sm, []).append(np.stack([va, vb]))
            for n in names:
                res[i][n].append(np.nanmean(np.stack(acc[n]), axis=0))
            for sm, v in sacc.items():
                side_res[i].setdefault(sm, []).append(np.nanmean(np.stack(v), axis=0))
        done += b
        if done % 200 < BATCH or done == B + 1:
            log(f"  {claim}: {done - 1} of {B} replicates ({time.time() - t0:.0f} s)")

    DEC.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, pl in enumerate(plans):
        ct = pl["ct"]
        r = {n: np.concatenate(res[i][n]) for n in names}
        row = dict(claim=claim, contrast=ct["name"], day=ct.get("day", ""), a=ct["a"],
                   b=ct["b"], models="+".join(ct["models"]), metric=ct["metric"],
                   n_pairs=len(pl["pairs"]), no_diff=bool(ct.get("no_diff", False)),
                   side_only=bool(ct.get("side_only", False)),
                   shared="|".join(pl["shared"][0]),
                   nonshared_classes="|".join(sorted(set(pl["classes"]) - set(pl["shared"][0]))))
        for n in names:
            pt, bs = r[n][0], r[n][1:]
            row[n] = pt
            row[n + "_lo"], row[n + "_hi"] = np.nanpercentile(bs, [2.5, 97.5])
        gs = r["G_shared"]
        excl = (row["G_shared_lo"] > 0) or (row["G_shared_hi"] < 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            sh = r["C"] / gs
        row["share_C"] = sh[0] if excl else np.nan
        row["share_C_lo"], row["share_C_hi"] = (np.nanpercentile(sh[1:], [2.5, 97.5])
                                                if excl else (np.nan, np.nan))
        row["share_note"] = "" if excl else "gap not distinguishable from zero"
        pub = ct.get("published", {})
        pm = pub.get(ct["metric"])
        if isinstance(pm, tuple) and None not in pm:
            row["published_A"], row["published_B"] = pm
            row["published_gap"] = pm[1] - pm[0]
        elif "recall_gap" in pub:
            row["published_gap"] = pub["recall_gap"]
        row["published_source"] = ct.get("source", "")
        if "published_gap" in row and not np.isnan(row["published_gap"]):
            pg = row["published_gap"]
            row["direction_reproduced"] = bool(np.sign(row["G"]) == np.sign(pg)) if pg else ""
            row["within_0.01"] = bool(abs(row["G"] - pg) <= 0.01)
        for sm, v in side_res[i].items():
            v = np.concatenate(v, axis=1)
            row[f"{sm}_A"], row[f"{sm}_B"] = v[0, 0], v[1, 0]
            g = v[1] - v[0]
            row[f"{sm}_gap"] = g[0]
            row[f"{sm}_gap_lo"], row[f"{sm}_gap_hi"] = np.nanpercentile(g[1:], [2.5, 97.5])
            ps = pub.get(sm)
            if isinstance(ps, tuple):
                row[f"{sm}_published_A"], row[f"{sm}_published_B"] = ps
        if ":" in ct["metric"]:   # 8.4 rule: fewer than 20 test units of the class
            cls = ct["metric"].split(":", 1)[1]
            row["min_test_units_class"] = min(
                len(np.unique(cells[k]["unit"][cells[k]["true"] == cls]))
                for p in pl["pairs"] for k in p)
        row["proxy"] = ct.get("proxy", "")
        rows.append(row)
    for ct in zero:
        rows.append(dict(claim=claim, contrast=ct["name"], day=ct.get("day", ""), a=ct["a"],
                         b=ct["b"], models="+".join(ct["models"]), metric=ct["metric"],
                         no_diff=bool(ct.get("no_diff", False)),
                         side_only=bool(ct.get("side_only", False)), G=0.0, G_lo=0.0, G_hi=0.0,
                         share_note="zero by construction: same records and one truth",
                         published_gap=ct.get("published", {}).get("recall_gap", np.nan),
                         published_source=ct.get("source", "")))
    pd.DataFrame(rows).to_csv(DEC / f"{tag}_gaps.csv", index=False)

    # 8.2: metric per arm, model, day over seeds (point estimates)
    arow = []
    for i, pl in enumerate(plans):
        ct = pl["ct"]
        if ct.get("side_only"):
            continue
        for side, arm in (("A", ct["a"]), ("B", ct["b"])):
            for m in ct["models"]:
                vals = []
                for ka, kb in pl["pairs"]:
                    k = ka if side == "A" else kb
                    if k[1] != m:
                        continue
                    C = mat(k, pl["classes"], pl["labels"]).sum(axis=1)
                    C = np.asarray(C).reshape(1, len(pl["classes"]), len(pl["labels"]))
                    vals.append(float(metric(C, pl["classes"], pl["labels"], ct["metric"])[0]))
                pm = ct.get("published", {}).get(ct["metric"])
                pv = pm[0 if side == "A" else 1] if isinstance(pm, tuple) else np.nan
                arow.append(dict(claim=claim, contrast=ct["name"], day=ct.get("day", ""),
                                 arm=arm, model=m, metric=ct["metric"], seeds=len(vals),
                                 mean=np.mean(vals), min=np.min(vals), max=np.max(vals),
                                 seed0=vals[0], published=pv))
    pd.DataFrame(arow).drop_duplicates(["day", "arm", "model", "metric"]).to_csv(
        DEC / f"{tag}_arms.csv", index=False)

    # 8.3 step 1: what differs, per pair of arms (first common seed of the first model)
    drow = []
    seen = set()
    for pl in plans:
        ct = pl["ct"]
        ka, kb = pl["pairs"][0]
        pair = (ct["a"], ct["b"], ct.get("day"))
        if pair in seen:
            continue
        seen.add(pair)
        for k, other in ((ka, ct["b"]), (kb, ct["a"])):
            c = cells[k]
            mapped = c["unit"] >= 0
            olk = lk_cache.get(other) if other in lk_cache else lookup(other)
            if olk is not None and mapped.any():
                absent = ~np.isin(c["unit"][mapped], olk["gconv"].to_numpy())
                absent_share = absent.mean()
            else:
                absent_share = np.nan
            cls, cnt = np.unique(c["true"], return_counts=True)
            drow.append(dict(claim=claim, day=ct.get("day", ""), arm=k[0], other=other,
                             seed=k[3], test_records=len(c["true"]),
                             mapped_records=int(mapped.sum()),
                             conversations=int(len(np.unique(c["unit"][mapped]))),
                             records_per_conv=(mapped.sum() / max(1, len(np.unique(
                                 c["unit"][mapped])))),
                             share_in_conv_absent_from_other=absent_share,
                             attempted=c["attempted"],
                             class_shares="|".join(f"{a}:{b / cnt.sum():.4f}"
                                                   for a, b in zip(cls, cnt))))
    pd.DataFrame(drow).to_csv(DEC / f"{tag}_differs.csv", index=False)
    log(f"{tag}: written {DEC}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("claims", nargs="+", choices=list(CLAIMS))
    ap.add_argument("--stage", choices=["flow", "conv"], default="flow")
    ap.add_argument("--variant", choices=["sched", "fold"], default="sched")
    a = ap.parse_args()
    for c in a.claims:
        if a.stage == "flow":
            run(c)
        else:   # both units from the same generator seed: the replicates are paired
            for level in ("flow", "conv"):
                run(c, "conv", a.variant, level)
