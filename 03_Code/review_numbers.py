"""
Phase 11 (review response): the numbers the revised manuscript quotes from the Phase 11
runs, as macros and one table. Called by make_numbers.py, which passes its macro
store and writer, so that every number still reaches the paper through one script.

Inputs, all under 04_Results/phase11/ (written by phase11_*.py):
    analysis/*.csv           phase11_analysis.py (core, payload, payload_h2, own, ttl,
                             ht_h2, models, loo, full)
    payload/payload_check.csv phase11_payload.py check
    nfs_idle_probe.csv       phase11_nfs_idle_probe.py
    split_probe.csv          phase11_split_probe.py
    loo/instrument_loo_*.csv phase11_instrument_loo.py
Outputs: macros named Rev..., and 05_Paper/tables/rev_robustness.tex.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P11 = ROOT / "04_Results" / "phase11"
AN = P11 / "analysis"
CIC, UNSW = "CICIDS2017", "UNSW-NB15"
# the RQ3 classes: attack classes with at least 20 common conversations in both NFS arms
RQ3_EXCLUDE = {"benign", "web_sqli", "heartbleed", "infiltration"}


def verdict(df, bench, setting, kind="raw"):
    r = df[(df["benchmark"] == bench) & (df["setting"] == setting) & (df["kind"] == kind)]
    assert len(r) == 1, (bench, setting, kind, len(r))
    r = r.iloc[0]
    return int(r["exporter_above"]), int(r["eligible"]), r["h2"]


def review(M, write, f3, s3, n, pct, word):
    outs = []
    src = "phase11/analysis/"

    # ------------------------------------------------------------------ H2 robustness (M3, m10, m11, m13, M5, M1)
    ms = pd.read_csv(AN / "ms_h2.csv")
    rows = []
    for bench, tag in ((CIC, "Cic"), (UNSW, "Unsw")):
        a, e, h = verdict(ms, bench, "grouped flow own")
        M(f"RevPrimAbove{tag}", a, src + "ms_h2.csv, raw shares, flow, own")
        M(f"RevElig{tag}", e, src + "ms_h2.csv")
        a, e, h = verdict(ms, bench, "grouped flow own", "per_df")
        M(f"RevPerdfAbove{tag}", a, src + "ms_h2.csv, per degree of freedom")
        M(f"RevPerdfHtwo{tag}", h, src + "ms_h2.csv")
    rows.append(("Primary setting",) + tuple(verdict(ms, b, "grouped flow own")[:2] for b in (CIC, UNSW)))
    rows.append(("Shares per degree of freedom",) + tuple(verdict(ms, b, "grouped flow own", "per_df")[:2]
                                                         for b in (CIC, UNSW)))
    bca = pd.read_csv(AN / "bca.csv")
    e = bca[(bca["statistic"] == "arm_minus_model") & bca["eligible"]]
    same = ((e["pct_lo"] > 0) == (e["bca_lo"] > 0)) & ((e["pct_hi"] < 0) == (e["bca_hi"] < 0))
    M("RevBcaSame", int(same.sum()), src + "bca.csv, eligible classes, same verdict under BCa")
    M("RevBcaN", len(e), src + "bca.csv")
    bc = {b: int((e[e["benchmark"] == b]["bca_lo"] > 0).sum()) for b in (CIC, UNSW)}
    rows.append(("BCa intervals",) + tuple((bc[b], len(e[e["benchmark"] == b])) for b in (CIC, UNSW)))
    g = bca[(bca["benchmark"] == UNSW) & (bca["cls"] == "generic") & (bca["statistic"] == "arm_share")].iloc[0]
    M("RevBcaGenericShare", f3(g["value"]), src + "bca.csv, UNSW-NB15 Generic exporter share")
    M("RevBcaGenericPctLo", f3(g["pct_lo"]), src + "bca.csv")
    M("RevBcaGenericPctHi", f3(g["pct_hi"]), src + "bca.csv")
    M("RevBcaGenericLo", f3(g["bca_lo"]), src + "bca.csv")
    M("RevBcaGenericHi", f3(g["bca_hi"]), src + "bca.csv")

    thr = pd.read_csv(AN / "thresholds.csv")
    for bench, tag in ((CIC, "Cic"), (UNSW, "Unsw")):
        t = thr[thr["benchmark"] == bench]
        M(f"RevThrHold{tag}", int((t["h2"] == "holds").sum()), src + "thresholds.csv")
        M(f"RevThrEligMin{tag}", int(t["eligible"].min()), src + "thresholds.csv")
        M(f"RevThrEligMax{tag}", int(t["eligible"].max()), src + "thresholds.csv")
    M("RevThrSettings", word(int(len(thr) / 2)), src + "thresholds.csv, settings per benchmark")
    rows.append(("Size and support thresholds varied",) + tuple(
        f"holds at {int((thr[thr['benchmark'] == b]['h2'] == 'holds').sum())} of "
        f"{int((thr['benchmark'] == b).sum())}" for b in (CIC, UNSW)))

    sub = pd.read_csv(AN / "h2_subsets_verdict.csv")
    for bench, tag in ((CIC, "Cic"), (UNSW, "Unsw")):
        s = sub[sub["benchmark"] == bench]
        M(f"RevSubHold{tag}", word(int((s["h2"] == "holds").sum())), src + "h2_subsets_verdict.csv")
        M(f"RevSubN{tag}", word(len(s)), src + "h2_subsets_verdict.csv")
        lo = s["exporter_above"].min()
        M(f"RevSubAboveMin{tag}", int(lo), src + "h2_subsets_verdict.csv")
    rows.append(("One arm left out, each in turn",) + tuple(
        f"holds in {int((sub[sub['benchmark'] == b]['h2'] == 'holds').sum())} of "
        f"{int((sub['benchmark'] == b).sum())}" for b in (CIC, UNSW)))
    u = sub[sub["benchmark"] == UNSW].set_index("setting")
    M("RevSubUnswNoCicAbove", int(u.loc["without cic_unsw", "exporter_above"]), src + "h2_subsets_verdict.csv")
    M("RevSubUnswNoCicModel", int(u.loc["without cic_unsw", "model_above"]), src + "h2_subsets_verdict.csv")

    ht = pd.read_csv(AN / "ht_h2_verdict.csv")
    for bench, tag in ((CIC, "Cic"), (UNSW, "Unsw")):
        a, e_, h = verdict(ht, bench, "full-data prevalence, flow")
        M(f"RevHtAbove{tag}", a, src + "ht_h2_verdict.csv")
        M(f"RevHtHtwo{tag}", h, src + "ht_h2_verdict.csv")
    rows.append(("Test set reweighted to full-data prevalence",) +
                tuple(verdict(ht, b, "full-data prevalence, flow")[:2] for b in (CIC, UNSW)))

    pay = pd.read_csv(AN / "payload_h2_verdict.csv")
    a, e_, h = verdict(pay, CIC, "payload-aware labels, flow")
    M("RevPayAbove", a, src + "payload_h2_verdict.csv")
    M("RevPayElig", e_, src + "payload_h2_verdict.csv")
    M("RevPayHtwo", h, src + "payload_h2_verdict.csv")
    pf = pd.read_csv(AN / "payload_h2.csv")
    pf = pf[(pf["setting"] == "payload-aware labels, flow") & (pf["kind"] == "raw")]
    M("RevPayPointAbove", int((pf["contrast"] > 0).sum()), src + "payload_h2.csv, point estimate above zero")
    M("RevPayModelAbove", int((pf["verdict"] == "model above exporter").sum()), src + "payload_h2.csv")
    a2, e2, h2 = verdict(pay, CIC, "payload-aware labels, conversation")
    M("RevPayAboveConv", a2, src + "payload_h2_verdict.csv")
    rows.append(("Payload-aware labels (\\cicids{} only)", (a, e_), None))

    tt = pd.read_csv(AN / "ttl_h2_verdict.csv")
    a, e_, h = verdict(tt, UNSW, "without TTL")
    M("RevTtlAbove", a, src + "ttl_h2_verdict.csv")
    M("RevTtlElig", e_, src + "ttl_h2_verdict.csv")
    rows.append(("TTL fields removed (UNSW-NB15 only)", None, (a, e_)))
    td = pd.read_csv(AN / "ttl_h2.csv").set_index(["setting", "cls"])
    M("RevTtlDosWith", f3(td.loc[("with TTL", "dos"), "contrast"]), src + "ttl_h2.csv")
    M("RevTtlDosWithout", f3(td.loc[("without TTL", "dos"), "contrast"]), src + "ttl_h2.csv")
    M("RevTtlMinLo", f3(td.loc["without TTL", "lo"].min()), src + "ttl_h2.csv, smallest lower bound")

    mv = AN / "models_h2_verdict.csv"
    if mv.exists():
        m5 = pd.read_csv(mv)
        r5 = tuple(verdict(m5, b, "five models")[:2] for b in (CIC, UNSW))
        rows.append(("Five models (adds logistic regression and $k$-NN)",) + r5)
        for (a, e_), tag in zip(r5, ("Cic", "Unsw")):
            M(f"RevFiveAbove{tag}", a, src + "models_h2_verdict.csv")
        mm = AN / "models_macro.csv"
        if mm.exists():
            mc = pd.read_csv(mm)
            mc = mc[(mc["unit"] == "flow") & (mc["population"] == "common")]
            low = mc.loc[mc.groupby(["benchmark", "arm"])["macro_fixed"].idxmin(), "model"]
            M("RevFiveLrLowest", str(int((low == "lr").sum())), src + "models_macro.csv, arms where logistic "
              "regression has the lowest macro F1, flow, common")
            M("RevFiveArms", str(len(low)), src + "models_macro.csv")
        r5d = tuple(verdict(m5, b, "five models", "per_df")[:2] for b in (CIC, UNSW))
        for (a, e_), tag in zip(r5d, ("Cic", "Unsw")):
            M(f"RevFivePerdfAbove{tag}", a, src + "models_h2_verdict.csv")
    lines = []
    for label, c, u_ in rows:
        cell = lambda v: "n/a" if v is None else v if isinstance(v, str) else f"{v[0]} of {v[1]}"  # noqa: E731
        lines.append(f"{label} & {cell(c)} & {cell(u_)} \\\\\n")
    body = ("\\begin{table}[t]\n\\caption{H2 under robustness settings, grouped split, flow unit, each "
            "arm's own population: eligible small classes for which the interval of the exporter share "
            "minus the model share lies above zero, out of the eligible classes. H2 holds when this is a "
            "strict majority. The rows on thresholds and on left-out arms count the settings in which H2 "
            "holds}\\label{tab:rev-robust}\n\\small\n"
            "\\begin{tabular}{@{}lll@{}}\n\\toprule\nSetting & \\cicids{} & UNSW-NB15 \\\\\n\\midrule\n"
            + "".join(lines) + "\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    outs.append(write("rev_robustness.tex", body, "04_Results/phase11/analysis"))

    # ------------------------------------------------------------------ Definition 1 tested directly (M2)
    eq = pd.read_csv(AN / "equivalence.csv")
    for bench, tag in ((CIC, "Cic"), (UNSW, "Unsw")):
        for unit, ut in (("flow", "Flow"), ("conversation", "Conv")):
            q = eq[(eq["benchmark"] == bench) & (eq["unit"] == unit)]
            pc, mac = q[q["cls"] != "MACRO_FIXED"], q[q["cls"] == "MACRO_FIXED"]
            M(f"RevEq{tag}{ut}Two", pct(pc["equiv_0.02"].mean()), src + "equivalence.csv, per class, margin 0.02")
            M(f"RevEq{tag}{ut}Five", pct(pc["equiv_0.05"].mean()), src + "equivalence.csv, per class, margin 0.05")
            M(f"RevEqMac{tag}{ut}Five", pct(mac["equiv_0.05"].mean()), src + "equivalence.csv, macro, margin 0.05")
            M(f"RevEqMac{tag}{ut}Two", pct(mac["equiv_0.02"].mean()), src + "equivalence.csv, macro, margin 0.02")
        M(f"RevEqTests{tag}", n(len(eq[(eq["benchmark"] == bench) & (eq["unit"] == "flow") &
                                     (eq["cls"] != "MACRO_FIXED")])), src + "equivalence.csv")

    # ------------------------------------------------------------------ prevalence (M5)
    h = pd.read_csv(AN / "ht.csv")
    h = h[(h["population"] == "own") & (h["cls"] != "benign")]
    p = h.pivot_table(index=["benchmark", "arm", "model", "seed", "unit", "cls"], columns="weights",
                      values=["precision", "recall", "f1"])
    d = (p.xs("full_prevalence", axis=1, level=1) - p.xs("subsample", axis=1, level=1)).reset_index()
    assert d["recall"].abs().max() < 1e-9
    for bench, tag in ((CIC, "Cic"), (UNSW, "Unsw")):
        q = d[(d["benchmark"] == bench) & (d["unit"] == "flow")]
        cm = q.groupby("cls")[["precision", "f1"]].mean()
        M(f"RevHtPrecMeanMax{tag}", f3(-cm["precision"].min()), src + "ht.csv, largest mean fall in precision")
        M(f"RevHtFoneMeanMax{tag}", f3(-cm["f1"].min()), src + "ht.csv, largest mean fall in F1")

    # ------------------------------------------------------------------ payload-aware labels (M1)
    chk = pd.read_csv(P11 / "payload" / "payload_check.csv")
    w = chk[chk["arm"] == "wtmc2021"].set_index("cls")["conversations_now_benign"]
    for c, tag in (("dos_slowhttptest", "Slowhttp"), ("dos_slowloris", "Slowloris"),
                   ("web_bruteforce", "WebBf"), ("web_xss", "Xss"), ("dos_goldeneye", "Golden"),
                   ("dos_hulk", "Hulk")):
        M(f"RevPayConv{tag}", n(w.get(c, 0)), "phase11/payload/payload_check.csv, WTMC-2021")
    f = pd.read_csv(AN / "payload_f1.csv")
    o = pd.read_csv(AN / "own_rq3_f1.csv")
    s = pd.read_csv(AN / "schedule_f1_check.csv")
    for df, tag in ((f, "Pay"), (o, "Own"), (s, "Sch")):
        x = df[df["arm"].isin(["nfs2023_nte", "nfs2023_te"])].pivot_table(
            index="cls", columns=["unit", "arm"], values="f1_mean")
        for c, ct in (("bot", "Bot"), ("ssh_patator", "Ssh")):
            M(f"Rev{tag}{ct}FlowNte", f3(x.loc[c, ("flow", "nfs2023_nte")]), src + f"{tag.lower()} RQ3 F1")
            M(f"Rev{tag}{ct}FlowTe", f3(x.loc[c, ("flow", "nfs2023_te")]), src + f"{tag.lower()} RQ3 F1")
        cl = [c for c in x.index if c not in RQ3_EXCLUDE]
        gap = (x[("flow", "nfs2023_te")] - x[("flow", "nfs2023_nte")]).reindex(cl)
        M(f"Rev{tag}MaxGapFlow", f3(gap.abs().max()), src + f"{tag.lower()} RQ3 F1, largest TE minus nTE")
    sch = s[s["arm"].isin(["nfs2023_nte", "nfs2023_te"])]
    assert abs(sch[(sch.cls == "bot") & (sch.unit == "flow") & (sch.arm == "nfs2023_te")]["f1_mean"].iloc[0]
               - pd.read_csv(ROOT / "04_Results" / "phase6" / "table_6_1.csv").set_index("cls").loc[
                   "bot", "f1_flow_grouped_te"]) < 1e-9, "schedule check does not reproduce RQ3"
    w4 = pd.read_csv(AN / "payload_h4.csv").iloc[0]
    for k, t in (("w_flow", "WFlow"), ("w_conversation", "WConv"), ("lo", "WDiffLo"), ("hi", "WDiffHi")):
        M(f"RevPay{t}", s3(w4[k]), src + "payload_h4.csv")
    sa = pd.read_csv(AN / "schedule_shares_check.csv")
    pa = pd.read_csv(AN / "payload_shares.csv")
    mm = sa.merge(pa, on=["benchmark", "unit", "cls"], suffixes=("_s", "_p"))
    mf = mm[(mm["unit"] == "flow") & (mm["cls"] != "benign")]
    M("RevPayShareFell", int((mf["arm_share_p"] < mf["arm_share_s"]).sum()), src + "payload_shares.csv")
    M("RevPayShareN", len(mf), src + "payload_shares.csv")
    mc = mm[(mm["unit"] == "conversation") & (mm["cls"] != "benign")]
    large = lambda sh, sp: int(((mc[sh] >= 0.25) & (mc[sp] >= 0.05)).sum())  # noqa: E731, the RQ4 rule
    M("RevSchLarge", word(large("arm_share_s", "arm_spread_s")), "schedule_shares_check.csv, RQ4 large-residual rule")
    M("RevPayLarge", word(large("arm_share_p", "arm_spread_p")), src + "payload_shares.csv, RQ4 large-residual rule")
    M("RevPaySpreadMax", f3(mc["arm_spread_p"].max()), src + "payload_shares.csv, conversation unit, attack classes")
    for c, ct in (("bot", "Bot"), ("ssh_patator", "Ssh")):
        r = mf.set_index("cls").loc[c]
        M(f"RevPayShare{ct}", f3(r["arm_share_p"]), src + "payload_shares.csv, flow")
        M(f"RevSchShare{ct}", f3(r["arm_share_s"]), src + "schedule_shares_check.csv, flow")

    # ------------------------------------------------------------------ exporter source traced (m8)
    ip = pd.read_csv(P11 / "nfs_idle_probe.csv")
    for arm, tag in (("nfs2023_nte", "Nte"), ("nfs2023_te", "Te")):
        q = ip[ip["arm"] == arm]
        M(f"RevIdle{tag}", n(len(q)), "phase11/nfs_idle_probe.csv")
        M(f"RevIdle{tag}Sixty", n((q["gap_s"] >= 60).sum()), "phase11/nfs_idle_probe.csv, gap of 60 s or more")
    q = ip[ip["arm"] == "nfs2023_nte"]
    M("RevIdleGapLoris", f"{q[q['cls'] == 'dos_slowloris']['gap_s'].median():.0f}", "phase11/nfs_idle_probe.csv")
    M("RevIdleGapHttp", f"{q[q['cls'] == 'dos_slowhttptest']['gap_s'].median():.0f}", "phase11/nfs_idle_probe.csv")
    sp = pd.read_csv(P11 / "split_probe.csv").set_index(["arm", "cls"])
    hk = sp.loc[("bccc2017", "dos_hulk")]
    M("RevHulkExtra", n(hk["extra_records"]), "phase11/split_probe.csv")
    M("RevHulkAfterRst", n(hk["prev_has_rst"]), "phase11/split_probe.csv")
    M("RevHulkAfterFin", n(hk["prev_fin_both"]), "phase11/split_probe.csv")
    M("RevHulkCarryRst", n(hk["cur_has_rst"]), "phase11/split_probe.csv")
    M("RevHulkNoPay", n(hk["cur_no_payload"]), "phase11/split_probe.csv")
    M("RevGoldenConvs", n(sp.loc[("bccc2017", "dos_goldeneye"), "conversations"]), "phase11/split_probe.csv")
    dd = sp.loc[("bccc2017", "ddos")]
    M("RevDdosExtra", n(dd["extra_records"]), "phase11/split_probe.csv")
    M("RevDdosAfterRst", n(dd["prev_has_rst"]), "phase11/split_probe.csv")
    M("RevDdosConvs", n(dd["conversations"]), "phase11/split_probe.csv")
    fb, fo = sp.loc[("bccc2017", "ftp_patator")], sp.loc[("cic_original", "ftp_patator")]
    M("RevFtpBcccExtra", n(fb["extra_records"]), "phase11/split_probe.csv")
    M("RevFtpBcccExtraRst", n(fb["cur_has_rst"]), "phase11/split_probe.csv")
    M("RevFtpBcccAfterRst", n(fb["prev_has_rst"]), "phase11/split_probe.csv")
    M("RevFtpOrigExtra", n(fo["extra_records"]), "phase11/split_probe.csv")
    M("RevFtpOrigSyn", n(fo["cur_has_syn"]), "phase11/split_probe.csv")
    M("RevFtpOrigNoPay", n(fo["cur_no_payload"]), "phase11/split_probe.csv")
    M("RevFtpOrigPk", f"{fo['cur_packets_median']:.0f}", "phase11/split_probe.csv")

    # ------------------------------------------------------------------ five models (M2), H4 power
    mw = AN / "models_w.csv"
    if mw.exists():
        w5 = pd.read_csv(mw).set_index(["benchmark", "models"])
        for bench, tag in ((CIC, "Cic"), (UNSW, "Unsw")):
            r = w5.loc[(bench, "five models")]
            M(f"RevFiveWFlow{tag}", f3(r["w_flow"]), src + "models_w.csv")
            M(f"RevFiveWConv{tag}", f3(r["w_conversation"]), src + "models_w.csv")
            M(f"RevFiveWDiffLo{tag}", s3(r["lo"]), src + "models_w.csv")
            M(f"RevFiveWDiffHi{tag}", s3(r["hi"]), src + "models_w.csv")
            M(f"RevFiveVerdict{tag}", r["verdict"], src + "models_w.csv")

    # ------------------------------------------------------------------ instrument without one arm (M4)
    lo_ = [P11 / "loo" / f"instrument_loo_{b}.csv" for b in (CIC, UNSW)]
    if all(p_.exists() for p_ in lo_):
        L = pd.concat([pd.read_csv(p_) for p_ in lo_])
        al = L[L["cls"] == "ALL"]
        c = al[al["benchmark"] == CIC]
        dr = c[c["variant"].str.startswith("drop:")]
        M("RevLooJaccMinCic", f3(dr["jaccard"].min()), "phase11/loo/instrument_loo_CICIDS2017.csv")
        pl = c[c["variant"] == "+cic_original"]
        M("RevLooJaccPlusOrig", f3(pl["jaccard"].iloc[0]), "phase11/loo/instrument_loo_CICIDS2017.csv")
        u_ = al[al["benchmark"] == UNSW]
        M("RevLooJaccMinUnsw", f3(u_["jaccard"].min()), "phase11/loo/instrument_loo_UNSW-NB15.csv")
        src_c = "phase11/loo/instrument_loo_CICIDS2017.csv"
        cc = L[L["benchmark"] == CIC].set_index(["variant", "cls"])["jaccard"]
        M("RevLooJaccOtherMinCic", f3(dr[dr["variant"] != "drop:bccc2017"]["jaccard"].min()), src_c +
          ", arms other than BCCC-CIC-IDS2017 left out")
        M("RevLooBenignDropBccc", f3(cc[("drop:bccc2017", "benign")]), src_c)
        M("RevLooSlowlorisDropBccc", f3(cc[("drop:bccc2017", "dos_slowloris")]), src_c)
        M("RevLooSlowhttpDropBccc", f3(cc[("drop:bccc2017", "dos_slowhttptest")]), src_c)
        M("RevLooHulkPlusOrig", f3(cc[("+cic_original", "dos_hulk")]), src_c)
        M("RevLooDdosPlusOrig", f3(cc[("+cic_original", "ddos")]), src_c)
        uu = L[(L["benchmark"] == UNSW) & ~L["cls"].isin(["ALL", "benign", "excluded"]) & (L["frozen"] >= 20)]
        M("RevLooAttackMinUnsw", f3(uu["jaccard"].min()), "phase11/loo/instrument_loo_UNSW-NB15.csv, "
          "attack classes with 20 or more conversations")
    ls = AN / "loo_scores.csv"
    if ls.exists():
        sc = pd.read_csv(ls)
        w = sc.groupby(["benchmark", "variant"])["w_conversation_own"].first()
        M("RevLooWMinCic", f3(w.loc[CIC].min()), src + "loo_scores.csv")
        M("RevLooWMaxCic", f3(w.loc[CIC].max()), src + "loo_scores.csv")
        base = sc[sc["variant"] == "frozen"].set_index(["benchmark", "arm", "cls"])["conv_f1"]
        oth = sc[sc["variant"] != "frozen"].set_index(["benchmark", "arm", "cls"])
        dlt = (oth["conv_f1"] - base.reindex(oth.index)).abs()
        M("RevLooFoneMaxChange", f3(dlt.max()), src + "loo_scores.csv, largest change in conversation F1")

    # ------------------------------------------------------------------ RQ5 decided contrasts (M6)
    v = pd.read_csv(ROOT / "04_Results" / "phase8" / "verdicts_sched.csv")["verdict"]
    und = v.str.contains("not estimable|zero by construction|not obtained")
    M("RevFiveDecided", str(int((~und).sum())), "phase8/verdicts_sched.csv, verdicts other than not "
      "estimable, zero by construction and data not obtained")
    M("RevFiveUndecided", word(int(und.sum())), "phase8/verdicts_sched.csv")
    M("RevFiveSurvivePct", pct((v == "survives").sum() / (~und).sum()), "phase8/verdicts_sched.csv, "
      "survives over decided")

    # ------------------------------------------------------------------ full data (M5)
    fu = AN / "full.csv"
    if fu.exists():
        fd = pd.read_csv(fu)
        fd = fd[(fd["population"] == "own") & (fd["unit"] == "flow") & (fd["cls"] != "benign")]
        pv = fd.pivot_table(index=["benchmark", "arm", "cls"], columns="data", values=["f1", "precision", "recall"])
        dlt = pv.xs("full", axis=1, level=1) - pv.xs("subsample", axis=1, level=1)
        sup = pv[("support", "subsample")] if ("support", "subsample") in pv.columns else None
        if sup is None:
            sp = fd[fd["data"] == "subsample"].set_index(["benchmark", "arm", "cls"])["support"]
            sup = sp.reindex(dlt.index)
        big = dlt[sup >= 20]
        M("RevFullPairs", str(len(big)), src + "full.csv, arm and attack class pairs with 20 or more test records")
        M("RevFullWithinFive", pct((big["f1"].abs() <= 0.05).mean()), src + "full.csv, share with flow F1 "
          "change of at most 0.05")
        M("RevFullFoneMedian", s3(big["f1"].median()), src + "full.csv, median change in flow F1")
        M("RevFullFoneMaxAbs", f3(big["f1"].abs().max()), src + "full.csv, flow, own, attack classes")
        M("RevFullPrecMaxFall", f3(-big["precision"].min()), src + "full.csv")
        fz = dlt.xs((UNSW, "fuzzers"), level=("benchmark", "cls"))
        M("RevFullFuzzRecallMin", f3(-fz["recall"].max()), src + "full.csv, UNSW-NB15 Fuzzers, smallest recall fall")
        M("RevFullFuzzRecallMax", f3(-fz["recall"].min()), src + "full.csv, UNSW-NB15 Fuzzers, largest recall fall")
    return outs
