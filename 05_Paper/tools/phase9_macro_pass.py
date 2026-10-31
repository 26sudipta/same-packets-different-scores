"""
Step 9.2, one pass: replace every number typed in 05_Paper/springer/manuscript.tex with a macro
from 05_Paper/tables/numbers.tex (written by 03_Code/make_numbers.py), a \\litnum{} for a number
quoted from a cited work, or a \\setnum{} or \\docnum{} for a design constant or a documented fact
that no result file holds; and replace the six pasted tables with \\input of the generated ones.

Each replacement must match exactly once, or the pass stops and writes nothing. Run once; the
git diff of the manuscript and 00_Plan/Phase9_Number_Audit.md (written here) are the record.

    python3 05_Paper/tools/phase9_macro_pass.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "05_Paper" / "springer" / "manuscript.tex"
TAB = ROOT / "05_Paper" / "tables"
AUDIT = ROOT / "00_Plan" / "Phase9_Number_Audit.md"

# (kind, old, new). kind: result, correction (the typed number was wrong or not regenerable),
# lit, setting, doc, table, preamble
R = []


def add(kind, old, new):
    R.append((kind, old, new))


# ---------------------------------------------------------------- preamble
add("preamble", r"""\newcommand{\pending}[1]{\textcolor{red}{[Pending: #1]}}""",
    r"""\newcommand{\pending}[1]{\textcolor{red}{[Pending: #1]}}
% Numbers (plan step 9.2). Every result number is a macro from numbers.tex, written by
% 03_Code/make_numbers.py; the comment on each definition names its source file.
% \litnum: a number quoted from a cited work. \setnum: a design constant of the protocol.
% \docnum: a fact recorded in a project note or the code, not in a result file.
% \VERIFYNUM: a number not yet traced; build.sh fails while any is left.
\input{../tables/numbers.tex}
\newcommand{\litnum}[1]{#1}
\newcommand{\setnum}[1]{#1}
\newcommand{\docnum}[1]{#1}
\newcommand{\VERIFYNUM}[1]{\textcolor{magenta}{[Verify: #1]}}
\graphicspath{{../figures/}}""")
add("preamble", r"""% the CSV files named in the comments, and are replaced by macros from
% 03_Code/make_numbers.py at the freeze (plan step 9.2).""",
    r"""% the CSV files named in the comments through macros from 03_Code/make_numbers.py
% (plan step 9.2); 05_Paper/build.sh checks that no number is typed.""")

# ---------------------------------------------------------------- abstract
add("result", "(pairwise Jaccard index 0.997 to 1.000 on CICIDS2017)",
    r"(pairwise Jaccard index \RQoneConvJaccardMin{} to \RQoneConvJaccardMax{} on CICIDS2017)")
add("result", "but not on how they are cut into flows (0.649 to 0.999), and the share of test conversations that leak into training under a random flow split ranges from 0.07 to 0.53.",
    r"but not on how they are cut into flows (\RQoneFlowJaccardMin{} to \RQoneFlowJaccardMax{}), and on CICIDS2017 the share of test conversations that leak into training under a random flow split ranges from \RQoneLeakCicMin{} to \RQoneLeakCicMax{} between releases.")
add("result", "than the model for six of seven small classes; on UNSW-NB15 it does so for two of six.",
    r"than the model for \RQtwoExporterAboveCic{} of \RQtwoEligibleCic{} small classes; on UNSW-NB15 it does so for \RQtwoExporterAboveUnsw{} of \RQtwoEligibleUnsw{}.")
add("result", "lowers flow-level F1 for Bot from 0.998 to 0.666, while conversation-level F1 stays at 0.999.",
    r"lowers flow-level F1 for Bot from \RQthreeFlowBotNte{} to \RQthreeFlowBotTe{}, while conversation-level F1 stays at \RQthreeConvBotTe{}.")

# ---------------------------------------------------------------- introduction
add("lit", "A systematic review of 51 flow collection studies", r"A systematic review of \litnum{51} flow collection studies")
add("lit", "report a macro-F1 spread of up to 8.77 percent between timeout settings",
    r"report a macro-F1 spread of up to \litnum{8.77} percent between timeout settings")
add("result", "checked by hand on 40 cases", r"checked by hand on \NhandCases{} cases")
add("result", "and 480,630 exact duplicate lines in the original UNSW-NB15 CSV files",
    r"and \RQoneUnswDups{} exact duplicate lines in the original UNSW-NB15 CSV files")
add("result", "seed shares over 330 trained cells", r"seed shares over \RQtwoCells{} trained cells")

# ---------------------------------------------------------------- related work
add("lit", "settings differ by up to 8.77 percent in macro-F1", r"settings differ by up to \litnum{8.77} percent in macro-F1")
add("lit", "later reviewed 276 studies", r"later reviewed \litnum{276} studies")
add("lit", "after their corrections more than 20 percent of the traffic traces", r"after their corrections more than \litnum{20} percent of the traffic traces")
add("lit", "lowers recall by about 20 percent on average", r"lowers recall by about \litnum{20} percent on average")
add("lit", "their HERA version has 1,381,043 flows against 1,668,530 in the original",
    r"their HERA version has \litnum{1{,}381{,}043} flows against \litnum{1{,}668{,}530} in the original")
add("lit", "report F1 above 0.997", r"report F1 above \litnum{0.997}")
add("lit", "fine-grained macro-F1 of 0.955", r"fine-grained macro-F1 of \litnum{0.955}")
add("lit", "yet they hold 6,972 and 10,943 attack flows", r"yet they hold \litnum{6{,}972} and \litnum{10{,}943} attack flows")
add("lit", "share 32.5 percent of their records", r"share \litnum{32.5} percent of their records")

# ---------------------------------------------------------------- problem statement
add("setting", "which is $0$ for $n = 1$ and, with $q = 0.3$, about $0.82$ for $n = 2$.",
    r"which is $0$ for $n = 1$ and, with $q = \SetTestFrac$, about $\SetLeakTwo$ for $n = 2$.")

# ---------------------------------------------------------------- data
add("result", "have the same row count, 2,830,743, but neither)", r"have the same row count, \NcicRows{}, but neither)")
add("result", "All 22 checks matched exactly, including the per-day flow and attack counts of five releases in Table 2 of Pekar and Jozsa",
    r"\NcountChecksMatched{} \NcountChecks{} checks matched exactly, including the per-day flow and attack counts of five releases in Table~\litnum{2} of Pekar and Jozsa")

# ---------------------------------------------------------------- method: clocks, key, sweep
add("result", "one hour behind the capture's local time, UTC minus 4;", r"one hour behind the capture's local time, UTC minus \NbcccOffset{};")
add("setting", "tuned on seed 0 of the grouped split", r"tuned on seed \setnum{0} of the grouped split")
add("correction", "with that offset its FTP-Patator window matches LYCOS to the microsecond on four windows.",
    r"with that offset, the first attack record in BCCC starts at the same microsecond as in LYCOS in \NbcccWindowsSame{} of the \NbcccWindows{} attack windows, and later in the other two.")
add("result", "is spread evenly over minus 0.5 to plus 0.5 seconds for 99.97 and 99.998 percent of paired flows on the two capture days. Treating them as truncated left about 45 percent of records unpaired on one day.",
    r"is spread evenly over minus \setnum{0.5} to plus \setnum{0.5} seconds for \NunswOffsetShareA{} and \NunswOffsetShareB{} percent of paired flows on the two capture days. Read as truncated, \NunswTruncOutside{} percent of the paired flows on each day would fall outside their one-second window.")
add("correction", "because Argus writes protocol names, 36 percent of which on UNSW-NB15 have no number that the system table resolves, while nProbe writes numbers.",
    r"because Argus writes protocol names and \NprotoUnresolved{} percent of the rows of the UNSW-NB15 ground truth file name a protocol that the system table cannot resolve to a number, while nProbe writes numbers.")
add("correction", "At most 69 of its records per day reach none, and between 0 and 2,234 per day reach more than one candidate, out of about 460,000 to 700,000.",
    r"At most \NnondefUnmappedMax{} of its records per day reach none, and between \NnondefAmbigMin{} and \NnondefAmbigMax{} per day reach more than one candidate, out of \NcicDayRecordsMin{} to \NcicDayRecordsMax{} records per day.")
add("correction", "We chose $T = 15$ s from a sweep over 5, 15, 30, 60, 120 and 300 seconds on every day. Every scored class is stable from 5 to 30 seconds. Above that, two attack tools reuse endpoint keys: Friday DDoS falls from 93,178 conversations at 30 seconds or less to 92,980 at 60 seconds and 63,540 at 120 seconds, and Wednesday DoS Hulk falls from 158,517 to 14,375 at 120 seconds. Benign counts move by well under one percent across the sweep. Fifteen seconds sits inside the stable range and is longer than every clock resolution except the minute clock, which does not define conversations.",
    r"We chose $T = \SetT$ s from a sweep over \SetSweep{} seconds on every day. On \cicids{} every scored class has the same number of conversations at every value from \NsweepCicStableMin{} to \NsweepCicStableMax{} seconds. Above that, two attack tools reuse endpoint keys: Friday DDoS falls from \NsweepDdosLow{} conversations at \NsweepCicStableMax{} seconds or less to \NsweepDdosSixty{} at \setnum{60} seconds and \NsweepDdosOneTwenty{} at \setnum{120} seconds, and Wednesday DoS Hulk falls from \NsweepHulkLow{} to \NsweepHulkOneTwenty{} at \setnum{120} seconds. On UNSW-NB15 no scored class changes by more than \NsweepUnswUpMax{} percent between \SetT{} and \NsweepCicStableMax{} seconds, but Generic is sensitive at the short end: at \NsweepCicStableMin{} seconds it has \NsweepGenericFivePct{} percent more conversations than at \SetT{} (\NsweepGenericFive{} against \NsweepGenericFifteen{} on one day). Benign counts move by at most \NsweepBenignMaxChange{} percent across the sweep. Fifteen seconds sits inside the stable range of \cicids{} and above the short end where UNSW-NB15 Generic splits, and is longer than every clock resolution except the minute clock, which does not define conversations.")

# ---------------------------------------------------------------- method: ground truth, hand check
add("result", "This removes 239,079 conversations on Thursday and 1,468 on Friday.",
    r"This removes \NexclThursday{} conversations on Thursday and \NexclFriday{} on Friday.")
add("result", "agree with the schedule on 0.99 or more of the conversations of every class. CRiSIS-2022 and both NFS-2023 releases agree on 0.32 of slowhttptest and 0.57 of slowloris conversations, and on 0.11 of web brute force and 0.03 of XSS conversations.",
    r"agree with the schedule on \NlabelGoodMin{} or more of the conversations of every class. CRiSIS-2022 and both NFS-2023 releases agree on \NlabelSlowhttp{} of slowhttptest and \NlabelSlowloris{} of slowloris conversations, and on \NlabelWebBf{} of web brute force and \NlabelXss{} of XSS conversations.")
add("result", "We hand checked 40 conversations, 20 per benchmark,", r"We hand checked \NhandCases{} conversations, \NhandPerBench{} per benchmark,")
add("result", "and the 40 cases are frozen as test fixtures", r"and the \NhandCases{} cases are frozen as test fixtures")

# ---------------------------------------------------------------- method: agreement, splits, protocol
add("setting", "Intervals are 95 percent percentile intervals from 1,000 bootstrap resamples of conversations.",
    r"Intervals are \setnum{95} percent percentile intervals from \SetBoot{} bootstrap resamples of conversations.")
add("setting", "a tail (at most 10 ms long and starting within one second", r"a tail (at most \SetTailMs{} ms long and starting within one second")
add("setting", "puts it in the lowest 30 percent of its class on that day, for seeds 0 to 4.",
    r"puts it in the lowest \SetTestPct{} percent of its class on that day, for seeds \SetSeeds{}.")
add("setting", "Seed: 0 to 4, selecting the split", r"Seed: \SetSeeds{}, selecting the split")
add("setting", "Each model has one hyperparameter grid of four settings", r"Each model has one hyperparameter grid of \SetGridSize{} settings")
add("setting", "by validation macro-F1 on 20 percent of the training conversations", r"by validation macro-F1 on \SetValPct{} percent of the training conversations")
add("setting", "LightGBM uses an L2 penalty of 1 on leaf values", r"LightGBM uses an L2 penalty of \setnum{1} on leaf values")
add("setting", "0.2 of benign conversations, and at most 20,000 conversations per attack class",
    r"\SetBenignFraction{} of benign conversations, and at most \SetAttackCap{} conversations per attack class")
add("setting", "macro-F1 over a fixed set of classes with at least 20 test conversations in every grouped seed",
    r"macro-F1 over a fixed set of classes with at least \SetMinTest{} test conversations in every grouped seed")
add("setting", "Intervals are 95 percent percentile intervals from a cluster bootstrap over conversations \\cite{field2007} with 1,000 resamples,",
    r"Intervals are \setnum{95} percent percentile intervals from a cluster bootstrap over conversations \cite{field2007} with \SetBoot{} resamples,")
add("setting", "A class counts as small when it has fewer than 5,000 scored conversations; it is eligible for H2 when it has at least 20 test conversations in every grouped seed.",
    r"A class counts as small when it has fewer than \setnum{5{,}000} scored conversations; it is eligible for H2 when it has at least \SetMinTest{} test conversations in every grouped seed.")

# ---------------------------------------------------------------- RQ1
add("result", "On \\cicids, flow counts differ between the seven releases by a factor of 1.54, from 1,837,498 records (LYCOS) to 2,826,174 (NFS-2023-TE) over the scored days, while conversation counts differ by less than 0.2 percent. Over all 21 pairs of releases the conversation Jaccard index lies between 0.997 and 1.000, and the flow Jaccard index between 0.649 and 0.999.",
    r"On \cicids, flow counts differ between the seven releases by a factor of \RQoneFlowRatio{}, from \RQoneFlowsMin{} records (\RQoneFlowsMinArm{}) to \RQoneFlowsMax{} (\RQoneFlowsMaxArm{}) over the scored days, while conversation counts differ by \RQoneConvSpread{} percent. Over all \RQonePairs{} pairs of releases the conversation Jaccard index lies between \RQoneConvJaccardMin{} and \RQoneConvJaccardMax{}, and the flow Jaccard index between \RQoneFlowJaccardMin{} and \RQoneFlowJaccardMax{}.")
add("result", "PortScan stays at 1.00 to 1.01 flows per conversation", r"PortScan stays at \RQonePortscanFpc{} flows per conversation")
add("result", "FTP-Patator is written as one record per conversation by four releases and as 2.00 to 2.39 records by the other three",
    r"FTP-Patator is written as one record per conversation by \RQoneFtpOneArms{} releases and as \RQoneFtpSplitRange{} records by the other \RQoneFtpSplitArms{}")
add("correction", "DoS Hulk separates the releases differently again: 1.00 in five of them, 2.21 in BCCC and 2.37 in NFS-2023-TE.",
    r"DoS Hulk separates the releases differently again: one record per conversation in \RQoneHulkOneArms{} of them, \RQoneHulkOrig{} in CIC original, \RQoneHulkBccc{} in BCCC and \RQoneHulkTe{} in NFS-2023-TE.")
add("result", "carries 1.61 flows per conversation in BCCC, 1.82 in WTMC-2021, CRiSIS-2022 and LYCOS, and 2.69 in NFS-2023-TE.",
    r"carries \RQoneMacroBccc{} flows per conversation in BCCC, \RQoneMacroCorrected{} in WTMC-2021, CRiSIS-2022 and LYCOS, and \RQoneMacroTe{} in NFS-2023-TE.")
add("result", "a single connection written as 11 or 12 records", r"a single connection written as \RQoneHeartbleedRecords{} records")
add("correction", "In NFS-2023-TE, 99 percent of the extra FTP-Patator records are tails of at most 10 ms",
    r"In NFS-2023-TE, \RQoneTailFtpTe{} percent of the extra FTP-Patator records are tails of at most \SetTailMs{} ms")
add("result", "and in BCCC 98 to 100 percent of the extra records", r"and in BCCC \RQoneTailBccc{} percent of the extra records")
add("result", "are cut by the 120 second active timeout: 99 to 100 percent of the extra slowloris records",
    r"are cut by the \NactiveTimeout{} second active timeout: \RQoneCutSlowloris{} percent of the extra slowloris records")
add("result", "the single Heartbleed connection becomes 11 records", r"the single Heartbleed connection becomes \RQoneHeartbleedCut{} records")
add("result", "(its longest record lasts 30,155 seconds)", r"(its longest record lasts \RQoneBcccLongest{} seconds)")
add("correction", "more often than the 120 second timeout predicts, mostly by idle gaps of 1 to 15 seconds.",
    r"more often than the \NactiveTimeout{} second timeout predicts: \RQoneIdleNfs{} percent of their extra records in these two classes are idle splits, which start more than one second after the previous record ends and, since they stay in the conversation, at most $T$ seconds plus the clock resolution after it.")
add("result", "BCCC splits DoS Hulk (2.21) but not DDoS (1.03) or GoldenEye (1.00)",
    r"BCCC splits DoS Hulk (\RQoneHulkBccc{}) but not DDoS (\RQoneDdosBccc{}) or GoldenEye (\RQoneGoldenBccc{})")
add("result", "And 37 percent of FTP-Patator conversations carry three records in CIC original, 39 percent in BCCC",
    r"And \RQoneFtpThreeOrig{} percent of FTP-Patator conversations carry three records in CIC original, \RQoneFtpThreeBccc{} percent in BCCC")
add("result", "agrees one to one on 0.635 of conversations and is many to one on 0.314.",
    r"agrees one to one on \RQoneOrigWtmcOneOne{} of conversations and is many to one on \RQoneOrigWtmcManyOne{}.")
add("correction", "is 0.44 for CIC original, 0.53 for NFS-2023-TE and 0.39 for BCCC, the three releases that write tail records, against 0.07 to 0.09 for WTMC-2021, CRiSIS-2022, LYCOS and NFS-2023-nTE. On UNSW-NB15 it is 0.04 and 0.08.",
    r"is \RQoneLeakOrig{} for CIC original, \RQoneLeakTe{} for NFS-2023-TE and \RQoneLeakBccc{} for BCCC, the three releases that write tail records, against \RQoneLeakLow{} for WTMC-2021, CRiSIS-2022, LYCOS and NFS-2023-nTE. On UNSW-NB15 it is \RQoneLeakUnswOrig{} for the original and \RQoneLeakUnswVthree{} for NF-UNSW-NB15-v3.")
add("result", "On \\cicids{} Krippendorff's alpha is 0.988 (interval 0.988 to 0.988) over 1,718,246 conversations, with observed agreement 0.995. By day it is 0.999 on Tuesday, 0.983 on Wednesday and 0.989 on Friday, but 0.470 on Thursday,",
    r"On \cicids{} Krippendorff's alpha is \RQoneAlphaCic{} (interval \RQoneAlphaCicLo{} to \RQoneAlphaCicHi{}) over \RQoneAlphaCicUnits{} conversations, with observed agreement \RQoneAlphaCicObs{}. By day it is \RQoneAlphaTuesday{} on Tuesday, \RQoneAlphaWednesday{} on Wednesday and \RQoneAlphaFriday{} on Friday, but \RQoneAlphaThursday{} on Thursday,")
add("result", "On UNSW-NB15 alpha is 0.768 (0.766 to 0.771) over 1,841,078 conversations, with observed agreement 0.980. NF-UNSW-NB15-v3 marks 37,579 records on one day and 4,445 on the other",
    r"On UNSW-NB15 alpha is \RQoneAlphaUnsw{} (\RQoneAlphaUnswLo{} to \RQoneAlphaUnswHi{}) over \RQoneAlphaUnswUnits{} conversations, with observed agreement \RQoneAlphaUnswObs{}. NF-UNSW-NB15-v3 marks \RQoneVthreeExtraA{} records on one day and \RQoneVthreeExtraB{} on the other")
add("result", "has 3.74 times as many attack flows as NF-UNSW-NB15-v3 (321,311 against 86,008), while attack conversations differ by a factor of 1.06.",
    r"has \RQoneUnswAttackRatio{} times as many attack flows as NF-UNSW-NB15-v3 (\RQoneUnswAttackOrig{} against \RQoneUnswAttackVthree{}), while attack conversations differ by a factor of \RQoneUnswAttackConvRatio{}.")
add("result", "The original CSV files hold 480,630 surplus lines that are identical to another line in all 49 fields (64,006 on the first capture day and 416,624 on the second)",
    r"The original CSV files hold \RQoneUnswDups{} surplus lines that are identical to another line in all \docnum{49} fields (\RQoneUnswDupsA{} on the first capture day and \RQoneUnswDupsB{} on the second)")
add("result", "the attack flow ratio falls to 1.16 and the attack flow Jaccard index rises from 0.239 to 0.862.",
    r"the attack flow ratio falls to \RQoneUnswAttackRatioDedup{} and the attack flow Jaccard index rises from \RQoneUnswAttackJaccard{} to \RQoneUnswAttackJaccardDedup{}.")
add("correction", "A further 24,016 pairs of records on the second day still share start time and duration after exact duplicates are removed; in a sample of 3,000 of them, 1,450 differ only in the attack category and 1,196 only in the windowed count features.",
    r"After exact duplicates are removed, a further \RQoneNearB{} records on the second day (\RQoneNearA{} on the first) repeat the five-tuple, start, end and byte count of an earlier record; of the first \RQoneNearSample{} of them on the second day, \RQoneNearAttackCat{} differ from that earlier record only in the attack category and \RQoneNearCt{} only in the windowed count features.")
add("result", "by a factor of more than seven between releases", r"by a factor of more than \RQoneLeakFactor{} between releases")

# ---------------------------------------------------------------- RQ2
add("result", "The grid has 330 cells:", r"The grid has \RQtwoCells{} cells:")
add("result", "Three cells rerun from scratch gave predictions identical row for row, and three cells drawn at random were recounted",
    r"Rerun from scratch, \RQtwoReproCells{} cells gave predictions identical row for row, and \RQtwoSanityCells{} cells drawn at random were recounted")
add("result", "ranges from 0.759 to 0.935 across arms for the random forest, from 0.777 to 0.928 for LightGBM and from 0.704 to 0.853 for the multilayer perceptron.",
    r"ranges from \RQtwoMacroRfMin{} to \RQtwoMacroRfMax{} across arms for the random forest, from \RQtwoMacroLgbmMin{} to \RQtwoMacroLgbmMax{} for LightGBM and from \RQtwoMacroMlpMin{} to \RQtwoMacroMlpMax{} for the multilayer perceptron.")
add("result", "(0.693 to 0.782 against 0.653 to 0.757)",
    r"(\RQtwoUnswVthreeMin{} to \RQtwoUnswVthreeMax{} against \RQtwoUnswDedupMin{} to \RQtwoUnswDedupMax{})")
add("result", "the model for six of the seven eligible classes, and every one of the six intervals excludes zero. The largest gaps are SSH-Patator (exporter share 0.970, model share 0.006) and Bot (0.841 against 0.038). Web XSS goes the other way (0.214 against 0.639). Web SQL injection, with 4 test conversations in its smallest seed,",
    r"the model for \RQtwoExporterAboveCic{} of the \RQtwoEligibleCic{} eligible classes, and every one of the \RQtwoExporterAboveCic{} intervals excludes zero. The largest gaps are SSH-Patator (exporter share \RQtwoSshArm{}, model share \RQtwoSshModel{}) and Bot (\RQtwoBotArm{} against \RQtwoBotModel{}). Web XSS goes the other way (\RQtwoXssArm{} against \RQtwoXssModel{}). Web SQL injection, with \RQtwoSqliTest{} test conversations in its smallest seed,")
add("result", "On UNSW-NB15 the exporter leads for two of six eligible classes, Shellcode and Analysis; the model leads for Generic, DoS and Backdoor, and Worms is undecided (interval of the difference $-0.160$ to $0.191$).",
    r"On UNSW-NB15 the exporter leads for \RQtwoExporterAboveUnsw{} of \RQtwoEligibleUnsw{} eligible classes, Shellcode and Analysis; the model leads for Generic, DoS and Backdoor, and Worms is undecided (interval of the difference \RQtwoWormsLo{} to \RQtwoWormsHi{}).")
add("result", "At the conversation unit the exporter leads for five of the seven eligible classes at both populations; DoS Slowhttptest turns to the model (difference $-0.254$, interval $-0.402$ to $-0.006$). Under the random flow split the exporter leads for five of seven,",
    r"At the conversation unit the exporter leads for \RQtwoExporterAboveConvOwn{} of the \RQtwoEligibleCic{} eligible classes on the own population and \RQtwoExporterAboveConvCommon{} on the common one; DoS Slowhttptest turns to the model (difference \RQtwoSlowhttpConv{}, interval \RQtwoSlowhttpConvLo{} to \RQtwoSlowhttpConvHi{}). Under the random flow split the exporter leads for \RQtwoExporterAboveRandomOwn{} of \RQtwoEligibleCic{},")
add("correction", "with the release as published, the exporter leads on point estimates for Shellcode, Analysis and Generic and the model for DoS and Backdoor, which is still no majority; with near repeats collapsed, the exporter leads for Shellcode and Analysis as in the primary set.",
    r"with the release as published, the exporter leads by the same interval rule for \RQtwoUnswReleasedLeadN{} of \RQtwoEligibleUnsw{} classes (\RQtwoUnswReleasedLead{}), the model for \RQtwoUnswReleasedModel{}, and \RQtwoUnswReleasedUndecided{} is undecided, which is still no majority; with near repeats collapsed, the exporter leads for \RQtwoUnswCollapsedLead{} as in the primary set, the model for \RQtwoUnswCollapsedModel{}, and \RQtwoUnswCollapsedUndecided{} is undecided.")
add("result", "(0.609 for Web brute force, 0.431 for FTP-Patator, 0.421 for DoS Slowloris)",
    r"(\RQtwoInterWebBf{} for Web brute force, \RQtwoInterFtp{} for FTP-Patator, \RQtwoInterSlowloris{} for DoS Slowloris)")
add("result", "(at most 0.096, for Web SQL injection) but reaches 0.396 for UNSW-NB15 Generic and 0.434 for its benign class",
    r"(at most \RQtwoSeedCicMax{}, for \RQtwoSeedCicMaxClass{}) but reaches \RQtwoSeedUnswGeneric{} for UNSW-NB15 Generic and \RQtwoSeedUnswBenign{} for its benign class")
add("result", "The own minus common difference is within 0.001 of zero for 10 of the 13 scored classes.",
    r"The own minus common difference is within \setnum{0.001} of zero for \RQtwoPopNearZero{} of the \RQtwoPopScored{} scored classes.")
add("result", "PortScan $-0.123$ (interval $-0.181$ to $-0.053$) and DoS Slowloris $-0.036$ ($-0.048$ to $-0.025$). DDoS ($-0.014$) has an interval that includes zero.",
    r"PortScan \RQtwoPopPortscan{} (interval \RQtwoPopPortscanLo{} to \RQtwoPopPortscanHi{}) and DoS Slowloris \RQtwoPopSlowloris{} (\RQtwoPopSlowlorisLo{} to \RQtwoPopSlowlorisHi{}). DDoS (\RQtwoPopDdos{}) has an interval that includes zero.")
add("result", "benign $-0.378$ ($-0.626$ to $-0.076$), Fuzzers $-0.362$ ($-0.394$ to $-0.282$) and Reconnaissance $-0.114$ ($-0.160$ to $-0.051$). The one positive difference with an interval above zero is small (Generic, 0.008).",
    r"benign \RQtwoPopUnswBenign{} (\RQtwoPopUnswBenignLo{} to \RQtwoPopUnswBenignHi{}), Fuzzers \RQtwoPopUnswFuzzers{} (\RQtwoPopUnswFuzzersLo{} to \RQtwoPopUnswFuzzersHi{}) and Reconnaissance \RQtwoPopUnswRecon{} (\RQtwoPopUnswReconLo{} to \RQtwoPopUnswReconHi{}). The one positive difference with an interval above zero is small (Generic, \RQtwoPopUnswGeneric{}).")
add("result", "is 0.886 (interval 0.780 to 0.919) at the flow unit and 0.820 (0.763 to 0.869) at the conversation unit, grouped split. The random forest or LightGBM ranks first and the multilayer perceptron last in most arms. On UNSW-NB15 there are only two primary arms and $W$ is 1.000 at every unit,",
    r"is \RQtwoWFlow{} (interval \RQtwoWFlowLo{} to \RQtwoWFlowHi{}) at the flow unit and \RQtwoWConv{} (\RQtwoWConvLo{} to \RQtwoWConvHi{}) at the conversation unit, grouped split. By mean macro-F1 the random forest or LightGBM ranks first and the multilayer perceptron last in \RQtwoMlpLastArms{} arm. On UNSW-NB15 there are only two primary arms and $W$ is \RQtwoWUnsw{} at every unit,")
add("result", "NFS-2023-TE scores near 0.667 F1 for SSH-Patator and Bot with every model. Without that arm, point estimates still put the exporter above the model for six of the seven eligible \\cicids{} classes, with smaller margins: the exporter share falls from 0.970 to 0.294 for SSH-Patator (model 0.153) and from 0.841 to 0.524 for Bot (model 0.132).",
    r"NFS-2023-TE scores between \RQtwoTeSmallMin{} and \RQtwoTeSmallMax{} F1 for SSH-Patator and Bot with every model. Without that arm, point estimates still put the exporter above the model for \RQtwoLoaoLead{} of the \RQtwoEligibleCic{} eligible \cicids{} classes, with smaller margins: the exporter share falls from \RQtwoSshArm{} to \RQtwoLoaoSshArm{} for SSH-Patator (model \RQtwoLoaoSshModel{}) and from \RQtwoBotArm{} to \RQtwoLoaoBotArm{} for Bot (model \RQtwoLoaoBotModel{}).")

# ---------------------------------------------------------------- RQ3
add("result", "covers the eleven attack classes with at least 20 common conversations in both arms",
    r"covers the \RQthreeClasses{} attack classes with at least \SetMinTest{} common conversations in both arms")
add("result", "With the rule on, eight of the eleven attack classes go from about 1.0 record per conversation to between 2.0 and 2.4.",
    r"With the rule on, \RQthreeDoubled{} of the \RQthreeClasses{} attack classes go from about one record per conversation to between \RQthreeDoubledMin{} and \RQthreeDoubledMax{}.")
add("result", "(share 1.00, 0.997, 1.00, 1.00 and 1.00)", r"(share \RQthreeFirstFin{})")
add("result", "a single forward packet of 60 to 66 bytes", r"a single forward packet of \RQthreeTailBytes{} bytes")
add("result", "(0.958 of Bot, 0.981 of SSH-Patator, 0.974 of Web brute force, 0.952 of Web XSS and 0.737 of GoldenEye conversations)",
    r"(\RQthreeSecondIdle{} conversations)")
add("result", "(later records with a median of 1 to 2 packets and 66 to 120 bytes)",
    r"(later records with a median of \RQthreeBenignTailPackets{} packets and \RQthreeBenignTailBytes{} bytes)")
add("result", "(grouped recall 0.999 for Bot, 0.998 for SSH-Patator, 0.891 for Web brute force) and almost never the second (0.000, 0.014 and 0.000; 0.263 for GoldenEye)",
    r"(grouped recall \RQthreeRecallFirstBot{} for Bot, \RQthreeRecallFirstSsh{} for SSH-Patator, \RQthreeRecallFirstWebBf{} for Web brute force) and almost never the second (\RQthreeRecallSecondBot{}, \RQthreeRecallSecondSsh{} and \RQthreeRecallSecondWebBf{}; \RQthreeRecallSecondGolden{} for GoldenEye)")
add("result", "Flow F1 on the grouped split falls from 0.998 to 0.666 for Bot and from 0.996 to 0.667 for SSH-Patator, while conversation F1 stays at 0.999 and 0.995.",
    r"Flow F1 on the grouped split falls from \RQthreeFlowBotNte{} to \RQthreeFlowBotTe{} for Bot and from \RQthreeFlowSshNte{} to \RQthreeFlowSshTe{} for SSH-Patator, while conversation F1 stays at \RQthreeConvBotTe{} and \RQthreeConvSshTe{}.")
add("result", "It is, for 10 of 11 classes. DoS Hulk is the exception (gap 0.066 at the flow unit and 0.128 at the conversation unit). TE produces about 1,748 benign test records per cell that models label as DoS Hulk, against about one in nTE,",
    r"It is, for \RQthreeSmallerAtConv{} of the \RQthreeClasses{} classes. DoS Hulk is the exception (gap \RQthreeHulkGapFlow{} at the flow unit and \RQthreeHulkGapConv{} at the conversation unit). TE produces about \RQthreeHulkFalseTe{} benign test records per cell that models label as DoS Hulk, against about \RQthreeHulkFalseNte{} in nTE,")
add("correction", "lifts leakage under the random flow split from near zero to about 0.70 for the classes it splits (Table~\\ref{tab:rq3}). That is what a random split with test share $q = 0.3$ gives when every conversation has two records: a test record's sibling lands in training with probability $1 - q = 0.7$. Yet the random split gains at most 0.006 F1 over the grouped split in either arm: across the eleven classes the change in leakage ranges from 0.003 to 0.798, and the change in this inflation from $-0.002$ to $0.009$.",
    r"lifts leakage under the random flow split from near zero to between \RQthreeLeakTeMin{} and \RQthreeLeakTeMax{} for the classes it doubles (Table~\ref{tab:rq3}). That is about what a random split with test share $q = \SetTestFrac$ gives when every conversation has two records: a test record's sibling lands in training with probability $1 - q = \SetTrainFrac$. Yet the random split gains at most \RQthreeInflMax{} F1 over the grouped split in either arm: across the \RQthreeClasses{} classes the change in leakage ranges from \RQthreeDLeakMin{} to \RQthreeDLeakMax{}, and the change in this inflation from \RQthreeDInflMin{} to \RQthreeDInflMax{}.")
add("result", "(Spearman $\\rho = 0.50$, one-sided permutation $p = 0.061$)",
    r"(Spearman $\rho = \RQthreeRhoPooled$, one-sided permutation $p = \RQthreePPooled$)")
add("result", "Per model, $\\rho$ is 0.51 for the random forest ($p = 0.058$), 0.11 for LightGBM ($p = 0.38$) and 0.67 for the multilayer perceptron ($p = 0.014$); on each arm's own population the pooled value is 0.53 ($p = 0.050$).",
    r"Per model, $\rho$ is \RQthreeRhoRf{} for the random forest ($p = \RQthreePRf$), \RQthreeRhoLgbm{} for LightGBM ($p = \RQthreePLgbm$) and \RQthreeRhoMlp{} for the multilayer perceptron ($p = \RQthreePMlp$); on each arm's own population the pooled value is \RQthreeRhoOwn{} ($p = \RQthreePOwn$).")
add("result", "come back to within 0.05 F1 of their nTE values", r"come back to within \RQthreeConvGapMaxOther{} F1 of their nTE values")


def tables(t):
    """Swap each pasted table for an \\input of the generated file, after checking that the two
    hold the same numbers in the same order."""
    num = re.compile(r"-?\d[\d{},]*\.?\d*")

    def block(txt, label):
        i = txt.index(r"\label{" + label + "}")
        a = txt.rfind(r"\begin{table", 0, i)
        b = txt.index(r"\end{table", i)
        b = txt.index("}", txt.index("{", b + 1)) + 1
        return a, b

    def nums(x):
        x = x.split(r"\toprule", 1)[1]
        return [v.replace("{,}", "").replace(",", "") for v in num.findall(x.replace("$-", "-"))]

    for f, lab in (("tab_arms", "tab:arms"), ("rq1_agreement", "tab:rq1-agreement"),
                   ("rq1_fpc_class", "tab:rq1-fpc"), ("rq1_alpha", "tab:rq1-alpha"),
                   ("rq2_shares", "tab:rq2-shares"), ("rq3_table", "tab:rq3")):
        g = (TAB / f"{f}.tex").read_text()
        a, b = block(t, lab)
        ga, gb = block(g, lab)
        same = nums(t[a:b]) == nums(g[ga:gb])
        assert same, f
        R.append(("table", t[a:b][:60] + " ...", rf"\input{{../tables/{f}.tex}}"))
        t = t[:a] + rf"\input{{../tables/{f}.tex}}" + t[b:]
    return t


def main():
    t = TEX.read_text()
    if r"\input{../tables/numbers.tex}" in t:
        sys.exit("already converted")
    for kind, old, new in R:
        c = t.count(old)
        if c != 1:
            sys.exit(f"{c} matches for: {old[:80]}")
    for kind, old, new in R:
        t = t.replace(old, new)
    t = t.replace("%% Generated by 03_Code/make_numbers.py from 04_Results/phase4 (rq1_agreement.tex), pasted\n"
                  "%% inline because the template asks for a single .tex file.\n", "")
    t = tables(t)
    TEX.write_text(t)
    kinds = {}
    for k, _, _ in R:
        kinds[k] = kinds.get(k, 0) + 1
    lines = ["# Phase 9, step 9.2: number audit of the manuscript", "",
             "Written by `05_Paper/tools/phase9_macro_pass.py` on the one pass that replaced every typed",
             "number in `05_Paper/springer/manuscript.tex`. Kinds: result (now a macro from",
             "`05_Paper/tables/numbers.tex`), correction (the typed text was wrong or could not be",
             "regenerated; the new text is a macro), lit (a number from a cited work, `\\litnum`), setting",
             "(a design constant, a macro from the code or `\\setnum`), doc (`\\docnum`, a documented fact",
             "with no result file), table (a pasted table replaced by the generated one after a",
             "number by number check).", "",
             "Counts: " + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())), ""]
    for i, (k, old, new) in enumerate(R, 1):
        if k == "preamble":
            continue
        lines += [f"## {i}. {k}", "", "Before:", "", "> " + old.replace("\n", " "), "", "After:", "",
                  "> " + new.replace("\n", " "), ""]
    AUDIT.write_text("\n".join(lines))
    print(f"{len(R)} replacements; audit in {AUDIT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
