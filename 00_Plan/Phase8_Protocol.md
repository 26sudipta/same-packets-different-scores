# Phase 8 protocol. RQ5, which published claims survive

Written on 27 October 2026 (about 19:45 UTC), before any Phase 8 number was computed. What was
already known at this time: the published numbers quoted below, read from the papers and the
authors' repositories; every Phase 4 to 7 result. Not yet computed: any rebuild, gap, decomposition
or conversation-level score of this phase. The table of claims (8.1) is part of this protocol so
that the set of claims cannot be chosen after seeing which survive.

**Question.** RQ5: which published cross-version improvement claims survive when the comparison is
defined (same conversations, one ground truth, conversation-grouped split, conversation scoring)?

**Framing rule (8.2 and 8.5).** We rebuild the type of comparison each paper made, on the releases
the authors published, with the authors' own code or a line-by-line port of it where it exists. We
do not claim to reproduce their exact experiment. Each paper is a contribution: its release is the
reason this test is possible. Sentences are about comparisons, never about authors.

## 8.1 The claims

| ID | Paper | Claim as published (source) | Releases | Examinable? |
| --- | --- | --- | --- | --- |
| R1 | Rosay et al. 2021, WI-IAT, LYCOS-IDS2017 | LYCOS-IDS2017 gives better results than CIC-IDS2017 for all metrics and all algorithms (section 6.3, Table 2). Binary metrics on the multi-class test set; e.g. RF accuracy 99.91 to 99.96 %, MCC 0.9983 to 0.9992; LDA accuracy 87.88 to 96.59 % (the text says 88.10 %; Table 2 says 87.88 %), FPR 22.61 to 6.35 % | `cic_original`, `lycos` | Yes. Authors' `create_datasets.py` and `analysis.py` are on disk |
| R2 | Engelen et al. 2021, WTMC | Per-class F1 of RF changes from Original to Final (Table II): Bot 0.60 to 0.99, Web Brute Force 0.77 to 0.95, XSS 0.18 to 0.79, Infiltration 0.81 to 0.91, Heartbleed 0.77 to 0.93, SQL Injection 0.11 to 0.00; aggregate metrics unchanged at 0.99 | `cic_original` (stands in for their "Original", which the paper says matches the public dataset), `wtmc2021` (their "Final") | Yes, Original against Final. Their Intermediate version was not released: not examinable |
| R3 | Lanvin et al. 2022, CRiSIS (LNCS 13857) | Q2: labelling the Thursday port scan (RD to RPD) costs about 20 % recall on average over four models; Q3: without duplicated packets (RPD to RP) recall returns from about 80 % to close to 100 %; Q4: RP is as good as RD (section 4.2, Figures 1 and 2) | `CRiSIS2022_RD`, `crisis2022_rpd`, `CRiSIS2022_RP` (Labels/Exp_* of gitlab.inria.fr/mlanvin/crisis2022) | Yes if Exp_RD (Thursday) and Exp_RP finish downloading (started 27 October). If Exp_RP does not arrive, Q3 and Q4 are "not examinable, data not obtained"; Q2 is then examined with `wtmc2021` (their D) in place of RD and reported as a proxy |
| R4 | Sarhan, Layeghy and Portmann 2022, MONET | NF-UNSW-NB15-v2 improves detection over UNSW-NB15: Extra Trees binary F1 0.92 to 0.97, AUC 0.9545 to 0.9845 (Table 8); multi-class detection rate of Fuzzers 50.50 to 80.57 %, Worms 5.26 to 85.98 % (Table 9) | `unsw_original` as released, NF-UNSW-NB15-v2 (downloaded 27 October from UQ RDM 8c6e2a00-ef9c-11ed-827d-e762de186848), `nf_unsw_v3` | Yes for 8.2 and 8.3 with v2 itself. v2 has no timestamps, so 8.4 uses v3 as the NetFlow arm, reported as a proxy for v2 |
| R5 | Pekar and Jozsa 2024, Computer Networks | Per day binary RF (Table 6): (a) CIC-IDS2017 Tuesday F1 0.8709 against at least 0.9993 for every refined version; (b) Thursday: CIC-IDS2017, WTMC-2021 and NFS-2023-TE lower (F1 0.9404, 0.9365, 0.9416) than CRiSIS-2022 and NFS-2023-nTE (0.9834, 0.9914); (c) TCP flag expiry "does not adversely affect" detection (TE against nTE) | `cic_original`, `wtmc2021`, `crisis2022_rpd`, `nfs2023_nte`, `nfs2023_te` | Yes. Authors' notebooks and result JSON files are on disk (FlowFrontiers/CyberML-DataQuality) |
| R6 | Janati Idrissi et al. 2025, FGCS | Best against worst timeout setting differs by up to 8.77 % macro-F1 (abstract) | none | No: needs re-extraction of the packet captures over their timeout grid (scope (a) excludes it) and the full text is not held |
| R7 | Pinto et al. 2025, ICISSP (HERA) | RF macro-F1 84.24 to 98.38 % (Table 10 of the arXiv version held) | none | No: HERA flows requested 27 October 2026, no reply |
| R8 | Xin, Goseland and Rahouti 2026, arXiv | 15-class macro-F1 0.955 on a new re-extraction (Phase 1 reading notes) | none | No: no flow files released, and the claim is not a cross-version comparison |
| R9 | Shafi et al. (NTLFlowLyzer, BCCC-CIC-IDS2017) | not read: only the publisher preview is held | `bccc2017` | No as a claim (claim text unverified); the release stays an arm in RQ1 to RQ4 |
| R10 | Vitorino et al. 2025, arXiv (HERA on IoT) | macro-F1 gains up to 68 points (26 October notes, not re-checked here) | none | No: other benchmarks, flows not held |
| R11 | Luay et al. 2025 (NF-v3) | no performance comparison is made (descriptive temporal analysis) | `nf_unsw_v3` | Nothing to examine |

Headline contrast per examinable claim (the "gap", new minus old, in the claim's own metric):

- R1: binary MCC, per model (RF, DT, LDA); accuracy and FPR reported beside it.
- R2: per-class F1 of RF for the six classes named above; macro-F1 over shared classes beside it.
- R3: binary recall (TPR) averaged over DT, RF and Bernoulli naive Bayes, per contrast Q2 (RPD minus
  RD), Q3 (RP minus RPD), Q4 (RP minus RD); per model beside it. Their SVC is left out: an RBF
  support vector machine on about 700,000 training rows is beyond the laptop; the claim is examined on
  three of their four models and the table says so.
- R4: binary F1 of Extra Trees; multi-class per-class detection rate (recall) of Fuzzers and Worms.
- R5: binary F1 of RF, contrasts (a) Tuesday nTE minus CIC, (b) Thursday nTE minus TE and CRiSIS
  minus WTMC, (c) TE minus nTE on each day.

## 8.2 Rebuild per flow, as published

Each arm under its own natural split, per flow, with the arm's own released labels, the authors'
preprocessing, features and model. Where the authors fixed a seed it is used; the model is also
refitted with seeds 0 to 4 and the mean over seeds is the rebuilt value (range beside it).

- R1: a port of `create_datasets.py` (pandas 3: `append` becomes `concat`, nothing else changes;
  `np.random.seed(2021)`, per class permutation with `random_state=0`, the authors' per class
  percentages so that both datasets get the same number of attack rows per class and attacks are half
  of each set) and of `analysis.py` (their drop lists, z-score fitted on training, multi-class
  training, binary metrics from the multi-class confusion matrix as in `print_metrics`). Models: LDA,
  DT (`random_state`), RF (100 trees, `random_state`). k-NN, QDA, linear SVM and their TensorFlow MLP
  are not rebuilt (time; the claim is "all algorithms", three are tested and the table says so).
- R2: RF, sklearn defaults, 75/25 stratified split with seed s, Attempted labels assigned to BENIGN,
  per-class F1. Their hyperparameters and feature list are on a project page we have not read; we
  drop flow ID, addresses, ports and timestamp (their shortcut learning section argues against
  them) and say that these choices are ours.
- R3: a port of `Evaluation_relabelling_CICIDS2017_reordered.ipynb` and `tools.preprocessData`:
  training rows are rows whose label is the same in both variants, 65 % of each attack drawn at random
  plus as many benign rows, the rest is the test set; StandardScaler; DT and RF with depth 15,
  BernoulliNB; binary recall on the test set. For RP (no R release held), "constant" rows are the
  rows not labelled port scan, which is what R against RP gives if P only adds the port scan label;
  this is checked on RD against RPD before it is used.
- R4: Extra Trees (sklearn defaults), flow identifiers, ports, timestamps and TTL features dropped,
  min-max scaling, 70/30 split, five splits (seeds 0 to 4), binary F1 and multi-class recall per class.
- R5: the authors' notebook cell exactly: per day, their 42 columns, `train_test_split(test_size=0.3,
  random_state=42, stratify=y)`, `RandomForestClassifier()` (unseeded in their code; seeds 0 to 4
  here), binary label "not BENIGN", Attempted rows removed for WTMC and CRiSIS, NaN and infinite rows
  removed for the CICFlowMeter arms.

**Fidelity check.** For every claim, the rebuilt values sit next to the published ones. A rebuilt gap
with the published sign is "reproduced in direction"; within 0.01 (or within the published
rounding) "reproduced". A claim that is not reproduced per flow is still carried through 8.3 and 8.4,
and the table says so.

## 8.3 Decompose each gap

Per claim and per pair (A old, B new), on the rebuilt test sets:

1. Tabulate what differs: test records, distinct conversations (instrument ids), records per
   conversation, class shares, and the share of test records in conversations absent from the other
   arm.
2. Classes are the canonical classes of Phase 3 (`instrument.canonical`). Shared classes are those
   present in both test sets. G (full gap) = G_shared (gap on shared classes only) + non-shared part.
3. Composition (class weight) effect on shared classes, symmetric (two-fold Oaxaca style): reweight
   each test record so that its class's total weight matches the other arm's class share,
   C = 1/2 [(M_B(w_B) - M_B(w_A)) + (M_A(w_B) - M_A(w_A))]. Within-class part W = G_shared - C.
   Share explained by class weights = C / G_shared.
4. Interval: paired cluster bootstrap over conversations (the Phase 5 scheme: strata = schedule class,
   B = 1000, generator seed 0), the same replicate weight for a conversation in both arms, metric
   averaged over model seeds inside each replicate. Rows of NF-UNSW-NB15-v2 (no timestamps, no
   conversation) are resampled as single records within their class; the table marks it.
5. A share is reported only when the interval of G_shared excludes 0; otherwise "gap not
   distinguishable from zero". The trap of the plan holds: the share is a measured quantity, and a
   share near 1 is not called an artifact unless its interval is near 1 too.

## 8.4 Redo each comparison at conversation level

Population: the Phase 5 subsample (benign fraction 0.2, attack cap 20,000 conversations per class,
`04_Results/phase5/settings.json`), restricted to conversations with at least one record in both arms
of the pair (instrument per-arm counts), schedule class not excluded. RP is attached to the frozen
conversations by key and time (each record joins the conversation of the same key whose span, widened
by T = 15 s, contains its start; unmatched records are counted and left out) and hand-checked on
twenty records before use (standing gate 3). RD shares its rows with RPD (checked), so RD and RPD are
the same arm under schedule labels.

Split: the instrument's grouped split (`g0` to `g4`, 30 % of conversations, stratified by schedule
class). Truth: schedule label (Phase 5 `sched`, Attempted kept as attack); sensitivity with Attempted
folded into benign (`sched_attempted_fold`) for R2 and R3, whose papers fold it. Model and features:
the claim's own (Rosay's and Lanvin's destination port is added back from the instrument's `dport`,
since the Phase 5 tables drop ports). R5 trains and tests per day, the others on all days. Scoring:
rule "any" (3.8). Reported for each pair: the gap per flow on this population and the gap per
conversation, beside the 8.2 gap, with the paired cluster bootstrap interval of 8.3 step 4.

**Verdict per claim** (the conversation-level gap in the claim's metric):

- "survives": same sign as published and 95 % interval excludes 0;
- "reverses": interval entirely of the opposite sign;
- "not supported": otherwise (undecided);
- for a no-difference claim (R3 Q4, R5 (c)): "survives" if the interval lies inside [-0.01, +0.01],
  "contradicted" if it excludes 0, "undecided" otherwise.

A class with fewer than 20 test conversations in any seed is "not estimable" at conversation level and
is counted as such, not as a verdict.

## 8.5 Tone

Every sentence that names a paper is reread before Gate 8: comparisons, not authors; each release
acknowledged as the contribution that makes the check possible; deviations from their setup stated
as ours.

## Amendments (28 October 2026, after the first 8.2 cells, before any 8.3 or 8.4 verdict)

Each amendment says what was known when it was made.

1. **R5 uses Exp_RP, not Exp_RPD.** The authors' notebook
   `3-CRiSIS-2022/1-analyse-dataset.ipynb` downloads `Labels/Exp_RP/*.csv`, and their
   `data-distribution/summary.json` gives 355,618 Thursday flows with 64,185 Port_scan, which is
   not our `crisis2022_rpd` (72,336 Port_scan on Thursday). Table 8.1 assumed the instrument arm.
   Known at this time: the 8.2 rebuilt F1 of the RPD arm per day (Thursday 0.995 against their
   0.9834 for RP). Change: the R5 CRiSIS arm is `crisis2022_rp` in 8.2, 8.3 and 8.4 (8.4 through
   the RP map of step 8.4); the RPD cells already computed stay as a labelled extra row ("our
   instrument arm"), not as the claim's test. If Exp_RP cannot be obtained, the R5 contrasts that
   involve CRiSIS are "not examinable, data not obtained" and the other R5 contrasts stand.
2. **Exp_RD and Exp_R.** The repository tree (one commit, c4aa1f3) gives Exp_RD the same blob as
   Exp_RPD on Monday, Tuesday, Wednesday and Friday, and Exp_R the same blob as Exp_RP on those days.
   Exp_RD Thursday equals Exp_RPD Thursday with the label `Port_scan` written back as `BENIGN`: the
   rebuilt file has the published blob id (b8f992a8). The same rebuild is applied to Exp_R Thursday
   from Exp_RP Thursday and kept only if its blob id matches (9847d24e). With Exp_R held, the RP
   experiment of R3 uses the authors' own definition of constant rows (same label in R and RP)
   instead of the stand-in "rows not labelled port scan" of 8.2.
3. **Naming.** In Lanvin et al.'s labels D marks the versions with duplicated packets kept (RD,
   RPD; notebook `..._reordered`), and the versions without D (R, RP) come from captures without
   duplicated packets (notebook `..._reordered_without_duplication`). Our instrument arm
   `crisis2022_rpd` therefore keeps duplicated packets; `releases.py` described it as
   "deduplicated", which was wrong and is corrected. No Phase 3 to 7 number depends on that note.
4. **Exp_RP not obtained (28 October 2026, after the R1, R2, R4 and R5 verdicts were computed).**
   Every download of the five Exp_RP files from gitlab.inria.fr was cut by the server or the route
   after 3 to 17 minutes, on three endpoints, over about four hours (logs in
   `Datasets/_downloads/crisis2022_extra/`). As 8.1 and amendment 1 say, R3 Q3, R3 Q4 and the R5
   contrasts that involve CRiSIS-2022 are "not examinable, data not obtained". This amendment chooses
   nothing: it records the fallback the protocol already fixed. Q2 uses the authors' RD (rebuilt and
   blob-checked, amendment 2), so the WTMC-2021 proxy of 8.1 is not needed. At conversation level Q2 is
   zero by construction (RD and RPD are one arm under schedule labels, 8.4) and is reported as "not
   supported (zero by construction)". If the files are obtained later, the R3 steps of
   `run_phase8.sh` and `run_phase8_rest.sh`, then `run_phase8_rp.sh`, produce the missing rows
   without other changes.
5. **R4, one spelling of Backdoor (29 October 2026, after every verdict was known, found in the
   Phase 9 code review).** UNSW-NB15 original labels 534 rows "Backdoors" (`UNSW-NB15_1.csv`) and
   1,795 rows "Backdoor" (files 2 to 4). Our R4 rebuild of 8.2 (our implementation of their setup,
   not the authors' code) trained the multi-class Extra Trees model and drew the stratified 70/30
   split on the raw strings, so it treated one class as two; scoring already mapped both to
   "backdoor". Change: the two spellings are one class for training and the split, and the run
   stops if any arm still has two spellings of one class (`phase8_flow.py`, `run_r4`). Known at
   this time: every 8.2, 8.3 and 8.4 result. What it can change: the R4 per-flow values of
   UNSW-NB15 original (binary F1, Fuzzers and Worms recall) through the split and the model. The
   scored claims do not involve Backdoor, and the 8.4 verdicts use schedule labels per
   conversation, so no verdict is expected to change; the second freeze run reports the new
   values and any change (`00_Plan/Phase9_Freeze_Notes.md`).

6. **Exp_RP and Exp_R obtained (30 October 2026, after every other verdict was known).** The five
   Exp_RP files and Exp_R Thursday were downloaded from gitlab.inria.fr in a cloud machine (the
   route from the laptop gave 1 to 4 KB/s on two networks) and moved to the laptop; each has the
   git blob id of the repository tree (Exp_RP: 0e15b227, 68c1beb3, e7566f14, 01e8ea94, 742b69cc;
   Exp_R Thursday: 9847d24e). The rebuild of Exp_R Thursday from Exp_RP Thursday that amendment 2
   allowed did not match (blob 074cd776: 1,789 rows differ in the last digits of float features),
   so the downloaded file is used, as amendment 2 required. The missing rows were produced by the
   Exp_RP steps of `run_all.sh` (`03_Code/run_phase8_rpfill.sh`), with no change to any choice of
   this protocol. The 20-case hand check of the RP map passed before any RP number was read
   (standing gate 3). Amendment 4 no longer applies.
