# Phase 6 protocol. RQ3, NFS-2023 nTE against TE

Written at 15:07 UTC on 27 October 2026, before any Phase 6 number was computed. What was
already known at this time, from Phase 5: TE gives flow-level F1 near 0.667 for SSH-Patator, Bot and
DoS GoldenEye on the grouped split, because about two records per attack conversation exist and one
looks benign (`03_Code/phase5_te_check.py`). Not yet looked at: leakage per class, and any
random-split against grouped-split comparison per class for these two arms.

**Question.** RQ3: does segmentation alone change leakage and scores? The pair differs in one
generator setting (NFStream `FlowExpirationManager`, TCP FIN/RST expiry, on in TE only). The released
CSVs also differ through a post-processing filter (flows of exactly one packet that is FIN or RST are
dropped) and the per-flow labeller (`02_Literature/Reading_Notes.md`, step 1.1). Phase 6 removes the
labeller difference by using the schedule ground truth (step 3.2) for both arms, and removes the
population difference by scoring on conversations present in every arm (`in_all`). What remains is
the expiry rule plus the filter it triggers. The paper calls it "one generator setting, with the
filter that responds to it", never a pure one-setting contrast.

**Data.** The Phase 5 tables of the two arms (`Datasets/_cache_phase5/main/CICIDS2017/`, the same
conversation subsample) and the Phase 5 cell scores (`04_Results/phase5/results.csv`). No new model
is trained.

**Classes.** CICIDS2017 attack schedule classes with at least 20 common-population conversations in
the subsample of both arms. Benign is reported, but not part of the 6.2 test.

**Measures, per class and arm, common population unless stated.**

1. Records per conversation: mean, and the share of conversations with two or more records.
2. Leakage under the random flow split: for seed s, the share of the class's test records whose
   conversation has at least one record in training (column r<s>); mean over seeds 0 to 4.
   The grouped split has zero leakage by construction; this is checked, not assumed.
3. F1 at the flow and the conversation unit (rule "any"), grouped and random split, mean over the
   three models and five seeds, with the min and max over cells.
4. Inflation: flow-level F1 on the random split minus flow-level F1 on the grouped split.
5. Mechanism of the extra TE records: for the classes where TE has more records per conversation,
   compare the first and the later records of a conversation (packets, duration, bytes, flag counts,
   expiration id in the raw CSV) to say what the extra record is.

**6.2 prediction, fixed now.** Across eligible classes, the change in leakage from nTE to TE
(leak TE minus leak nTE) is positively associated with the change in inflation (inflation TE minus
inflation nTE). Statistic: Spearman rho across classes. One-sided permutation p value (exact when
there are at most 8 classes, otherwise 100,000 random permutations, seed 6). Verdict:
"supported" if rho > 0 and p < 0.05; "direction only" if rho > 0 and p >= 0.05; "not supported"
if rho <= 0. Robustness, reported but not used for the verdict: the same per model, and on the own
population. Reported whatever the outcome, including if leakage barely changes.

**Second prediction, fixed now (descriptive).** At the conversation unit on the grouped split, the
F1 gap between nTE and TE is smaller than at the flow unit for most eligible classes, because a
conversation is detected if any of its records is. Reported as a count of classes, no test.

**Outputs.** `03_Code/phase6_rq3.py`, `04_Results/phase6/*.csv`, one table for 6.1, the 6.2 result,
and a figure of leakage change against inflation change.
