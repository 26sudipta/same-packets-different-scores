# Phase 7 protocol. RQ4, the remedy

Written on 27 October 2026 (15:45 UTC), before any Phase 7 number was computed. What was
already known at this time, from Phase 5 (`04_Results/phase5/kendall_w.csv`): on CICIDS2017,
grouped split, common population, the mean over seeds of Kendall's W is 0.886 at the flow unit and
0.820 at the conversation unit, with separate (unpaired) intervals 0.780 to 0.927 and 0.763 to 0.869.
The paired interval of the difference, which the H4 rule uses, had not been computed. From Phase 6:
the NFS-2023 TE trailing-ACK record and about 1,748 benign records per cell predicted DoS Hulk in TE.
Not yet looked at: exporter shares at the conversation unit against the flow unit, class by class,
and every sensitivity variant below.

**Question.** RQ4: does scoring per conversation with a conversation-grouped split remove the
exporter effect, and what remains?

**Data.** The Phase 5 arm tables, predictions and cell scores. No new model is trained, except for
the two label variants of 7.3 marked "retrained".

## 7.1 H4, fixed in PROGRESS_LOG before the grid (Phase 5 protocol), made exact here

- Statistic: Kendall's W of the model ranking (three models ranked by macro-F1 over the fixed class
  set of Phase 5) with the arms as raters, computed per seed and averaged over the five seeds
  (`w_seed_mean`, the Phase 5 headline). Secondary: W on the seed means.
- Setting: CICIDS2017, primary arm set (seven arms), grouped split, common population.
- Difference d = W(conversation) minus W(flow). Paired cluster bootstrap over conversations, the
  Phase 5 scheme (stratified by schedule class, B = 1000, generator seed 0), the same replicate
  weights for both units.
- Verdict: "supported" if the 2.5th percentile of d is above 0; "reversed" if the 97.5th
  percentile is below 0; otherwise "not supported (undecided)".
- UNSW-NB15 is reported with the same computation but marked "not informative": with two primary
  arms W is 1 whenever the two arms agree, which Phase 5 already showed.

Alongside H4 (descriptive, not a hypothesis): per class, the exporter (arm) share of the sum of
squares at the conversation unit minus at the flow unit, same setting, paired interval. "Falls",
"rises" or "undecided" by the same interval rule. Classes: those with at least 20 test conversations
in every grouped seed (the Phase 5 fixed set), both benchmarks.

## 7.2 The residual

Setting: conversation unit, grouped split, common population, so every arm is scored on the same
test conversations with the same truth. A class has a **large residual** if its exporter share is at
least 0.25 and the spread of its arm means of F1 (max minus min over arms of the mean over models and
seeds) is at least 0.05. The second condition keeps out classes where every arm scores near 1 and a
large share of a tiny sum of squares means nothing. For every large-residual class: the arm(s)
responsible (leave-one-arm-out shares, point estimates), and a mechanism from the Phase 4 step 4.4
table or Phase 6, or the words "not explained".

## 7.3 Sensitivity

**Conclusions counted** (the "headline result" for this table):

- K1: the H4 verdict (CICIDS2017).
- K2: the H2 verdict per benchmark in the primary setting (grouped, flow, own population).
- K3: the H2 verdict per benchmark at the conversation unit (grouped, own population).
- K4: per class verdicts at the conversation unit, grouped, common population: exporter above
  model, model above exporter, or undecided (the H2 class rule applied to every class in the fixed
  set, small or not).
- K5: per class direction of the exporter share from flow to conversation (7.1).
- K6: the set of large-residual classes (7.2).

For each variant: whether K1, K2, K3 change, and how many of the K4, K5 class verdicts change out of
how many, and which classes enter or leave K6. A class verdict that becomes not estimable is counted
separately, not as a change.

**Variants.** Everything not named stays at the Phase 5 choice (T = 15 s, rule "any", schedule
labels, Attempted kept, conversation unit).

1. Baseline: must reproduce the Phase 5 point estimates (W, shares, H2) to 1e-9. If it does not,
   stop and find out why.
2. T = 5, 30, 60 and 120 s. Conversations rebuilt with `instrument.build_conversations` on the same
   records (first checked: T = 15 rebuilds the frozen conversations exactly). Predictions and the
   grouped split stay as trained (the split was drawn on T = 15 conversations); a T conversation is
   scored on its test records. Truth of a T conversation: the attack class with the most test
   records among its records, benign only if all are benign; the number of rebuilt conversations
   whose constituents carry two schedule classes is reported. The flow unit does not change except
   through the resampling clusters.
3. Aggregation rule "majority" and rule "bytes" (`conv_scoring.RULES`).
4. Episode unit (step 3.6) in place of the conversation, resampled by episode. Expected to be
   mostly not estimable on CICIDS2017 (often one episode per class); that count is itself the result
   and goes into 7.4.
5. Attempted folded into benign (`sched_attempted_fold`), scoring only: truth changes, predictions do
   not. Then **retrained**: CICIDS2017, every arm, three models, grouped split, seeds 0 to 4, the
   Phase 5 hyperparameters (no new tuning), trained and scored on the folded labels.
6. Own labels (each arm's released labels, canonical classes, conversation truth "any attack record
   makes an attack conversation"), scoring only. Then **retrained** on CICIDS2017 as in 5, trained
   and scored on own labels. UNSW-NB15 own-label retraining only if the laptop time allows; if not,
   the scoring-only result stands and the table says so.

A retrained variant changes the training labels, so its K2 can change; the scoring-only variants
can change only the conversation-unit conclusions. If any variant flips K1 or K2, it goes in the
abstract (plan, step 7.3 trap).

## 7.4 The reporting rule

Written from 7.1 to 7.3: a checklist for producers and for evaluators, worked on Heartbleed,
Infiltration and one class that passes. The number of independent units per class (conversations and
episodes, test side, per seed) is taken from the instrument, not estimated.
