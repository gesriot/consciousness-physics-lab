# B006: fallible diagnosis – development outcome

## Result and scope

None of the four preregistered global probabilistic configurations passed the
strict development admission rule. No configuration was selected for confirmation,
no confirmation cohort was run, and neither closure conclusion is evaluated.
This is a negative result for these designs under this gate, not evidence that
reliability modelling is unnecessary or that a different design would succeed.

The first saved development comprises 12 independent seeded series and 82,944
dependent episode conditions. Each series contains 32 sequential episodes on one
fixed graph; physical faults reset between episodes but adaptive counters persist.
Environments, budgets and policies within a series are paired, not independent
replications. The factorial comparison shares physical draws across check-error
levels. All summaries below are development descriptions, not held-out estimates.

The common admission checks passed: D made 1,225 false accusations under moderate
check error, exposure covered all registered fault/scope combinations, and every
P/Pxprime placebo action comparison had zero mismatches. These facts do not rescue
the failed low_zero gate. K3of5 was the selected constant repetition rule and the
strongest constant control under the registered lexicographic ranking.

## Laboratory and interpretation limits

Four environments combine low/high false-objection frequency with zero/moderate
check error. D retains the B005 rule; R first repeats the initiating report;
K repeats checks; P chooses tests and diagnoses using posterior probabilities.
All use inherited provenance labels L. Px perturbs active reliability estimates;
Pxprime applies the same perturbation to unused auxiliary estimates. N does nothing.
The complete probabilities, costs, seeds and finite selection grid are in the
[protocol](../../protocols/B006-fallible-diagnosis.md).

Checks return fallible binary reports. An accusation authorizes a certified
catalogue replacement followed by paid descendant rebuilding. This repair is
not fallible: a false accusation can waste resources without creating an incorrect
final value. Lost-good counts are zero throughout by this construction, not a
learned protection against destructive revision. False accusations require both
an innocent ground being named and at least one descendant actually being revised;
attempted accusations are recorded separately.

False route-conflict signals are reports from the signal channel. Policies cannot
bypass paid checks by inspecting stored endpoint values to falsify those reports.
Thus the experiment does not establish robustness to arbitrary numerical evidence.
D/K do not repeat an exhausted conflict report; R/P can. Policy contrasts include
these action-space differences, not only probabilistic confidence.

Learning uses fractional posterior responsibilities only after observable
resolution: a paid observation repeat reporting no discrepancy, or a paid repaired
descendant agreeing with the initiating public evidence. A confident diagnosis
alone does not train. No evaluator truth enters adaptation, but this remains
self-training and can reinforce a wrong diagnosis; it does not identify true
channel reliabilities from an independent supervised sample.

## Strict low_zero admission failures

Admission required P to be no worse than D in each of four primary metrics in
every budget/branch cell. A small regression is a failure, not an equivalence
margin. The complete failing cells follow; unlisted metrics passed the comparison.

| Configuration | Budget | Signal | Metric | D | P |
|---|---:|---|---|---:|---:|
| beta8_t80 | 2 | measurement | remaining_bad | 698 | 868 |
| beta8_t80 | 2 | conflict | remaining_bad | 790 | 956 |
| beta8_t80 | 8 | measurement | remaining_bad | 9 | 195 |
| beta8_t80 | 8 | conflict | remaining_bad | 225 | 381 |
| beta8_t80 | 8 | conflict | cost | 84225 | 87136 |
| beta8_t80 | 32 | measurement | remaining_bad | 0 | 140 |
| beta8_t80 | 32 | conflict | remaining_bad | 0 | 102 |
| beta8_t80 | 32 | conflict | cost | 99962 | 109150 |
| beta8_t95 | 2 | measurement | remaining_bad | 698 | 952 |
| beta8_t95 | 2 | conflict | remaining_bad | 790 | 994 |
| beta8_t95 | 8 | measurement | remaining_bad | 9 | 884 |
| beta8_t95 | 8 | conflict | remaining_bad | 225 | 994 |
| beta8_t95 | 32 | measurement | remaining_bad | 0 | 884 |
| beta8_t95 | 32 | conflict | remaining_bad | 0 | 994 |
| beta99_t80 | 2 | measurement | remaining_bad | 698 | 699 |
| beta99_t80 | 8 | measurement | false_accusations | 0 | 1 |
| beta99_t80 | 32 | measurement | false_accusations | 0 | 1 |
| beta99_t80 | 32 | conflict | remaining_bad | 0 | 40 |
| beta99_t95 | 2 | measurement | remaining_bad | 698 | 703 |
| beta99_t95 | 2 | measurement | cost | 24585 | 24860 |
| beta99_t95 | 2 | conflict | remaining_bad | 790 | 848 |
| beta99_t95 | 2 | conflict | cost | 23906 | 24242 |
| beta99_t95 | 8 | measurement | remaining_bad | 9 | 15 |
| beta99_t95 | 8 | conflict | remaining_bad | 225 | 243 |
| beta99_t95 | 8 | conflict | cost | 84225 | 86727 |
| beta99_t95 | 32 | conflict | cost | 99962 | 104749 |

The beta99_t80 setting is close in some cells but still fails: one extra remaining
bad claim at budget 2, one false accusation at each of budgets 8 and 32 in the
measurement branch, and 40 remaining bad claims instead of zero in the budget-32
conflict branch. Calling it successful would relax the registered gate after data.

## Paired primary tradeoffs

Each tuple below is (false accusations, remaining bad claims, lost good, cost).
Costs are laboratory work units, not elapsed seconds. Every row pools the same
12 paired independent series within one cell. The complete P-versus-D/R/K cell
comparisons, including signed differences, are in [analysis.json](analysis.json).
The following tables show every P configuration against D without collapsing
budgets, signal branches or environments into a single superiority score.

### P_beta8_t80

| Environment | Budget | Signal | D tuple | P tuple | Relation |
|---|---:|---|---|---|---|
| high_moderate | 2 | conflict | 123/436/0/21023 | 16/438/0/20770 | tradeoff |
| high_moderate | 2 | measurement | 74/306/0/22861 | 15/370/0/19033 | tradeoff |
| high_moderate | 32 | conflict | 135/408/0/50626 | 21/279/0/83634 | tradeoff |
| high_moderate | 32 | measurement | 89/116/0/63484 | 24/198/0/60327 | tradeoff |
| high_moderate | 8 | conflict | 135/410/0/49899 | 20/328/0/66419 | tradeoff |
| high_moderate | 8 | measurement | 89/119/0/63156 | 23/206/0/55902 | tradeoff |
| high_zero | 2 | conflict | 0/357/0/23620 | 0/431/0/23530 | tradeoff |
| high_zero | 2 | measurement | 0/267/0/24176 | 0/356/0/23014 | tradeoff |
| high_zero | 32 | conflict | 0/0/0/82668 | 0/112/0/94600 | control_dominates |
| high_zero | 32 | measurement | 0/0/0/75410 | 0/104/0/70573 | tradeoff |
| high_zero | 8 | conflict | 0/86/0/76629 | 0/244/0/78858 | control_dominates |
| high_zero | 8 | measurement | 0/4/0/75082 | 0/141/0/65786 | tradeoff |
| low_moderate | 2 | conflict | 136/963/0/19108 | 21/977/0/19359 | tradeoff |
| low_moderate | 2 | measurement | 50/786/0/23642 | 10/898/0/19866 | tradeoff |
| low_moderate | 32 | conflict | 144/903/0/41389 | 25/643/0/81736 | tradeoff |
| low_moderate | 32 | measurement | 53/323/0/72783 | 14/399/0/67577 | tradeoff |
| low_moderate | 8 | conflict | 144/906/0/40825 | 18/777/0/58763 | tradeoff |
| low_moderate | 8 | measurement | 53/329/0/72209 | 16/476/0/62978 | tradeoff |
| low_zero | 2 | conflict | 0/790/0/23906 | 0/956/0/23634 | tradeoff |
| low_zero | 2 | measurement | 0/698/0/24585 | 0/868/0/22130 | tradeoff |
| low_zero | 32 | conflict | 0/0/0/99962 | 0/102/0/109150 | control_dominates |
| low_zero | 32 | measurement | 0/0/0/82176 | 0/140/0/75187 | tradeoff |
| low_zero | 8 | conflict | 0/225/0/84225 | 0/381/0/87136 | control_dominates |
| low_zero | 8 | measurement | 0/9/0/81438 | 0/195/0/73494 | tradeoff |

### P_beta8_t95

| Environment | Budget | Signal | D tuple | P tuple | Relation |
|---|---:|---|---|---|---|
| high_moderate | 2 | conflict | 123/436/0/21023 | 0/445/0/19286 | tradeoff |
| high_moderate | 2 | measurement | 74/306/0/22861 | 1/418/0/16163 | tradeoff |
| high_moderate | 32 | conflict | 135/408/0/50626 | 0/445/0/59621 | tradeoff |
| high_moderate | 32 | measurement | 89/116/0/63484 | 1/399/0/36498 | tradeoff |
| high_moderate | 8 | conflict | 135/410/0/49899 | 0/445/0/54723 | tradeoff |
| high_moderate | 8 | measurement | 89/119/0/63156 | 1/399/0/35497 | tradeoff |
| high_zero | 2 | conflict | 0/357/0/23620 | 0/445/0/23352 | tradeoff |
| high_zero | 2 | measurement | 0/267/0/24176 | 0/407/0/22210 | tradeoff |
| high_zero | 32 | conflict | 0/0/0/82668 | 0/445/0/63892 | tradeoff |
| high_zero | 32 | measurement | 0/0/0/75410 | 0/386/0/51449 | tradeoff |
| high_zero | 8 | conflict | 0/86/0/76629 | 0/445/0/62592 | tradeoff |
| high_zero | 8 | measurement | 0/4/0/75082 | 0/386/0/51449 | tradeoff |
| low_moderate | 2 | conflict | 136/963/0/19108 | 0/994/0/17834 | tradeoff |
| low_moderate | 2 | measurement | 50/786/0/23642 | 0/971/0/18084 | tradeoff |
| low_moderate | 32 | conflict | 144/903/0/41389 | 0/994/0/51603 | tradeoff |
| low_moderate | 32 | measurement | 53/323/0/72783 | 0/904/0/34230 | tradeoff |
| low_moderate | 8 | conflict | 144/906/0/40825 | 0/994/0/46419 | tradeoff |
| low_moderate | 8 | measurement | 53/329/0/72209 | 0/909/0/33894 | tradeoff |
| low_zero | 2 | conflict | 0/790/0/23906 | 0/994/0/23078 | tradeoff |
| low_zero | 2 | measurement | 0/698/0/24585 | 0/952/0/21458 | tradeoff |
| low_zero | 32 | conflict | 0/0/0/99962 | 0/994/0/57778 | tradeoff |
| low_zero | 32 | measurement | 0/0/0/82176 | 0/884/0/37232 | tradeoff |
| low_zero | 8 | conflict | 0/225/0/84225 | 0/994/0/54378 | tradeoff |
| low_zero | 8 | measurement | 0/9/0/81438 | 0/884/0/37232 | tradeoff |

### P_beta99_t80

| Environment | Budget | Signal | D tuple | P tuple | Relation |
|---|---:|---|---|---|---|
| high_moderate | 2 | conflict | 123/436/0/21023 | 119/432/0/21073 | tradeoff |
| high_moderate | 2 | measurement | 74/306/0/22861 | 74/295/0/22872 | tradeoff |
| high_moderate | 32 | conflict | 135/408/0/50626 | 132/396/0/53443 | tradeoff |
| high_moderate | 32 | measurement | 89/116/0/63484 | 89/98/0/56524 | P_dominates |
| high_moderate | 8 | conflict | 135/410/0/49899 | 134/399/0/51701 | tradeoff |
| high_moderate | 8 | measurement | 89/119/0/63156 | 89/101/0/55845 | P_dominates |
| high_zero | 2 | conflict | 0/357/0/23620 | 0/357/0/23620 | equal |
| high_zero | 2 | measurement | 0/267/0/24176 | 0/267/0/23998 | P_dominates |
| high_zero | 32 | conflict | 0/0/0/82668 | 0/20/0/81492 | tradeoff |
| high_zero | 32 | measurement | 0/0/0/75410 | 0/0/0/56042 | P_dominates |
| high_zero | 8 | conflict | 0/86/0/76629 | 0/86/0/76629 | equal |
| high_zero | 8 | measurement | 0/4/0/75082 | 0/4/0/55714 | P_dominates |
| low_moderate | 2 | conflict | 136/963/0/19108 | 129/953/0/19071 | P_dominates |
| low_moderate | 2 | measurement | 50/786/0/23642 | 50/768/0/23498 | P_dominates |
| low_moderate | 32 | conflict | 144/903/0/41389 | 144/903/0/42401 | control_dominates |
| low_moderate | 32 | measurement | 53/323/0/72783 | 53/293/0/68692 | P_dominates |
| low_moderate | 8 | conflict | 144/906/0/40825 | 144/906/0/41504 | control_dominates |
| low_moderate | 8 | measurement | 53/329/0/72209 | 53/299/0/68118 | P_dominates |
| low_zero | 2 | conflict | 0/790/0/23906 | 0/790/0/23906 | equal |
| low_zero | 2 | measurement | 0/698/0/24585 | 0/699/0/24074 | tradeoff |
| low_zero | 32 | conflict | 0/0/0/99962 | 0/40/0/97610 | tradeoff |
| low_zero | 32 | measurement | 0/0/0/82176 | 1/0/0/77395 | tradeoff |
| low_zero | 8 | conflict | 0/225/0/84225 | 0/225/0/84225 | equal |
| low_zero | 8 | measurement | 0/9/0/81438 | 1/9/0/76657 | tradeoff |

### P_beta99_t95

| Environment | Budget | Signal | D tuple | P tuple | Relation |
|---|---:|---|---|---|---|
| high_moderate | 2 | conflict | 123/436/0/21023 | 41/423/0/22704 | tradeoff |
| high_moderate | 2 | measurement | 74/306/0/22861 | 39/313/0/23152 | tradeoff |
| high_moderate | 32 | conflict | 135/408/0/50626 | 49/138/0/95041 | tradeoff |
| high_moderate | 32 | measurement | 89/116/0/63484 | 50/65/0/76160 | tradeoff |
| high_moderate | 8 | conflict | 135/410/0/49899 | 55/242/0/78062 | tradeoff |
| high_moderate | 8 | measurement | 89/119/0/63156 | 52/76/0/69024 | tradeoff |
| high_zero | 2 | conflict | 0/357/0/23620 | 0/385/0/23720 | control_dominates |
| high_zero | 2 | measurement | 0/267/0/24176 | 0/271/0/24226 | control_dominates |
| high_zero | 32 | conflict | 0/0/0/82668 | 0/0/0/85449 | control_dominates |
| high_zero | 32 | measurement | 0/0/0/75410 | 0/0/0/57426 | P_dominates |
| high_zero | 8 | conflict | 0/86/0/76629 | 0/93/0/77738 | control_dominates |
| high_zero | 8 | measurement | 0/4/0/75082 | 0/4/0/56638 | P_dominates |
| low_moderate | 2 | conflict | 136/963/0/19108 | 54/956/0/21678 | tradeoff |
| low_moderate | 2 | measurement | 50/786/0/23642 | 27/792/0/24087 | tradeoff |
| low_moderate | 32 | conflict | 144/903/0/41389 | 48/463/0/95948 | tradeoff |
| low_moderate | 32 | measurement | 53/323/0/72783 | 29/194/0/88098 | tradeoff |
| low_moderate | 8 | conflict | 144/906/0/40825 | 56/657/0/73611 | tradeoff |
| low_moderate | 8 | measurement | 53/329/0/72209 | 28/219/0/80183 | tradeoff |
| low_zero | 2 | conflict | 0/790/0/23906 | 0/848/0/24242 | control_dominates |
| low_zero | 2 | measurement | 0/698/0/24585 | 0/703/0/24860 | control_dominates |
| low_zero | 32 | conflict | 0/0/0/99962 | 0/0/0/104749 | control_dominates |
| low_zero | 32 | measurement | 0/0/0/82176 | 0/0/0/79385 | P_dominates |
| low_zero | 8 | conflict | 0/225/0/84225 | 0/243/0/86727 | control_dominates |
| low_zero | 8 | measurement | 0/9/0/81438 | 0/15/0/78207 | tradeoff |

For example, beta8_t80 in high_moderate at budget 32 reduces conflict-branch false
accusations from 135 to 21 and remaining bad claims from 408 to 279, but increases
cost from 50,626 to 83,634. In the measurement branch it reduces accusations from
89 to 24 and cost from 63,484 to 60,327, but leaves 198 bad claims instead of 116.
These are different tradeoffs, not uniform improvement. Relations against R and
K3of5 must likewise be read cell by cell in the saved analysis.

## A registered one-step stopping limitation

In low_zero, series 20270501 episode 0, beta8_t95 at budget 8 spent only 60
of 656 units after two correct source reports. Its posterior assigned the true
culprit probability 0.8022284122562674, below the 0.95 decision threshold; ten
incorrect claims remained. Every affordable one-step classification-loss gain
rounded to zero, so the policy stopped rather than exhausting its budget.
The current leader remains among the most likely hypotheses after every next
answer; one possible report produces an exact tie, resolved in favor of the same
leader by the registered ordering. Another positive report about the true culprit
would raise its probability to 0.970105, past the decision threshold. The acquisition loss values a change
in classification, whereas the stopping rule requires sufficient confidence.
This is a consequence of the registered myopic objective and threshold, not an
implementation deviation. It helps explain the failed design, and is not a
counterfactual experiment demonstrating a replacement's advantage. Distinguishing
action/exposure gates passed; no configuration or objective was changed to turn
the failed admission into confirmation.

## Calibration, abstention and secondary outcomes

Calibration is computed for issued diagnoses; abstentions are counted separately,
not scored as incorrect diagnoses or silently dropped from coverage. Fixed-control
confidence was estimated on development itself and is marked development_in_sample.
It is not an independently validated probability. Series-level Brier means and
standard errors use series as the replication unit; pooled Brier values and bins
are descriptive. No calibration result here confirms H3.

The table illustrates the same high_moderate, budget-32 measurement cell for all
four P configurations and constant controls. SE is across series, not episodes.

| Policy | Diagnoses | Abstentions | Brier mean across series | SE | Series with diagnoses |
|---|---:|---:|---:|---:|---:|
| D | 162 | 14 | 0.269269 | 0.003911 | 12 |
| R | 169 | 7 | 0.160309 | 0.018906 | 12 |
| K3of5 | 153 | 23 | 0.143853 | 0.022121 | 12 |
| P_beta8_t80 | 87 | 89 | 0.222793 | 0.026669 | 12 |
| P_beta8_t95 | 6 | 170 | 0.116711 | 0.115749 | 4 |
| P_beta99_t80 | 176 | 0 | 0.470782 | 0.019185 | 12 |
| P_beta99_t95 | 166 | 10 | 0.301952 | 0.039776 | 12 |

Secondary outcomes retain explicit denominators for true counterexamples corrected
and false objections rejected. Depth and private/shared scope are stratified in
analysis.json; depth zero includes clean cases and is not a fault-localization
success. The following same-cell breakdown for beta8_t80 illustrates why the
aggregate remaining-bad count is not a pure diagnosis measure.

| Stratum | Episodes | True faults | Corrected | False objections | Rejected | Independent series |
|---|---:|---:|---:|---:|---:|---:|
| depth=0 | 16 | 0 | 0 | 0 | 0 | 8 |
| depth=1 | 56 | 23 | 15 | 33 | 15 | 12 |
| depth=2 | 33 | 4 | 2 | 29 | 12 | 11 |
| depth=3 | 54 | 27 | 14 | 27 | 3 | 12 |
| depth=4 | 33 | 5 | 1 | 28 | 0 | 11 |
| scope=private | 96 | 27 | 15 | 60 | 13 | 12 |
| scope=shared | 96 | 32 | 17 | 57 | 17 | 12 |

Stratum counts overlap between the depth and scope views and must not be added.
Independent-series counts can be below 12 when a stratum is absent from a series.
Full diagnostic probability bins, abstention confidence and all branch/budget
strata are retained in the machine-readable analysis, including negative results.

## Evidence and reproduction

The first development and its independent audit were committed before this report.
[Gate](development/gate.json), [summary](development/summary.json),
[manifest](development/manifest.json) and [independent audit](development/audit.json)
retain the admission decision and data identities. Complete ordinary and noisy
replay archives are also retained; identical ordinary blobs are shared by Git.

| Run | Dependent conditions | Protected comparison | Reconstruction |
|---|---:|---|---|
| First development | 82,944 | reference | independent PASS |
| Ordinary replay | 82,944 | PASS, all archive and summary bytes identical | first reconstruction reused by verified identity |
| Last-bit replay | 82,944 | PASS, 3,175,174 numerical differences across 1,340 paths | independent PASS |

Both replays used --runtime-override. All three summary files are byte-identical;
the noisy numerical differences occur in diagnostics and propagated physical
values, not in protected choices, verdicts, integer counters or admission.
See the [ordinary comparison](compare-ordinary.json),
[identity verification](replay-ordinary/identity-verification.json),
[noisy comparison](compare-noise.json) and [noisy audit](replay-noise/audit.json).
Repeating the same series is a reproducibility check, not additional independent
evidence. Each completed experimental run preceded the next one. Read-only
audit and comparison work could overlap a later run.

The auditor independently reconstructs policy choices, posterior updates,
learning, action prices and outcome ledgers. It shares the physical executor;
gate/closure accounting uses the common function over independently reconstructed
primary metrics. This is independent policy reconstruction, not a third physical
model or an independently designed hypothesis-selection criterion.

Both complete test suites passed 648 main plus 30 utility tests:
[ordinary log](logs/check.log) and [one-bit log](logs/check-noise.log).
The first data survived an interrupted environment and were restored with every
compressed and canonical hash checked. A later ordinary replay lost its external
output mount after seven completed series and was restarted in the working
project environment; its [failure log](logs/ordinary-storage-failure.log) and
[recovery record](logs/recovery.txt) are retained. This was a storage interruption,
not a changed parameter or discarded scientific outcome.

Reproduce into fresh output directories:

```sh
python3 -m binding.run_b006 --phase development --output /tmp/b006-repeat --replay binding/results/B006/development --runtime-override
python3 -m binding.run_b006 --phase development --output /tmp/b006-noise --replay binding/results/B006/development --runtime-override --noise
python3 -m binding.audit_b006 binding/results/B006/development
python3 -m binding.audit_b006 binding/results/B006/replay-noise
./scripts/check.sh
./scripts/check-noise.sh
```

Saved-data analysis does not simulate new episodes:

```sh
python3 -m binding.analyze_b006 binding/results/B006/development --output binding/results/B006/analysis.json
```

The failed admission is the stopping result. Confirmation and closure must not be
inferred from development contrasts. No conclusion about consciousness follows.
