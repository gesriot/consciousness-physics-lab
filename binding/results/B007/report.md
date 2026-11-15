# B007: corrected repetition and failed reliability learning

## Decision and scope

All four preregistered configurations failed development admission because learned reliability did not move closer to truth in both moderate-error environments. No configuration was selected, no confirmation cohort was run, and no third correction is attempted. The reliability-model question in this laboratory is closed by the task boundary, not by a proof that reliability models cannot work. Neither general superiority nor the formal confirmatory closure criterion has been established.

The first development contains 12 independent seeded series, 32 episodes per series, four paired environments, three budgets and 26 policies: 119,808 dependent episode conditions. Policies, environments, budgets and episodes within a series are not independent replicates. Aggregated counts below describe these development conditions; they are not estimates from a fresh confirmation sample.

## What changed and what passed

The correction retains D's candidate queue, requires direct BAD evidence before accusation and direct GOOD before clearing, aligns action scores with decision thresholds, looks ahead at most two reports, and retains informative fallback. Corroborating reports remain eligible even at a certain posterior. D and K can also repeat an exhausted conflict observation. Consequently the contrast is less confounded by available actions than in B006. All four configurations passed low_zero safety on false accusations and lost-good counts, with zero unsupported accusations and zero premature stops. This admission check deliberately does not demand equal cost or equal residual error; it is not the old B006 four-metric gate.

D made 1,225 false accusations in moderate environments. K3of5 was the selected constant rule and the strongest constant comparator under the registered ranking. Every P differed in real actions and repetition patterns from D, R and K3of5; placebo comparisons had zero differences. Active near-error-free interventions changed actions well beyond the registered 10% floor. Px uses probabilities 0.999999/0.000001, not literally error-free likelihoods. Pstar uses evaluator-provided true rates as a privileged control, never as an admissible learned policy.

## Low_zero cost and residual error by cell

The safety gate above does not constrain remaining_bad or cost. The complete low_zero budget-by-branch comparison follows for D and all four P configurations. Each row pools 12 paired series; these counts must not be interpreted as independent repetitions. Neither a cost increase nor a residual-error increase in this table was removed or used to retune admission after outcomes.

| Budget | Signal | Policy | Remaining bad | Cost |
|---:|---|---|---:|---:|
| 2 | measurement | D | 698 | 24585 |
| 2 | measurement | P_beta2_t80 | 826 | 25223 |
| 2 | measurement | P_beta2_t95 | 956 | 25325 |
| 2 | measurement | P_beta4_t80 | 754 | 23989 |
| 2 | measurement | P_beta4_t95 | 785 | 23732 |
| 2 | conflict | D | 790 | 23906 |
| 2 | conflict | P_beta2_t80 | 993 | 24581 |
| 2 | conflict | P_beta2_t95 | 994 | 24924 |
| 2 | conflict | P_beta4_t80 | 931 | 24269 |
| 2 | conflict | P_beta4_t95 | 956 | 24556 |
| 8 | measurement | D | 9 | 81438 |
| 8 | measurement | P_beta2_t80 | 30 | 88436 |
| 8 | measurement | P_beta2_t95 | 125 | 92217 |
| 8 | measurement | P_beta4_t80 | 12 | 83651 |
| 8 | measurement | P_beta4_t95 | 55 | 89548 |
| 8 | conflict | D | 225 | 88489 |
| 8 | conflict | P_beta2_t80 | 296 | 93741 |
| 8 | conflict | P_beta2_t95 | 340 | 95465 |
| 8 | conflict | P_beta4_t80 | 267 | 90980 |
| 8 | conflict | P_beta4_t95 | 293 | 94300 |
| 32 | measurement | D | 0 | 82176 |
| 32 | measurement | P_beta2_t80 | 0 | 89184 |
| 32 | measurement | P_beta2_t95 | 0 | 100254 |
| 32 | measurement | P_beta4_t80 | 0 | 83954 |
| 32 | measurement | P_beta4_t95 | 0 | 92032 |
| 32 | conflict | D | 0 | 105210 |
| 32 | conflict | P_beta2_t80 | 0 | 116951 |
| 32 | conflict | P_beta2_t95 | 0 | 126055 |
| 32 | conflict | P_beta4_t80 | 0 | 110592 |
| 32 | conflict | P_beta4_t95 | 0 | 118692 |

## Reliability gate

The registered operational measure is mean squared error over six channel rates, series and budgets separately in each moderate environment. It averages squared errors, not the square of pooled mean error. Both environments had to improve; the following table includes all configurations, not a selected winner.

| Configuration | Initial MSE | Final low_moderate | Final high_moderate | Px action changes / 2058 |
|---|---:|---:|---:|---:|
| beta2_t80 | 0.037683333 | 0.035543048 | 0.056281600 | 1776 |
| beta2_t95 | 0.037683333 | 0.026323773 | 0.038690953 | 1983 |
| beta4_t80 | 0.004350000 | 0.008814562 | 0.019950908 | 1730 |
| beta4_t95 | 0.004350000 | 0.005778437 | 0.008605666 | 1898 |

Both beta2 variants improve low_moderate but worsen high_moderate. Both beta4 variants worsen both environments. A more permissive aggregate across environments would conceal failures and was not substituted after outcomes.

The next table gives final mean rates, with initial and true values for comparison. TP means a BAD report conditional on a bad target; FP means a BAD report conditional on a good target. These are descriptive means over paired conditions, not independent calibration trials.

| Configuration | Environment | Channel | Rate | Initial | Final mean | Truth |
|---|---|---|---|---:|---:|---:|
| beta2_t80 | high_moderate | check_claim | fp | 0.333333 | 0.224102 | 0.180000 |
| beta2_t80 | high_moderate | check_claim | tp | 0.666667 | 0.688092 | 0.880000 |
| beta2_t80 | high_moderate | check_source | fp | 0.333333 | 0.249373 | 0.120000 |
| beta2_t80 | high_moderate | check_source | tp | 0.666667 | 0.829625 | 0.820000 |
| beta2_t80 | high_moderate | repeat_observation | fp | 0.333333 | 0.200952 | 0.150000 |
| beta2_t80 | high_moderate | repeat_observation | tp | 0.666667 | 0.472019 | 0.900000 |
| beta2_t80 | low_moderate | check_claim | fp | 0.333333 | 0.237927 | 0.180000 |
| beta2_t80 | low_moderate | check_claim | tp | 0.666667 | 0.790642 | 0.880000 |
| beta2_t80 | low_moderate | check_source | fp | 0.333333 | 0.254981 | 0.120000 |
| beta2_t80 | low_moderate | check_source | tp | 0.666667 | 0.878865 | 0.820000 |
| beta2_t80 | low_moderate | repeat_observation | fp | 0.333333 | 0.255535 | 0.150000 |
| beta2_t80 | low_moderate | repeat_observation | tp | 0.666667 | 0.580505 | 0.900000 |
| beta2_t95 | high_moderate | check_claim | fp | 0.333333 | 0.230684 | 0.180000 |
| beta2_t95 | high_moderate | check_claim | tp | 0.666667 | 0.697346 | 0.880000 |
| beta2_t95 | high_moderate | check_source | fp | 0.333333 | 0.213081 | 0.120000 |
| beta2_t95 | high_moderate | check_source | tp | 0.666667 | 0.813510 | 0.820000 |
| beta2_t95 | high_moderate | repeat_observation | fp | 0.333333 | 0.227084 | 0.150000 |
| beta2_t95 | high_moderate | repeat_observation | tp | 0.666667 | 0.564442 | 0.900000 |
| beta2_t95 | low_moderate | check_claim | fp | 0.333333 | 0.235835 | 0.180000 |
| beta2_t95 | low_moderate | check_claim | tp | 0.666667 | 0.792348 | 0.880000 |
| beta2_t95 | low_moderate | check_source | fp | 0.333333 | 0.214535 | 0.120000 |
| beta2_t95 | low_moderate | check_source | tp | 0.666667 | 0.853006 | 0.820000 |
| beta2_t95 | low_moderate | repeat_observation | fp | 0.333333 | 0.264187 | 0.150000 |
| beta2_t95 | low_moderate | repeat_observation | tp | 0.666667 | 0.633890 | 0.900000 |
| beta4_t80 | high_moderate | check_claim | fp | 0.200000 | 0.176330 | 0.180000 |
| beta4_t80 | high_moderate | check_claim | tp | 0.800000 | 0.777580 | 0.880000 |
| beta4_t80 | high_moderate | check_source | fp | 0.200000 | 0.142280 | 0.120000 |
| beta4_t80 | high_moderate | check_source | tp | 0.800000 | 0.856083 | 0.820000 |
| beta4_t80 | high_moderate | repeat_observation | fp | 0.200000 | 0.143525 | 0.150000 |
| beta4_t80 | high_moderate | repeat_observation | tp | 0.800000 | 0.636286 | 0.900000 |
| beta4_t80 | low_moderate | check_claim | fp | 0.200000 | 0.187225 | 0.180000 |
| beta4_t80 | low_moderate | check_claim | tp | 0.800000 | 0.855439 | 0.880000 |
| beta4_t80 | low_moderate | check_source | fp | 0.200000 | 0.168769 | 0.120000 |
| beta4_t80 | low_moderate | check_source | tp | 0.800000 | 0.902750 | 0.820000 |
| beta4_t80 | low_moderate | repeat_observation | fp | 0.200000 | 0.181459 | 0.150000 |
| beta4_t80 | low_moderate | repeat_observation | tp | 0.800000 | 0.749839 | 0.900000 |
| beta4_t95 | high_moderate | check_claim | fp | 0.200000 | 0.174415 | 0.180000 |
| beta4_t95 | high_moderate | check_claim | tp | 0.800000 | 0.806979 | 0.880000 |
| beta4_t95 | high_moderate | check_source | fp | 0.200000 | 0.138778 | 0.120000 |
| beta4_t95 | high_moderate | check_source | tp | 0.800000 | 0.875334 | 0.820000 |
| beta4_t95 | high_moderate | repeat_observation | fp | 0.200000 | 0.150368 | 0.150000 |
| beta4_t95 | high_moderate | repeat_observation | tp | 0.800000 | 0.741345 | 0.900000 |
| beta4_t95 | low_moderate | check_claim | fp | 0.200000 | 0.183788 | 0.180000 |
| beta4_t95 | low_moderate | check_claim | tp | 0.800000 | 0.858240 | 0.880000 |
| beta4_t95 | low_moderate | check_source | fp | 0.200000 | 0.137500 | 0.120000 |
| beta4_t95 | low_moderate | check_source | tp | 0.800000 | 0.887936 | 0.820000 |
| beta4_t95 | low_moderate | repeat_observation | fp | 0.200000 | 0.181785 | 0.150000 |
| beta4_t95 | low_moderate | repeat_observation | tp | 0.800000 | 0.782032 | 0.900000 |

In particular, repeat-observation TP in high_moderate moves away from the true 0.9 in every configuration. Learning only after observable resolution prevents training on an unverified confidence statement, but does not make posterior responsibilities true labels. Resolution is policy-selected and still noisy; self-training can reinforce an incorrect explanation. This mechanism is compatible with the observed drift, not a separately isolated causal explanation.

| Configuration | Environment | Resolved episode-conditions | Unresolved episode-conditions |
|---|---|---:|---:|
| beta2_t80 | low_moderate | 441 | 588 |
| beta2_t80 | high_moderate | 317 | 712 |
| beta2_t95 | low_moderate | 443 | 586 |
| beta2_t95 | high_moderate | 350 | 679 |
| beta4_t80 | low_moderate | 491 | 538 |
| beta4_t80 | high_moderate | 476 | 553 |
| beta4_t95 | low_moderate | 499 | 530 |
| beta4_t95 | high_moderate | 450 | 579 |

The saved field named resolved_series_conditions is an episode-condition count, not a count of independent series. Each row's resolved plus unresolved count is 1,029 signaled episode-conditions.

## Performance and the price of caution

Totals below pool both moderate environments, all budgets and branches. Such totals expose tradeoffs but cannot establish cellwise dominance. Lost-good is zero for every row because repair is certified by the laboratory, not because a policy learned immunity to harmful repairs.

| Policy | False accusations | Remaining bad | Cost | Repeated checks |
|---|---:|---:|---:|---:|
| D | 1225 | 6005 | 552157 | 0 |
| R | 462 | 6780 | 597503 | 71 |
| K3of5 | 393 | 5023 | 1073356 | 29732 |
| P_beta2_t80 | 54 | 4823 | 1373767 | 7808 |
| P0_beta2_t80 | 3 | 5140 | 1374915 | 8850 |
| Pstar_beta2_t80 | 110 | 4088 | 1023584 | 4940 |
| P_beta2_t95 | 20 | 4595 | 1327494 | 13606 |
| P0_beta2_t95 | 0 | 5040 | 1520918 | 19570 |
| Pstar_beta2_t95 | 16 | 3950 | 1118012 | 8928 |
| P_beta4_t80 | 92 | 4331 | 1059107 | 5350 |
| P0_beta4_t80 | 26 | 4375 | 1117295 | 6026 |
| Pstar_beta4_t80 | 110 | 4088 | 1023584 | 4940 |
| P_beta4_t95 | 34 | 4111 | 1127821 | 9319 |
| P0_beta4_t95 | 9 | 4252 | 1212906 | 11436 |
| Pstar_beta4_t95 | 16 | 3950 | 1118012 | 8928 |

Every P reduces false accusations relative to D, but spends more. Relative to its own P0, every P leaves fewer bad records and makes more false accusations; cost falls for all four. Learning changes the balance rather than providing uniform improvement. Pstar is not universally safer than P: at threshold 0.8 it makes 110 false accusations, more than either learned counterpart, while leaving fewer bad records. At threshold 0.95 it makes 16. Exact knowledge of channel rates does not remove finite-data ambiguity or make a policy optimal. Pstar's two priors coincide within each threshold because its active rates are overridden and frozen; these are duplicated controls, not independent replications.

The saved [analysis](analysis.json) contains every P comparison against D, R, K3of5, matched P0 and matched Pstar, as well as calibration, abstention, depth and scope breakdowns. Brier summaries exclude abstentions while reporting them separately; the scoring sample must not be confused with all episodes.

## Evidence and reproduction

The [protocol](../../protocols/B007-reliability-correction.md) and its amendments were registered before scientific data; no post-outcome changes were made. The first [manifest](development/manifest.json), [gate](development/gate.json), [summary](development/summary.json) and [independent audit](development/audit.json) are retained. Audit passed all 119,808 conditions. Saved-data analysis authenticates archives and embedded intake without rerunning policies. Ordinary and numerical-noise replay are pending; their status will be finalized after comparison and audit. Full repeat archives are not retained; replay manifests, comparisons and logs provide compact evidence.

B006 portability fixes authenticate saved canonical intake, regenerate only for numerical diagnostics and reject structural/discrete drift. Runtime overrides do not authorize changing source or protocol. The separate portability record preserves the owner's macOS observations; it does not rewrite historical outcomes.

No result here tests attention mechanisms or establishes consciousness. The result concerns fallible diagnosis with a fixed candidate queue, three report channels, certified repairs and limited budgets.
