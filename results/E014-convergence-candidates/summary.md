# E014: convergence-candidate development comparison

V2 is C1: require ordinary contraction Delta_n <= 0.5 Delta_previous. The rounding envelope remains in U, but rounding alone cannot qualify a decision. This is a development selection, not validation or a continuous-domain certificate.

## Map and reference audit

The bounded half-step-shifted map evaluated 62,451 distinct points and 249,804 eligible point/grid steps. The four batches contained 13,756, 4,744, 12,256, 31,695 points. All candidates share every FV value and uncertainty.

The 60-digit audits inspected 15,766 points, with zero failed references and no fallback. Dimensionless checks passed. Refinement compares all candidate flags and coverage, so the node set is a development-dependent set, not independent sampling.

| Candidate | Dangerous misses at 512 | 1024 | 2048 | 4096 | Total |
|---|---:|---:|---:|---:|---:|
| C0 | 857 | 2,124 | 3,391 | 4,343 | 10,715 |
| C1 | 0 | 0 | 0 | 0 | 0 |
| C2 | 0 | 0 | 0 | 0 | 0 |
| C3 | 0 | 0 | 0 | 0 | 0 |

A dangerous miss is a converging step whose reference error exceeds U. Every ordinary-only C1 step is covered at these nodes. C2 and C3 also have zero dangerous misses. The complete compressed maps preserve every coverage miss, flag, value, coordinate, layer ratio and parent refinement rectangle. The audit retains a separate dangerous-miss list.

## Archived operational cost replay

No saved intake, artifact, probe or reference was regenerated. Replay uses the exact archived full paths, consumes only the available prefix at each grid, and preserves the original futility and gates. C0 exactly reproduced all 2,688 original rows, including operational FV prefixes, step evidence, decision cells and verdicts.

All four candidates produced the same operational results in both replay sets:

| Development archive | Rows | Accurate accepts | False accepts | Decision cells |
|---|---:|---:|---:|---:|
| E009 | 1,152 | 721 | 0 | 1,271,808 |
| E011 | 1,536 | 560 | 0 | 3,286,016 |
| combined | 2,688 | 1,281 | 0 | 4,557,824 |

Each C1/C2/C3 replay lost zero accurate accepts, gained zero, created zero false accepts, and changed zero numerical or combined violation verdicts. The prediction that C1 would lose low-frequency roundoff acceptances was not supported by these operational archives. Roundoff differences on later shadow grids need not affect an already stopped operation. Decision-cell totals are replay cost measures, not fresh runtime measurements.

## Selection, reproducibility and limits

C0 is excluded by its dangerous misses. C1, C2 and C3 tie on zero lost accurate accepts, so the registered simplicity order selects C1. There was no threshold adjustment or candidate tuning. C3’s layer-width threshold remains a physical heuristic, not a proven error bound.

Protocol registration: c298b26; typography-only correction before evaluation: 932a221; implementation and tests: ad257ce. Both complete runs use the pinned Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0 environment. Their metadata records source hashes, protocol hash, platform and BLAS details. Source archive byte hashes are stored in each cost replay.

The exact repeat audit compares all points, references, candidate flags, refinements, batch audits, cost replay rows, summaries and selection, excluding elapsed time and runtime provenance. Every exact comparison passed; see audit.json and audit.log. Full repository checks passed: 99 laboratory tests and 30 vendor tests.

Files: first/ and repeat/ contain full compressed maps and cost replays, compressed reference batches, dimensionless checks and summaries. first-run.log and repeat-run.log retain execution output. check.log retains the complete repository check.

Adaptive corner testing cannot exclude isolated interior transitions. The shifted map and historical replay are dependent development evidence. Independent held-out validation must follow selection. No conclusions about consciousness follow from these numerical results.

## Owner's macOS replay observation

The owner repeated the experiment on macOS arm64 with Python 3.12.13,
NumPy 2.3.5 and SciPy 1.17.0. The unmodified run stopped during cost replay:
exact equality of recomputed `uncertainty_estimate` and `discrepancy` failed
because complex magnitude (`hypot`) can differ in its last bit across platforms.
The map had not yet been written, so that failure also prevented its retention.

With numerical comparisons relaxed to 1e-12 while flags and verdicts remained
exact, the principal result agreed: C0 had 10,715 dangerous misses, C1–C3 had
zero, C1 was selected, and none of the 2,688 cost-replay decisions changed.
The adaptive map nevertheless differed: 53,304 macOS nodes versus 62,451 Linux
nodes. On 41,482 common nodes, C0 flags and coverage agreed, but C1 flags differed
on 6,209 steps and C2 flags on 74 steps.

In the fully resolved region both intergrid differences can be rounding noise.
For example, at |zT| = 1e-5 and grid 4,096 the Linux differences were
2.04e-11 and 1.97e-11, giving a ratio of 1.03, whereas macOS noise produced a
ratio no greater than 0.5. All these steps were covered: the observed coverage
safety result did not change. C1 convergence and therefore adaptive refinement
can nevertheless be controlled by rounding noise in this region. This is an
owner-reported cross-platform observation, not an independent random sample or
a reason to modify the selected C1 rule after seeing these results.

### Portability correction

Cost replay now compares finite floating measurements with
|a - b| <= 1e-12 + 1e-12 max(|a|, |b|). Flags, verdicts, cell counts, step
counts and archived input prefixes remain exact. The legacy counter
`baseline_rows_exact` denotes exact discrete reproduction, not bitwise equality
of floating measurements; new replay output records this distinction explicitly.
The completed map is written before the cost audit, and `--runtime-override`
records an explicit reason through the shared environment gate. No candidate
rule, uncertainty formula, refinement rule or selection criterion changes.

The correction passes eight focused tests, including rejection of changed
flags and nonfinite measurements and preservation of a map after simulated
cost failure. Replaying all 2,688 archived inputs retains 1,281 accurate accepts,
zero false accepts and 4,557,824 decision cells for each candidate. The retained
portability-fix cost summary and test log document this check; it is not a newly
generated map or a replacement for the owner's macOS observation.
