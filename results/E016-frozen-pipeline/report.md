# E016: held-out confirmation of the frozen pipeline

Both preregistered policies produced zero false acceptances on 240 new requests
and 2,160 paired input conditions. No decisive step missed its intrinsic
numerical uncertainty: 782 distinct V2 task/model/grid steps and 207 distinct
third-solver tasks were covered. Strict acceptance retained 138 of 185 accurate
clean answers; allowing third-solver abstention retained 145. These are sampled
outcomes, not a certified bound or a validation of the trusted physical definition.

## Registration and implementation

The base is 2647515. Owner-reported E014/E015 observations were recorded in
bbbfa93; E014 portability corrections are in 2367b42. The E016 protocol was
committed in 91e4cb7 before implementation or any held-out generation, while
assembly work continued. The pre-outcome clarification a9329b7 fixed original
uniform E008/E010/E013 fingerprint probes, graded V2 only, and separate intrinsic
numerical versus wrong-model errors. It did not use held-out outcomes.

Pipeline assembly is 6a51588; a concurrent audit-script integration mismatch was
corrected separately in 47ffc97 before generation. The independent archived-input
audit 0c78bd2 freshly reproduced all 1,536 E011 C1 paths, including 4,892 numeric
fields exactly, 864 model-gate cases and 288 E013 third-solver decisions.
E011's historical probes used graded volumes; restoring the explicitly requested
original uniform probe arithmetic changed none of these archived gate decisions.
The immutable pipeline parameter manifest documents delegated constants; legacy
solver/probe formulas remain their effective implementations, with regression
checks against the manifest. It is not a second set of runtime tuning controls.

The runner and tests were committed as 9426d8a before generating requests. Both
--replay and --runtime-override were present in that first implementation commit.
No E016 decision rule or parameter changed after generation. The protocol's
clarification, code integration correction and retained negative outcomes are
not hidden as if development had been a single uninterrupted commit.

## Design and reference

Task seed 20261127 and condition-order seed 20261128 generate five cohorts of
48 requests: W broad, H thin layer, M intermediate frequency, O nearly imaginary,
S strict tolerance. All nine inputs and both policies are paired on each request.
The five local defects retain E006 definitions, including the previous clean
modal value as the stale input. The two common-model controls are scaled sphere
and slab; the mixture retains the E013 binary64 component parameters.

There are no exact request duplicates within E016 or matches against saved
E005, E006, E007, E008, E009, E010, E011 and E013 task archives. E012/E014/E015
are development maps rather than independent task intakes. Absence of exact
matches does not establish independence of numerical mechanisms or domains.

The preselected 60-digit audit checked 60 requests using exact rational
log-magnitude coordinates and task IDs, not a floating modulus threshold.
All passed; maximum analytic discrepancy was 4.745320846912229e-16. No
reference replacement was required. Intake and reference audit were persisted
before pipeline outcomes. Intrinsic references for all four solved models were
computed only after the nine component decisions for each request and never
entered decisions. Original-sphere truth alone determines artifact accuracy.

## Primary outcomes by cohort

Every row below represents 48 independent requests and 432 paired inputs.
A/V/I denotes acceptance, violation detected, and inconclusive.

| Cohort | Strict A/V/I | Soft A/V/I | False acceptances strict/soft | Accurate numerical violations | Accurate model violations |
|---|---:|---:|---:|---:|---:|
| W | 208/214/10 | 208/214/10 | 0/0 | 0 | 39 |
| H | 88/125/219 | 110/125/197 | 0/0 | 0 | 96 |
| M | 175/253/4 | 175/253/4 | 0/0 | 0 | 11 |
| O | 155/188/89 | 176/188/68 | 0/0 | 0 | 65 |
| S | 0/399/33 | 0/399/33 | 0/0 | 0 | 5 |

There were 216 violations on individually accurate values: 83 scaled-sphere
and 133 slab values. All were model-check violations, not falsely asserted
numerical errors. All 240 scale controls failed the first moment, second moment
and high-frequency checks; all 240 slabs failed the latter two. Mixtures passed
all four checks. No accurate clean value received a violation. Request binding
passed every input because the registered faults change values/models, not the
trusted original request. This series therefore does not newly stress corrupt
request hashes.

### Decisive numerical coverage

The primary solver-U comparison uses the solved model's intrinsic truth; a
wrong model's bias against the original sphere is a separate quantity. V2's
raw V2_decisive_misses field counts original-sphere discrepancies, so its 439
entries must not be described as intrinsic numerical coverage failures.

| Cohort | V2 decisive evaluations | Distinct V2 steps | Distinct sphere steps | Intrinsic numerical misses | Third decisive evaluations | Distinct third tasks | Third misses |
|---|---:|---:|---:|---:|---:|---:|---:|
| W | 424 | 187 | 49 | 0 | 421 | 47 | 0 |
| H | 146 | 61 | 21 | 0 | 254 | 29 | 0 |
| M | 432 | 193 | 49 | 0 | 432 | 48 | 0 |
| O | 338 | 149 | 39 | 0 | 341 | 38 | 0 |
| S | 426 | 192 | 54 | 0 | 397 | 45 | 0 |

In total, V2 made 1,766 decisive evaluations, comprising 782 distinct
(task, model, grid) steps; 212 of the distinct steps used the original sphere.
Collocation made 1,845 decisive evaluations on 207 distinct tasks. Repeated
faults do not increase the independent sample size. There were zero intrinsic
coverage misses for sphere, scaled sphere, slab or mixture, and zero for
collocation. The 439 original-sphere discrepancies on wrong-model V2 paths
instead express common model bias. For example, all 48 middle-frequency
mixture paths differ from the sphere beyond their small numerical U, but 31
submitted mixtures are still within the much larger task tolerance.

## Acceptance cost and predictions

| Cohort | Accurate clean | Inaccurate clean | Strict accurate clean accepts | Soft accurate clean accepts | Cost of strictness |
|---|---:|---:|---:|---:|---:|
| W | 45 | 3 | 45 | 45 | 0 |
| H | 48 | 0 | 15 | 18 | 3 |
| M | 46 | 2 | 46 | 46 | 0 |
| O | 46 | 2 | 32 | 36 | 4 |
| S | 0 | 48 | 0 | 0 | 0 |

W and M retained all accurate clean answers under both policies. In H the
strict policy accepted 15 and the soft policy 18 of 48 accurate clean answers;
V2 itself abstained on many thin-layer paths, so relaxing collocation did not
recover all remaining answers. In O the corresponding counts were 32 and 36
of 46. The observed cost of strictness is seven accurate clean acceptances,
not a new estimate derived from the development map.

The middle-frequency mixture prediction held: 31 accurate mixtures were
accepted and all 17 inaccurate mixtures received a violation under both
policies; all 48 passed the four finite probes. Across cohorts the mixture
produced 117 strict and 127 soft acceptances, 67 violations, and 56 versus 46
abstentions, with zero false acceptances. The soft policy's ten extra mixture
acceptances were accurate; this does not eliminate the untested risk of common
model error where collocation abstains.

In S all 48 clean modal outputs were inaccurate at the registered 4096-mode
limit. Neither policy accepted any of the 432 input conditions in that cohort.
There is no denominator of accurate clean S answers from which to infer an
acceptance rate. The broad prediction that S would often bind was borne out
more strongly here, without changing the design to make that happen.

### Abstention reasons

Counts below are nonexclusive among each policy's inconclusive rows. Model
check failure is always a violation, hence contributes zero to abstention.

| Cohort | Strict V2/resolution/third | Soft V2/resolution/third |
|---|---:|---:|
| W | 2/10/4 | 2/10/4 |
| H | 197/0/136 | 197/0/114 |
| M | 0/4/0 | 0/4/0 |
| O | 62/6/61 | 62/6/40 |
| S | 0/33/4 | 0/33/4 |

## Rounding-noise diagnostics

Both intergrid differences were no greater than R_n on 88 decisive evaluations:
16 in W, zero in H/M, 16 in O and 56 in S. These represent 33 distinct
(task, model, grid) steps: 6, 0, 0, 6 and 21 respectively. Such decisions
can change across platforms, as the owner's E014 observations demonstrated.
All were covered on this run. C1/C2 flags differed on zero of 3,742 operational
eligible evaluations and zero of 1,766 decisive evaluations. This negative
finding does not contradict E014's differences on late diagnostic grid paths
or authorize a future convergence-rule change without another registration.

## Work

Both policies reuse the same component evaluations. Their incremental decision
cost relative to the common recorded work is zero; do not sum policy totals.
Mixture FV cells and modal terms count both slab components. Probe calls are
fresh per input and include repeated zero-frequency calls. Timings are retained
but are not treated as deterministic outcomes.

| Cohort | Total modal terms | Total FV cells | Third matrix solves | Sum of degrees |
|---|---:|---:|---:|---:|
| W | 10,234,349 | 6,955,008 | 864 | 82,944 |
| H | 10,223,142 | 9,502,720 | 864 | 82,944 |
| M | 10,182,997 | 6,854,656 | 864 | 82,944 |
| O | 10,292,001 | 7,923,712 | 864 | 82,944 |
| S | 11,206,656 | 6,884,352 | 864 | 82,944 |

The total is 52,139,267 modal terms, 38,120,448 FV cells and 4,320
collocation matrix solves with summed degrees 414,720. The modal total includes
122 terms for the one initial stale value, excluded from the cohort subtotals.
Verification used 6,171,648 FV cells; probes used 31,948,800 FV cells and
49,152,000 modal terms. Clean generation, common-model generation and fault
injection used 426,735, 1,706,940 and 853,470 modal terms respectively.

## Reproduction and limits

Both complete runs used the pinned Linux x86_64 Python 3.12.14, NumPy 2.3.5,
SciPy 1.17.0 environment with one BLAS thread, without runtime override. Exact
fresh replay produced zero non-timing scientific mismatches; only environment,
source/git provenance, replay provenance and fields ending in seconds are
excluded by the comparison. All summaries are recalculated from retained rows. A separate raw-row audit
recomputed decisions, coverage, noise and work without the production summary
function. Its independent 80-digit model formulas differed from the stored
60-digit values by at most 1.4168149750966945e-56; no accuracy label changed.
Intakes, 60-digit audit values, full operational paths, probes, collocation
values, conditions, model references, costs and source hashes are retained.

The pre-run check passed 126 laboratory tests and 30 vendor tests. Final check
output is retained separately. Commands from a clean committed checkout:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m lab.pipeline_holdout --output results/local/e016.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m lab.pipeline_holdout --replay results/local/e016.json --output results/local/e016-repeat.json
python3 scripts/audit_e016.py results/local/e016.json results/local/e016-repeat.json --output results/local/e016-audit.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ./scripts/check.sh
```

The sampled compound pipeline met its zero-false-acceptance prediction, but
neither finite testing nor heuristic U establishes universal safety. Forty
accurate clean answers were not accepted even under the soft policy, and
47 under strict. Uniform fingerprints remain finite probes, third collocation
has fixed resolution and a nonzero rounding floor, and the original request,
physical definition and decision logic remain trusted. No component was tuned
from E016 and no inference about consciousness follows.

