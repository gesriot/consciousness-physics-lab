# E007: registered three-outcome verifier stress test

This protocol is committed separately BEFORE implementation and generation of
E007 data. E005/E006 are development evidence. No new sample has been generated.

## Frozen design and predictions

Use Python 3.12.14 / NumPy 2.3.5 / SciPy 1.17.0. Task seed 20261114;
order seed 20261115. Generate 192 independent tasks, first 96 broad with
log10|zT| uniform[-6,6], next 96 enriched with log10|zT| uniform[5,6].
Within each task draw log10(T) uniform[-2,2], log10|zT| as above, phase uniform
[-pi/2,pi/2], log10(tol) uniform[-4,-1], in that order. z=|zT|exp(i phase)/T.
Budget4096; planner target .5tol, unchanged from E006. Initial stale input
Request(1,1,.01), updated to the clean answer after each task.

Use exactly E006's six input conditions: clean, conjugate, renormalized,
time_units, near_tolerance, stale_value (clean plus FIVE injected defects;
the task's phrase “six defects” is interpreted as the existing six conditions).
Independently permute fault order, then D/D3 order within each fault using the
order seed. Reuse one clean modal execution per task and share artifacts across
D/D3; record additional injection modes separately.

D is frozen lab.fault_holdout.verify(...,'D'): binary acceptance, same grid
schedule, same planner, final guard, and U. Do not change those modules.
D3 uses grids64,128,256,512,1024,2048,4096; decisions start at512.
At grid n, Δn=|FVn-FVn/2|, Rn=32 eps n² max(1,|FVn|).
Convergence is Δn<=max(.5Δn/2,Rn); U=2Δn+Rn, UNCHANGED.
d=|supplied-FVn|. If converging and d+U<=tol: accept.
If converging and d-U>tol: violation_detected. Otherwise: inconclusive.
Stop at first decisive grid>=512. Final acceptance requires BOTH accept and
the unchanged resolution guard. An accurate guard failure is reported separately,
with clean and accidentally accurate corrupted values distinguished.

Numerical futility heuristic (not a proof of impossibility): at grid1024 or2048,
after testing for decisive verdict, stop inconclusive if U>=4tol AND the last
TWO contraction ratios Δn/Δn/2 and Δn/2/Δn/4 are >=.5.
Zero denominator yields ratio0 if numerator0, otherwise infinity.
This rule targets persistent nonasymptotic uncertainty, not task11, and never
turns an inconclusive into acceptance. Prediction: high-frequency enrichment
increases such stops and saves cells, but may sacrifice accurate acceptances
available on the full path. Report this loss rather than tune the rule.
Prediction: D3 reduces faulty-value decision cells; U coverage may fail when
delta/h is small. Zero misses is NOT assumed or a certified error bound.

## Shadow and measurement

For every task compute all seven FV grids through4096 as a separate shadow
path, never exposed to the decision loop. Shadow artifacts for each supplied
value use the same three-outcome tests but do not stop for futility; record
whether ANY grid would accept, terminal verdict, and first decisive verdict.
Measure accurate early rejects/inconclusives that the full path would accept.
Report shadow cells once per task (8128), separately from actual D and D3
decision work per artifact. Shadow does not count as decision savings and the
total research computation includes it. Time full shadow, D, D3 separately.
Do not reuse the shadow cache inside decisions.

Primary measures: FV error>|U| (U nonnegative), on ALL unique task/grid shadow
steps >=512 and on D/D3 decisive decision steps; false acceptances; D3
violation_detected on accurate values, both all records and guard-passing.
Oracle analytic(original z,T) is evaluated only after decisions; it never
controls stopping, guard, U, or planner. Stratify unique coverage by sampling
cohort AND delta/h=n/sqrt(15|zT|), bins [0,.5),[.5,1),[1,2),[2,4),[4,infinity).
Repeated faults/conditions are not independent coverage samples.
Secondary: accurate inconclusives, clean accurate losses versus accidentally
accurate corruption, decision cells, cells saved versus full path per record,
matched D comparison, futility stops and lost accurate full-path acceptances,
and descriptive wall/kernel times. Independent sampling unit=task (192),
not 2304 correlated condition records. No p-values or population guarantee.

## Audit and reporting

Record git_head of committed implementation, protocol_commit (first introducing
this file), protocol_sha256, all relevant source hashes, environment, seeds,
raw tasks/steps/artifacts, decisions, timing/accounting and summary in results.json.
Commit implementation before data generation, then commit raw results/summary
without squashing. Run tests, reproduce numerical results ignoring runtime/
provenance-only fields, save logs. Preserve every negative result.
Update README and MANUSCRIPT sections10/11 with actual evidence, not predictions.
No conclusions about consciousness. Do not touch separate attention materials.
