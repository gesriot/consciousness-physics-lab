# E009: nested surface refinement and observed-order uncertainty

This protocol must be committed before implementation and new E009 data.
E005–E008 are development data. No task outcomes for the seeds below have been
viewed. All constants below remain fixed even if coverage or acceptance worsens.

## Frozen factorial design and numerical definition

Four paired cells: uniform/frozen, surface/frozen, uniform/order, surface/order.
Uniform is exactly the original finite-volume solver. Surface uses 64 base cells
with faces r_j=(1-exp(-4*j/64))/(1-exp(-4)), j=0..64. Every refinement splits
EACH physical cell at its midpoint, not merely at a mapped-coordinate midpoint.
Thus every grid is nested, n=64..4096, and the base grading coefficient is 4.
Use midpoint cell centres, exact spherical shell volumes, face-area conductance
divided by centre distance, and a half-cell Dirichlet surface distance. This is
a second-order family under uniform subdivision of its fixed base mesh.
The coefficient 4 is an a-priori moderate grading: surface width at 4096 is
about 1.9e-5, versus the development range minimum layer width 2.58e-4. It is not
fit to individual tasks. The max/min cell ratio is about 52 and is held fixed.
Test order against the analytic sphere BEFORE the E009 experiment, at fixed
positive real and complex zT=1,10,100; tests are not a new parameter search.

Frozen uncertainty and convergence are unchanged: Delta_n<=max(.5 Delta_n/2,Rn),
U=2 Delta_n+Rn, Rn=32 eps n^2 max(1,abs(FV_n)). Observed-order variant uses
p=log2(Delta_n/2/Delta_n), band [1.5,2.5], F=2,
U=F Delta_n/(2^p-1)+Rn. The band is symmetric around formal order 2; F=2
provides a factor-two Richardson margin without claiming certification. These
choices use scheme order and the development lesson of unresolved layers only.
If either difference is zero, p is undefined, or p<=0, report p=null where
undefined and use the frozen U as a diagnostic fallback, NEVER a decisive
order verdict. For finite p>0 outside the band retain the order U for coverage,
but allow no decisive verdict. No roundoff exception to the registered band.
Roundoff floor is the old heuristic, not a proof on the graded mesh.

Keep D3's three verdicts, first decision at512, unchanged futility ratios and
thresholds at1024/2048, resolution guard and .5tol planner. In the order cell
only U and convergence qualification change. Do not alter production D3.
Predictions: grading removes low-delta/h coverage misses and improves high
cohort clean acceptance; observed order improves acceptance but may cause
coverage misses. Failure of either prediction is retained, not retuned.

## Sampling and measurement

Seeds 20261118 (tasks), 20261119 (order). 192 independent tasks: 96 broad
log10|zT| uniform[-6,6], 96 high uniform[5,6]. Draw log10T uniform[-2,2],
log10|zT|, phase uniform[-pi/2,pi/2], log10tol uniform[-4,-1] in that order.
Same six E007 inputs: clean, conjugate, renormalized, time_units, near_tolerance,
stale_value. Permute faults and then four factorial cells per fault. One modal
execution per task, shared fault artifact per input across all four cells.
Initial stale request (1,1,.01); update from clean receipt after each task.

All operational decisions precede oracle and shadow evaluation. Compute separate
full shadow to4096 for EACH mesh once per task, never reuse inside decisions.
Analyze each estimator on that mesh's shadow values. Independent unit is task,
not 4608 records or repeated grids. Coverage is 768 unique task/grid steps PER
factorial cell, and separately unique decisive task/grid steps (union of input
conditions). Record misses/counts/max ratio by cohort and delta/h_s bins
[0,.5),[.5,1),[1,2),[2,4),[4,infinity), h_s actual surface cell width and
delta=1/sqrt(15|zT|). Record fallback and convergence qualification explicitly.
Primary: coverage misses all/decisive, false acceptance, violation_detected on
accurate artifacts (also guard-passing). Secondary: clean accurate acceptance
per cohort, inconclusive, decision cells, futility stops and accurate acceptances
lost relative to any shadow accept, with losses specifically at futility stops.
Report both mesh shadow costs, injections, modal execution, decision work and
wall times; no shadow savings counted as operational savings. No p-values.

## Audit

Pinned Python3.12.14, NumPy2.3.5, SciPy1.17.0. Record platform/BLAS metadata,
protocol commit/hash, source hashes and committed implementation head. Commit
protocol, then implementation/tests before experiment, then data and observations
separately. Preserve full tasks, steps, shadow, raw rows, summaries and negatives.
Save check log and exact same-platform repeat comparison. Never use analytic
in operational decisions. No conclusions about consciousness or completeness.
