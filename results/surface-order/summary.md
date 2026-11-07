# E009: surface resolution and observed-order gate

192 paired tasks; 4608 dependent condition records. Implementation 49b2e5c, preregistration 2f09c38. Oracle is evaluated only after operational decisions. Exact same-platform non-timing repeat passes.

## uniform/frozen

Coverage misses 88/768, decisive 0/189; maximum error/U 3.01102004.
False acceptances 0; violations on accurate 0; inconclusive 303; decision cells 5385216.
Clean broad: accurate accepted 79/87; all tasks 96.
Clean high: accurate accepted 58/91; all tasks 96.
Futility stops 66; accurate shadow acceptances lost 0 (at futility 0); lost shadow violations 31; reversals 0.

Coverage by cohort and actual surface delta/h_s:
- broad:[0,.5): 14/25 misses.
- broad:[.5,1): 0/21 misses.
- broad:[1,2): 0/25 misses.
- broad:[2,4): 0/23 misses.
- broad:[4,infinity): 0/290 misses.
- high:[0,.5): 74/145 misses.
- high:[.5,1): 0/96 misses.
- high:[1,2): 0/96 misses.
- high:[2,4): 0/47 misses.
- high:[4,infinity): 0/0 misses.

## surface/frozen

Coverage misses 0/768, decisive 0/236; maximum error/U 0.204515896.
False acceptances 0; violations on accurate 0; inconclusive 2; decision cells 1271808.
Clean broad: accurate accepted 87/87; all tasks 96.
Clean high: accurate accepted 90/91; all tasks 96.
Futility stops 0; accurate shadow acceptances lost 0 (at futility 0); lost shadow violations 0; reversals 0.

Coverage by cohort and actual surface delta/h_s:
- broad:[0,.5): 0/0 misses.
- broad:[.5,1): 0/0 misses.
- broad:[1,2): 0/3 misses.
- broad:[2,4): 0/10 misses.
- broad:[4,infinity): 0/371 misses.
- high:[0,.5): 0/0 misses.
- high:[.5,1): 0/0 misses.
- high:[1,2): 0/9 misses.
- high:[2,4): 0/65 misses.
- high:[4,infinity): 0/310 misses.

## uniform/order

Coverage misses 88/768, decisive 0/118; maximum error/U 3.01102004.
False acceptances 0; violations on accurate 0; inconclusive 509; decision cells 5218304.
Clean broad: accurate accepted 76/87; all tasks 96.
Clean high: accurate accepted 24/91; all tasks 96.
Futility stops 156; accurate shadow acceptances lost 34 (at futility 34); lost shadow violations 42; reversals 0.

Coverage by cohort and actual surface delta/h_s:
- broad:[0,.5): 14/25 misses.
- broad:[.5,1): 0/21 misses.
- broad:[1,2): 0/25 misses.
- broad:[2,4): 0/23 misses.
- broad:[4,infinity): 0/290 misses.
- high:[0,.5): 74/145 misses.
- high:[.5,1): 0/96 misses.
- high:[1,2): 0/96 misses.
- high:[2,4): 0/47 misses.
- high:[4,infinity): 0/0 misses.

## surface/order

Coverage misses 0/768, decisive 0/207; maximum error/U 0.499701861.
False acceptances 0; violations on accurate 0; inconclusive 0; decision cells 1285120.
Clean broad: accurate accepted 87/87; all tasks 96.
Clean high: accurate accepted 91/91; all tasks 96.
Futility stops 0; accurate shadow acceptances lost 0 (at futility 0); lost shadow violations 0; reversals 0.

Coverage by cohort and actual surface delta/h_s:
- broad:[0,.5): 0/0 misses.
- broad:[.5,1): 0/0 misses.
- broad:[1,2): 0/3 misses.
- broad:[2,4): 0/10 misses.
- broad:[4,infinity): 0/371 misses.
- high:[0,.5): 0/0 misses.
- high:[.5,1): 0/0 misses.
- high:[1,2): 0/9 misses.
- high:[2,4): 0/65 misses.
- high:[4,infinity): 0/310 misses.

## Interpretation and cost

Grading removed observed coverage misses and improved clean acceptance. The order estimator did NOT improve uniform-grid acceptance: its order band withheld many decisions. This negative result is retained without retuning. On the graded mesh the order variant accepted all 178 clean accurate artifacts, versus 177 with the frozen estimate. Zero misses in this sample is not a certified bound or evidence about consciousness.

Actual decision FV cells 13160448; separate shadow cells 3121152; total 16281600. Modal execution terms 241100, additional injection terms 482200; initial stale execution is separately recorded in results.json. Shadow work is not counted as decision savings.

Raw tasks, both mesh paths, all estimator steps, fallback status, strata and paired artifacts are in results.json. Recorded timings are descriptive. Prior E005–E008 outputs were not changed.

## Post-result reporting audit

The supplementary reporting-audit.json derives decisive-step cohort/stratum
counts and maximum ratios from the preserved raw results. It also deduplicates
injection timings across factorial cells and includes initial stale modal terms.
Each mesh's shadow costs 1,560,576 cells. Separate mesh wall times and initial
stale execution time were not recorded; pooled shadow time is available. This
is a measurement limitation, not zero cost. No scientific outcomes were rerun,
replaced or tuned for this reporting correction.
