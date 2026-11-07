# Development analysis: E005 and E006

This retrospective analysis uses already inspected data, not a held-out sample.
Executed with Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0:
`python3 -m lab.development_analysis --output results/development-verifier/analysis.json`.
FV values are the saved observations; only analytic references and verdicts are
recomputed. Repeated task/grid evidence is deduplicated across ablations/faults.

- 586 distinct experiment/task/grid steps; U covers all 586 errors. Maximum
  |FV-G|/U = 0.6451869355442745 at E006 task 76, grid 512, nonconverging.
- E006 clean task 11 is accurate (0.958087724 tolerance). FV error at 4096 is
  8.830650835e-6 versus U=5.279638783e-5; acceptance would require U<=1.489935353e-5.
  Its recent FV errors shrink approximately fourfold per grid doubling whereas
  U=2Δ allows only twofold convergence. This is sensitivity, not permission to
  tune U to task 11.
- Of 558 guard-passing E006 records, all 212 inaccurate values would receive
  violation_detected at the first decisive step; no accurate value would.
  Three accurate records remain inconclusive. Their aggregate FV cost changes
  from 2,239,616 to 766,080 cells. These are correlated records from 96 tasks.
- Eight accurate rejections split into one clean value and seven accidentally
  accurate corrupted values. Five corrupted accurate values fail the guard
  at the 4096-mode limit; two fail the FV check.
- The original E006 tuple has six input conditions: clean plus five defects,
  not six distinct defects. E007 preserves these exact six conditions.

No formula, threshold, or old data was changed. No claim of certified U coverage
or an independently preregistered E005/E006 trial follows from these observations.
