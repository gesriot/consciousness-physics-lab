# B002 comparison correction

The confirmation comparator now treats the uniform raw step and safety margins as diagnostics only when both protected verdicts are inconclusive and both admission flags are false. This includes a raw accept becoming inconclusive and a numeric decision margin becoming null. The additional diagnostic_guarded_raw_changes counter reports these differing leaves separately.

Protected verdicts, safety.passed, the safety multiplier, grid configuration, cross-instrument convergence_pass, decision_pass and finite flags remain exact. Active raw flags remain exact when the guard does not veto both sides. Prices, selected records, withdrawals, integer totals, summary and closure stay exact; per-cell confirmation and selection fractions are now also exact. Structural mismatches and non-finite numerical drift still fail.

Related comparisons were inspected: qualification already separates unguarded_verdict from protected verdicts and convergence/decision/finite flags; exposure compares operational projections and its protected step flags explicitly. Their semantics were retained. The independent audits intentionally reconstruct raw diagnostics and were not relaxed. No solver, guard, archived trace or B002 outcome was changed.

Validation: 89 binding tests pass, including 6 new comparator regression methods covering protected and raw flips, active flags, cross-instrument flags, prices, selections, totals, schema changes and non-finite values. The fixture exercises both revision attempts and ordinary trace traversal. Tests are recorded in tests.log. Fresh 12-series repeat and numerical-noise comparisons both pass decisions and intake. Repeat has 0 changed numerics; noise has 386010 with maximum absolute difference 9.094947017729282e-13. Noise includes 2106 changed guarded raw diagnostic leaves and 0 raw verdict changes. Results are recorded beside this note.

The reported macOS counts of 1042 differing diagnostic fields and 314 raw verdict changes are owner observations, not a locally reproduced macOS run. The local comparisons use the existing archived datasets and do not replace or amend their original comparison artifacts.

Reproduction commands:

```sh
python3 -m unittest discover -s tests -p 'test_binding*.py' -v
python3 -m binding.revision_b002 --compare binding/results/B002/confirmation-first binding/results/B002/confirmation-repeat --output binding/results/B002/comparison-correction/confirmation-repeat.json
python3 -m binding.revision_b002 --compare binding/results/B002/confirmation-first binding/results/B002/confirmation-noise --output binding/results/B002/comparison-correction/confirmation-noise.json
```
