# Archived-input replay verification

Implementation commit: 51593d0. Both replays ran in the pinned Linux x86_64
runtime, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0. This is a same-platform
verification of the replay implementation, not an independent reproduction of
the owner's macOS observation.

- E009: 192 archived requests, 4,608 operational records. Decisions comparison
  passed with atol = rtol = 1e-12; zero separately listed diagnostic differences.
- E010: 96 archived requests, 288 geometry records. Decisions comparison passed
  with atol = rtol = 1e-12; zero separately listed diagnostic differences.
- A second comparison with atol = rtol = 0 also passed for both studies, with
  zero diagnostic differences. Thus all compared non-timing scientific fields
  match exactly here, while execution provenance differs as expected.
- Protocol identities are unchanged. Provenance differences are the new
  implementation commit, replay metadata and changed source hashes; they are
  retained in the comparison logs. No whole-file exact-reproduction claim is
  made.
- `step1-check.log` records 67 laboratory and 30 solver checks passing.

Raw outputs are `e009-replay.json` and `e010-replay.json`; run summaries are in
`e009-run.log` and `e010-run.log`. Toleranced comparisons are in the matching
`*-comparison.log` files; zero-tolerance comparisons are in `*-zero-tolerance.log`.
The original result files were not overwritten.

Commands from repository root (choose fresh output paths when repeating):

```
python3 -m lab.surface_experiment --replay results/surface-order/results.json --output results/platform-replay/e009-replay.json
python3 -m lab.geometry_invariants --replay results/E010-geometry-invariants/results.json --output results/platform-replay/e010-replay.json
python3 scripts/compare_experiment_runs.py results/surface-order/results.json results/platform-replay/e009-replay.json --mode decisions --rtol 1e-12 --atol 1e-12
python3 scripts/compare_experiment_runs.py results/E010-geometry-invariants/results.json results/platform-replay/e010-replay.json --mode decisions --rtol 1e-12 --atol 1e-12
```

The comparison tolerances are fixed reporting conventions introduced after the
owner's observation; no experimental estimator, threshold or invariant was
changed. Full scientific equivalence still requires inspecting separately
reported diagnostics, not merely the exit status.
