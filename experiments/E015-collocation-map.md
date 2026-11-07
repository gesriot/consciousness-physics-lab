# E015: registered coverage map of the independent collocation solver

This protocol precedes new implementation and evaluations. E013 is known
 development evidence, not a holdout. Do not fit the solver or uncertainty to it.
Use the unchanged lab.radial_collocation degrees 64 and 128, T = 1,
Delta = |G128 - G64|, R = 32 eps 128^4 max(1, |G128|), U = 2 Delta + R.
Convergence requires finite values and Delta <= tolerance / 8.

## Predictions and domain

Prediction: resolved low and intermediate frequencies are covered; refinement
will locate loss of convergence at tighter tolerances. Unresolved degrees can
agree falsely, so convergence need not ensure coverage. Retain any dangerous
miss and any coverage sustained only by the roughly 2e-6 rounding term.
Neither zero dangerous misses nor a smooth sampled boundary proves a theorem.

Use exactly the E012 base nodes: log10|zT| = -6 + i/20 for i = 0..360,
phase = -pi/2 + j*pi/36 for j = 0..36. The three-dimensional map consists of
seven fixed tolerance planes 10^k for k = -7..-1, with shared solver values.
There is no continuous tolerance grid or interpolation claim. Use E012 integer
coordinates and three refinement levels. Compare all seven (converging, covered)
flags at rectangle corners; refine their union, splitting flagged rectangles
into four and deduplicating coordinates globally. Interior-only transitions
can be missed. Each new point gets fresh degree 64 and 128 evaluations.

## Reference, controls and reproducibility

Use E012 analytic reference and its exact-binary64, 60-digit sphere formula.
Audit every exact coordinate x > 2240 (log magnitude > 8), plus floor(10% of
remaining batch size) without replacement, continuous default_rng seed 20261128.
Persist each audit before collocation evaluation. Threshold is
1e-13 + 1e-12 |G60|. Any failed audit replaces all references in that batch
with once-rounded 60-digit values, retaining every discrepancy.

Before mapping check dimensionless equivalence at magnitudes .01, 1, 1e4,
phases 0 and .7, both degrees, and T = .125, 3, 8 against T = 1.
Allowance is each degree's 32 eps degree^4 max(1, |G|). Test cached and freshly
constructed operators give exactly equal responses. Failure stops mapping.
Use one BLAS thread. Retain metadata and runtime override if needed.

Primary endpoint is converging AND |G128 - G| > U on each tolerance plane.
Secondary endpoints are nonconvergence, covered nonconvergence, and coverage
held only by rounding: 2 Delta < error <= U. Also measure U > tolerance,
which alone makes acceptance impossible, and oracle-reference verdicts under
the frozen decision rule. Keep these distinct from a submitted artifact's
actual verdict. Report coordinate extrema and counts by level and plane for
all regions, phase-column first/last samples and transition brackets, not
continuous boundary estimates. Retain full compressed points, audits and
refinement records, including all miss coordinates.

Implement focused formula, negative false-agreement, rounding-only, tolerance,
refinement-union, reference-fallback and exact audit-boundary tests. Commit
protocol, implementation/tests, then runs/report separately. Repeat the entire
map with fresh responses and identical coordinates, and compare every scientific
field exactly, excluding timing/provenance only. Retain complete check logs.
Explain where E013's independent defense is supported on sampled coordinates
and where the compound pipeline lacks independent support. No retuning,
production verifier changes, continuous-domain guarantee or consciousness claim.
