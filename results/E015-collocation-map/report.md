# E015: independent collocation coverage and abstention map

## Result

The frozen degrees 64 and 128 cover all 21,587 evaluated coordinates on the
seven tolerance planes. There are zero coverage misses and zero dangerous
converging misses among 151,109 point-plane classifications. The maximum
error/U is 0.28860307684076564. These are dependent adaptive samples, not
independent trials or a continuous-domain guarantee.

The map has 13,357 base points, followed by 1,164, 2,357 and 4,709 new points.
The first three levels refine 290, 588 and 1,176 rectangles. Solver values
are shared across the seven fixed planes; no continuous tolerance grid exists.
The fourth pass evaluates the final children without further splitting.

| Tolerance | Nonconverging, necessarily abstain | U > tolerance, no acceptance possible | Oracle-reference inconclusive | Oracle-reference accept |
| --- | ---: | ---: | ---: | ---: |
| 1e-7 | 13,519 | 21,587 | 21,587 | 0 |
| 1e-6 | 12,084 | 21,587 | 21,587 | 0 |
| 1e-5 | 10,593 | 9,837 | 10,593 | 10,994 |
| 1e-4 | 8,834 | 7,500 | 8,834 | 12,753 |
| 1e-3 | 5,491 | 0 | 5,491 | 16,096 |
| 1e-2 | 0 | 0 | 0 | 21,587 |
| 1e-1 | 0 | 0 | 0 | 21,587 |

Every nonconverging point remains covered. Oracle-reference verdicts evaluate
an exact target submitted as the artifact; they do not assert what arbitrary
artifacts or the compound pipeline would do. There are no oracle-reference
violation verdicts. Nonconvergence forces abstention for every artifact.
U > tolerance only prevents acceptance: a sufficiently discrepant artifact can
still receive a violation verdict when the convergence gate passes.

## Rounding and observed boundaries

At 631 coordinates, coverage holds only after adding the rounding term:
2 Delta < error <= U. Their sampled log magnitudes span -5.95 to 3.1 and
phases span both domain endpoints. These are all base nodes, appearing in all
seven planes. R = 1.9073486328125e-6 throughout this map. Thus tolerances
1e-7 and 1e-6 cannot produce acceptance anywhere, regardless of artifact.
The 631 count describes the relative comparison to tiny differences, not
errors of order R or observed failures of the full envelope.

The union-refined phase columns provide first-loss-of-convergence brackets.
The following ranges enclose the leftmost and rightmost endpoints of those
brackets across the 289 sampled phase columns, in log10 |zT|:

| Tolerance | Range of first-loss bracket endpoints |
| --- | --- |
| 1e-7 | 3.825 to 4.09375 |
| 1e-6 | 4.0625 to 4.3125 |
| 1e-5 | 4.36875 to 4.5875 |
| 1e-4 | 4.65625 to 4.99375 |
| 1e-3 | 5.26875 to 6.74375 |

No sampled convergence loss occurs at 1e-2 or 1e-1. These are sampled
brackets, not interpolated boundaries or universal magnitude cutoffs.
Convergence can re-enter: at tolerance 1e-4 converging samples extend to
log magnitude 6.23125 despite earlier nonconvergence. The full
[transition record](boundaries.json.gz) retains every sampled state change,
while [first-loss brackets](first-loss-brackets.json) and the
[plot](first-loss.svg) show only the first loss in each phase column.
Corner-driven refinement cannot detect isolated interior transitions.

The preregistered false-agreement concern did not produce an uncovered
converging sample. Loose-tolerance convergence is not accurate layer resolution:
at zT = 1e12 on the real axis, G128 = 9.156007011382628e-5 versus reference
7.745964692414833e-7, with error 9.07854736445848e-5 and U =
5.513888144990898e-4. The 1e-2 and 1e-1 gates pass and coverage holds, while
all five tighter tolerance gates abstain. No thresholds were retuned.

## Interpretation for the compound pipeline

E013 is known development evidence: its matched two-slab model passed the
finite invariant probes and caused 23 false acceptances under D plus gates;
the third implementation detected those 23. E015 adds direct numerical
coverage evidence for the frozen independent solver at these sampled sphere
coordinates. Where its gate converges and |v - G128| + U <= tolerance,
the observed coverage implies |v - G| <= tolerance by the triangle inequality.
This supports a numerical defense against common implementation/model error,
conditional on the trusted sphere definition and unchanged arithmetic.

It does not validate all possible model faults or all E013 inputs, which are
not these grid nodes. In the nonconverging regions the compound pipeline has
no affirmative independent third-solver verdict and must retain abstention
when its earlier gates would otherwise pass. In the acceptance-impossible
regions it cannot independently approve an artifact, although a violation
may still be detected if converging. Outside evaluated coordinates, between
tolerance planes, beyond this domain, or under a shared wrong mathematical
definition, this map provides no independent guarantee. A base-policy rejection
still remains a rejection rather than being relabeled as third-solver evidence.

## Audit, chronology and reproduction

All 48 dimensionless/cache controls pass. The four reference batches contain
3,999, 116, 235 and 470 audits, totaling 4,820, with zero failures and no fallback.
The largest analytic-versus-60-digit discrepancy is 9.520509789948165e-16.
The exact-coordinate threshold x > 2240 determines mandatory audits.

Protocol commit: 4327b73. Implementation/test commit: f7a42ac. Eight focused
tests pass. The complete repository check passes 109 main tests and 30 vendor
tests; see [check log](check.log). Exact repeat matches every point, flag,
reference audit, refinement and summary, and the retained source hashes match.
The standalone audit also recomputes all 151,109 classifications from values.

An execution bookkeeping error supplied an unnecessary runtime override to
the first run and repeat, claiming the pinned runtime was unavailable even
though the recorded versions match. Both original datasets and metadata are
retained in [the superseded directory](../E015-superseded-override/). Fresh
canonical run and repeat were then executed without an override, using Python
3.12.14, NumPy 2.3.5 and SciPy 1.17.0. Their scientific results match each other
and the superseded run exactly. This was a metadata-label error, not a data
discrepancy or registration violation. Protocol and implementation remained
unchanged. Canonical run wall time was approximately 17 seconds; concurrent
repeat timing is not a standalone performance benchmark.

Run from the repository root, with fresh output directories:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m lab.collocation_map results/E015-new
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m lab.collocation_map results/E015-new-repeat
python3 scripts/audit_e015.py results/E015-new results/E015-new-repeat
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 bash scripts/check.sh
```

The committed map and audit files retain all point coordinates, degrees,
responses, uncertainty components, tolerance flags, region extrema by level,
and parent refinements. Reruns require a clean tracked tree. No solver changes,
production V2 integration, or claims about consciousness are made here.

## Owner's macOS replay observation

The owner repeated E015 with `--runtime-override` on macOS arm64, Python 3.12.13,
NumPy 2.3.5 and SciPy 1.17.0. All principal measures agreed: zero dangerous
misses and uncovered nodes, the same nonconvergence and acceptance-impossible
counts, and the same reference-audit subset. Only the diagnostic count of nodes
whose coverage depends solely on the rounding term differed: 393 rather than
631. That comparison involves rounding-scale quantities and is platform
dependent; it is diagnostic, not a primary safety endpoint. These are reported
owner observations, not a new run performed for this report.
