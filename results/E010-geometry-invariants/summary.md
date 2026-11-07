# E010: shared mean-preserving geometry errors

## Findings

The preregistered slab and cylinder faults preserve normalization and mean time
while changing the transfer function. Across 96 independent tasks with three
paired geometries, the hash, normalization, first moment and joint E008 checks
accepted all 288 conditions. D accepted 89 clean, 90 slab and 90 cylinder values.
The analytic sphere reference labels 90 clean, 58 slab and 64 cylinder artifacts
accurate at the original task tolerance.

| Policy | Clean accepted / false accepts | Slab accepted / false accepts | Cylinder accepted / false accepts |
|---|---:|---:|---:|
| D | 89 / 0 | 90 / 33 | 90 / 27 |
| Hash alone | 96 / 6 | 96 / 38 | 96 / 32 |
| Normalization alone | 96 / 6 | 96 / 38 | 96 / 32 |
| First moment alone | 96 / 6 | 96 / 38 | 96 / 32 |
| Old joint invariants alone | 96 / 6 | 96 / 38 | 96 / 32 |
| Second moment alone | 96 / 6 | 0 / 0 | 0 / 0 |
| High frequency alone | 96 / 6 | 0 / 0 | 0 / 0 |
| All probes and hash alone | 96 / 6 | 0 / 0 | 0 / 0 |
| D + hash, normalization, first moment, or old joint invariants | 89 / 0 | 90 / 33 | 90 / 27 |
| D + second moment, high frequency, or all checks | 89 / 0 | 0 / 0 | 0 / 0 |

Each new check detected all 192 injected geometry faults and had zero clean
probe false rejections. Each rejected all 70 inaccurate fault artifacts, including
all 60 that D accepted, but also rejected 122 accurate fault artifacts (58 slab,
64 cylinder). Model identity and pointwise task accuracy are different targets.
For each new D combination there are no remaining inaccurate accepted artifacts;
the clean acceptance count is unchanged. D alone rejected one accurate value in
each geometry and five inaccurate values in each faulty geometry.

Neither new check alone catches the six inaccurate clean artifacts: these come
from the task's modal budget/resolution rather than a wrong global geometry.
Definition probes are supplements to task-specific verification, not substitutes.

**Reporting correction:** the raw summary's field `clean_false_rejected` counts
all clean rejections, not just accurate clean artifacts rejected. For D and every
D combination that raw count is 7: six inaccurate clean artifacts correctly
rejected and one accurate clean artifact falsely rejected. Standalone probes
reject none. The preserved archive is not rewritten. `reporting-correction.json`,
derived by `scripts/audit_e010_reporting.py`, provides unambiguous separate counts.

## Probe margins and costs

Maximum clean second-moment relative error was 0.000227235, below allowance .02.
Maximum clean high-frequency error was 0.0217348 (modal) and 0.00369197 (FV),
below allowance .03. The modal finite-frequency probe includes a normalization
tail magnified by sqrt(zT)=100; its agreement is not a certified bound.
Slab second-moment errors were about .160124, cylinder about .066844.
Slab high-frequency errors were .261031 modal/.254811 FV; cylinder errors
.101515 modal/.089286 FV. These exceed the frozen allowances without tuning.

| Work across 288 conditions | Modal terms | FV cells | Recorded seconds |
|---|---:|---:|---:|
| Artifact execution | 373056 | 0 | 0.224876 |
| D verification | 0 | 440320 | 0.187424 |
| Old normalization + first moment | 2359296 | 294912 | 0.080361 |
| Second moment | 2359296 | 2359296 | 0.163288 |
| High frequency | 1179648 | 1179648 | 0.082083 |

Hash comparisons took 0.005044 seconds. The second moment costs 8192 modal terms
and 8192 FV cells per condition, high frequency 4096 and 4096, and old joint
probes 8192 and 1024. The shared zero/shift probes serve normalization and first
moment jointly, not two separately executed copies. All probes run even when D
rejects, and no response is cached. Root constants are cached; their initial
construction is inside the first timed call, not a discarded warm-up. Timings
are descriptive single-run measures, not independent samples or speed claims.

## Scope

The checks derive from the stipulated diffusion PDE, spherical volume weighting,
surface boundary condition and original mean time. They do not borrow a solver's
geometry label as evidence. The original request digest, original T, sphere
specification and verdict logic remain trusted and outside the fault boundary.
Hash checks cannot detect a shared geometry error with unchanged requests.

This fault family tests two alternative dimensions, not arbitrary wrong models.
Faults matching normalization, first moment and second moment evade moment-only
checks. A model can also match the leading high-frequency coefficient, or match
all finitely sampled probes while differing at the task frequency. No completeness,
certification or general correctness claim follows from these results. There is
no conclusion about consciousness.

## Reproduction and evidence

Protocol commit: e5a9c9c. Implementation commit used by both runs: ec296a3.
The full check log records 65 lab tests and 30 vendored solver tests passing.
`results.json` and `repeat.json` retain every task, digest, hook dimension, D step,
probe, standalone/combined verdict, oracle error, timing, cost and provenance.
`run.log` and `repeat.log` retain summaries; `reproduction.log` records identical
provenance and PASS for exact non-timing comparison. Pinned Python 3.12.14,
NumPy 2.3.5, SciPy 1.17.0. No production source, attention state or old result
archive was changed. No output-dependent thresholds or parameters were adjusted.

Commands from repository root:

```
bash scripts/check.sh
python3 -m lab.geometry_invariants --output /tmp/e010-new.json
python3 scripts/compare_experiment_runs.py results/E010-geometry-invariants/results.json /tmp/e010-new.json --ignore-git-head
python3 scripts/audit_e010_reporting.py results/E010-geometry-invariants/results.json
```

Use a new output path. The exact repeat in this archive was generated at the
same implementation commit; the later reproduction command permits only the
expected metadata change in the enclosing results/documentation commit.

### Cross-platform comparison convention

The owner reports a macOS arm64 replay with Python 3.12.13, NumPy 2.3.5 and
SciPy 1.17.0: decisions unchanged, numerical differences at rounding level.
The raw owner replay was not supplied for independent inspection. The earlier
relative-only comparison rejected near-zero numerical fields despite matching
decisions. For the revised comparison use `--mode decisions --atol 1e-12
--rtol 1e-12`: finite floats satisfy |a-b| ≤ atol + rtol max(|a|,|b|), while
operational verdicts, flags and integer counters remain exact. Provenance is
separate. These are reporting tolerances, not changes to invariant thresholds.
A pass is conditional operational reproducibility, not a bitwise or general
cross-platform guarantee. The E009 report documents the shadow-path diagnostic
separation used for experiments with those registered paths.
