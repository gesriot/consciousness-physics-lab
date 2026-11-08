# E012: a coverage map reveals a rounding-triggered failure

## Main finding

The preregistered prediction failed. Among **26,023 evaluated nodes** and
**104,092 eligible (node, grid) pairs**, **10,468 pairs at 4,159 distinct nodes**
were simultaneously converging and uncovered. All 10,468 passed convergence
through its **rounding exception**, not its ordinary ratio threshold.
The frozen numerical check therefore does not support the proposed
all-tolerances coverage implication even at the evaluated nodes.

These are dependent map entries, not independent trials. The full verifier's
request, resolution and invariant gates were not re-evaluated by this coverage
experiment. The result is a failure of the frozen numerical component, not a
claim that all 10,468 entries are actual false acceptances of the entire pipeline.

## Registered design and execution

The base map contains 361 log-magnitude coordinates from -6 to 12 and 37 phases
from -90 to 90 degrees. Its 13,357 nodes gained 1,796, 3,634 and 7,236 nodes
at the three registered refinement levels. Of 12,960 base rectangles, 450
were refined; levels one and two refined 908 and 1,806 child rectangles.
No threshold, estimator or refinement rule changed after seeing the map.

The original sphere FV arithmetic was reused unchanged on grids 64 through
4096. Only mesh geometry constants were cached. The cache-equivalence test
checks identical numerical values. All four eligible grids were evaluated,
even if an operational run would have stopped before reaching them.

Dimensionlessness passed 54 FV and 18 modal comparisons. Maximum absolute
changes under time rescaling were 5.2153e-12 and 1.7347e-18, respectively.
The FV tolerance was the registered conditioning-aware rounding envelope,
not a claim of single-ulp equality; the largest error/envelope was 0.0003281.

The 60-digit reference audit checked **11,848 nodes**, including every
magnitude above 1e8 and the seeded subset of the remainder. There were zero
reference failures and no fallback replacements. Maximum discrepancy was
6.7413e-16. Each batch's audit was saved before its FV paths were evaluated.
Adaptive later batches are outcome-selected by the registered rule; their
random audit subsets are not independent samples of the whole domain.

## Coverage and abstention

| Grid | Node-grid pairs | All misses | Converging misses | Covered, not converging | Max error/U |
|---|---:|---:|---:|---:|---:|
| 512 | 26,023 | 11,940 | 870 | 4,477 | 170.8795 |
| 1024 | 26,023 | 9,354 | 2,103 | 5,510 | 61.1726 |
| 2048 | 26,023 | 8,226 | 3,336 | 5,035 | 21.4040 |
| 4096 | 26,023 | 6,502 | 4,159 | 4,167 | 7.3517 |
| Total | 104,092 | 36,022 | 10,468 | 19,189 | 170.8795 |

The earliest sampled dangerous log-magnitudes were 11.85625, 11.55625,
11.25625 and 10.95 for the four grids, respectively. These are map-node
observations, not certified transition boundaries between nodes.

For all misses, delta/h_s ranged from 0.0017166 to 0.316444 across grids.
The maximum miss layer ratios were 0.316444, 0.314921, 0.315669 and 0.314150.
Dangerous misses occupy much thinner layers: their maximum layer ratios
were 0.00202557, 0.00572238, 0.0161662 and 0.0460004.
The positive ratio-minus-0.5 deficit on *every* miss was at least 1.105713.
Thus the ordinary ratio filter rejected all misses. The exception
Delta <= R nevertheless marked some as converging once the absolute
successive-grid difference became sufficiently small at extreme frequency.
A small difference between two unresolved grids is not an error bound.

The [four-panel map](coverage-map.png) shows all evaluated nodes; it is a
scatter map with adaptive density and no interpolation. The two complete
compressed maps retain every value and flag. Each run's misses.csv lists
all missed coordinates and diagnostics.

## Constructed numerical-gate counterexample

After mapping, a separate illustrative witness was constructed from the
largest error/U dangerous pair. It is not another independent trial and was
not used to retune any rule. At log10|zT|=12 and phase=-pi/2 on grid 512:

- error relative to the sphere is 7.7271835261e-7;
- U is 4.5220066351e-9;
- the successive-difference ratio is approximately 2;
- the rounding exception still declares convergence.

Supply that FV512 value and choose tol=(error+U)/2=3.8862017962e-7.
The unchanged numerical decision loop actually accepts at its first eligible
grid, 512, although the supplied value's true error exceeds tolerance.
This demonstrates failure of the conditional numerical acceptance guarantee.
It does **not** claim acceptance after the full pipeline's other gates,
which this constructed artifact does not exercise.

Full witness: [constructed-witness.json](constructed-witness.json).
Reproduce it with scripts/witness_e012.py on the primary map; the selection
rule and numerical call are retained in that post-map script.

## Repeat, checks and limitations

The full repeat reproduced all 26,023 nodes, FV values, flags, references,
refinement choices and summaries **exactly**, excluding elapsed time and
provenance metadata from scientific equality. Primary and repeat calculation
times were 34.47 and 34.87 seconds on this environment; these are descriptive
single-run timings, not a performance benchmark. The pinned numerical runtime
was used. The full check passed 92 laboratory tests and 30 vendor tests;
the six E012-specific tests also passed separately.

Test-discovery provenance: the 92-test laboratory total included eight then-untracked E013 development smoke tests, including a fixed z=1 evaluation, before E013 preregistration. Thus 92 is the discovered working-tree suite total, not a count attributable solely to the committed E012 tree. These smoke tests did not change the E012 map, its protocol or its thresholds. The original full-suite log is retained unchanged; the E013 report separately records that experiment's preregistration departure.
 The full metadata and source digests are in each map.

The finite map cannot exclude isolated interior transitions or certify the
continuum between sampled nodes. Its adaptive density must not be interpreted
as a probability distribution. No repairs are fitted in this experiment:
the rounding-exception failure is retained as the primary negative finding.
An independently registered change would be needed to assess a replacement.
There are no conclusions about consciousness.

## Reproduction and files

Run from the repository root with clean tracked implementation files:

```sh
python3 -m lab.coverage_map results/E012-coverage-map/primary
python3 -m lab.coverage_map results/E012-coverage-map/repeat
python3 scripts/audit_e012.py results/E012-coverage-map/primary results/E012-coverage-map/repeat results/E012-coverage-map/repeat-comparison.json
python3 scripts/witness_e012.py results/E012-coverage-map/primary/map.json.gz results/E012-coverage-map/constructed-witness.json
./scripts/check.sh
```

Output directories must not already exist, to prevent overwriting evidence.
The original saved directories should be retained; use fresh paths for a new
run. `primary/` and `repeat/` include full compressed maps, per-batch reference
audits, dimensionless checks, summaries, all-miss CSVs and run logs.
`repeat-comparison.json` records the exact comparison, `tests.log` the six
focused tests and `check.log` the full repository check at this stage.

## Сообщение владельца о межплатформенном повторе

Владелец сообщил о проверке на macOS arm64, Python 3.12.13, NumPy 2.3.5,
SciPy 1.17.0. Карта E012 совпала по всем 36 022 промахам покрытия, всем
10 468 опасным промахам и всем флагам сходимости. В уточняющих партиях число
сверяемых узлов составило 1117/1120, 2247/2249 и 4482/4484: сравнение
вещественного модуля с 10⁸ по-разному классифицировало граничные узлы.
Для новых карт используется точная целочисленная координата log10; архив E012
не изменён. Добавлен явный параметр отклонения среды --runtime-override.
E013 с --replay дал PASS: все рабочие решения совпали. Обновлённое сравнение
decisions также дало PASS на прежних повторах E009 и E011. Это сообщение
владельца; исходные macOS-файлы здесь не получены и заново не проверены.
