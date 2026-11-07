# E014: development comparison of convergence candidates

This protocol precedes implementation and outcomes. E005–E013 are development
evidence. This experiment selects a candidate, not independent validation.
Frozen uncertainty is U=2 Delta_n+R_n, R_n=32 eps n² max(1,|F_n|).
All candidates share each fresh graded sphere FV path at64..4096 and inspect
all eligible grids512..4096, including grids not operationally reached.

## Candidates and prediction

C0 is frozen E011: ordinary contraction Delta_n<=.5 Delta_previous OR
Delta_n<=R_n. C1 permits ordinary contraction only. C2 permits ordinary
contraction OR rounding with ordinary contraction on the immediately previous
pair of differences on the same path: Delta_previous<=.5 Delta_preprevious.
This is not the previous candidate flag or any earlier successful pair.
C3 permits ordinary contraction OR rounding with delta_layer/h_surface>=1,
where delta_layer=sqrt(kappa/|z|). For sphere kappa=1/(15T).
The physical threshold one requires a surface cell no wider than the diffusion
layer before rounding can bypass contraction. It is a resolution heuristic,
not a coverage theorem, and is fixed without fitting new outcomes.

Rounding alone in C1 is inconclusive, reported as rounding-limit if it reaches
the last grid without another stopping reason. C2/C3 similarly label a terminal
blocked rounding exception. All numerical accept/violation inequalities,
resolution/model/hash gates, first eligible grid and futility rules stay frozen.
Prediction: C1 removes dangerous misses but loses accurate low-frequency
roundoff acceptances; C2/C3 reduce that cost. Retain contrary outcomes.

## Shifted domain and audit

Use log10|q| in[-6,12] and phase in[-pi/2,pi/2], T=1. Interior base nodes are
shifted by half E012 base spacing: log=-6+.025+.05*i, i=0..359, and
phase=-pi/2+pi/72+j*pi/36, j=0..35. Append both original domain boundaries
on each axis, giving362*38 base nodes. This bounded convention uses half-width
boundary cells and does not wrap magnitudes or phases. The integer lattice is
log=-6+x/320 and phase=-pi/2+y*pi/576. Base x nodes0,8,24,..5752,5760;
y nodes0,8,24,..568,576. Refine any rectangle whose four corners differ in
any candidate's complete (converging,covered) vector, splitting four ways for
three levels, globally deduplicating. No interior-transition guarantee.

Use E012 dimensionless and cached-geometry controls. Reference auditing uses
its exact binary64-to-60-digit formula and threshold1e-13+1e-12|G60|.
Audit every x>4480 (log|q|>8), plus floor10% of remaining new sorted nodes
without replacement per batch using continuous default_rng seed20261127.
The coordinate threshold, rather than floating abs(q), avoids boundary noise.
Persist audit before FV evaluation; any failure replaces the entire batch with
once-rounded high-precision references, retaining all discrepancies.

## Development cost replay and selection

Read unchanged archived results/surface-order/results.json, surface/frozen rows,
and results/E011-extrapolation/results.json, all V rows. Never regenerate tasks,
artifacts, probes or inputs. Replay sequential decisions on archived full shadow
values, using only the prefix available at each grid, with unchanged futility
and gates. Check C0 exactly reproduces archived operational values/prefixes and
verdicts; a mismatch blocks selection. For E011 shared models use the solved
model's kappa: scaled sphere1/(15000T), slab1/(3T). These are development costs,
not validation; shadow-based replay measures decision cell counts, not runtime.
Report all candidate verdicts, accurate accepts lost versus C0, newly false
acceptances, gained accurate accepts, violation changes and decision cells,
per experiment and combined. Preserve every row and source byte hashes.

Primary map endpoint is converging AND error>U, with complete coordinates and
candidate flags. Report covered but nonconverging counts and all coverage misses.
Select V2 only among candidates with zero dangerous misses, then minimize total
lost accurate accepts across the two replay sets. Break ties by simplicity in
fixed order C1,C3,C2,C0. If none qualify, select none; do not tune thresholds.
Report new false accepts and violation changes even if they contradict predictions.
This discrete development selection provides no continuous-domain guarantee.

Commit protocol, then implementation/tests, then results/report separately.
Test candidate distinctions with synthetic negative controls, local refinement,
boundary/audit thresholds, prefix replay, futility and gates. Retain compressed
full map, reference audits, replay rows, source/protocol/environment metadata,
exact independent full repeat and non-timing comparison. Record runtime override
if pinned runtime unavailable. Run complete repository checks. No claims about
consciousness. Main manuscript integration follows this experiment.
