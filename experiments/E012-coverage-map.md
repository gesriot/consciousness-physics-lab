# E012: registered coverage map of the frozen sphere verifier

This protocol precedes map evaluation. E007–E011 are development evidence.
No estimator, threshold, mesh, or refinement criterion will be fitted to outcomes.
Prediction: no converging, uncovered eligible steps. A counterexample is the
primary result and will be retained with its coordinates.

## Domain and deterministic refinement

Set T=1 and q=zT. Base log10|q|=-6+i/20, i=0..360, phase=-pi/2+j*pi/36,
j=0..36: 13,357 points. Evaluate frozen E011 meshes 64..4096; eligible n>=512.
Use the unchanged E011 sphere FV solver, U=2|F_n-F_(n/2)|+R,
R=32 eps n² max(1,|F_n|), convergence Delta_n<=max(.5 Delta_previous,R).
All eligible steps are inspected, including those an operational run never
reaches. This endpoint is not a count of actual decisions.

Use an integer lattice with eight subdivisions per base interval. Compare each
rectangle's four corners' complete (converging,covered) vectors. If any component
differs, split into four, adding missing edge midpoints and centre. Repeat on
children for at most three levels; unflagged rectangles are not refined.
Deduplicate globally. Record all coordinates and parent rectangles. This corner
test cannot exclude isolated interior transitions.

Before mapping, test dimensionless equivalence for magnitudes .01,1,1e4, phases
0,.7 and T=1 versus T'=.125,3,8: modal4096 and FV64,512,4096. Modal allowance
128 eps max(1,|F|); FV allowance is the frozen rounding envelope 32 eps n²
max(1,|F|), accounting for conditioning. Record actual discrepancies. Failure
stops mapping, not a threshold adjustment. Cache identical mesh geometry only;
never reuse response values across repeats. Test cached/uncached exact equality.

## Reference and provenance

Reference is analytic. Audit every |q|>1e8 point against the direct 60-digit
sphere formula, plus a 10% random subset of other points in each new sorted
coordinate batch. NumPy default_rng seed20261124 continues across batches;
select floor(.1*N) without replacement. Convert binary64 inputs exactly.
Threshold: 1e-13+1e-12*|G60|. If any audited point fails, evaluate G60 for every
point in that batch and use it rounded once, recording all replacements.
Persist the batch audit before FV evaluation. Refinement uses prior coverage,
not reference errors. Retain coordinates, q, all FV values/steps, metadata,
protocol/source hashes, references and discrepancies. Runtime deviations require
an explicit reason.

## Endpoints and interpretation

Primary: count and coordinates of (point,n) with convergence and |F_n-G|>U.
Secondary: all coverage misses with delta/h_s, delta=sqrt(1/(15|q|)); ratio
Delta_n/Delta_previous and signed deficit ratio-.5 on misses; covered but not
converging region; counts by grid/level; layer-ratio extrema. Positive deficit
means the ordinary ratio threshold was missed; record rounding exceptions too.
Zero denominators yield null ratio and an explicit flag. Retain full miss list.

Zero dangerous misses would imply by triangle inequality that at tested nodes,
convergence and |v-F_n|+U<=tol entail |v-G|<=tol, for any v and positive tol.
This is restricted to evaluated nodes, implementation and arithmetic, not a
continuous-domain proof. Adaptive nodes are dependent, not independent trials.
No conclusions about consciousness.

Run a complete exact-coordinate repeat with the same algorithm, retain both
compressed datasets and exact non-timing comparison. Tests cover dimensions,
endpoints, refinement, reference fallback and frozen formulas. Commit protocol,
implementation/tests and outcomes separately. Retain full repository check log.
