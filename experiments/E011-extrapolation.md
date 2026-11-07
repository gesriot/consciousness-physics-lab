# E011: frozen verification beyond its development domain

This protocol is fixed before task generation, reference audits, or outcomes.
E005–E010 are development data, not validation data. One pipeline V is evaluated;
no parameter, allowance, stopping rule, sample, or fault will be retuned here.

## Sampling and predictions

Task seed 20261122; condition-order seed 20261123; NumPy default_rng. There are
192 independent tasks, in cohort order A, B, C, K, 48 per cohort. For each task
draw four independent uniform variates in order: log10 T, log10 |zT|, phase,
log10 tolerance. T ranges from 1e-2 to 1e2 in every cohort. The ranges are:

- A: log10 |zT| in [6,8], phase in [-pi/2,pi/2], log10 tolerance in [-4,-1].
- B: log10 |zT| in [-6,6], phase uniform on the two disjoint intervals
  [-pi/2,-pi/2+.05] and [pi/2-.05,pi/2]. Map one uniform variate u: for u<.5,
  phase=-pi/2+.1*u; otherwise phase=pi/2-.1*(1-u). Tolerance in [-4,-1].
- C: log10 |zT| in [-6,6], phase in [-pi/2,pi/2], log10 tolerance in [-6,-4].
- K: the broad E009 design: log10 |zT| in [-6,6], phase in [-pi/2,pi/2],
  log10 tolerance in [-4,-1].

Set z=|zT| exp(i phase)/T. Store exact binary64 requests before any injection.
A predicts possible undercoverage when the layer remains unresolved, which
should yield inconclusive rather than unsafe acceptance; the fixed
high-frequency probe allowances are also tested here. B may introduce
nonmonotonic convergence. C should frequently hit the 4096-mode limit and
must not accept inaccurate clean values. K should resemble broad E009 within
sampling variation, not reproduce its exact counts. These are predictions,
not criteria for discarding results. The probe frequency remains 1e4/T in all
cohorts, so this experiment does not claim to validate an asymptotic probe at
frequencies beyond that fixed probe.

## Frozen operational pipeline

Use E009's nested graded mesh, base 64 cells, grading coefficient 4, midpoint
subdivision to 4096, and frozen (not observed-order) uncertainty. At grid n,
Delta is the absolute difference from the preceding grid, R=32 eps n²
max(1,|value_n|), U=2 Delta+R. Convergence requires
Delta_n <= max(.5 Delta_previous,R). The first decisive grid is 512.
Accept if converging and discrepancy+U<=tol; violation_detected if converging
and discrepancy-U>tol; otherwise inconclusive. Preserve the E007/E009 futility
rule at 1024 and 2048 exactly, and the .5 tol modal planner, budget 4096,
resolution guard, and independent original-request digest.

All four E008/E010 invariants are checked even if another gate fails. Use
fresh modal and graded finite-volume evaluations, with the same shared model
hook as the artifact and verifier. Normalization and first moment: modal4096,
FV512, z=0 and i*1e-6/T_original, allowances 2e-4 and relative 2e-3.
Second moment: both4096, z=0 and i*.01/T_original, estimate
2*(Re G(0)-Re G(ih))/(h*T_original)^2, target20/7, relative allowance .02.
High frequency: both4096, z=1e4/T_original, compare100 G to sqrt(3/5),
relative allowance .03. No response cache or discarded warm-up probes.
The mesh change relative to E010 is declared in advance; no margins change.

V accepts only if digest, resolution, all invariants and numerical acceptance
pass. A failed digest or invariant yields violation_detected with reason
model_or_request_violation, even when the individual supplied value is within
tolerance. Otherwise a numerical violation retains violation_detected;
a numerical acceptance blocked solely by resolution yields inconclusive.
Retain the numerical verdict, each gate and combined verdict separately.
Thus the primary accurate-violation endpoint explicitly includes model
violations on coincidentally accurate individual values; also report its
numerical-only and clean subsets. Gate failures are not relabeled as errors
in the discretization uncertainty estimate.

## Conditions and accounting

Eight paired conditions per task, randomly permuted using the order seed:
clean, conjugate, renormalized, time_units, near_tolerance, stale_value,
model_scale, slab. The first six use exactly E006 injection definitions with
one shared clean receipt; initial stale request is (z=1,T=1,tol=.01), updated
from the clean receipt after each task. The last two are explicit experiment
hooks, never trusted receipts: scale multiplies model T by1000 in modal,
graded FV and all probes; slab uses d=1, kappa=1/(3T), eigenvalues
((n-.5)pi)^2, weights2/eigenvalue, and graded FV volumes dr and face area1.
Sphere has d=3, kappa=1/(15T), volume r²dr and face area r².
Original request, digest, target definition and invariant comparison stay fixed.
A shared fault changes every model-dependent solver path but not these targets.

Time and count clean execution once per task, injection work once per condition,
shared-model artifact work, numerical decisions, all probe calls and full shadow
paths separately. Cached mesh/eigenvalue constants are allowed; response values
are not reused across operational decisions/probes. Shadow data are never used
in decisions. One full shadow per task/model (sphere, scaled sphere, slab) is
computed after all operational conditions. Count all model paths separately.
All 1536 condition rows are paired; they are not 1536 independent observations.

## Reference validation before decisions

Generate and save the intake and a separate reference-audit artifact before
any operational decisions. For all 96 A/B requests compare analytic against
60-decimal-digit mpmath evaluation of 3*(coth(x)/x-1/x²), x=sqrt(15 z T),
using exact binary64 inputs converted to high precision. Zero, if needed, is1.
Store high-precision decimal real/imaginary values, analytic values and absolute
error. An analytic value fails if nonfinite or error >1e-13+1e-12*|reference|.
If ANY A/B request fails, all A/B accuracy and sphere coverage references use
the 60-digit values rounded once to binary64; otherwise analytic is used.
C/K use analytic. The entire reference audit is completed and persisted before
any decisions. Neither reference enters V. Report fallback and maximum error.
For shared-model coverage also evaluate exact scaled-sphere and slab closed
forms at60 digits after operational decisions; slab is tanh(sqrt(3zT))/sqrt(3zT).

## Endpoints and audit

Primary per cohort: false acceptances (accepted and error>tol); unique decisive
coverage misses; combined violation_detected on accurate values. Coverage means
FV discretization error relative to the model actually solved, not model bias.
For the primary coverage denominator take the union of operational decisive
(task,model,grid) tuples, with no duplication across paired artifact defects.
Also report sphere-only coverage, since that matches the E009 estimator claim,
and original-sphere discrepancy for shared faults separately as model bias.

Secondary: coverage on all four decision-eligible grids, stratified by model
and delta/h_surface bins [0,.5),[.5,1),[1,2),[2,4),[4,infinity), with layer
width sqrt(kappa/|z|); clean accurate acceptance; each clean invariant false
rejection; numerical and combined inconclusive counts; modes/cells/timing.
Report max error/U, counts and denominators, not p-values. Discretization
coverage at a model-biased decisive step does not certify the intended sphere.

Commit protocol first, then implementation/tests before generation. Store
protocol and source hashes, implementation head, runtime/platform/BLAS metadata,
all requests, reference audit, steps, probes, full shadows, raw rows and summary.
Use the existing pinned runtime when available, otherwise record explicit
runtime override. Add exact-input replay and compare all non-timing numerical
outcomes on a same-platform repeat, retaining the full repeat and comparison.
Run the full scripts/check.sh and save its log. Negative results remain.
Finite probes and heuristic U are not proofs. No conclusions about consciousness.
