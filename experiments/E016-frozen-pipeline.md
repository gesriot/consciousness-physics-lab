# E016: held-out confirmation of the frozen pipeline

This protocol is committed before implementation, task generation and outcomes.
E005–E015 are development data. E014 selected C1; E015 mapped collocation alone.
This experiment tests their composition without tuning any component or policy
in response to these outcomes. Failures remain results, not repair opportunities
within this series. There are 240 independent requests; nine conditions and two
policies on each request are paired, not independent samples. No consciousness
claims, significance tests or continuous-domain guarantees are intended.

## Frozen components and policies

The single pipeline module assembles V2: the E009 graded grid, frozen
U = 2 Delta_n + R_n with R_n = 32 eps n² max(1, |F_n|), C1 ordinary contraction
Delta_n <= .5 Delta_previous only, first eligible grid 512, grids through 4096,
three numerical verdicts, unchanged futility and resolution rules, and the
.5 tolerance modal planner. Request binding and the four E008/E010 probes
(normalization, first moment, second moment, high-frequency coefficient) keep
their original sizes and allowances. Model-corrupted calls share the selected
model, while independent sphere collocation is unchanged from E013/E015.

Both policies require request binding, all four model checks, resolution and V2
acceptance. P-strict additionally requires collocation acceptance. P-soft
requires that collocation has not detected a violation; inconclusive is allowed.
A model/request failure or a numerical component's violation is retained as a
violation, with separate component reasons. Otherwise an unmet acceptance
condition is inconclusive. Compute all components even after earlier failure.
No policy changes the underlying component verdicts or arithmetic.

## Requests, artifacts and order

Use default_rng task seed 20261127 and independent condition-order seed
20261128. Generate 48 consecutive requests in each of W, H, M, O, S.
For each request draw, in order, log10 T uniformly in [-2, 2], one uniform
binary64 u in [0, 1) for log magnitude, one uniform v for phase, and log10
 tolerance uniformly over the specified interval. Set z = magnitude exp(i phase)/T,
with budget 4096. Magnitude log intervals are W [-6, 6], H [6, 12],
M [-1, 3], O [-6, 12], S [-6, 6]. Tolerance log intervals are [-4, -1]
except S [-6, -4]. Phase is -pi/2 + pi v except O, where it is
-pi/2 + .1 v if v < .5, otherwise pi/2 - .1(1-v).
Store requests, original digests, sampled coordinates and exact u integer ratio
before solving. Check exact binary64 request overlap internally and against
available saved E005–E015 task intakes; maps are development coordinates, not
independent request samples. Report the archives checked and any unavailable
intake, rather than claiming seeds prove independence.

Nine inputs per request: clean; E006 conjugate, renormalized, time_units,
near_tolerance and stale_value; E008 common model_scale; E010 slab; E013 fixed
two-slab mixture. Preserve E006 injection definitions, including additive
1.25 tolerance and the previous request's clean modal value for stale_value.
Initialize that value from Request(1, 1, .01) with the .5 tolerance planner.
A shared clean receipt supplies the first six inputs. Common-model artifacts,
verification and fingerprints use their respective model consistently; the
original request and sphere collocation do not change. The mixture parameters
are the unchanged E013 positive branch. Randomly permute nine conditions per
request using only the condition-order generator. Save all supplied artifacts.

## Independent reference and coverage

The primary truth is analytic(original z, T), never used in policy decisions.
Audit with the exact binary64-input 60-decimal-digit sphere formula for every
sample whose exact coordinate lo + (hi-lo) u exceeds 8, using integer-rational
arithmetic, and for every remaining task with task_id divisible by 10. This
selection is fixed before data; no comparison of floating abs(zT) chooses audits.
Allowance is 1e-13 + 1e-12 |G60|. If any audit fails, replace references for all
240 requests with once-rounded 60-digit values, retaining all discrepancies.
Persist the intake and audit before pipeline outcomes. Mark failures honestly.

For every operational decisive V2 step (accept or violation), record its
reference error |F_n - G|, U, coverage and model. Distinguish sphere numerical
coverage from wrong-model discrepancy against the original sphere; the latter
is not a claim that U bounds model error. Also retain per-row collocation
reference error, U, convergence and verdict, and coverage on its decisive steps.
Shared requests and duplicated component evaluations remain explicitly paired.

Record on every V2 step whether both consecutive differences are <= R_n;
count this on decisive steps separately because it can vary across platforms.
Compute C2 in shadow from the same prefix, never using it for stopping, and
count C1/C2 flag disagreements on all operational eligible steps and separately
on decisive steps. No E016 result can change C1, C2, U or stopping rules.

## Endpoints and predictions

Primary measures, by cohort and policy: false acceptances (accepted artifact
error > tolerance, predicted zero); decisive component coverage misses; violations
on accurate supplied values, separately numerical and model/request reasons.
Report clean accurate-value violations separately from accurate corrupt-model
values. A model mismatch can rightly violate a definition even if its value is
incidentally within tolerance. Keep V2 and third coverage endpoints separate.

Secondary measures: clean accurate acceptances and strictness cost; accurate
and inaccurate clean counts; abstentions with nonexclusive V2, resolution,
model checks and third-component reasons; modal terms, FV cells, collocation
matrix solves and degrees, with one-time generation and per-policy reuse made
explicit. Do not double-count reused components across policies. Retain all
component traces, gates, work counters, row verdicts and summary definitions.

Predictions by cohort and policy:

- W: P-soft accepts at least as many accurate clean inputs as P-strict; both
  reject detectable local defects and model-scale/slab mismatches. Strictness
  loses acceptances where collocation's uncertainty exceeds tolerance.
- H: P-strict accepts few or no clean inputs beyond collocation's convergence
  range; P-soft accepts where V2 and the resolution/model gates allow it.
  Neither may accept an inaccurate input. Common model faults in collocation's
  abstention region are the specific risk for P-soft, not assumed harmless.
- M: the mixture passes finite model probes, but both policies detect it where
  its bias exceeds tolerance and collocation resolves that discrepancy. At
  biases below tolerance, acceptance is not a false acceptance.
- O: nearly imaginary requests may increase numerical abstention. P-strict
  cannot accept more clean inputs than P-soft; no inaccurate acceptance is
  predicted under either policy.
- S: the 4096-mode limit often leaves clean modal answers inaccurate, especially
  at low/moderate frequencies. Both policies must refuse such answers. Some
  high-frequency answers can still be accurate at this absolute tolerance;
  blanket clean inaccuracy is not assumed. Collocation's fixed rounding floor
  makes P-strict particularly restrictive near the tight end.

## Reproduction and retention

Implementation must support --replay and --runtime-override in its first commit.
Replay reads exact saved requests, coordinates and condition order, regenerates
fresh numerical evaluations and compares every non-timing scientific field.
Keep protocol/source/environment hashes, implementation commit, full original
and replay results, intake, reference audits and comparison report. Use one BLAS
thread. Commit protocol, implementation/tests, then outcomes/report separately.
Run ./scripts/check.sh and retain the full log. Update README and the manuscript
with actual results; reproducibility machinery belongs in reports, not the
scientific narrative. No post-outcome tuning or silent data exclusion.
