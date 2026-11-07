# E013: matched slab mixtures and a definition-derived third solver

This protocol precedes implementation, task generation and outcomes. E005–E012
are development evidence. No parameter, threshold or sample changes after viewing
outcomes. The target is the original sphere; paired model conditions and policy
rows are not independent observations. Production solvers remain unchanged.

## Definition and predictions

For a slab with mean xT, the normalized response is tanh(sqrt(3xs))/sqrt(3xs),
s=zT. A two-slab mixture with positive weights a,1-a has the four sphere
properties exactly when

- weights sum to one;
- a*x1+(1-a)*x2=1;
- a*x1^2+(1-a)*x2^2=25/21;
- a/sqrt(x1)+(1-a)/sqrt(x2)=3/sqrt(5).

Derive the solution at 60 decimal digits from the initial positive branch
(a,x1,x2)=(.19,.098,1.21), retaining the equations and residuals. The asymptotic
condition follows from the slab coefficient 1/sqrt(3x). The sphere second
moment is20/7; a slab's is(12/5)x². A separate calculation, not a task experiment,
will test three positive slabs with the additional condition sum(w*x³)=25/17:
the slab third moment is306/35, the sphere third moment is90/7. Fix the smallest
support at .03 and solve for the other two supports and three positive weights.
A definition-only calculation before registration found positive weights
(.0570782983283,.195718071725,.747203629947) and supports
(.03,.323459350380,1.251306569240), with residuals below4.4e-55 at60 digits.
This is a mathematical construction, not a task outcome. The two-slab branch
has a=.189790853209, x1=.098260113626, x2=1.211231856719.
Finite linear response features do not identify an unrestricted positive-mixture
class: on more supports than constraints, a null vector of the feature matrix
perturbs strictly positive weights in two directions while preserving those
features. This does not assert every target has every finite-support realization.

Prediction: request hash and all four finite-probe tests pass the two-slab
mixture; numerical agreement D can accept it when internally consistent.
False acceptances should concentrate at intermediate |s| where model bias exceeds
tolerance. A one-slab control should fail the second-moment/high-frequency gates.
The third sphere solver should reject inaccurate mixtures when resolved and may
abstain in unresolved layers. Negative results remain unchanged.

## Frozen task design and common hooks

96 independent tasks, task seed20261125, condition-order seed20261126. Draw in
order log10T uniformly[-2,2], log10|s| ([-6,6] for the first48 tasks,
[-1,3] for the last48), phase uniformly[-pi/2,pi/2], log10tol uniformly[-4,-1].
Store binary64 inputs and independent request digests before decisions. Audit
exact binary64 request overlap (real/imaginary z,T,tol) within E013 and against
the saved E009/E010/E011 intakes; report duplicates rather than assuming new
seeds certify disjointness. Shared frequency domains do not imply identical tasks. Permute
clean sphere, two-slab mixture and one-slab control per task. Use the .5tol modal
planner, budget4096, resolution guard and uniform-grid adaptive verifier D from
E010 (the unchanged E006 verify/D and E005 refine_value functions). D is binary:
acceptance requires convergence and d+U<=tol; otherwise refinement continues
to4096 and the final nonacceptance is reject, not a proved numerical violation.
This deliberately retains the D requested here rather than relabeling E011 V. All model-dependent
artifact, numerical-verifier and probe calls go through one experiment-only
model selector: clean uses existing sphere calls; slab uses E010 uniform slab calls;
mixture sums the same two slab calls at x1*T and x2*T with their fixed weights.
No response cache. Count component solves, not only mixture wrapper calls.

All E010 probes and allowances remain unchanged: normalization/first moment
modal4096/FV512 at0 and i*1e-6/T, allowances2e-4 and relative2e-3; second moment
both4096 at0 and i*.01/T, relative allowance.02; high frequency both4096 at1e4/T,
relative allowance.03 to sqrt(3/5). All probes run even after earlier rejection.

## Third implementation and decisions

A separate module uses only the radial equation and NumPy/SciPy linear algebra,
not modal, finite-volume, analytic, mixture, or experiment code. Set w(r)=r*u(r).
Then w''=15s*w, w(0)=0, w(1)=1 and G=3*integral_0^1 r*w(r)dr. Use Chebyshev
Lobatto collocation at degrees64 and128 and independently constructed
Clenshaw-Curtis quadrature for the integral. Both evaluations are fresh for each
condition and cannot pass through the model selector. No analytic expression
enters its decisions. Constants such as differentiation matrices may be cached.

Let delta be the absolute difference of the two values and
R=32*eps*128^4*max(1,|G128|), Uthird=2*delta+R. Qualify convergence only if both
values are finite and delta<=tol/8. For discrepancy d from the submitted value,
accept if qualified and d+Uthird<=tol; violation_detected if qualified and
d-Uthird>tol; otherwise inconclusive. This two-resolution estimate is heuristic,
not a certified bound. The numerical definition of the sphere, intake and third
solver must remain trusted; independence is from implementation, not definition.
An erroneous shared definition would defeat this defense too.

Policies: D; D+hash+four invariants; that policy plus third implementation; third
implementation alone. D and D+gates retain binary accept/reject; a gate failure rejects even a
coincidentally accurate artifact. In D+gates+third, a base-policy rejection
remains reject; otherwise use the third verdict. Retain the third verdict
separately in every row, including when a base policy rejects. Do not relabel
binary D rejection as demonstrated numerical violation.

## Measurement and reproducibility

Evaluate analytic(original z,T) only after all condition verdicts for a task.
Report per condition and policy: accepted/inconclusive/violation counts, false
acceptances (accepted and error>tol), clean violations on accurate artifacts,
clean nonacceptance separately, accurate mixture violations and nonacceptance,
third inconclusive count/fraction. Report error and tolerance per raw row, every
probe and both third values, its difference, floor and uncertainty. Count modal
terms, FV cells, third matrix solves/degrees and wall time separately, including
all duplicate zero probes and both mixture components. Policies reuse recorded
components but their incremental and full costs must be explicit.

Retain all raw rows, intake, parameters/derivation, metadata, source/protocol
hashes and committed implementation head; exact-input replay reads stored
requests. Store full same-platform replay and compare all non-timing scientific
fields exactly; separately report platform/provenance differences. Save the full
check log. No statistical significance tests, no consciousness claims. Protocol,
implementation/tests, and results have separate commits, after E012.
