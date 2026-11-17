# B007: one correction of fallible diagnosis

## Scope and sequence

B006 development exposed five design defects: a classification-loss selector
stopped below its decision threshold, exclusion could accuse an untested ground,
P changed ground order as well as repetition, strong priors barely adapted, and
fixed rules could not repeat exhausted conflict messages. Its four-measure
low-zero gate also conflated safety admission with confirmatory dominance.
These corrections are made after viewing B006 development, not independently
predicted discoveries. This is the only correction attempt in this laboratory.
If admission fails, or a fixed rule dominates everywhere in confirmation, close
the reliability-model question here without a third version.

Commit this protocol before policy code and new runs. Use B006 laboratory,
physical recipes, independent binary report streams, 32 episodes, four
environments, costs, budgets 2/8/32, certified repair and observable-resolution
learning by import. Development reuses viewed seeds 20270501–20270512.
Confirmation uses new seeds 20270701–20270712 only after a committed freeze.
One series is an independent unit; its episodes and policy conditions are
paired dependent observations. This is functional reliability modeling, not an
attention schema, cortex model or consciousness theory.

## Corrected common action space

All active policies may repeat a measurement or conflict message after exhausting
grounds. D and R use one report, K requires k equal reports among at most m with
(1,1), (2,3), (3,5). R first repeats the signal. Ground order for every policy is
D's deterministic cost/order rule on the same remaining candidates, with the
same affordability filter. No policy receives evaluator truth. N does nothing.

P keeps the current D-selected ground until its posterior is at least threshold
(to accuse) or at most one minus threshold (to clear and advance), with at least one direct
good report required for clearing. Without this corroboration, small initial
per-ground priors would clear every untested ground; this pre-data clarification
prevents a degenerate queue. Accusation
additionally requires an actual direct bad report about that ground. An
observation diagnosis requires its posterior to reach threshold. P can choose
only the current ground check or a signal repeat; it cannot jump to another
ground because its hypothesis probability happens to be high. Every repetition
and every reconstructed descendant is paid. No suspicion alone withdraws claims.

For a posterior p and current ground h, let gap be the minimum of
max(0,t-p(observation)), max(0,p(h)-(1-t)), and max(0,t-p(h)); include the
last term only after a direct bad report for h and the clearing term only after
a direct good report for h; omit both ground terms when
there is no current ground. This potential includes clearing the current ground,
which advances the fixed queue on the next real iteration.

For each first action a, enumerate outcomes y with predictive probability q(y).
Impossible zero-probability outcomes are omitted. Its one-step value is
max(0,gap(p)-sum q(y) gap(p_y))/cost(a). For each y, choose an affordable
second action among the SAME current-ground and observation tests that minimizes
expected gap after that action rounded to 9 decimal places (ties by price, kind, id); permit no second action
if no eligible one is affordable. Hypothetical queue advancement is not used:
lookahead ends once any acceptance/clearing threshold is reached. Let g_y be that
terminal or expected second-step gap and c_y its cost (zero if no second action).
The two-step value is max(0,gap(p)-sum q(y)g_y)/(cost(a)+sum q(y)c_y).
The first action score is the larger one- and two-step value. Scores are rounded
to 9 decimal places and resolved by kind/id/candidate. If all rounded scores are
zero, use expected entropy reduction per price among eligible actions. If
that too rounds to zero, choose the first eligible action deterministically.
A test is informative iff its bad-report likelihood differs across hypotheses
with positive posterior support. This mathematical test avoids treating
normalization roundoff as information. Clear all already-exonerated current grounds before
constructing this inventory. Stop below threshold only when budget cannot afford
an eligible allowed action or no allowed action can change the posterior or
supply necessary decision evidence;
record the reason and remaining budget. There is no blind continuation on an
already resolved decision. A current-ground test is also eligible when it can
provide missing direct good/bad evidence for an already threshold-qualified
clearing/accusation, with strictly positive probability of the required report.
Qualification uses the same rounded probability rule as actual decisions. This
pre-data clarification applies to every P variant and both lookahead levels,
not only Pstar: corroboration can change the permitted decision even when it
cannot change a certain posterior. A contradictory noisy report reopens the
posterior calculation. Stopping must not ignore such an affordable action.

## Models and learning

Four global configurations cross sensitivity/false-positive Beta priors
(4,1)/(1,4) or (2,1)/(1,2) with thresholds .80 or .95. Their initial effective
sample sizes are 5 or 3, permitting movement over 32 episodes. The hypothesis
prior and fractional self-training rule are imported from B006. Train only after
observably resolved episodes: latest paid signal repeat dismisses discrepancy,
or paid descendant repair agrees with the initiating observation/comparison.
An inferred diagnosis alone is not resolution; no hidden labels enter training.

P learns channel reliability. P0 freezes the same initial values and hypothesis
prior. Pstar receives true channel probabilities in advance and does not learn;
this privileged bound is not an eligible candidate. Exact 0/1 probabilities
are retained; impossible outcome branches are omitted, never epsilon-smoothed. Px fixes near-perfect active
reliability: true-positive probability 0.999999 and false-positive probability
0.000001, retaining the B006 regularization. This operational near-error-free
intervention avoids undefined updates after reports declared impossible by the
intervention; it is not literally an error-free likelihood. Pxprime alters the same number of unused parameters; its active
model and learning remain P's. Both true and inferred rates are reported, with
truth accessible only to evaluation or explicitly privileged Pstar.

## Admission and freeze

Select K globally and the strongest fixed rule among D/R/selected K by
lexicographic totals false accusations, remaining bad claims, lost good claims,
then cost, with textual ties. Select one qualifying P configuration identically.
All gates apply before freeze:

1. B006 exposure: each fault/scope has at least 20 affected and one inaccurate
claim; both branches, depths 1–4, low/high false objections, both report-error
directions in source and local checks, and a moderate D false accusation occur.
2. In every low_zero budget/branch cell P has no more false accusations or lost
good claims than D. Remaining bad claims and price are reported, not admission
criteria. This separates safety admission from confirmatory dominance.
3. Zero accusations without direct bad reports and zero below-threshold stops
while an affordable action can change posterior probability.
4. In moderate environments, mean squared error of the six channel probability
estimates, pooled over series and budgets at episode 32 separately in each
moderate environment, is strictly smaller than at initialization. Report each channel/direction and each environment,
including worsening components and unresolved counts; these quantitative gates
are registered operationalizations, not selected afterward.
5. Px differs in actual actions in at least 10% of signaled moderate paired
episodes. Pxprime has zero protected-action or outcome differences. P differs
from each D/R/selected K in at least one repetition decision.

If no configuration qualifies, retain all outcomes, close the question and do
not run confirmation. Do not expand the finite prior grid after outcomes.
Freeze gate evidence, selected configuration, fixed comparator and calibration
estimates in a separate commit before new confirmation seeds.

## Measures, closure and predictions

Primary measures and closure are unchanged from B006: compare the selected fixed
rule against P in every environment × budget × branch cell on false accusations,
remaining incorrect accepted claims, lost initially correct claims and total
price. If fixed is no worse everywhere, the reliability model is unnecessary
here. Otherwise list the cells and price of each advantage; non-dominance is not
universal superiority. Secondary measures include P versus P0 and Pstar, false
objections dismissed without revision, genuine counterexamples corrected,
repetitions, calibration/Brier scores with abstentions separate, depth and scope.

Predictions: low_zero P is no less safe than D; moderate P makes fewer false
accusations than D and no more errors than K at lower price; Pstar is no worse
than P and P no worse than P0. Px loses benefits and Pxprime retains them.
Frequent false objections favor R and P approaches it. Preserve failures.

## Integrity and reproduction

From first code commit support noise and --runtime-override. Authenticate stored
canonical intake by manifest SHA256; replay and audit use it, while regenerated
seed floats have diagnostic tolerance of 8 ULP. Integers and structure remain
strict. Comparisons protect decisions/counts/prices and report floating drift.
First runs and both repeats execute sequentially. Write completed series at once.
An independent auditor reconstructs transitions without importing simulation.
Keep full first-run archives only; repeat manifests, hashes, comparisons, audits
and logs remain. Data/audits precede numerical reports. No historical outcomes,
lab/attention files or manuscript sections 3–4 change. Limits: 50 MiB per blob,
60 MB Git growth; deterministic gzip level 9, timestamp zero.
