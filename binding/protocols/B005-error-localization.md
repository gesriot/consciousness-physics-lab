# B005: locating a faulty premise

## Scope and laboratory

This experiment studies functional provenance in scientific diagnosis, not neural
synchronization or consciousness. One independent series contains paired branches
of dimensioned passive-sphere calculations with shared material/method records,
separate radius records and step-specific method records. Accepted claims include
diffusivity, response, inverse diffusivity, predictions and two-branch comparisons.
The graph contains multiple parents, common ancestors and depths one through four.
The laboratory and its tests are committed before this protocol and experimental
code. The hidden evaluator never supplies a culprit or truth flag to a policy.

There is at most one changed source or step-method record in each fault condition.
The alternatives are radius-unit confusion, material-version mixing, a response
method outside its declared domain, and an intermediate inference method error.
Shared and private instances are paired. A false-measurement condition changes the
observation only; clean conditions have no damage. Accuracy and signal tolerance
are abs(error) <= 1e-10 + 1e-5*abs(reference). Observations are synthetic independent
observations of specified claim quantities; they are not noisy solver reports.
The returned initial signal names claims and their observed discrepancy, never a
source. Alternative-route conflicts name both endpoints of a quantity comparison.

## Information and permitted checks

G stores direct parent/source links. L stores the transitive set of source and
method IDs, including every local inference method, without parent pointers.
F stores only direct source attributes, T stores time order, and N never revises.
Each record has a fixed 16384-byte capacity, with actual serialized bytes reported;
Numerical payloads use eight-byte binary64 slots and metadata uses canonical JSON;
these encoded sizes are measured, not Python heap estimates. No eviction or
information is compressed into identifiers. Common physical recipe
execution is outside memory, accessible only through paid actions. No policy can
read the evaluator, clean catalog or recipe graph through that interface.

G and L use identical set operations and score rules on the provenance available
to them. For multiple independently failed observations they prioritize their
common candidate grounds. For an unresolved two-route conflict neither endpoint
is known wrong: its initial candidate grounds must be the union, not intersection,
since an intersection would exclude a fault private to one route by construction.
Checks that justify a premise remove it from the candidates. A direct local
rederivation justifies only its own step method, NOT all ancestors. This distinction
prevents a correct calculation from exonerating a wrong input. No recursive oracle
silently reads all correct source records.

A source check compares one observed record with its current authoritative catalog
entry and costs 30 ticks for radius/material records and 100 for method records.
A local-claim check repeats one operation using its currently stored direct parents
and observed inputs, with the specified correct local operation; its full numerical
cost is paid (forward/arithmetic one evaluation; inverse 66 evaluations),
plus one tick for retrieving its specified local method. It tests
only the local step-method candidate. Intermediate checks are alternatives to a
method-record check. Repeated observation costs twice the full deepest-claim unit.
These explicit costs are laboratory tariffs, not measured wall-clock times.

Every affordable available test partitions the remaining candidate set into the hypotheses
it can falsify and the rest. Selection maximizes min(k,n-k)/cost; singleton sets
have score 1/cost. Scores are rounded to nine decimals before opaque-ID ties.
A local check has singleton coverage, not an unjustified global group-testing
power. An additional deterministic rule prefers a local check to an equal-score
source read. F checks direct grounds of signalled claims, then remaining records;
T uses reverse ingestion order, with stable ID ties: source records arrive in
catalog order before the randomly ordered claims; each local-check action uses
its claim timestamp and each source-check action its source timestamp. All fallback checks are paid.

Gx and Lx remove the provenance of the signalled endpoints and replace it with an
unrelated request marker; edits do not know the hidden culprit. Gxprime/Lxprime
change only unrelated request metadata, which is not candidate provenance.
True recipes used for physical execution are unaffected. The irrelevant controls
must have identical actions. G versus L is expected to coincide, not required to
differ for admission. Actual action differences and record bytes will be reported.

## Resolution and budget

After a verified culprit, replace that source/step record with the paid corrected
record. Select affected claims through the policy's provenance, then rederive in
causal order through the common execution interface. Rebuilding a selected claim
recursively from its current parents charges every numerical operation required;
shared work within that action is counted once, never reused for free across
policies. If all candidates are exonerated, repeat the measurement; a false
observation is corrected, not attributed to an innocent source. A clean condition
has no signal and pays no diagnosis. A genuine conflict with no localized culprit
remains unresolved rather than making a false accusation by fiat.

Do not continue blind checking after resolution. Do not withdraw a claim unless
its full rederivation is affordable. If diagnosis runs out of money, claims whose
provenance intersects unexonerated hypotheses become suspected, remain accepted
and are counted separately; a journal without provenance conservatively suspects
all its claims while candidates remain. Budgets are 2, 8 and 32 times the full cost of unique numerical operations and source reads
for the deepest claim, determined from the fixed graph before any fault. Every
check, rederivation and repeated observation is charged. Report unused budget,
actual price and all temporarily/finally withdrawn true claims separately.

## Development and freeze

Twelve development seeds 20270401 through 20270412 and twelve new confirmation
seeds 20270413 through 20270424 are the independent units. Faults, scopes, nine
memory policies, three budgets and two signal branches are dependent paired
conditions. Do not multiply the effective sample by conditions, depths or repeats.

Exposure per physical fault and attainable depth requires at least ten affected
accepted claims and five incorrect claims across development; intermediate faults
begin at depth two. Require signals at depths two through four, both shared/private
culprits of every type, intermediate checks, G/F and L/F action differences,
G/Gx and L/Lx differences, and exact G/Gxprime and L/Lxprime agreement. Degenerate
designs may be corrected only on development, with an explicit change record.
A failed necessary gate blocks confirmation. Commit a freeze with the passing
checks and code identity before confirmation. No parameter changes after examining
confirmation outcomes. Negative results are retained.

## Measures and forecasts

Primary cells are corruption x budget x signal branch, aggregating the registered
shared/private cases: final incorrect accepted claims; final correct withdrawals;
false culprit accusations; total charged cost. Also report each scope separately,
localization rate/check count by type/depth/scope, unresolved and suspected counts,
false-observation behavior and storage sizes. Treat equal action patterns as
repeated applications of a mechanism, not independent confirmations.

Forecast: G and L coincide while the labels fit; both diagnose cheaper than F/T
at deep signals and do not increase false accusations. At root signals F is no
worse. Relevant provenance damage loses benefit, placebo edits preserve it, N
retains all faults without withdrawing true claims. Zero false accusations for
all policies would not support a differential advantage on that measure.

Closure 1: if L is no worse than G on all four primary measures in every primary
cell, explicit links add no necessary information beyond inherited labels in this
laboratory. Closure 2: if F is no worse than G AND L in every primary cell,
provenance does not improve localization here. State each outcome explicitly.

## Reproducibility

Record protocol commit/hash, source hashes, input hashes and actual runtime.
Implement --runtime-override and --last-bit-noise from the first code commit.
Perturb numerical outputs by one nextafter step, not decisions. Exactly compare
choices, verdicts, flags, integer accounting and input/protocol identity; summarize
numeric drift separately. Run original, ordinary replay and noisy replay strictly
sequentially with independent audits. Write each series immediately. Preserve full
compact archives only for first runs, with shared inputs and hashed references.
Every manifest records compressed-file SHA256 and canonical decompressed-JSON
SHA256. Cross-platform equality concerns the latter; compression bytes can differ.
