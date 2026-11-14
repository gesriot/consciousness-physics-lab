# B004: source reconciliation through derived claims

## Question and scope

EXP-004 asks whether explicit provenance helps reconcile incompatible results from
mixed units, regimes and table versions. B001–B003 tested instrument failure and
sampling. Their outcomes are not new B004 observations. This laboratory concerns
truth maintenance, not neural synchrony or consciousness.

## Laboratory and independent units

Each independent series has twelve material/query groups. Each contains a primary
chain of three accepted claims: sphere response from properties, diffusivity
inferred from that response, and response at a different radius/frequency from the
inferred diffusivity. A second table supplies one independent root response for
the same first query, using diffusivity directly instead of conductivity/density/
heat capacity. Thus there are 48 accepted records, including 12 at each deeper
level. The reference path has no visible trusted flag. All IDs and ingestion times
are randomized independently of damage and signal. Ingestion order is not causal
order: these are imported scientific records. The target request is version 2.

Sources carry explicit units and versions. The current authoritative table and
exact response define the hidden evaluator; policies cannot read it. A common
paid service rederives an opaque claim ID from the current table and the exact
model, following the full original recipe. This service knows the recipe but
returns no parent chain to a memory that lacks it. All initial claims are accepted
by the same finite-positive admission rule; this is a study of revision, not a
claim that initial source validation is adequate. Units/dimensions, clean chains,
scaling, independent 60-digit references and full solver costs are tested first.

## Corruptions and signals

Two primary chains in distinct groups are chosen independently per series. Three
separate paired fault conditions: radius magnitude in millimetres labelled metres
(factor 1000); a diffusivity from version 1 (factor 4) mixed into the requested
version-2 table; high-frequency leading response used at zT uniformly in log space
from 1 to 10, below its declared regime zT >= 100. The last corruption remains in
(0,1), permitting inverse inference. All descendants use the observed parent
value. Other source properties vary log-uniformly over ordinary positive ranges.
The clean counterfactual uses identical inputs and record order.

External revision names the two suspect source records and their applicable
versions, not hidden accuracy labels. A conflict signal is generated from accepted
root claims for the same quantity when their values differ by more than the
registered tolerance. It names both sides, not the culprit. To locate the fault,
a memory pays for an independent root rederivation of one side (side and pair
order depend only on random opaque IDs), compares both values with that answer,
and then obtains the mismatching root and its direct source attributes. A correct
source is not blamed by fiat. If no conflict exists, no signal is generated.
Clean external notices deliberately name two correct sources (false notification
control); clean conflict runs test false alarms. N observes signals but does not
act or pay for diagnosis.

## Equal-capacity memories and actions

Every memory retains all 48 records with an equal 2048-byte logical slot limit, with identical
numerical payloads, statement units/modes and timestamps. No eviction; this is equal permitted capacity, not a claim of identical Python RAM. G has parent
and direct-source links. F has units, regime and version of direct raw sources but
no parent claim IDs or inherited ancestry. T has a time journal without usable
provenance. N never revises. G1 uses direct-source roots plus their immediate
children, frozen from the initially implicated roots, with no recursive expansion.
G traverses all descendants. Gx severs every root-to-child link by replacing it
with a typed opaque external endpoint; Gxprime performs the same number of edits
on irrelevant request pointers. Both edits are independent of the fault target.
They do not alter the true recipes used by the common verifier. Gxprime must match
G; this registered invariant placebo is exempt from the requirement to differ.

Each active policy prioritizes its implicated set by depth, then opaque ID. The
remainder is a common deterministic blind scan (ID order) for G/G1/F/Gx/Gxprime;
T uses reverse ingestion order throughout. This continuation is explicit: policies
check further records even after their candidate set is exhausted. Each candidate
is withdrawn for one bookkeeping tick and rederived if the remaining budget covers
the complete recipe. Revalidation accepts the corrected result. If the next full
rederivation is unaffordable, that last withdrawal stands and the series stops;
no skipped expensive claims or uncharged cached work. Each claim is considered
at most once. N does nothing. Trace includes candidates, chosen action, charged
cost, repaired/withdrawn status and actual instrument evaluations.

A numerical evaluation costs one tick; inverse uses a fixed 66 evaluations
(two bracket endpoints and 64 log-space bisections), so a full depth-1/2/3
rederivation costs 1/67/68 ticks, plus one withdrawal tick. One budget unit is
69 ticks, the full deepest rederivation plus withdrawal. Limits 2/8/32 therefore
mean 138/552/2208 ticks, including conflict diagnosis. Actual costs are reported,
not assumed equal merely because limits match. No work is padded to spend money.
Scores are rounded to nine decimal places before stable ID tie-breaking. A
last-bit repeat perturbs numerical outputs with nextafter and compares protected
actions exactly; raw numbers are diagnostics. Explicit --runtime-override
permits environment differences, never changed inputs/protocol/source hashes.

## Development, freeze and confirmation

Initial development seeds 20270301..20270312; held-out confirmation 20270313..
20270324. They are independent series, not the dependent conditions or depths.
Fault conditions share common inputs per series; all seven memories, three budgets
and two signal branches are paired. Clean controls are also paired. Development
may change the mechanism, but each change and reason is recorded before freeze.
No confirmation data may be generated before the freeze commit.

Exposure gate for every corruption and every depth separately: at least ten
unique affected accepted claims and at least five initially inaccurate claims,
counted across development series, not policies/budgets/repeats. A failed fault
is excluded and reported. Before freeze inspect actual actions: G versus F on
depths 2–3, G versus G1 at depth 3, G1 versus F at depth 2, G versus Gx, F versus T, and active versus N
must differ at least once in an admitted condition; deep chains and conflicts
must exist. Gxprime must preserve G actions. Diagnose structural degeneracy and
record a development correction, or exclude an untestable comparison. Do not
retune B003 on its old confirmation outcomes.

## Measures, forecasts and closure

Primary per fault x budget x signal cell: inaccurate/incompatible claims still
accepted, unnecessary withdrawal of initially correct claims, and total paid
cost. Report final incorrect acceptances after repair, correct original records
withdrawn (including temporary withdrawals), and correct records left withdrawn
separately; replacing an inaccurate claim by a correct one is not a correct loss.
Accuracy uses abs error <= 1e-10 + 1e-5*abs(reference), shared with conflict
resolution. Record unresolved conflicts and initially exposed errors separately.
Secondary: culprit source localization precision/recall; candidate precision and
recall by depth; false-notice losses; solver evaluation counts; randomization,
capacity and graph-edit checks. Summaries are paired by independent series with
mean and standard error, not by treating claims as independent replications.

Forecasts before development: F no worse than G at depth 1; G fewer remaining
incompatible claims at depths 2–3 than F/T at equal price; G1 intermediate; Gx
loses advantage, Gxprime preserves it; N loses no correct claims and retains every
initial incompatibility. Actual-cost differences qualify any equal-price claim.
The closure rule is fixed now: if F is no worse than G on both remaining
incompatibilities and unnecessary correct withdrawals in every primary cell,
explicit links are unnecessary for reconciliation in this laboratory. Otherwise
report where the conjunction fails; that alone does not prove any forecast.

## Reproducibility and compact storage

Store common inputs, initial claims and hidden evaluator values once per series.
Conditions reference them by canonical SHA256. Store only action deltas and
summaries per condition, with content-addressed reuse of identical condition
outputs; no duplicated ordinary traces. Save deterministic gzip (-9, mtime=0).
First runs alone retain full archives. Ordinary and noisy repeats run sequentially
from saved intake, followed by independent audit and exact decision comparisons.
Retain their manifests, audits, comparisons, logs and SHA256, not duplicate data.
Source inventory, protocol/freeze hashes, environment and runtime overrides are
explicit. Results are written per completed series. Before a data commit measure
new Git object sizes; limit 60 MB and individual blobs 50 MiB. If exceeded, report
before committing. Never alter lab/ or attention/. Final check.sh and
check-noise.sh logs belong in the report, not the manuscript.

## Pre-data clarification of source identity

A root has two typed direct supports: its versioned property table and its declared
calculation-method record. Regime corruption belongs to the latter, not to an
otherwise correct property table. Revision notices include the applicable version.
Paid conflict diagnosis compares provenance metadata to the current catalogue and
identifies the table or method support separately. A resolved conflict means the
remaining accepted pair no longer disagrees; accuracy is evaluated separately.

Pure immutable rederivations may be memoized across paired conditions to avoid
repeated physical work. Every requesting policy is still charged the complete
registered evaluation count; actual cache-miss evaluations are reported separately.
This optimization does not expose verifier outputs before the action is paid.
