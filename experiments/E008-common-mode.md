# E008: registered common-mode corruption and independent anchors

Preregistered before implementation/data, after E007. Pinned environment:
Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0. No production solver edits.

## Design and predictions

96 independent tasks, seed20261116, order seed20261117. In each task draw
log10(T) uniform[-2,2], log10|zT| uniform[-6,5], phase uniform[-pi/2,pi/2],
log10(tol) uniform[-4,-1], in that order; z=|zT|exp(i phase)/T. Budget4096.
The four paired conditions are clean, request_time (shared Request T×1000),
request_conjugate (shared Request z conjugated), model_scale (Request intact,
shared solver mean-time scale×1000, equivalently diffusivity/1000).
Permute condition order independently per task using order RNG. Independent
unit=task, not 384 condition records or repeated solver probes.

Save request_hash(original Request) to an independent-channel record BEFORE
constructing corrupted requests. This record and original T are outside the
fault injection scope. This models a trusted intake channel, not cryptographic
attestation or an attacker able to corrupt both channels.

Use frozen E006 D: planner .5tol, final_guard and allocation.refine_value,
unchanged grids, convergence heuristic and U. Both solver calls see the shared
corrupted Request in request faults. For model_scale, keep the caller Request
and guard intact, but explicit experiment-only modal/FV TEST HOOK wrappers
multiply mean_time by1000. Scoped replacement of allocation.response directs
D to the FV hook; never modify production source or the analytic oracle.
Return explicit experimental untrusted outcomes, never fake trusted Receipts.
Clean uses the same wrappers with scale1. Guard is evaluated with the operational
Request (corrupted in request faults); independent anchors are separate defenses.

Prediction: agreement can accept internally consistent but wrong answers under
all three corruptions. Not every corruption must succeed: small response
changes, loose tolerances, mode budgets and FV convergence affect counts.
Evaluate error ALWAYS against analytic(original z,original T), only AFTER all
verdicts. Oracle does not determine planner, guards, probes or acceptance.

## Frozen defenses

Hash defense: compare hash of operational Request to the independently saved
original digest, never compare two corrupted objects.
Definition invariants: G(0)=1 and -G'(0)=T_original. For EACH solver probe G(0)
and G(i h), with h=1e-6/T_original. Estimate -G'(0)=-Im G(i h)/h
(complex-step derivative, real-coefficient transfer function). Probe modal at
4096 modes, FV at512 cells, regardless of task tolerance. Require for BOTH:
|G(0)-1|<=2e-4 and |estimated_mean/T_original-1|<=2e-3.
These a-priori allowances cover finite modal normalization tail (~1.5e-4),
FV discretization, rounding and derivative truncation in the clean model;
they are operational tolerances, not rigorous guarantees. Test clean false
rejections explicitly. No parameter changes after observing data.

Record D, D+hash, D+invariants, D+both, and each defense alone. Run all defenses
regardless of D verdict; reuse the exact same artifact and D verdict across
four policy comparisons. Compute probes per condition without cache, count
8192 modal terms and1024 FV cells per condition separately from task execution
and D cells. No shadow path. Timings descriptive, not independent samples.

## Audit

Report actual inaccurate acceptances, accurate rejections, clean false rejections,
wrong values blocked, and remaining misses per condition/policy. Retain negatives.
Raw original/operational Requests, independent digest, hook scale, probes,
outcomes and D steps in results.json. Record protocol introducing commit/hash,
committed implementation git_head, source hashes, seeds and exact environment.
Protocol-only commit, implementation-only commit BEFORE data, results commit.
Run scripts/check.sh and full non-timing numerical reproduction with saved logs.
No conclusions about consciousness or parameter retuning.
