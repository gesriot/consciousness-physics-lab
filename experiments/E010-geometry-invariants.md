# E010: mean-preserving shared geometry errors

Commit this protocol before implementation and E010 data. E005–E009 are
prior development only. No retuning after any E010 outcome. Production code,
attention state and old archives remain unchanged. Hooks are TEST HOOKS only.

## Definition-derived predictions

The requested model is diffusion in a unit sphere, uniform volume average,
surface value one, diffusivity kappa=1/(15 T). Derive checks from that definition,
not either numerical solver. In radial dimension d, Lf=f''+(d-1)f'/r.
The local mean exit time solves kappa Lm1=-1, m1(1)=0, regular at zero,
so m1(r)=(1-r²)/(2 d kappa), and its volume mean is
T=1/[d(d+2)kappa]. The second moment solves kappa Lm2=-2 m1.
Integrating twice and averaging with d r^(d-1) gives
<m2>=4/[d²(d+2)(d+4)kappa²], hence G''(0)/T²=4(d+2)/(d+4).
Sphere: 20/7; cylinder: 8/3; slab: 12/5. The normalization is one.
The thin surface layer has width sqrt(kappa/z) and surface/volume ratio d,
so sqrt(zT)G(z) tends to d sqrt(kappa T)=sqrt(d/(d+2)) along positive real z.
Sphere: sqrt(3/5); cylinder: sqrt(1/2); slab: sqrt(1/3).
These predictions are checked against the sphere analytic function in tests.

Faults: both modal and finite-volume solvers compute slab d=1 or cylinder d=2,
with kappa=1/[d(d+2)T], preserving G(0)=1 and -G'(0)=T. Clean d=3 uses
production solver calls. Request and saved original digest are untouched.
Modal roots are (n-.5)pi, J0 zeros, n pi respectively; weights 2d/root².
FV uses uniform cells, volumes diff(r^d)/d and face areas r^(d-1).
The same geometry hook applies to artifact, D, and every probe. Constants/root
arrays may be cached; response/probe results may not. No forged trusted receipts.
Prediction: hash and both E008 invariants miss these faults; D can accept an
internally consistent wrong geometry. New checks should detect geometry errors,
not merely incorrect individual task values, with no clean false rejection.

## Frozen design and defenses

96 independent tasks, task seed20261120, order seed20261121. Draw log10T in
[-2,2], log10|zT| in[-6,5], phase in[-pi/2,pi/2], log10tol in[-4,-1],
in that order, uniform and independent. Pair clean/slab/cylinder for each task;
permute their order per task. Budget4096. Preserve frozen E006 D, planner .5tol,
resolution guard, adaptive uniform grids, U and convergence rule.
Independent digest and original T are saved before injection.

Run each defense even when D rejects. Retain standalone hash, normalization,
first moment, old joint invariants, second moment, high frequency and all checks;
report D alone and D combined with each. Same artifact/D verdict is reused.
E008 probes: modal4096 and FV512 at zero and i*1e-6/T_original;
normalization allowance2e-4, first-moment relative allowance2e-3, both solvers.
Second moment: each solver at zero and i*.01/T_original, modal4096/FV4096;
2*(Re G(0)-Re G(ih))/(h² T_original²), relative error to20/7 <=.02.
Subtraction cancels modal normalization tail; no normalization/renormalization.
High frequency: both solvers modal4096/FV4096 at real z=1e4/T_original;
compare sqrt(1e4)*G to sqrt(3/5), relative absolute deviation <=.03.
Allowances are a-priori operational margins for truncation, discretization and
finite-frequency bias, not certified error bounds. No parameter search.

Per condition: old checks8192 modal terms/1024 FV cells; second moment8192/8192;
high frequency4096/4096. Count all probes, including zero probes, with no cache
and no discarded warm-up evaluations. Time hash and each complete probe group,
artifact, and D separately. Root construction occurs inside timed solver calls.
Record standalone and combined detection, false acceptance, accurate rejection,
clean false rejection and work. Accurate faulty-geometry task values can occur
at small frequency or loose tolerance and must remain distinguished from clean.
Oracle analytic(original z,original T) is called only after ALL verdicts.

## Limits and audit

A trusted intake digest, original T, sphere definition and uncorrupted comparison
logic are assumed. Hash cannot attest model semantics. Finite probes do not
prove geometry or correctness. Faults matching normalization, first and second
moments evade moment-only checks; some can also match the leading asymptotic
coefficient or all sampled probe values. This is no completeness claim.

Pinned Python3.12.14, NumPy2.3.5, SciPy1.17.0. Save raw requests, probes, values,
D steps, policy verdicts, all timings/work, summary and platform/BLAS metadata,
protocol commit/hash and implementation/source hashes. Protocol-only commit,
implementation/tests commit before experiment, results/observations commit.
Run full scripts/check.sh, repeat same platform and compare all non-timing data
exactly. Preserve failures and negative findings. No consciousness claims.
