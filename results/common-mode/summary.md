# E008: common-mode agreement is not an independent specification

Protocol: 43f2b4401b3ec6a25ea399818b228c185a5fff0b.
Implementation before generation: a04f712137de38af5a2e21906fc4e9a861a7b963.
Protocol SHA-256: f3ac8c87edfea045d3126f711dda624977377fa2ea067b5ae3f965f8aa80ae0f.
Pinned Python 3.12.14 / NumPy 2.3.5 / SciPy 1.17.0.

96 independent fresh tasks; 384 paired condition records. Policies share each
condition's artifact and D decision and are not independent replications.
All accuracy labels use analytic of the ORIGINAL user Request, after decisions.
Frozen E006 D uses the unchanged final guard and FV refinement, not that oracle.

## Actual common-mode failure

- Clean: 92 accurate, all92 accepted; 4 inaccurate budget-limited outcomes rejected.
- Shared T×1000 Request: 79 inaccurate, D falsely accepts63; D accepts72 total.
  Of17 accidentally accurate outcomes, D accepts9 and rejects8.
- Shared conjugated-z Request: 49 inaccurate, D falsely accepts45; D accepts92 total.
  All47 accidentally accurate outcomes are accepted by D.
- Shared model scale×1000 with intact Request: 79 inaccurate, D falsely accepts63;
  D accepts72 total, including9 of17 accidentally accurate outcomes.

Thus corruption does not always create an inaccurate value, and an inaccurate
value is not always accepted. FV rejects24/96 in each scale condition, while
the resolution guard rejects4/96. In clean/conjugate conditions each rejects4.
Matching scale-condition counts do not make them independent experiments: they
are paired algebraically equivalent changes to solver inputs, while request
integrity and the guard's request differ.

## Defense results and costs

The original request digest was recorded in an independent intake list BEFORE
corruption. Its comparison rejects all96 in each request-corruption condition,
with zero clean false rejections. It rejects no model-scale cases and therefore
leaves63 wrong model-scale acceptances. Comparing two corrupted requests would
not supply this protection.

Definition-derived probes use G(0)=1 and -G'(0)=T_original for BOTH modal and FV,
with complex-step h=1e-6/T_original, normalization allowance2e-4 and relative
mean allowance2e-3. These frozen numerical allowances are not certified bounds.
Invariants reject all96 in both scale conditions, with zero clean false
rejections. They reject no conjugation conditions and leave45 wrong conjugated
acceptances: the local frequency request changes while these model invariants
remain intact. Normalization alone is insensitive to a time-scale change;
the first-moment check provides the scale detection.

D+both has zero inaccurate acceptances on this sample, retains all92 accurate
clean acceptances, and adds zero clean rejections (0/96 defense-alone false
rejections, 0/92 added accurate-clean rejection). It rejects all81 accidentally
accurate corrupted outcomes as well. This conservative integrity/model criterion
is not the same as merely checking whether an answer happens to meet tolerance.

Task execution costs506,372 modal terms. D costs985,088 FV cells. Defense probes
add3,145,728 modal terms and393,216 FV cells, computed per condition without cache.
Total research computation:3,652,100 modal terms and1,378,304 FV cells. Do not
multiply these by four policy comparisons. Descriptive wall times: execution
0.014414s, D verification0.246060s, defense0.107934s. These are not speed guarantees.

## Limits and verification

No production solver was modified. Explicit experiment-only hooks multiply the
mean-time argument in both numerical solvers. A scoped dependency substitution
feeds the FV hook into frozen D and is restored after each call; untrusted
experimental outcomes never masquerade as trusted Receipts. The shared-model
fault leaves Request/guard untouched. Both solver probes share the same faulty
model scale, but compare against independent original T and definition-derived
targets. The anchor channel itself is assumed trusted and uncorrupted; neither
hashes nor these two invariants protect against arbitrary adaptive corruption
or prove full model correctness. Zero observed misses is not a general guarantee.

No parameters were retuned and no implementation/metadata fix was needed after
outcomes. All raw records and audit hashes are in results.json. scripts/check.sh
passed43 laboratory and30 solver tests; checks.txt contains the full log. Full
reproduction matches every non-timing field, including provenance, exactly;
see reproduction.txt. Run: python3 -m lab.common_mode --output results/local/e008.json.
No conclusions about consciousness follow from these operational fault tests.
