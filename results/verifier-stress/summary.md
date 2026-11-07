# E007: three-way decisions and boundary-layer stress

Protocol commit: 0bcda74f7861cbb85517c796031319a42bf73409. Execution code: 6409dd0e210506ed21c65a773d5eeeb3bab85365.
Protocol SHA-256: 87213436eb555b5b6c4e94dd6aef280b8cb036a1bba697725d34b3d9184f1c2a.
192 independent tasks (96 broad, 96 high-frequency), 6 input conditions, 2 policies.
The 2,304 records are paired/dependent, not independent replications.

Primary negative result: 101/768 unique task/grid U-coverage misses across 86 tasks;
maximum error/U=3.0823350154740305. All misses have delta/h<0.5.
Broad cohort: 11/20 misses in that bin; enriched cohort: 90/149.
Every other registered bin has zero misses (see JSON for denominators and empty bin).
No miss occurs at decisive D steps (0/161 unique task/grids) or D3 steps (0/176).
Neither policy falsely accepts an inaccurate value. D3 makes zero violation claims
on accurate values, including the guard-passing subset. This does not certify U.

Both policies accept 573/759 accurate records. Both reject 32 clean accurate
and 154 accidentally accurate corrupted records. Seventeen accurate records
fail the resolution guard; seven are accepted by FV but blocked by the guard.
D3 detects 286 inaccurate violations and leaves 179 accurate records inconclusive.
D is binary: its rejections are not interpreted as evidence of violation.

D3 stops for futility 84 times. Compared with all shadow grids, it loses zero
accurate acceptances, but 34 subsequent violation decisions. This is a real
cost of the heuristic, even though final acceptance is not weakened.
No decisive reversal occurred on the full paths.

D decision cells: 7,257,088. D3 decision cells: 5,313,536, saving 1,943,552
(26.78%) versus D.
D3 saves 4,049,920 cells versus its per-record full-path equivalent 9,363,456.
Separate full shadow paths cost 1,560,576 cells (once per task); actual total
research FV computation is 14,131,200 cells, not the D3-only cost.
Measured verification time: D 0.804073s; D3 0.687278s;
shadow 0.141590s. These are descriptive, not a speed guarantee.

The unchanged U fails in the unresolved boundary-layer regime. No parameter was
retuned after outcomes. delta=1/sqrt(15|zT|) is the registered magnitude proxy,
not the phase-dependent physical decay length. Coverage bins are repeated grids
within tasks and carry no independent-sample confidence claim.

Checks: scripts/check.sh passed (37 laboratory + 30 solver tests).
Full numerical reproduction passed excluding only timing fields; see logs.
Run: python3 -m lab.verifier_stress --output results/local/e007.json.
