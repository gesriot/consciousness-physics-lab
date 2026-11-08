# E014 supplemental fresh-solve reproducibility audit

This is a post-result reproducibility check, not a new preregistered experiment
or validation. The original E014 protocol, maps, candidate definitions and
selection are unchanged. Implementation was committed as de01542 before this
supplemental run.

The audit reads the exact saved binary64 requests in the original E009 and E011
archives, with no sampling or input regeneration. It freshly evaluates the
unchanged graded FV solvers for 192 E009 surface paths and 576 E011 paths
(192 tasks times three solved models: sphere, scaled sphere and slab).
All 768 paths, comprising 5,376 FV solves and 6,242,304 solved cells, exactly
match the archived values at every grid. Archive byte hashes also match the
original E014 cost replay. These comparisons were persisted before applying
any candidate decision to the freshly evaluated paths.

The unchanged C0, C1, C2 and C3 rules then consume fresh path prefixes with
unchanged supplied artifacts, accuracy labels, resolution gates, hash checks
and invariant gates. All 2,688 condition rows match the original replay exactly
for every candidate: 10,752 candidate records, including every eligible decision
step, verdict, combined verdict, stopping reason and decision-cell count.
All E009, E011 and combined aggregate endpoints and cost fields also match
exactly. There are no differing rows and no numerical path discrepancies.

This audits the FV numerical replay from saved inputs. It does not freshly
regenerate artifacts or re-execute gates, and its solve-cell count is additional
reproducibility work rather than an operational policy cost. The original
conclusion remains: C1, C2 and C3 incur zero observed lost accurate accepts on
these archives, and the registered simplicity tie-break selects candidate C1.

The complete freshly computed paths, original requests, grid comparisons and
source/environment metadata are retained in fresh-solve-audit/fresh-paths.json.gz.
All fresh candidate rows and summaries are in
fresh-solve-audit/fresh-cost-replay.json.gz. The small machine-readable summary
is fresh-solve-audit/summary.json; execution output is fresh-solve-audit.log.
