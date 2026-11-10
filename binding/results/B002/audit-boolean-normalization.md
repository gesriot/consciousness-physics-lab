# Audit-only scalar Boolean normalization

The initial confirmation audit rejected a stored false safety flag because the
recomputed conjunction returned a numerical-library Boolean scalar instead of a
built-in Boolean. Both values were false. For example, series 04, permissive,
external, budget 8, G, candidate 0:22 has decision margin 1.2067403982959605e-5
below threshold 3.0517578125e-5. No numerical verdict or recorded outcome changed.

The auditor now explicitly converts its derived conjunction to bool before the
existing exact comparison. A synthetic regression uses values 1, 0.999, 0.9989,
0.99889 with tolerance 0.00004; it accepts the correctly stored false flag and
still rejects a mutated true flag. Initial failure evidence is retained separately.
The confirmation numerical source files and all experiment outputs are unchanged.
