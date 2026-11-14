# Owner-reported B004 replay

The owner reports a confirmation replay on macOS arm64 with Python 3.12.13,
NumPy 2.3.5 and SciPy 1.17.0. With --runtime-override it took 43 seconds and
about 127 MB peak memory. audit_b004 passed all 12 series and 2016 conditions.
The comparison returned decisions_pass=true, with empty differing_decisions
and numerical_diagnostics. summary.json matched byte for byte.

The decompressed contents of all 12 series archives also matched the first
run byte for byte, whereas the .gz files differed. The owner's macOS zlib
version 1.2.12 produced different compressed bytes without changing outcomes.
Starting with B005, manifests record both compressed-file SHA-256 and
SHA-256 of decompressed canonical JSON; cross-platform comparison uses the
latter rather than treating compression differences as numerical differences.

Both ./scripts/check.sh and ./scripts/check-noise.sh passed on that platform:
519 main and 30 additional tests each, about 3 minutes 40 seconds per suite.
The owner also reports that the B003 path correction works.

These are the owner's observations, not a new local execution or independent
inspection of the macOS archives. No B004 outcome has been changed.
