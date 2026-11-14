# Development corrections

The first development run used code 3c7c2af and passed exposure/discernibility,
but independent archive audit failed for the journal policy. The source dictionary
was traversed in generator insertion order, whereas canonical JSON sorts its keys.
The archive therefore did not explicitly retain the journal's source-ingestion
order. The first failure was series 20270401, false_measurement/private, T/32/
measurement: the first 43 actions matched, and action 44 selected a different
source in the auditor. This is a representation defect, not numerical drift.

The failed first archives and audit log are retained. Before freezing, an explicit
source_order list is added and verified. The journal uses that list in execution
and independent audit, with a JSON-roundtrip regression test. A new development
run and its sequential replays are required. No confirmation data have been run.
