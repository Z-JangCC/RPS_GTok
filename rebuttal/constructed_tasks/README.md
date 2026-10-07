# All-21 constructed graph-level proxy task

This directory contains a leakage-audited auxiliary task for materialized
graphs whose source files do not provide a verified graph-level label. It is
not a recovery of semantic labels. Each `*.jsonl` sidecar is keyed by
`graph_id` and stores a continuous topology-only target:

`log1p(mean effective resistance over deterministic within-component pairs)`

with normalized cycle rank as the explicitly recorded degenerate fallback. The
generator reads only `num_nodes`, `edges`, and `directed`; it does not read
`y`, node/edge attributes, metadata, or graph IDs for the target value. Source
records are unchanged (`y=null`). Graph IDs are sorted by SHA-256 only to form
deterministic, graph-disjoint non-empty train/validation/test partitions. The
optional binary regime uses the train median only and is not used in E16.

`scripts/audit_constructed_graph_tasks.py` recomputes every target from the
source records and checks sidecar alignment, non-empty splits, null source
labels, and the forbidden-field policy. The audit covers all 21 datasets.

E16 compares raw edge list, canonical edge list, and RPS-GTok++ in one shared
plain Transformer with a frozen union vocabulary, three seeds, and an 8,192
token budget. The authoritative aggregate is
`../tables/Table_E16_constructed_topology_proxy_notrunc_aggregate.csv`; all
21×3 rows have three seeds and zero test truncation. Results are structural
proxy evidence only and must not be described as real semantic classification.
