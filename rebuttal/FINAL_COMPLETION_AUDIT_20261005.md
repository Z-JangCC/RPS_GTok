# Final completion audit (2026-10-05)

This document separates the requested rebuttal scope from claims that remain
externally impossible or intentionally out of scope.

## Verified complete

- All five reviewer issue groups have a point-by-point response in
  `FINAL_REBUTTAL_RESPONSE_DRAFT.md` and the reviewer response files.
- Canonicalized edge/adjacency/traversal controls, payload footprint, schema
  round trips, bounded runtime, fallback flags, and exact-completion fields are
  covered by authoritative tables and tests.
- Source labels are recovered only after source-index/shape/topology checks;
  ambiguous Planetoid/OGBN and partial COLLAB joins are quarantined.
- Raw converted records are checked for node-label fields; datasets containing
  such fields are structural-only unless a source-aligned, leakage-free task
  is separately authorized (`materialized_label_field_audit.csv`).
- For the remaining unlabeled corpus, a separate topology-only constructed task
  is available for all 21 datasets. Labels are sidecars keyed by `graph_id` and
  recomputed from `num_nodes/edges/directed` only; source records remain
  `y=null`, and `constructed_task_audit.csv` verifies alignment and forbidden
  fields.
- The all-21 constructed-task rerun contains 21 datasets × 3 views × 3 seeds
  under a shared plain Transformer, train-only union vocabulary and an 8,192-
  token budget with zero test truncation.
- E14 and E15 are post-schema, union-vocabulary, shared-plain, equal-parameter,
  three-seed and zero-truncation runs, checked by `matched_consumer_runs.json`.
- E15 checks a physical `pandapower.runpp` task with target leakage removed.
- Final gate: evidence audit, matched-consumer audit, 29 tests, release verify,
  links, and manuscript PDF all pass.
- A full materialized-corpus nauty certificate audit now processes all available
  graphs in each of the 21 datasets with an orientation-preserving directed
  gadget, eight spawn-isolated workers and a 300-second per-graph budget. Completed
  pairs all certify equality; the current v12 table covers 974 graphs with
  974 completed pairs, zero timeouts and zero errors under a uniform 300-second
  per-graph budget. It is generated with a direct simple-graph path,
  WL-refined attributed fallback and the spawn-safe executor.

## Explicitly not claimed

- Universal semantic superiority: E13 MUTAG is negative; E14 is a fair RPS
  win on MOLHIV; E15 is a small raw-vs-RPS improvement while canonical wins.
- Full 21-dataset real semantic classification: many converted records have no
  verified domain task. The new all-21 results are explicitly a topology-only
  constructed proxy and must not be described as recovered semantic labels.
- Polynomial-time or arbitrary-graph exact completion: the materialized 21-
  dataset corpus is 100% completed under the documented 300-second budget, but
  this is an empirical resource-bounded result rather than a worst-case
  complexity theorem.
- Automatic lossless identifier-only autoregressive generation: only complete
  identifier-plus-payload streams are guaranteed by the executable audit.

## Remaining external dependency

The source-label dependency remains for datasets whose local converted files do
not contain an auditable domain-level target. The scalable exact backend and
documented resource budget are now present; no exact-canonicalization blocker
remains for the current materialized 21-dataset corpus.
