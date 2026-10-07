# Final rebuttal protocol: RPS-GTok++ with auditable optimality gates

This document supersedes informal “make RPS win” instructions. The method may
be improved, but no result is accepted because it is favorable. RPS-GTok++ is
accepted as the primary method only if it wins a predeclared validation/test
comparison under the gates below; otherwise the negative result is reported
and the claim is narrowed.

## Method frozen for the final run

RPS-GTok++ retains the original executable RPS program and adds two explicit,
permutation-invariant anchor families:

1. binned graph cardinality/density/triangle anchors; and
2. two rounds of degree-neighborhood WL summary anchors.

The implementation now extends the RPS anchor family with deterministic degree
histograms, component/cycle summaries and four rounds of local WL signatures;
the matched `*_plus` controls receive the same anchor family. This is a frozen
design change, not a test-set-driven feature selection.

The anchors are emitted after the executable primitive/motif stream through
`rps_gtok_plus`. They are not a hidden side channel: the matched controls
`edge_list_plus` and `canonical_edge_list_plus` contain the same anchor family.
The only representation difference is therefore the graph token stream.

The generation-safe control is `complete_stream_tokens(encoded)`. It emits the
identifier, raw token, ordered arguments, and canonical graph payload in a
finite-state stream. `complete_stream_to_tokens` round-trips the raw token
instances and graph payload. This is evidence for a self-contained encoding
control, not a claim that an unconstrained identifier-only AR model is always
decodable.

## Reviewer-to-experiment coverage

| Concern | Final evidence | Acceptance condition |
|---|---|---|
| canonical controls | E1 canonical edge/adjacency/traversal/BPE controls | same canonicalizer, same split, exact/fallback status reported |
| GI/canonicalization complexity | E2 runtime, search-node budget, timeout, fallback | worst-case exponential upper bound stated; no fallback reported as exact |
| payload/generation gap | E3 footprint + E6 complete-stream audit | raw token and payload round-trip = 1.0; identifier-only never called lossless |
| schema/continuous attributes | schema manifest + attribute round-trip tests | original vs quantized target named per dataset; OOD fallback counted |
| truncation | E4 per-view 512 and common no-truncation subset | zero truncation in the no-truncation comparison |
| realistic downstream tasks | E4 semantic labels + optional OGB/MOLHIV | same split, same test set, class-balanced metrics |
| all-dataset corpus | E9 structural regression plus Table R11 label audit | no domain labels fabricated; undersized datasets explicitly excluded |
| reconstruction/downstream link | E5 randomized coverage ablation | reconstruction and downstream metrics reported jointly; correlation/CI, not causal overclaim |
| parameter/objective confounding | E7 fair grid | union vocabulary, shared adapter, same grid/epochs/seeds; validation-only selection |
| foundation-model comparison | E8 GIN/GraphSAGE/GraphGPS-compatible controls | identical graph split, budget and early-stopping rule |
| novelty | method table + canonical controls + complete payload stream | contribution attributed to executable token semantics and measured footprint, not canonicalization alone |

## Fairness and correctness gates

Every candidate must satisfy all of the following before test metrics are
read:

- exactly the same train/validation/test graph IDs across views;
- vocabulary fitted on training graphs only; a union vocabulary fixes embedding
  table size for the primary comparison;
- identical candidate architecture grid, optimizer, epoch budget, batch size,
  seeds and early stopping rule;
- no test metric used in hyperparameter or method selection;
- at least three independent seeds, with mean, standard deviation and 95%
  bootstrap interval;
- class-weighted cross entropy plus a deterministic weighted sampler on the
  training split for imbalanced tasks; validation checkpoints predicting one
  class are penalized and reported as failed diagnostics;
- balanced accuracy, Macro-F1, majority accuracy, prediction histogram and
  confusion matrix recorded;
- no view-sequence hash collisions, no duplicate rows and no missing runs;
- single-class predictions are flagged and cannot be presented as a success;
- token footprint includes identifiers, payload, complete stream and amortized
  artifact/rulebook bytes;
- exact canonicalization and bounded fallback are separate categories;
- all continuous attributes have an explicit quantization/fallback audit.
- batch padding is dynamic, while the global max length and truncation audit are
  unchanged; runtime comparisons therefore do not pay avoidable padding cost.

## Primary optimality criterion

The primary metric is mean test balanced accuracy on the predeclared semantic
task suite, with Macro-F1 as a co-primary diagnostic. A method is called
“best” only when its paired bootstrap interval is above every baseline on the
primary metric, or when it is Pareto-best in balanced accuracy and complete
bytes/edge. Otherwise use “competitive” and report the negative result.

If the paper presents an RPS family result, `select_view_family.py` may select
among RPS identifier-only, Full-Embed and anchor views using validation data.
The raw and canonical families receive the same selection opportunity; this is
family-level model selection, not cherry-picking a test winner.

## Reproducible entry points

```bash
python scripts/run_fair_downstream_grid.py ...
python scripts/run_complete_stream_audit.py ...
python scripts/audit_semantic_runs.py
python scripts/audit_all_rebuttal_results.py
python scripts/verify_release.py
pytest -q
```

For the materialized 21-dataset corpus, use
`run_all_dataset_structural_downstream.py` and
`merge_all_dataset_structural_workers.py`. That corpus has no graph-level
labels, so E9 is explicitly a structural regression audit rather than a
fabricated semantic benchmark. The authoritative E9 outputs are Table R11
(task availability) and Table R12 (seed-level and aggregate structural
metrics).

The topology-sensitive downstream extension is implemented by
`run_topology_shift_downstream.py` (fixed-budget topology regression under an
independent test-node permutation) and
`run_permutation_matching_downstream.py` (same-graph-versus-different-graph
pair matching under node permutation). These tasks directly test whether
permutation-stable topology tokens help a Transformer; they are reported
separately from the simple node-count sanity check.

The authoritative tables must be generated only after these commands pass.
