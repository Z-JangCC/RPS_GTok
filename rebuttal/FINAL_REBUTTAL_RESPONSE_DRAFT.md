# KDD27 rebuttal response draft

## Response to Reviewer 3HWf

We thank the reviewer for identifying three evaluation risks: weak topology
surrogates, 512-token truncation, and Full-Embed comparability. We corrected the
semantic pipeline after discovering that the first table mixed short,
single-seed pilot runs with corrected runs and allowed majority-like collapse.
The authoritative semantic table now uses three seeds, class-weighted loss,
weighted sampling, balanced accuracy, Macro-F1, label/prediction histograms,
confusion matrices, view hashes, and validation-only selection (Table R4).

Evidence: [Table R4 aggregate](tables/Table_R4_semantic_downstream_corrected_aggregate.csv) and [locked no-truncation control](tables/Table_R5_truncation_semantic_locked_aggregate.csv).

The shared-plain no-truncation IMDB control has zero truncation and zero
prediction collapse. It reports balanced accuracy 0.632 for raw edge list,
0.597 for canonical edge list, and 0.535 for RPS-GTok. We retain this
negative result and therefore do not claim universal downstream superiority.
Full-Embed is reported separately as an enriched-input auxiliary, never as the
primary tokenizer-only comparison.

For the same canonical relabeling control, Table R1b applies canonicalization
to edge, adjacency and DFS/BFS serializations. Raw stability is approximately
0.024, canonical controls are 0.725--0.778, and the cache-corrected RPS
representation audit is reported separately with exact/fallback status.

Evidence: [cache-corrected representation audit](tables/Table_E1_cachefixed_permutation_audit.csv).

## Response to Reviewer 8tZd

The technical contribution is the executable graph-token contract: a frozen
vocabulary-indexed primitive fragment, explicit token-local payload, complete
primitive fallback, deterministic serialization, and a schema-aware expansion
interpreter. Canonicalization is adopted as a reference frame, not claimed as
a new polynomial-time graph-canonization algorithm.

Reconstruction/downstream linkage is reported as association, not causality.
The coverage ablation changes strict reconstruction from 1.0 to 0.5 and 0.125
under controlled token deletion. The fixed-budget topology control and the
all-dataset structural audit are retained as negative controls because simple
node/edge statistics are too easy for raw edge lists.

The fair GIN reference is now complete for MUTAG, PROTEINS, IMDB-BINARY and
COLLAB. Mean balanced accuracy is 0.876, 0.700, 0.621 and 0.648 respectively.
These are model-level references and are not conflated with tokenizer-only
comparisons.

Evidence: [matched GIN reference](tables/Table_E8_gin_foundation_baseline.csv).

## Response to Reviewer bYQg

We agree that exact permutation-stable reversible serialization entails a
canonical reference and inherits graph-isomorphism/canonical-labeling
complexity. The implementation explicitly records search budget, branch count,
exact completion, fallback and timeout; fallback rows are never reported as
exact. Runtime tables cover high-symmetry and large graph stress cases.

The reconstruction claim applies to complete token instances containing the
identifier and expansion payload. The complete-stream audit serializes raw
token instances, ordered arguments and graph payload, and recovers them exactly.
Unconstrained identifier-only autoregressive generation is not claimed to be
automatically lossless.

The paired reconstruction/downstream ablation was corrected so the same
`rps_gtok_full` sequence and deterministic drop mask are used for decoding and
consumer training. Reconstruction decreases monotonically under deletion, but
the small-data consumer collapses; this is reported as a diagnostic association
and not as a causal downstream claim.

Evidence: [paired-ablation audit](experiments/E5_paired_semantic/PAIRED_ABLATION_AUDIT.md) and [fixed-budget control](tables/Table_R13_fixed_budget_topology_aggregate.csv).

Schema and attribute retention are enumerated in Tables R8 and R10. Continuous
attributes are reported under the declared raw/quantized policy; round-trip
tests cover directed edges, duplicate edges, self-loops, node/edge types and
attributes.

The first permutation audit also exposed two implementation defects: a
projection mismatch and a content-insensitive RPS cache key. Both are fixed.
The cache now fingerprints edges, types, attributes and direction. The cycle
anchor now uses cyclomatic rank rather than a traversal-dependent cycle-basis
member. All pre-fix permutation results are excluded from authoritative tables.

## Response to Reviewer HFMW

We added canonicalized reversible baselines, complete payload accounting,
corrected semantic controls, bounded runtime audits, and matched GIN references.
The revised contribution separates reference-frame canonicalization from the
executable primitive-fragment and payload contract. The authoritative-results
index explicitly excludes pilots, failed runs, partial exact-only workers and
pre-cachefix permutation outputs.

## Response to Reviewer Rkxp

The complete footprint reports identifier bytes, token-local payload bytes,
complete stream bytes and amortized artifact/rulebook bytes separately. The
compact topology/attribute audit additionally reports 27.417 raw/canonical edge
context bytes, 16.208 topology-only RPS-GTok identifier-plus-codebook bytes,
and 334.417/467.958/802.375 RPS-GTok topology/attribute/full bytes.
21-dataset token protocol has 20 evaluable datasets; the three-record IEEE
Power Grid sample is explicitly excluded from held-out metric aggregates.

The converted 21-dataset JSONL initially had no graph-level y labels. We did
not impute them: source-index/topology recovery now provides audited labels for
the eligible TU, GNNBenchmark, and OGB sources, while PCBA remains official
multilabel and ambiguous Planetoid/OGBN joins remain quarantined. For the
remaining unsupported records we report the structural regression audit rather
than fabricating semantic labels. Across its 16 evaluable datasets, RPS-GTok
has lower MAE on 6/16 while raw edge list is lower on 10/16; this result is
dataset-dependent.

Runtime/scalability tables report canonicalization time, search budget,
fallback, timeout and exact completion. Dense COLLAB, OGBG-PPA and highly
symmetric graphs are retained as explicit outliers.

The higher-budget all-21 nauty audit now processes all 974 materialized graphs
in each of the 21 datasets, including superpixel, road, IEEE, synthetic, AST
and directed KG records. Simple graphs use a direct nauty path; attributed or
multigraph cases use a WL-refined incidence fallback that preserves node/edge
attributes, parallel edges, self-loops and source/destination roles. Eight
spawned workers use a uniform 300-second hard budget per graph: all 974 pairs
finish with certificate equality 1.000, including all 191 directed inputs,
with zero timeout and zero error. This certifies the full materialized corpus;
it does not imply a polynomial worst-case guarantee or arbitrary-graph
completion within a finite budget.

Evidence: [runtime scaling](tables/Table_R2_runtime_scaling_pilot.csv), [dataset runtime audit](experiments/E2_runtime_dataset_fast2/summary.csv), [full all-21 certificate audit](tables/Table_E11_nauty_all21_full_v12.csv), and [schema/attribute audit](tables/Table_R10_schema_attributes.csv).

The lower-budget 60-second audits are retained as resource-sensitivity
diagnostics, not merged with the uniform 300-second full-corpus table.

## Final claim boundary

The revised evidence supports executable payload semantics, complete footprint
accounting, conditional permutation stability, and dataset-dependent RPS
utility. It does not support universal semantic superiority, universal exact
canonicalization, or automatic lossless identifier-only generation.

### External-blocker remediation (2026-10-05)

We audited the missing-label source rather than imputing labels. Source-index
plus topology/shape joins recover sampled targets for MUTAG, PROTEINS,
IMDB-BINARY, ENZYMES, CIFAR10, MNIST, OGBG-MOLHIV, OGBG-MOLPCBA and OGBG-PPA;
PCBA retains its official multilabel vector. The IEEE cases have a
separate declared `pandapower.runpp` mean-voltage target. The machine-readable
audits are `audits/label_recovery_audit.csv` and
`tables/Table_R11b_label_recovery_expanded.csv`; labeled copies are in
`data_21_labeled/`. COLLAB and ambiguous ego joins remain excluded.

For exact permutation evidence, we added an independently certified nauty
backend with a direct simple-graph path and an orientation-preserving,
WL-refined attributed incidence fallback. The full materialized 21-dataset
audit uses all 974 graphs, eight isolated workers and a uniform 300-second
per-graph budget. All 974 original/permuted pairs complete with certificate
equality 1.000, including 191 directed inputs. This repairs the exact-evidence
path while keeping the absence of a polynomial worst-case guarantee explicit.

We also rebuilt the MOLHIV semantic control from the official source and after
the final schema-deserialization fix rather
than using the imbalanced 60-graph prefix. The authoritative E14 uses 120
positive and 120 negative source-indexed graphs with a 168/36/36 split, a
frozen union vocabulary, the same plain adapter, equal 247,842 parameters,
zero truncation and zero prediction collapse. Raw, canonical, and RPS-GTok
obtain mean balanced accuracy 0.750, 0.722 and 0.769, respectively, over three
seeds. This is a dataset-specific fair gain, not a universal superiority claim.

For the reviewer’s concern that raw edge lists may be sufficient for toy
statistics, E15 evaluates a physical task: pandapower load perturbations with
`runpp` mean bus voltage targets and case-disjoint topology splits. The
authoritative frozen-union/shared-plain comparison uses exactly 70,433 consumer
parameters and zero truncation: raw MAE 0.0394, canonical MAE 0.0383, and
RPS-GTok MAE 0.0391. RPS-GTok improves over raw by about 0.8%, while canonical
remains stronger; this is task-dependent topology value, not universal
superiority.

Evidence: [post-schema MOLHIV table](tables/Table_E14_ogbg_molhiv_balanced_post_schema_aggregate.csv), [post-schema physical-task table](tables/Table_E15_powerflow_perturbed_post_schema_aggregate.csv), and [matched-consumer audit](audits/matched_consumer_runs.json).

For the remaining materialized graphs without auditable domain labels, we add a
separate topology-only proxy rather than inventing semantic classes. E16 writes
labels to an external sidecar keyed by `graph_id`; each target is recomputed from
`num_nodes`, `edges` and `directed` only, while the model input records retain
`y=null` and exclude node/edge attributes and metadata from label construction.
The full 21-dataset control uses raw edge list, canonical edge list and
RPS-GTok under one shared plain Transformer, train-only union vocabulary,
three seeds and an 8,192-token budget. All 63 aggregate rows pass the
alignment/leakage audit and have zero test truncation. These results are
structural proxy evidence, not recovered semantic labels.

Evidence: [constructed-task audit](audits/constructed_task_audit.csv), [constructed-task protocol](constructed_tasks/protocol.json), and [all-21 no-truncation aggregate](tables/Table_E16_constructed_topology_proxy_notrunc_aggregate.csv).
