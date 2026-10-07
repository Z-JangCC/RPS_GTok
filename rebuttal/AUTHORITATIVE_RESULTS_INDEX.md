# Authoritative rebuttal results index

This index prevents historical pilots, failed runs, and cache-invalidated
permutation results from being cited accidentally.

## Authoritative

| Evidence | Status | Scope |
|---|---|---|
| `Table_R1b_canonical_controls_21.csv` | authoritative bounded canonical control | 21 dataset samples; exact/fallback status retained |
| `experiments/E1_21_canonical_controls_bpe_full/summary.csv` + `protocol.json` | authoritative frozen-BPE canonical control | 21 datasets × 6 graphs; BPE fitted before permutation evaluation |
| `Table_R2_runtime_scaling_pilot.csv` + `E2_runtime_dataset_fast2` | authoritative runtime diagnostics | synthetic and dataset-level bounded runtime |
| `Table_R3_full_footprint_pilot.csv` / `Table_R9_token_level_21_fast60.csv` | authoritative footprint | 20 evaluable datasets; IEEE excluded |
| `Table_R14_footprint_attribute_breakdown.csv` + `protocol.json` | authoritative compact topology/attribute footprint control | 24-graph topology/attribute audit; varint IDs, amortized codebook and explicit attribute component |
| `Table_R4_semantic_downstream_corrected_aggregate.csv` | authoritative semantic pilot | MUTAG/PROTEINS/IMDB, 3 seeds |
| `Table_R5_truncation_semantic_locked_aggregate.csv` | authoritative no-truncation | IMDB 60/20/20 shared-plain, 3 seeds |
| `Table_R8_schema_contract.csv` / `Table_R10_schema_attributes.csv` | authoritative schema audit | all materialized datasets |
| `Table_R11_all_dataset_task_availability.csv` | authoritative label audit | all 21 datasets; all y are null |
| `Table_R12_all_dataset_structural_downstream_aggregate.csv` | authoritative structural audit | 16 evaluable datasets, raw/RPS, 3 seeds |
| `E6_complete_stream` | authoritative payload audit | MUTAG/IMDB controls, exact raw-token/payload round-trip |
| `E10_topology_shift/MUTAG_post_cachefix_v1` | authoritative pilot | cache-fixed, 3 seeds, negative/simple-target sanity control |
| `E11_permutation_matching/MUTAG_exact_pair_input_audit.csv` | authoritative input audit | exact retained subset; token equality only |
| `Table_E8_gin_foundation_baseline.csv` | authoritative matched GIN reference | MUTAG/PROTEINS/IMDB-BINARY/COLLAB, 3 seeds |
| `Table_R13_fixed_budget_topology_aggregate.csv` | authoritative fixed-budget topology control | MUTAG budgets 32/128/512, raw/RPS, 3 seeds |
| `Table_E1_cachefixed_permutation_audit.csv` | authoritative cache-fixed representation audit | MUTAG/PROTEINS/IMDB-BINARY/COLLAB, 20-graph sample, 3 permutations |
| `Table_E5_paired_reconstruction_downstream_aggregate.csv` | authoritative paired diagnostic | MUTAG/PROTEINS capped subsets; same sequence/drop mask; negative causal control |
| `audits/label_recovery_audit.csv` + `data_21_labeled/` | authoritative label recovery | exact TU joins for MUTAG/PROTEINS/IMDB-BINARY/ENZYMES; partial COLLAB excluded |
| `Table_R11b_label_recovery_expanded.csv` | authoritative expanded source audit | TU/GNNBenchmark/OGB/Planetoid/OGBN/pandapower task availability and eligibility |
| `Table_E11_nauty_exact_semantic.csv` | authoritative semantic exact certificate audit | nauty-colored incidence certificates; completed pairs only, timeout rows separated |
| `Table_E11_nauty_all21_full_v12.csv` | authoritative full materialized-corpus exact certificate audit | all 974 materialized graphs in each of 21 datasets; direct simple-graph fast path plus WL-refined attributed incidence fallback, 300 s/graph hard timeout, 8 spawned isolated workers, 974/974 completed certificates |
| `Table_E14_ogbg_molhiv_balanced_post_schema_aggregate.csv` | authoritative balanced OGB control | post-schema-fix source indices, frozen union vocabulary, shared plain adapter, equal 247,842 parameters, 168/36/36 split |
| `Table_E15_powerflow_perturbed_post_schema_aggregate.csv` | authoritative physical downstream control | post-schema pandapower `runpp`, case-disjoint split, frozen union vocabulary, shared plain adapter, zero truncation |
| `audits/matched_consumer_runs.json` | authoritative fairness gate | E14/E15 equal parameter counts, union vocabulary, complete seed/view grids |
| `audits/materialized_label_field_audit.csv` | authoritative leakage gate | raw converted node-label fields are flagged and kept structural-only |
| `audits/constructed_task_audit.csv` + `constructed_tasks/` | authoritative constructed-label audit | 21 topology-only sidecars; source `y=null`, graph-ID alignment, train-only thresholds, forbidden-field policy |
| `Table_E16_constructed_topology_proxy_notrunc_aggregate.csv` | authoritative all-21 constructed-task control | 21 datasets × 3 views × 3 seeds, shared plain Transformer, 8,192-token budget, zero test truncation |
| `rebuttal_number_manifest.json` | generated citation-number manifest | all numerical values cited in the final response draft |
| `final_rebuttal_gate_report.json` | final automated submission gate | evidence audit, pytest, release verify, links, manuscript PDF |
| `FINAL_COMPLETION_AUDIT_20261005.md` | scope/completion audit | verified requirements, explicit non-claims, remaining external dependencies |
| `reviewer_responses/REBUTTAL_SHORT_2500_CHARS.md` | submission-length response | one <2,500-character response per reviewer |
| `reviewer_responses/REBUTTAL_FULL_RESPONSE_ANONYMOUS.md` | anonymous-code long response | detailed reviewer-by-reviewer protocols, code entry points, results and claim boundaries |
| `reviewer_responses/REBUTTAL_DRAFT.md` | formal response entry point | points to the canonical short and unrestricted long versions |
| `reviewer_responses/REBUTTAL_STRATEGY.md` / `REVIEWER_RESPONSE_MATRIX.md` | response planning/audit | cross-reviewer priorities and issue-to-evidence traceability |

## Diagnostic only

- all `*_pilot`, `*_full`, `*_no_truncation` files not listed above;
- `E7` runs before shared adapter/vocabulary correction;
- E10/E11 permutation results generated before the content-fingerprint cache fix;
- initial concatenation-head E11 aggregate;
- bounded/fallback rows presented as exact canonicalization;
- partial exact-only worker directories with no complete `results.csv`.
- `E5_paired_semantic/MUTAG_corrected_v2`: valid reconstruction trend but collapsed downstream consumer; diagnostic only.
- partial COLLAB label join in `audits/collab_partial_label_join.jsonl` is quarantined and must not be used.
- `Table_E13_recovered_semantic_aggregate.csv` is diagnostic only: the valid 30-epoch MUTAG rerun is outside the primary operating-regime evidence, and the incomplete CIFAR run is quarantined.
- `Table_E14_ogbg_molhiv_balanced_aggregate.csv` is diagnostic only because it used per-view vocabularies; cite only the union-vocabulary table.
- `Table_E14_ogbg_molhiv_balanced_union_aggregate.csv` is diagnostic until the post-schema-deserialization rerun completes; an edge-row/attribute alignment defect was fixed afterward.
- `Table_E14_ogbg_molhiv_balanced_union_aggregate.csv` is diagnostic because it predates the schema-deserialization fix; cite only the post-schema table.
- `Table_E15_powerflow_perturbed_aggregate.csv` is diagnostic only because it used per-view vocabularies; cite only the union-vocabulary table.
- `Table_E15_powerflow_perturbed_union_aggregate.csv` is diagnostic until the post-schema-deserialization rerun completes.
- `Table_E15_powerflow_perturbed_union_aggregate.csv` is diagnostic because it predates the schema-deserialization fix; cite only the post-schema table.
- `Table_E11_nauty_all21_bounded3.csv`, `Table_E11_nauty_all21_full_v3.csv`, `Table_E11_nauty_all21_full_v4.csv`, `Table_E11_nauty_all21_full_v5.csv` and `Table_E11_nauty_all21_full_v7.csv` are superseded by the direct-fast-path/WL-refined spawn-safe full-corpus v12 audit; retain them only as reproducibility pilots.

## Current claim boundary

The evidence supports executable payload semantics, complete footprint
accounting, exact certificate invariance on the full materialized 21-dataset
corpus under the documented budget, and RPS-GTok utility in the reported
operating regimes. Claims remain scoped to those regimes, and no polynomial
worst-case canonicalization guarantee is asserted.
