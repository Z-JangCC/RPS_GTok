# Rebuttal evidence summary

Submission-length responses are in
[`REBUTTAL_SHORT_2500_CHARS.md`](REBUTTAL_SHORT_2500_CHARS.md); the unrestricted
anonymous-code response is in
[`REBUTTAL_FULL_RESPONSE_ANONYMOUS.md`](REBUTTAL_FULL_RESPONSE_ANONYMOUS.md).

_Evidence package generated from the revised independent implementation._

## Main verified findings

1. Canonicalized edge, adjacency, traversal and frozen-BPE controls achieve the same permutation-stability range in the corrected E1 pilot, while raw controls do not.
2. RPS-GTok identifier and full-token views are stable in the same pilot.
3. The final nauty certificate backend completes all 974 materialized graphs in the 21-dataset corpus under a documented 300-second/graph budget; this is an empirical corpus result, not a polynomial worst-case guarantee.
4. The schema layer preserves loops, duplicate edges, directed edges and attributes in round-trip tests.
5. Identifier context cost and complete payload footprint are separately recorded.
5a. A compact 24-graph audit reports 27.417 raw/canonical topology context bytes,
16.208 topology-only RPS-GTok identifier-plus-codebook bytes, and RPS-GTok
topology/attribute/full components of 334.417/467.958/802.375 bytes.
6. The materialized parent corpus supports a 21-dataset fast60 token-level protocol with per-graph self-contained payload footprint, strict-audit, canonical-completion and permutation fields. The final rerun after deterministic stream serialization is in the token_level_21_fast60_final artifact.
7. The semantic evidence now uses corrected MUTAG, PROTEINS and IMDB-BINARY 60/20/20 runs with three seeds, class-weighted loss, balanced accuracy, prediction histograms, confusion matrices and view-sequence hashes. The latest corrected run per dataset is reported with mean±standard deviation, with conclusions stated by operating regime.
8. Controlled token-deletion ablations reduce strict reconstruction from 1.0 to 0.5 and 0.125 at nominal 10% and 20% deletion in the synthetic pilot.
9. Vocabulary sensitivity saturates at 72 active codes for requested sizes 128-1024 on the 48-graph pilot.
10. The final token protocol has 20 evaluable datasets; the three-record IEEE power-grid sample is explicitly excluded because the fixed 70/15/15 split leaves no test graph (see Table_R9_excluded_datasets.csv). Dense COLLAB and OGBG-PPA are reported as runtime-sensitive cases with their timings retained explicitly.
11. The authoritative IMDB-BINARY no-truncation control is now the shared-plain 60/20/20, three-seed run with 0% truncation and both prediction classes represented; 700-graph stress runs remain runtime diagnostics and are not mixed into the performance table.
12. The initially observed RPS identifier stability of 0.125 was a projection-measurement bug. The authoritative post-cachefix representation audit is Table E1, which reports raw/canonical/RPS equality together with exact-pair fractions; pre-cachefix permutation outputs are excluded.
13. The final rebuttal protocol adds RPS-GTok invariant anchors, a complete identifier+payload stream control, validation-only fair hyperparameter selection with a union vocabulary, and a matched GIN reference. These protocol additions define the evidence-supported operating regimes.
14. The locked shared-plain IMDB no-truncation control (60/20/20, three seeds) has 0% truncation and both prediction classes represented. Its balanced-accuracy means are 0.632 (raw edge list), 0.597 (canonical edge list), and 0.535 for the identifier-only projection; this is a payload-boundary diagnostic, while complete-stream validity is audited separately at 100%.
15. The materialized 21-dataset corpus contains no graph-level `y` labels. We therefore added a clearly labeled all-dataset structural regression audit (log1p node count), not fabricated semantic labels. Sixteen datasets are evaluable; five are excluded for insufficient graph counts. The raw edge-list and RPS-GTok runs use the same split, three seeds, bounded canonicalization, sparse patch budget and regression metrics. RPS-GTok leads on 6/16 datasets, identifying the structural regimes where executable invariant context is most useful.
16. Matched GIN references are complete for MUTAG, PROTEINS, IMDB-BINARY and COLLAB; these are reported as model-level baselines, not attributed to tokenizer-only effects. Fixed-budget MUTAG topology controls at 32/128/512 tokens characterize the simple edge-count regime, while E16 adds broader topology-dependent evidence.
17. After the continuation audit, a content-fingerprint cache fix superseded pre-fix permutation-sensitive outputs. The post-fix representation audit reports exact-subset token equality for canonical/RPS and separately labels bounded cases; only post-fix numbers are authoritative.
18. The post-schema fair-consumer audit freezes a union vocabulary and shared plain adapter. In balanced official OGB-MOLHIV, RPS-GTok reaches 0.769 mean balanced accuracy versus 0.750 raw and 0.722 canonical with equal 247,842 parameters, 0% truncation and no collapse. In case-disjoint pandapower load-perturbation prediction, RPS-GTok reaches 0.0391 MAE versus 0.0394 raw and 0.0383 canonical with equal 70,433 parameters and 0% truncation.
19. The final v12 nauty audit covers all 974 materialized graphs across all 21 datasets, including 191 directed inputs, with 974/974 certificate equality and a 100% valid-certificate rate under the uniform 300-second budget.

## Claim boundaries

- Canonicalization is adopted, not claimed as a new canonization algorithm.
- Permutation stability is conditional on exact canonicalization completion.
- Reconstruction applies to complete token instances with payloads.
- Identifier-only autoregressive generation is not claimed to be automatically lossless.
- ID Tok./edge measures model-facing context cost; complete storage additionally includes payload and metadata.
- Shared downstream claims require union vocabulary, shared adapter and equal total parameter count; per-view-vocabulary runs are diagnostic only.

## Evidence files

- [Method specification](../REVISED_METHOD_SPEC.md)
- [Experiment plan](../REBUTTAL_EXPERIMENT_PLAN.md)
- [Phase-0 status](../PHASE0_STATUS.md)
- [21-dataset token table](../tables/Table_R9_token_level_21_fast60.tex)
- [Semantic table](../tables/Table_R4_semantic_downstream_all.tex)
- [Truncation table](../tables/Table_R5_truncation_semantic.tex)
- [Locked truncation table](../tables/Table_R5_truncation_semantic_locked_aggregate.csv)
- [Semantic run correction audit](../experiments/E4_semantic_tasks/SEMANTIC_RUN_CORRECTION.md)
- [Coverage ablation](../tables/Table_R7_reconstruction_ablation.tex)
- [Schema contract](../tables/Table_R8_schema_contract.tex)
- [Vocabulary sweep](../tables/Table_R6_vocabulary_sensitivity.tex)
- [Authoritative results index](../AUTHORITATIVE_RESULTS_INDEX.md)
- [All-dataset task availability](../tables/Table_R11_all_dataset_task_availability.csv)
- [All-dataset structural downstream](../tables/Table_R12_all_dataset_structural_downstream_aggregate.csv)
- [Cache-fixed representation audit](../tables/Table_E1_cachefixed_permutation_audit.csv)
- [Canonical-BPE pilot](../experiments/E1_21_canonical_controls_bpe_full/summary.csv)
- [GIN foundation reference](../tables/Table_E8_gin_foundation_baseline.csv)
- [Paired reconstruction audit](../tables/Table_E5_paired_reconstruction_downstream_aggregate.csv)
- [Matched consumer audit](../audits/matched_consumer_runs.json)
- [Balanced MOLHIV control](../tables/Table_E14_ogbg_molhiv_balanced_post_schema_aggregate.csv)
- [Pandapower physical control](../tables/Table_E15_powerflow_perturbed_post_schema_aggregate.csv)
- [Full 21-dataset nauty audit](../tables/Table_E11_nauty_all21_full_v12.csv)
- [Constructed-label leakage audit](../audits/constructed_task_audit.csv)
- [All-21 constructed topology task](../tables/Table_E16_constructed_topology_proxy_notrunc_aggregate.csv)
