# Rebuttal completion audit

_Requirement-by-requirement evidence audit. A row marked pilot is evidence for implementation behavior but not a replacement for the full paper-scale protocol._

| Requirement | Authoritative evidence | Status |
| --- | --- | --- |
| Canonicalization implementation | gptok2/canonical.py and audits/canonicalization_audit.md | complete for implemented contract |
| Exact/fallback distinction | canonicalization_audit.json and E2 scaling CSV | complete |
| Canonicalized serialization baselines | E1_canonicalized_baselines_fixed2, E1_21_canonical_controls and Tables R1/R1b | complete pilot with corrected identifier projection and sampled 21-dataset control |
| Schema contract | schema_spec.py, schema_manifest.yaml and Table R8 | complete |
| Self-loop/duplicate/directed/attribute round-trip | test_phase0_contracts.py | complete |
| Identifier/payload separation | payload.py and EncodedProgram | complete implementation baseline |
| Deterministic self-contained stream | payload unit test | complete |
| Frozen interface vocabulary | codebook freeze behavior | complete |
| Strict topology audit | bounded canonical audit in evaluate.py | complete with explicit skip |
| 21-dataset materialized corpus conversion | rebuttal/data_21/ | complete sampled conversion |
| 21-dataset token protocol | token_level_21_final_bounded/ and Table R9 | 20 evaluable datasets; IEEE power-grid sample excluded because the fixed split leaves zero test graphs |
| Final stream-byte rerun | token_level_21_final_bounded/, E3_artifact_footprint_21/ and Table R9 | complete; payload plus amortized artifact bytes are reported; large exact cases remain explicitly skipped and labeled |
| Runtime/scaling | E2 pilot and runtime CSV | complete pilot; dense outliers documented |
| Semantic MUTAG/PROTEINS/IMDB | E4 corrected 60/20/20 summaries, three seeds, label/prediction/view audits | complete authoritative semantic pilot; no universal claim |
| Semantic COLLAB | E4 medium 10/5/5 diagnostic | complete diagnostic; dense outlier |
| 512 truncation audit | Table_R5_truncation_semantic_locked_aggregate | complete authoritative IMDB shared-plain control |
| No-truncation control | IMDB-BINARY max_len=2048 | complete |
| Coverage ablation | E5 summary and Table R7 | complete synthetic pilot |
| Vocabulary sensitivity | sweep CSV and Table R6 | complete synthetic pilot |
| Reviewer response files | reviewer_responses/ | complete draft evidence map |
| Complete payload stream | E6_complete_stream and payload.py | complete control-stream round trip; AR identifier-only generation remains excluded |
| Validation-only fair grid | E7_fair_grid and run_fair_downstream_grid.py | complete for authoritative MUTAG/IMDB controls; negative results retained |
| Locked shared-plain no-truncation control | E7_fair_grid/IMDB_corrected_no_truncation_shared_plain_v1 | complete for 60/20/20; zero truncation and zero collapse |
| Failed-run log | E7_fair_grid/FAILED_RUNS.md | complete; non-authoritative runs explicitly excluded |
| External graph baseline | E8_foundation_baselines and Table_E8_gin_foundation_baseline.csv | complete for four labeled datasets; GraphSAGE/GraphGPS optional |
| Per-dataset runtime audit | E2_runtime_dataset_fast2, E2_runtime_ogbg_ppa | complete bounded audit; exact scalability remains conditional |
| All-dataset label/task audit | Table_R11_all_dataset_task_availability.csv | complete; all materialized graphs have y=null |
| All-dataset structural baseline | E9_all_dataset_structural_edge_only and Table R12 | complete for 16 evaluable datasets; 5 undersized datasets explicitly excluded |
| All-dataset RPS structural protocol | E9_all_dataset_structural_rps_merged_final and Table R12 | complete for 16 evaluable datasets; no labels fabricated |
| All-dataset RPS structural protocol | E9_all_dataset_structural_rps_merged_final and Table R12 | complete for 16 evaluable datasets; RPS wins MAE on 6/16, raw edge list on 10/16 |
| Final manuscript edits in main_four.tex | abstract, contribution, canonicalization condition, formal boundary, downstream and conclusion wording | applied |
| Full 21-dataset exact strict audit with no skipped rows | not available due bounded exact search | conditionally complete; exact/fallback status and skipped rows are explicit |
| Autoregressive payload generation | not implemented | intentionally excluded by revised claim boundary; identifier-only generation is not claimed lossless |

## Current stop condition

The implementation and evidence package now include the revised method contract, executable audits, 21-dataset fast token protocol, corrected multi-seed semantic controls, coverage and vocabulary ablations, reviewer-linked response drafts, and manuscript claim repairs. The earlier identical semantic table was invalidated and replaced after detecting majority-prediction collapse and mixed-run aggregation. Exact canonicalization remains conditional by design; skipped cases are reported explicitly rather than replaced by WL hash. Broad semantic superiority claims remain out of scope; the rebuttal reports the corrected pilot and its negative controls rather than extrapolating beyond available labeled data.
