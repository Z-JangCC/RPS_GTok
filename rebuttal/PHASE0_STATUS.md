# Phase 0 implementation status

_Generated during the active rebuttal execution. This file records verified progress and unresolved gates; it is not a claim that the complete rebuttal is finished._

## Completed in this turn

| Area | Evidence |
| --- | --- |
| Canonicalizer module | gptok2/canonical.py |
| Executable schema contract | gptok2/data/schema_spec.py and configs/schema_manifest.yaml |
| Canonicalization integration | GPTok2Tokenizer.fit and encode |
| Canonicalized baseline views | TokenViewBuilder canonical edge/adjacency/DFS/BFS views |
| Identifier/payload footprint utility | gptok2_tokenizer/payload.py |
| Strict topology metric | structure_fidelity now reports strict_topology_exact and WL hash separately |
| Complete token payload | EncodedProgram now stores graph payload and token-instance payload statistics |
| Schema round-trip | self-loop, duplicate edge, directed edge, node/edge attribute test |
| Frozen interface vocabulary | codebook freeze prevents inference-time interface mutation |
| Truncation audit fields | sequence_audit in downstream training metrics |
| Phase-0 audit runner | scripts/run_phase0_audit.py |
| E1 pilot | rebuttal/experiments/E1_canonicalized_baselines |
| E2 pilot | rebuttal/experiments/E2_runtime_scaling_pilot |
| E4 semantic pilot runner | scripts/run_semantic_downstream.py |

## Verified commands

~~~~text
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests -q
13 passed

PYTHONDONTWRITEBYTECODE=1 python scripts/reproduce_smoke.py
tokenizer smoke passed
consumer smoke passed
13 tests passed

python scripts/run_phase0_audit.py --out rebuttal/audits --permutations 8
8 adversarial graph families audited

python scripts/run_canonicalization_isolation.py \
  --out rebuttal/experiments/E1_canonicalized_baselines \
  --graphs 24 --permutations 6
240 per-graph/method rows written

python scripts/run_runtime_scaling.py \
  --out rebuttal/experiments/E2_runtime_scaling_pilot \
  --max-search-nodes 1000
23 scaling rows written
~~~~

## Pilot findings

- Raw edge, adjacency, DFS and BFS views had mean permutation stability 0.0 on the corrected synthetic permutation audit.
- Canonicalized edge, adjacency, DFS and BFS views had mean permutation stability 1.0 on the same pilot.
- RPS-GTok full and identifier-only views had mean permutation stability 1.0 on the same pilot.
- Highly symmetric or large tie groups trigger the explicit search budget and become non-certified fallback cases.
- A 10-node regular graph exceeded the 100,000 search-node budget in the Phase-0 audit.
- The runtime pilot therefore records exact completion and fallback rather than treating all stable outputs as theorem-level guarantees.
- The corrected E1 pilot reports raw serialization stability 0.0, canonicalized serialization stability 1.0, and RPS-GTok stability 1.0 on 24 synthetic graphs with six permutations per graph.
- The first RPS identifier-stability number (0.125) was a measurement bug: the baseline view stripped all action payloads while EncodedProgram used code-aware identifiers. After unifying both projections through the same payload splitter and preserving permuted attributes, RPS-GTok identifier stability is 1.0 in E1 fixed2.
- A sampled 21-dataset canonical-control audit now reports mean stability 0.955 for RPS identifiers, 0.725--0.778 for canonicalized serialization controls, and near-zero raw-control stability; see E1_21_canonical_controls and Table_R1b.
- The E2 pilot shows explicit fallback on larger path, cycle, regular and complete graph tie groups under a 1,000 search-node budget.
- A PROTEINS semantic pilot now compares raw edge list, canonical edge list, RPS identifier-only and RPS Full-Embed under the same small Transformer and zero truncation. The pilot is diagnostic only because it uses 10/5/5 graphs and two epochs.
- The initial semantic summaries exposed a serious evaluation defect: three-epoch single-seed runs on imbalanced 60/20/20 splits could produce identical majority-like predictions. Those summaries are now historical diagnostics only. The corrected runs materialize each view once, use three seeds, class-weighted loss, balanced accuracy, prediction histograms and view-sequence hashes; only the latest `_corrected_vN` run per dataset is used in Table_R4_semantic_downstream_all.
- A COLLAB tiny run completed separately because its large graph sequences and patch proposal cost are extreme; its explicit truncation rates are retained in the same table rather than silently pooling it with the smaller datasets.
- The semantic pilot exposed and led to fixes for three independent scalability bottlenecks: quadratic frontier scoring, dense refinement over non-neighbors and unbounded motif-score work.
- The materialized schema-v2 corpus in the parent project was converted into rebuttal/data_21/ for all 21 named datasets. The pilot-20 token protocol completed all 21 dataset runners and produced rebuttal/experiments/token_level_21_pilot20/.
- The 21-dataset fast60 protocol completed all 21 runners and produced rebuttal/experiments/token_level_21_fast60/ with per-graph payload, strict-audit, canonical-completion and permutation-stability fields.
- The fast60 summary now has a generated LaTeX artifact at rebuttal/tables/Table_R9_token_level_21_fast60.tex, including identifier cost, payload footprint, expansion match, permutation stability and canonical exact-completion columns.
- After deterministic complete-stream serialization was added, the 21-dataset protocol was rerun as token_level_21_fast60_final; Table R9 now reflects the final stream byte accounting.
- The final bounded-strict protocol has 20 datasets with held-out test graphs; `ieee_power_grid` has only three records and leaves zero test graphs under the predefined 70/15/15 split. It is now explicitly excluded from metric aggregates and listed in Table_R9_excluded_datasets.csv rather than reported as a zero-valued success. Table R9 uses canonicalizer-bounded strict audit rather than unbounded VF2 search.
- Deterministic self-contained stream serialization now has a direct round-trip unit test and is used in EncodedProgram complete_token_bytes.
- The first full pilot showed that strict topology audits are skipped for graphs above the configured exact-isomorphism size limit; those rows are explicitly marked rather than filled with WL-hash results.
- Large ogbg-ppa and dense collab samples remain the main runtime-heavy cases even after the fast proposal and sparse scoring fixes.
- Corrected semantic downstream runs completed at 60/20/20 for MUTAG, PROTEINS and IMDB-BINARY (three seeds and four views per dataset). Table_R4_semantic_downstream_all.csv/.tex now reports seed-level rows and the corrected aggregate table reports mean±standard deviation. Per-run protocol manifests are stored beside each summary.
- The old IMDB-BINARY no-truncation and COLLAB pilot results remain diagnostic artifacts; they are not pooled with the corrected table. The corrected truncation audit is Table_R5_truncation_semantic.tex.
- Runtime instrumentation now uses a stable sparse node-ID state during merge contractions, avoiding full NetworkX graph relabeling on each MERGE_NODE.

## Remaining blocking work

1. Expand E1 from synthetic graphs to the full 21-dataset token-level protocol.
2. Complete strict attribute, direction, multiedge and self-loop reconstruction metrics.
3. Implement complete payload serialization and full bytes-per-edge accounting.
4. Extend the corrected multi-seed semantic protocol to the full paper-scale dataset sizes before making any broad downstream claim.
5. Add full-scale semantic graph labels and parameter-matched Token-only/Full-Embed comparisons.
6. Add coverage-controlled reconstruction ablations.
7. Generate final TeX tables, figures and reviewer-linked response files.
