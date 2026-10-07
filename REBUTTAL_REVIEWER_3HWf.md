# Rebuttal response to Reviewer 3HWf

This file is the independent detailed response for Reviewer 3HWf. All paths below are direct paths in the anonymous repository: [rebuttal/](https://anonymous.4open.science/r/RPS_GTok/rebuttal/).

We sincerely thank the reviewer for noting that the overall design is coherent at a high level. We have carried out the following supplementary experiments and provided further explanations regarding the issues you raised.

## Deeper motivation and technical distinction

An isomorphic triangle after node-ID permutation can yield different raw edge or DFS/BFS sequences, so an ordered Transformer sees two inputs. Canonicalization fixes order but not executable arguments; primitive compilation is order-sensitive; BPE does not enforce graph-state preconditions. RPS-GTok couples canonical ordering, executable primitives, payload-bearing fragments and primitive fallback. This makes the Transformer input both permutation-stable and executable.

## Expanded downstream evaluation

The original topology-surrogate evaluation is complemented with source-aligned PROTEINS and MOLHIV labels and a physical power-flow task. The corrected semantic aggregate is [Table R4](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R4_semantic_downstream_corrected_aggregate.csv). PROTEINS BA is 0.6562 for raw edge, 0.6875 for canonical edge and 0.8125 for RPS-GTok. Balanced official MOLHIV BA is 0.7500, 0.7222 and 0.7685, respectively. In the case-disjoint PowerFlow task, MAE is 0.03942, 0.03829 and 0.03910.

Commands:

```bash
cd RPS_GTok_Review
PYTHONPATH=. python scripts/run_semantic_downstream.py --help
PYTHONPATH=. python scripts/audit_matched_consumer_runs.py
```

These results show that the executable invariant interface is useful beyond density, edge-count and triangle-statistic targets.

## Truncation and Full-Embed fairness

At the original 512-token budget, raw and canonical edge views have 10% test truncation on IMDB, whereas RPS-GTok has 0%; the corrected per-view data are in [Table R5](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R5_truncation_semantic.csv). The locked no-truncation control sets all views to 0% truncation, and E14/E15 also have 0% train/validation/test truncation. Full-Embed is auxiliary; the primary comparison uses Token-only views, shared adapters, a train-only union vocabulary and equal parameters. Therefore the source-aligned downstream gains are not attributed to an additional feature channel or discarded tokens.

## Canonical baselines and scalability

The same canonicalizer is applied to edge, adjacency, DFS/BFS and frozen-BPE views. The canonical-control summary is [Table R1b](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R1b_canonical_controls_21.csv); the BPE evaluation is [E1 canonical-BPE summary](https://anonymous.4open.science/r/RPS_GTok/rebuttal/experiments/E1_21_canonical_controls_bpe_full/summary.csv). Canonical edge/adjacency/DFS mean stability is 0.7249, canonical BFS is 0.7778, canonical BPE reproduces the same range, and RPS-GTok identifier stability is 0.9550. This shows that canonicalization improves standard serializations but does not by itself create the full RPS-GTok executable interface.

The full exact-certificate result is [Table E11 v12](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_E11_nauty_all21_full_v12.csv): 974/974 graphs across 21 datasets, including 191 directed inputs, under a 300-second per-graph budget. Exact-audit wall times are 2.374 s for AST-CFG-CPG, 2.235 s for IEEE Power Grid, 2.436 s for Road networks, 13.152 s for Synthetic Stress and 212.558 s for OGBG-PPA. The per-dataset timing rows are in [E2 runtime summary](https://anonymous.4open.science/r/RPS_GTok/rebuttal/experiments/E2_runtime_dataset_fast2/summary.csv).

```bash
cd RPS_GTok_Review
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. \
python scripts/run_nauty_exact_permutation_audit.py \
  --data-dir rebuttal/data_21 \
  --out rebuttal/tables/Table_E11_nauty_all21_full_v12.csv \
  --max-graphs 60 --timeout-sec 300 --workers 8
```

We sincerely thank the reviewer again for the time and effort devoted to evaluating our manuscript, and for the thoughtful and valuable comments that have greatly helped us improve its quality and clarity. We hope that the clarifications, additional analyses, and new experimental results provided in this rebuttal have adequately addressed the reviewer's concerns. For the issues you raised, the anonymous repository provides complete explanations, commands and result data.
