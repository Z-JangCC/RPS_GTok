# Rebuttal response to Reviewer HFMW

This file is the independent detailed response for Reviewer HFMW. All paths point to the anonymous repository: [rebuttal/](https://anonymous.4open.science/r/RPS_GTok/rebuttal/).

We sincerely thank the reviewer for identifying graph representations consumed by sequence models as a well-motivated problem. We have carried out the following supplementary experiments and provided further explanations regarding the issues you raised.

## Canonicalized reversible baselines

The same schema-aware canonicalizer is applied to reversible edge, adjacency, DFS/BFS and frozen-BPE views. The direct control reports mean stability 0.0238 for raw edge/adjacency, 0.0000 for raw DFS/BFS, 0.7249 for canonical edge/adjacency/DFS, 0.7778 for canonical BFS and 0.9550 for RPS-GTok identifier. The canonical-BPE evaluation reproduces the canonical range. These results show that canonical serialization can provide reconstruction plus improved stability, while RPS-GTok adds executable fragments, payloads and fallback.

Evidence: [Table R1b](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R1b_canonical_controls_21.csv) and [canonical-BPE evaluation](https://anonymous.4open.science/r/RPS_GTok/rebuttal/experiments/E1_21_canonical_controls_bpe_full/summary.csv).

## Complete reconstructable footprint

Identifier Tok./edge and Tok./node values measure model-facing context, not complete lossless storage. The complete footprint is

$$B_{\mathrm{total}}=B_{\mathrm{id}}+B_{\mathrm{payload}}+B_{\mathrm{stream}}+B_{\mathrm{artifact}}.$$

The compact topology/attribute evaluation reports 27.417 raw/canonical edge topology-context bytes, 16.208 topology-only RPS-GTok identifier-plus-amortized-codebook bytes, and RPS-GTok topology/attribute/full components of 334.417/467.958/802.375 bytes. The codebook compresses repeated topology structures in the model-facing stream, while attribute bytes remain explicit semantic information. See [Table R14](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R14_footprint_attribute_breakdown.csv) and [R3/R9 footprint evidence](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/).

## Realistic downstream evaluation

The original topology-statistics evaluation is supplemented by source-aligned PROTEINS and MOLHIV classification and a case-disjoint pandapower task. PROTEINS BA is 0.6562/0.6875/0.8125, MOLHIV BA is 0.7500/0.7222/0.7685, and PowerFlow MAE is 0.03942/0.03829/0.03910 for raw/canonical/RPS-GTok. E14/E15 use shared adapters, union vocabulary, three seeds, equal parameters (247,842/70,433) and 0% truncation. See [E14](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_E14_ogbg_molhiv_balanced_post_schema_aggregate.csv) and [E15](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_E15_powerflow_perturbed_post_schema_aggregate.csv).

## Canonicalization scalability

The exact backend certifies 974/974 materialized graphs across 21 datasets, including 191 directed inputs, under a 300-second per-graph budget. Wall times are 2.374 s for AST-CFG-CPG, 2.235 s for IEEE, 2.436 s for Road, 13.152 s for Synthetic Stress and 212.558 s for OGBG-PPA. The per-dataset runtime rows are in [E2 runtime summary](https://anonymous.4open.science/r/RPS_GTok/rebuttal/experiments/E2_runtime_dataset_fast2/summary.csv); the environment is Linux x86-64/Python 3.12.2 with eight isolated workers.

```bash
cd RPS_GTok_Review
PYTHONPATH=. python scripts/run_dataset_runtime_audit.py --help
```

We sincerely thank the reviewer again for the time and effort devoted to evaluating our manuscript, and for the thoughtful and valuable comments that have greatly helped us improve its quality and clarity. We hope that the clarifications, additional analyses, and new experimental results provided in this rebuttal have adequately addressed the reviewer's concerns. For the issues you raised, the anonymous repository provides complete explanations, commands and result data.
