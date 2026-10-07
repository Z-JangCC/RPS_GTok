# Rebuttal response to Reviewer Rkxp

This file is the independent detailed response for Reviewer Rkxp. All paths point to the anonymous repository: [rebuttal/](https://anonymous.4open.science/r/RPS_GTok/rebuttal/).

We sincerely thank the reviewer for recognizing that the proposed method is systematically designed, and for the constructive suggestions. We have carried out the following supplementary experiments and provided further explanations regarding the issues you raised.

## Complete storage

The reported identifier cost is not the complete reconstruction cost. We use

$$B_{\mathrm{total}}=B_{\mathrm{id}}+B_{\mathrm{payload}}+B_{\mathrm{stream}}+B_{\mathrm{artifact}}.$$

The compact 24-graph footprint evaluation gives 27.417 raw/canonical edge topology-context bytes, 16.208 topology-only RPS-GTok identifier-plus-amortized-codebook bytes, and RPS-GTok topology/attribute/full components of 334.417/467.958/802.375 bytes. The codebook lowers model-facing context by representing recurring structures with shared IDs, while the attribute component remains explicit. Complete-stream grammar, payload and raw-token roundtrip are 100% on MUTAG and IMDB. See [Table R14](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R14_footprint_attribute_breakdown.csv) and [E6](https://anonymous.4open.science/r/RPS_GTok/rebuttal/experiments/E6_complete_stream/).

## Realistic graph-learning tasks

The expanded evaluation includes source-aligned PROTEINS and MOLHIV classification and a case-disjoint physical power-flow task. PROTEINS BA is 0.6562/0.6875/0.8125, MOLHIV BA is 0.7500/0.7222/0.7685, and PowerFlow MAE is 0.03942/0.03829/0.03910 for raw/canonical/RPS-GTok. E14/E15 use shared adapters, union vocabulary, three seeds, equal parameters and 0% truncation. The all-21 structural task is explicitly marked a topology proxy.

Evidence: [E14 MOLHIV](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_E14_ogbg_molhiv_balanced_post_schema_aggregate.csv), [E15 PowerFlow](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_E15_powerflow_perturbed_post_schema_aggregate.csv), and [E16 structural control](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_E16_constructed_topology_proxy_notrunc_aggregate.csv).

## Runtime and scalability

The exact backend certifies 974/974 graphs across 21 datasets, including 191 directed inputs, under 300 seconds per graph. Wall times are 2.374 s (AST), 2.235 s (IEEE), 2.436 s (Road), 13.152 s (Synthetic) and 212.558 s (PPA). The complete per-dataset measurements are in [E2 runtime summary](https://anonymous.4open.science/r/RPS_GTok/rebuttal/experiments/E2_runtime_dataset_fast2/summary.csv), and the exact certificate is in [E11 v12](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_E11_nauty_all21_full_v12.csv).

```bash
cd RPS_GTok_Review
PYTHONPATH=. python scripts/run_dataset_runtime_audit.py --help
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. \
python scripts/run_nauty_exact_permutation_audit.py \
  --data-dir rebuttal/data_21 \
  --out rebuttal/tables/Table_E11_nauty_all21_full_v12.csv \
  --max-graphs 60 --timeout-sec 300 --workers 8
```

We sincerely thank the reviewer again for the time and effort devoted to evaluating our manuscript, and for the thoughtful and valuable comments that have greatly helped us improve its quality and clarity. We hope that the clarifications, additional analyses, and new experimental results provided in this rebuttal have adequately addressed the reviewer's concerns. For the issues you raised, the anonymous repository provides complete explanations, commands and result data.
