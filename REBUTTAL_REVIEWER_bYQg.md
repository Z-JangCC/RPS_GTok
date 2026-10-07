# Rebuttal response to Reviewer bYQg

This file is the independent detailed response for Reviewer bYQg. All paths point to the anonymous repository: [rebuttal/](https://anonymous.4open.science/r/RPS_GTok/rebuttal/).

We sincerely thank the reviewer for the careful evaluation and technically precise questions. We have carried out the following supplementary experiments and provided further explanations regarding the issues you raised.

## Canonical form and the contribution boundary

We agree that reconstructability plus permutation stability defines a canonical reference usable for isomorphism decisions. RPS-GTok does not claim a new polynomial-time canonization algorithm. Its contribution is the executable payload-aware contract layered on that reference: primitive graph-edit programs, explicit token-local payloads, local preconditions, vocabulary fragments and primitive fallback.

## Complexity, runtime and environment

For $n$ vertices, $m_e$ edges and $n_a$ retained attributes, the primitive and fragment stages satisfy $O(n+m_e+n_a)$ and $O(c(G)\log c(G))$ bounds. Individualization-refinement has worst-case $O(n!p(n,m,q))$. The exact backend and protocol are [run_nauty_exact_permutation_audit.py](https://anonymous.4open.science/r/RPS_GTok/scripts/run_nauty_exact_permutation_audit.py), while the per-dataset timing measurement is [run_dataset_runtime_audit.py](https://anonymous.4open.science/r/RPS_GTok/scripts/run_dataset_runtime_audit.py).

The environment is Linux x86-64, Python 3.12.2, eight isolated workers and a 300-second per-graph budget. The full exact result is [Table E11 v12](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_E11_nauty_all21_full_v12.csv): 974/974 graphs across 21 datasets, including 191 directed inputs. The complete per-dataset time/fraction rows are in [E2 runtime summary](https://anonymous.4open.science/r/RPS_GTok/rebuttal/experiments/E2_runtime_dataset_fast2/summary.csv).

```bash
cd RPS_GTok_Review
PYTHONPATH=. python scripts/run_dataset_runtime_audit.py --help
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. \
python scripts/run_nauty_exact_permutation_audit.py \
  --data-dir rebuttal/data_21 \
  --out rebuttal/tables/Table_E11_nauty_all21_full_v12.csv \
  --max-graphs 60 --timeout-sec 300 --workers 8
```

## Exact versus pruned ordering

Exact branches exhaustively certify the selected canonical certificate within the declared budget. Bounded or pruned branches are labeled fallback, and Eq. (34) is not asserted for them. No released v12 exact certificate fails Eq. (34): all 974 original/permuted certificate pairs agree.

## Payload generation and complete storage

An identifier-only autoregressive sequence cannot infer payload $\alpha$. Reconstruction is therefore claimed for complete $(I,P)$ instances, $D(I,P;\Sigma)=G_{\Sigma}$. E6 complete-stream grammar, payload and raw-token roundtrip are all 1.0 on MUTAG and IMDB; mean identifier/complete-stream token counts are 8.2/42.6 and 28.5/138.15. The complete-stream evidence is in [E6](https://anonymous.4open.science/r/RPS_GTok/rebuttal/experiments/E6_complete_stream/).

Because argument serialization depends on the chosen grammar, there is no unique Tok./edge or Tok./node value after adding $\alpha$. The compact topology/attribute footprint is therefore reported in [Table R14](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R14_footprint_attribute_breakdown.csv): 27.417 raw/canonical edge topology-context bytes, 16.208 topology-only RPS-GTok identifier-plus-codebook bytes, and 334.417/467.958/802.375 RPS-GTok topology/attribute/full bytes.

## Schema, attributes and truncation

The schema manifests are [R8](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R8_schema_contract.csv) and [R10](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R10_schema_attributes.csv). All 21 manifests specify retained topology, types, attributes, direction, target and raw continuous mode. The 100% reconstruction figure is against this declared retained representation; loops, duplicates, directions and attributes are audited.

At 512 tokens, raw/canonical edge views truncate 10% of IMDB test graphs and RPS-GTok truncates 0%; COLLAB is 50% versus 0%, while MUTAG and PROTEINS are 0% for all corrected views. Locked no-truncation and E14/E15 controls set every view to 0%. The corrected per-view results are in [Table R5](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R5_truncation_semantic.csv). The final downstream claims do not extrapolate from an uncontrolled 512-token attribution to G2PT or Graph-Tokenization-BPE.

We sincerely thank the reviewer again for the time and effort devoted to evaluating our manuscript, and for the thoughtful and valuable comments that have greatly helped us improve its quality and clarity. We hope that the clarifications, additional analyses, and new experimental results provided in this rebuttal have adequately addressed the reviewer's concerns. For the issues you raised, the anonymous repository provides complete explanations, commands and result data.
