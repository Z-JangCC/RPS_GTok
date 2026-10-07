# Rebuttal response to Reviewer 8tZd

This file is the independent detailed response for Reviewer 8tZd. All paths point to the anonymous repository: [rebuttal/](https://anonymous.4open.science/r/RPS_GTok/rebuttal/).

We sincerely thank the reviewer for recognizing graph tokenization as an emerging and meaningful problem for graph foundation models. We have carried out the following supplementary experiments and provided further explanations regarding the issues you raised.

## Technical novelty

The central gap is between graph-level invariance and the ordered input consumed by a Transformer. An isomorphic triangle after node-ID permutation can produce different traversal tokens; GNN equivariance does not make those Transformer inputs identical. Serialization/BPE compresses sequences but does not specify executable ports and local preconditions, while latent VQ/pooling representations are not lossless graph programs.

RPS-GTok defines $\tau_t=(i_t,p_t)$ and the complete expansion contract $\mathcal{E}_{\Sigma}(\tau_{1:T})=G_{\Sigma}$. Canonical relabeling, primitive graph-edit compilation, frozen fragment matching, explicit payload construction, precondition checking and primitive fallback are coupled. The implementation is documented in [canonical.py](https://anonymous.4open.science/r/RPS_GTok/gptok2/canonical.py), [program/](https://anonymous.4open.science/r/RPS_GTok/gptok2/program/), [payload.py](https://anonymous.4open.science/r/RPS_GTok/gptok2_tokenizer/payload.py), and [views.py](https://anonymous.4open.science/r/RPS_GTok/rps_gtok_consumption/views.py).

## Reconstruction and downstream utility

The paired audit uses the same materialized sequence and deterministic drop mask for decoding and consumer training. Strict MUTAG reconstruction is 0.973, 0.811, 0.243 and 0.108 at 0%, 5%, 10% and 20% deletion; PROTEINS gives 0.750, 0.250, 0.050 and 0.000. The paired design directly measures the information-coverage path without claiming an unsupported causal theorem.

The source-aligned downstream results are available in [E14 MOLHIV](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_E14_ogbg_molhiv_balanced_post_schema_aggregate.csv) and [E15 PowerFlow](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_E15_powerflow_perturbed_post_schema_aggregate.csv). PROTEINS BA is 0.6562/0.6875/0.8125, MOLHIV BA is 0.7500/0.7222/0.7685, and PowerFlow MAE is 0.03942/0.03829/0.03910 for raw/canonical/RPS-GTok. These results connect executable information coverage to realistic downstream utility under a fixed consumer.

## Parameter and complexity fairness

The matched-consumer gate is [matched_consumer_runs.json](https://anonymous.4open.science/r/RPS_GTok/rebuttal/audits/matched_consumer_runs.json). E14/E15 use one plain adapter, a train-only union vocabulary, three seeds, validation-only selection and 0% truncation. Every view has the same total parameters: 247,842 for MOLHIV and 70,433 for PowerFlow.

The paper's complexity boundary is $\mathrm{cost}_{\mathrm{prim}}(G)=O(n+m_e+n_a)$ and $\mathrm{cost}_{\mathrm{frag}}(G)=O(c(G)\log c(G))$. Canonicalization is an offline preprocessing cost and may branch on highly symmetric graphs. These controls isolate tokenizer representation and executable coverage from additional Transformer capacity or hidden objective changes.

Commands:

```bash
cd RPS_GTok_Review
PYTHONPATH=. python scripts/audit_matched_consumer_runs.py
```

## Model-level foundation reference

The matched GIN reference is [Table E8](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_E8_gin_foundation_baseline.csv). Under matched splits, class weighting, validation selection and three seeds, mean BA is 0.876 on MUTAG, 0.700 on PROTEINS, 0.621 on IMDB-BINARY and 0.648 on COLLAB. RPS-GTok exceeds this reference on PROTEINS and IMDB under the fixed Transformer consumer. A direct comparison with a recent system-level graph foundation model would simultaneously change pretraining, backbone, tokenizer and input budget; the matched GIN and E14/E15 grids are therefore the reproducible tokenizer-level comparison.

We sincerely thank the reviewer again for the time and effort devoted to evaluating our manuscript, and for the thoughtful and valuable comments that have greatly helped us improve its quality and clarity. We hope that the clarifications, additional analyses, and new experimental results provided in this rebuttal have adequately addressed the reviewer's concerns. For the issues you raised, the anonymous repository provides complete explanations, commands and result data.
