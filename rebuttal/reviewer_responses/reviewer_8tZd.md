# Response to Reviewer 8tZd

We sincerely thank the reviewer for recognizing graph tokenization as an
emerging and meaningful problem for graph foundation models, and for the careful
questions. We have carried out the following supplementary experiments and
provided further explanations regarding the issues you raised.

## Technical novelty and deeper motivation

Graphs have no intrinsic left-to-right order, whereas a Transformer consumes an
ordered sequence. An isomorphic triangle after node-ID permutation can therefore
produce different edge-list or traversal tokens. GNN equivariance does not make
those Transformer inputs identical. Serialization/BPE compresses sequences but
does not specify executable ports and preconditions; latent VQ/pooling codes are
not lossless graph programs. RPS-GTok couples canonical relabeling, primitive
graph-edit compilation, frozen fragment vocabulary, payload construction,
precondition checking and primitive fallback through
$\tau_t=(i_t,p_t)$ and
$\mathcal{E}_{\Sigma}(\tau_{1:T})=G_{\Sigma}$. This is the mechanism-level
novelty rather than canonicalization alone.

## Reconstruction and downstream utility

The paired audit uses the same materialized sequence and deterministic drop mask
for decoding and consumer training. Strict MUTAG reconstruction is 0.973, 0.811,
0.243 and 0.108 at 0%, 5%, 10% and 20% deletion; PROTEINS gives 0.750, 0.250,
0.050 and 0.000. Source-aligned downstream controls then give PROTEINS BA
0.6562 (raw), 0.6875 (canonical), 0.8125 (RPS-GTok); balanced MOLHIV gives
0.7500, 0.7222 and 0.7685; physical power-flow MAE is 0.03942, 0.03829 and
0.03910. The paired design directly measures the information-coverage path,
establishing the evidence link without claiming a causal theorem.

## Parameters and complexity

E14/E15 use a shared plain adapter, train-only union vocabulary, three seeds,
validation-only selection and 0% truncation. All views have equal total
parameters: 247,842 for MOLHIV and 70,433 for power flow. These checks isolate
representation effects from consumer capacity. The paper gives
$\mathrm{cost}_{\mathrm{prim}}(G)=O(n+m_e+n_a)$ and
$\mathrm{cost}_{\mathrm{frag}}(G)=O(c(G)\log c(G))$; canonicalization is an
offline preprocessing stage whose empirical runtime is reported separately.

## Foundation-model reference

The matched GIN reference uses the same splits, class weighting, validation
selection and three seeds, with mean BA 0.876 (MUTAG), 0.700 (PROTEINS), 0.621
(IMDB-BINARY) and 0.648 (COLLAB). RPS-GTok exceeds this reference on PROTEINS
and IMDB under the fixed Transformer consumer. A direct system-level comparison
with recent graph foundation models would change pretraining, backbone,
tokenizer and input budget simultaneously; the matched GIN and E14/E15 grids
therefore provide the reproducible tokenizer-level comparison.

We sincerely thank the reviewer again for the time and effort devoted to evaluating our manuscript, and for the thoughtful and valuable comments that have greatly helped us improve its quality and clarity. We hope that the clarifications, additional analyses, and new experimental results provided in this rebuttal have adequately addressed the reviewer’s concerns. For the issues you raised, our anonymous repository linked in the submission provides more detailed explanations, concrete experimental settings, and complete result data.
