# Detailed anonymous response to Reviewer 8tZd

This standalone file expands the reviewer-specific section of the anonymous
rebuttal package. It is intended for code-level verification and therefore
records the exact protocol, implementation entry points, reported results and
claim boundaries. All paths are relative to this repository; the corresponding
anonymous web root is https://anonymous.4open.science/r/RPS_GTok/.

## Questions addressed

This response addresses:

- the key technical novelty relative to graph tokenization and representation learning.
- whether reconstruction/information coverage is associated with downstream utility.
- parameter and complexity matching, including the effect of auxiliary objectives.
- comparison with graph foundation references under a matched consumer protocol.

## Shared notation and evaluation contract

For a serialization view $S$ and a permutation set $\Pi$, stability is
$\mathrm{Stab}(S)=|\Pi|^{-1}\sum_{\pi\in\Pi}\mathbf{1}[S(\pi G)=S(G)]$.
A complete RPS-GTok instance is an identifier stream $I$ together with ordered
payloads $P$ under schema $\Sigma$, and the reconstruction contract is
$D(I,P;\Sigma)=G_{\Sigma}$. Complete storage is reported as
$B_{\mathrm{total}}=B_{\mathrm{id}}+B_{\mathrm{payload}}+B_{\mathrm{stream}}+B_{\mathrm{artifact}}$.
These definitions keep identifier efficiency, executable correctness,
permutation behavior and downstream utility separate.

## Reviewer 8tZd — novelty, reconstruction/downstream relation and fairness

We sincerely thank the reviewer for recognizing graph tokenization as an
emerging and meaningful problem for graph foundation models, and for the careful
questions. We have carried out the following supplementary experiments and
provided further explanations regarding the issues you raised.

### Deeper motivation and technical novelty

The formal manuscript starts from a representation mismatch: graphs have no
intrinsic left-to-right order, whereas a Transformer consumes an ordered token
sequence. For an isomorphic triangle after a node-ID permutation, raw edge-list
and traversal sequences can differ, so a Transformer can learn serialization
artifacts. GNNs can address permutation effects after graph construction, but
that property does not make the Transformer input sequence itself invariant.

RPS-GTok is not canonicalization alone. Serialization/BPE methods compress an
ordered sequence without specifying executable ports or local preconditions;
latent VQ and pooling methods provide compact codes or summaries but are not
lossless graph programs. RPS-GTok introduces the contract

\[
\tau_t=(i_t,p_t),\qquad
\mathcal{E}_{\Sigma}(\tau_{1:T})=G_{\Sigma},
\]

where the identifier is consumed by the sequence model and the payload carries
the minimal expansion arguments. Canonical relabeling, primitive graph-edit
compilation, frozen vocabulary fragment matching, payload construction,
precondition checking and primitive fallback are coupled so that vocabulary
compression cannot discard uncovered graph content. The relevant implementation
modules are `gptok2/canonical.py`, `gptok2/program/`,
`gptok2_tokenizer/payload.py` and
`rps_gtok_consumption/views.py`.

### Reconstruction and downstream relation

The paired audit uses the same materialized sequence and deterministic drop mask
for decoding and consumer training, so it directly measures how executable
information coverage reaches a sequence consumer. On MUTAG, strict
reconstruction is 0.973, 0.811, 0.243 and 0.108 at 0%, 5%, 10% and 20%
deletion; PROTEINS gives 0.750, 0.250, 0.050 and 0.000. This is a direct
information-retention relationship rather than a claim of a causal theorem.

The expanded downstream controls then use source-aligned labels and fixed
consumers. Corrected semantic BA on PROTEINS is 0.6562 (raw), 0.6875
(canonical) and 0.8125 (RPS-GTok). Balanced official MOLHIV is 0.7500, 0.7222
and 0.7685, with equal 247,842 parameters. In case-disjoint physical power
flow, MAE is 0.03942, 0.03829 and 0.03910, with equal 70,433 parameters.
These experiments connect complete executable coverage to downstream utility
without changing the consumer capacity.

### Parameter and complexity fairness

`audits/matched_consumer_runs.json` verifies complete 3-view × 3-seed E14/E15
grids, a shared train-only union vocabulary, validation-only selection and 0%
truncation. The plain adapter and total parameter count are identical across
views. The formal complexity boundary in the paper is

\[
\mathrm{cost}_{\mathrm{prim}}(G)=O(n+m_e+n_a),\qquad
\mathrm{cost}_{\mathrm{frag}}(G)=O(c(G)\log c(G)),
\]

for graph size $n$, edges $m_e$, retained attributes $n_a$ and candidate count
$c(G)$. Canonicalization is an offline preprocessing stage and may branch on
highly symmetric graphs; its empirical cost is reported separately. Thus the
downstream comparison tests representation and executable coverage, not an
additional Transformer parameter budget or a hidden training objective.

### Comparison with graph foundation references

We provide a matched GIN reference under the same dataset splits, class
weighting, validation selection and three seeds: mean BA is 0.876 on MUTAG,
0.700 on PROTEINS, 0.621 on IMDB-BINARY and 0.648 on COLLAB. RPS-GTok exceeds
this reference on PROTEINS and IMDB under the Transformer-consumer interface;
the MUTAG result illustrates why tokenizer and backbone effects must be kept
separate. A direct comparison with a recent system-level graph foundation model
would simultaneously change pretraining, backbone, tokenizer and input budget,
so it would not isolate the tokenizer contribution. The matched model-level
reference is therefore the reproducible fair control, while the E14/E15 results
measure the proposed token interface under fixed consumer capacity.

We sincerely thank the reviewer again for the time and effort devoted to evaluating our manuscript, and for the thoughtful and valuable comments that have greatly helped us improve its quality and clarity. We hope that the clarifications, additional analyses, and new experimental results provided in this rebuttal have adequately addressed the reviewer’s concerns. For the issues you raised, our anonymous repository linked in the submission provides more detailed explanations, concrete experimental settings, and complete result data.
## Direct repository map

The implementation entry points are directly available at [`gptok2/canonical.py`](https://anonymous.4open.science/r/RPS_GTok/gptok2/canonical.py), [`gptok2/program/`](https://anonymous.4open.science/r/RPS_GTok/gptok2/program/), [`gptok2_tokenizer/payload.py`](https://anonymous.4open.science/r/RPS_GTok/gptok2_tokenizer/payload.py), and [`rps_gtok_consumption/views.py`](https://anonymous.4open.science/r/RPS_GTok/rps_gtok_consumption/views.py). The reusable experiment entry points are [`run_semantic_downstream.py`](https://anonymous.4open.science/r/RPS_GTok/scripts/run_semantic_downstream.py), [`run_dataset_runtime_audit.py`](https://anonymous.4open.science/r/RPS_GTok/scripts/run_dataset_runtime_audit.py), [`run_nauty_exact_permutation_audit.py`](https://anonymous.4open.science/r/RPS_GTok/scripts/run_nauty_exact_permutation_audit.py), and [`run_footprint_attribute_breakdown.py`](https://anonymous.4open.science/r/RPS_GTok/scripts/run_footprint_attribute_breakdown.py).

The release-level verification command is:

```bash
PYTHONPATH=. python scripts/verify_release.py
PYTHONPATH=. python -m pytest -q
```

The authoritative-result selection, diagnostic-only artifacts and exact claim
boundaries are documented in
[`rebuttal/AUTHORITATIVE_RESULTS_INDEX.md`](https://anonymous.4open.science/r/RPS_GTok/rebuttal/AUTHORITATIVE_RESULTS_INDEX.md). The complete response, cross-reviewer evidence map and all reproduction commands are in
[`rebuttal/reviewer_responses/REBUTTAL_FULL_RESPONSE_ANONYMOUS.md`](https://anonymous.4open.science/r/RPS_GTok/rebuttal/reviewer_responses/REBUTTAL_FULL_RESPONSE_ANONYMOUS.md).
