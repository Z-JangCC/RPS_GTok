# Detailed anonymous response to Reviewer bYQg

This standalone file expands the reviewer-specific section of the anonymous
rebuttal package. It is intended for code-level verification and therefore
records the exact protocol, implementation entry points, reported results and
claim boundaries. All paths are relative to this repository; the corresponding
anonymous web root is https://anonymous.4open.science/r/RPS_GTok/.

## Questions addressed

This response addresses:

- the canonical-form/graph-isomorphism boundary and the precise contribution beyond GI hardness.
- worst-case complexity, per-dataset runtime, canonicalization fraction and measurement environment.
- exact versus bounded individualization-refinement and when Eq. (34) is asserted.
- autoregressive identifier streams, payload generation and complete storage accounting.
- schema coverage, continuous attributes, reconstruction target and truncation controls.

## Shared notation and evaluation contract

For a serialization view $S$ and a permutation set $\Pi$, stability is
$\mathrm{Stab}(S)=|\Pi|^{-1}\sum_{\pi\in\Pi}\mathbf{1}[S(\pi G)=S(G)]$.
A complete RPS-GTok instance is an identifier stream $I$ together with ordered
payloads $P$ under schema $\Sigma$, and the reconstruction contract is
$D(I,P;\Sigma)=G_{\Sigma}$. Complete storage is reported as
$B_{\mathrm{total}}=B_{\mathrm{id}}+B_{\mathrm{payload}}+B_{\mathrm{stream}}+B_{\mathrm{artifact}}$.
These definitions keep identifier efficiency, executable correctness,
permutation behavior and downstream utility separate.

## Reviewer bYQg — GI hardness, payload generation, schema and formal scope

We sincerely thank the reviewer for the careful evaluation and technically
precise questions. We have carried out the following supplementary experiments
and provided further explanations regarding the issues you raised.

### Canonical form and the paper's gap

We agree that reconstructability plus permutation stability defines a canonical
reference usable for isomorphism decisions. RPS-GTok does not claim a new
polynomial-time canonization algorithm. The gap is the executable contract
layered on that reference: primitive graph-edit programs, explicit token-local
payloads, local preconditions, vocabulary fragments and primitive fallback. A
canonical serialization alone is stable but does not define this payload-aware
execution interface.

### Complexity, runtime and measurement environment

For $n$ vertices, $m_e$ edges and $n_a$ retained attributes, the paper gives
$\mathrm{cost}_{prim}(G)=O(n+m_e+n_a)$ and
$\mathrm{cost}_{frag}(G)=O(c(G)\log c(G))$. The individualization-refinement
canonicalizer has worst-case $O(n!\,p(n,m,q))$, where $p$ is polynomial; this
is the explicit boundary rather than a polynomial guarantee. The exact backend
uses a direct simple-graph path and a WL-refined attributed-incidence path,
with directed source/destination role gadgets. The measurement environment is
Linux x86-64, Python 3.12.2, eight isolated workers and a 300-second per-graph
budget. The reproducible entry points are `scripts/run_dataset_runtime_audit.py`
and `scripts/run_nauty_exact_permutation_audit.py`.

The per-dataset runtime measurements give total tokenization time and the
canonicalization fraction as follows (seconds/percentage): AST-CFG-CPG 4.938/
10.78, CIFAR10-Superpixels 99.021/1.45, CiteSeer 41.168/1.19, COLLAB
129.989/1.12, Cora 24.919/1.52, ENZYMES 2.839/5.27, FB15k-237 67.574/0.69,
IEEE Power Grid 0.385/6.45, IMDB-BINARY 31.443/0.61, MNIST-Superpixels
59.569/1.42, MUTAG 0.396/5.64, OGBG-Code2 3.274/12.07, OGBG-MOLHIV
0.410/10.15, OGBG-MOLPCBA 0.495/6.60, OGBN-Arxiv 194.531/0.43, PROTEINS
4.867/5.62, PubMed 57.995/0.72, Road networks 3.948/5.87, Synthetic Stress
0.217/7.04 and WN18RR 13.700/0.68. The separate full exact-certificate run
verifies 974/974 certificates, including 191 directed inputs.

### Exact versus pruned canonical ordering

The exact branch exhaustively resolves the remaining candidate orders within its
declared budget. A bounded or pruned research branch is marked fallback and is
not used to assert Eq. (34). Thus Eq. (34) is guaranteed only for exact-completed
inputs; no released v12 exact certificate fails it, since all 974 original/
permuted certificate pairs agree. This separates runtime fallback from a false
claim of exact canonical minimality.

### Autoregressive payload boundary and complete footprint

An identifier-only autoregressive sequence cannot infer $\alpha$ or the ports
and attributes required for expansion. Reconstructability is therefore defined
for the complete instance $(I,P)$:

\[
D(I,P;\Sigma)=G_{\Sigma},\qquad
B_{total}=B_{id}+B_{payload}+B_{stream}+B_{artifact}.
\]

E6 serializes complete identifier-plus-payload streams; grammar, payload and
raw-token roundtrip are all 1.0 on MUTAG and IMDB. Mean identifier/complete
stream token counts are 8.2/42.6 for MUTAG and 28.5/138.15 for IMDB. If $\alpha$
is serialized, its cost depends on the argument grammar, so no unique Tok./edge
or Tok./node exists across self-contained encodings. The compact 24-graph audit
reports raw/canonical edge topology context 27.417 bytes per graph and
topology-only RPS-GTok identifier plus amortized codebook 16.208 bytes. Its
complete executable topology component is 334.417 bytes; with attributes,
467.958 bytes are attribute payload and 802.375 bytes are full bytes. This
separates codebook-compressed model context from explicit semantic payload.

### Schema and continuous attributes

R8/R10 provide one manifest per all 21 datasets. Topology-only schemas are used
for COLLAB and IMDB-BINARY; topology plus node-type schemas for Cora, ENZYMES,
MUTAG and PROTEINS; directed typed schemas for FB15k-237, OGBG-Code2,
OGBN-Arxiv and WN18RR; the remaining manifests retain the stated node/edge
types and attributes, including IEEE, superpixel, molecular, road and synthetic
fields. Continuous mode is `raw` in all released manifests. Consequently the
100% reconstruction result is evaluated against the declared retained raw
representation, not an unstated original-feature target or hidden quantization.
Roundtrips cover loops, duplicate edges, directions, node/edge types and
attributes, and expansion is graph-equivalent under the canonical emission order.

### Truncation and the G2PT/BPE attribution

The corrected per-view measurements show 10% test truncation for raw and canonical
edge views and 0% for RPS-GTok on IMDB; COLLAB is 50% versus 0%, while MUTAG and
PROTEINS are 0% for all corrected views. The locked no-truncation control sets
all views to 0%, and the E14/E15 downstream reruns also have 0% train,
validation and test truncation. Thus the final downstream claims do not rely on
the original 512-token attribution. The original G2PT/Graph-Tokenization-BPE
rows are not used as isolated causal evidence without the same locked rerun;
the anonymous repository contains the corrected per-view audit and the
no-truncation comparison rather than extrapolating from mean length.

The permutation audit also led to a content-fingerprinted cache and invariant
cycle anchors; all pre-fix permutation numbers are excluded from the authoritative
index.

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
