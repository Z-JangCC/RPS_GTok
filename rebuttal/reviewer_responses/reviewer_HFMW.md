# Detailed anonymous response to Reviewer HFMW

This standalone file expands the reviewer-specific section of the anonymous
rebuttal package. It is intended for code-level verification and therefore
records the exact protocol, implementation entry points, reported results and
claim boundaries. All paths are relative to this repository; the corresponding
anonymous web root is https://anonymous.4open.science/r/RPS_GTok/.

## Questions addressed

This response addresses:

- canonicalized reversible edge, adjacency, traversal and BPE baselines.
- complete reconstructable footprint including payloads and artifacts.
- semantic/physical downstream tasks under a fair shared Transformer consumer.
- runtime and scalability of canonicalization on symmetric and large inputs.

## Shared notation and evaluation contract

For a serialization view $S$ and a permutation set $\Pi$, stability is
$\mathrm{Stab}(S)=|\Pi|^{-1}\sum_{\pi\in\Pi}\mathbf{1}[S(\pi G)=S(G)]$.
A complete RPS-GTok instance is an identifier stream $I$ together with ordered
payloads $P$ under schema $\Sigma$, and the reconstruction contract is
$D(I,P;\Sigma)=G_{\Sigma}$. Complete storage is reported as
$B_{\mathrm{total}}=B_{\mathrm{id}}+B_{\mathrm{payload}}+B_{\mathrm{stream}}+B_{\mathrm{artifact}}$.
These definitions keep identifier efficiency, executable correctness,
permutation behavior and downstream utility separate.

## Reviewer HFMW — canonicalized baselines, footprint and scalability

We sincerely thank the reviewer for identifying graph representations consumed
by sequence models as a well-motivated problem, and for the careful suggestions.
We have carried out the following supplementary experiments and provided further
explanations regarding the issues you raised.

### Canonicalized reversible baselines

We apply the same schema-aware canonicalizer to reversible edge, adjacency,
DFS/BFS and frozen-BPE views. The corrected 21-dataset control reports mean
stability 0.0238 for raw edge/adjacency, 0.0000 for raw DFS/BFS, 0.7249 for
canonical edge/adjacency/DFS, 0.7778 for canonical BFS and 0.9550 for the
RPS-GTok identifier view. The six-graph-per-dataset frozen-BPE evaluation reproduces
the corresponding canonical range. Thus the reviewer’s direct baseline is
valid: canonical serialization can combine reconstruction with improved
stability, but it does not make the baseline sequence near-perfect on the full
evaluation. RPS-GTok’s additional contribution is the executable primitive-fragment
interface, with payloads, local preconditions and primitive fallback.

### Complete reconstructable footprint

The reported identifier Tok./edge and Tok./node values are model-facing context
metrics, not complete lossless storage. We now report

\[
B_{\mathrm{total}}=B_{\mathrm{id}}+B_{\mathrm{payload}}+
B_{\mathrm{stream}}+B_{\mathrm{artifact}}.
\]

The compact 24-graph measurement gives 27.417 bytes for raw/canonical edge topology
context and 16.208 bytes for topology-only RPS-GTok identifier plus amortized
codebook (40.9% lower). The complete executable topology component is 334.417
bytes; with attributes, 467.958 bytes are attribute payload and 802.375 bytes
are full bytes. The codebook lowers model-facing context by representing
repeated substructures with shared IDs, while the attribute component is
explicit semantic information. E6 independently verifies grammar, payload and
raw-token roundtrip at 100% on MUTAG and IMDB.

### Realistic downstream evaluation

The original topology-statistics tasks are supplemented by source-aligned
PROTEINS and balanced official MOLHIV classification, plus a case-disjoint
pandapower load-perturbation task. Corrected PROTEINS BA is 0.6562 (raw),
0.6875 (canonical) and 0.8125 (RPS-GTok). E14 balanced MOLHIV uses 120 official
graphs per class, a 168/36/36 split, a train-only union vocabulary, a shared
plain adapter and 247,842 equal parameters; BA is 0.7500, 0.7222 and 0.7685,
with Macro-F1 0.7480, 0.7196 and 0.7670. E15 computes the physical target with
`pandapower.runpp`, removes the target feature from inputs, and splits by case
rather than perturbation; MAE is 0.03942, 0.03829 and 0.03910 with 70,433 equal
parameters. Both controls have 0% train/validation/test truncation. These tasks
test utility beyond density, edge count and triangle statistics while keeping
the sequence consumer fixed.

### Canonicalization scalability

The exact backend covers all 974 materialized graphs across all 21 datasets,
including 191 directed inputs, with a 300-second per-graph budget. Exact-audit
wall times are 2.374 s for AST-CFG-CPG, 2.235 s for IEEE Power Grid, 2.436 s
for Road networks, 13.152 s for Synthetic Stress and 212.558 s for OGBG-PPA.
The per-dataset runtime measurements record total tokenization time and
canonicalization fraction for every dataset; its environment is Linux x86-64
with Python 3.12.2, eight isolated workers and the same thread controls. The
reproducible entry points are `scripts/run_dataset_runtime_audit.py` and
`scripts/run_nauty_exact_permutation_audit.py`. These results establish the
empirical finite-budget boundary; the method does not claim a uniform
polynomial worst-case guarantee for highly symmetric graphs.

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
