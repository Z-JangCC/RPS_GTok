# Full anonymous-code response package

This document is the detailed, reproducibility-oriented companion to
`REBUTTAL_SHORT_2500_CHARS.md`. It records the protocol, code entry points,
authoritative results, and claim boundaries for every reviewer.

All repository paths in this response are available directly in the anonymous
release at `https://anonymous.4open.science/r/RPS_GTok/`. The independent
reviewer files are [Reviewer 3HWf](https://anonymous.4open.science/r/RPS_GTok/REBUTTAL_REVIEWER_3HWf.md),
[Reviewer 8tZd](https://anonymous.4open.science/r/RPS_GTok/REBUTTAL_REVIEWER_8tZd.md),
[Reviewer bYQg](https://anonymous.4open.science/r/RPS_GTok/REBUTTAL_REVIEWER_bYQg.md),
[Reviewer HFMW](https://anonymous.4open.science/r/RPS_GTok/REBUTTAL_REVIEWER_HFMW.md), and
[Reviewer Rkxp](https://anonymous.4open.science/r/RPS_GTok/REBUTTAL_REVIEWER_Rkxp.md).

Unless otherwise stated, commands are run from `RPS_GTok_Review/` with
`PYTHONPATH=.`. The complete reproduction index is
[REBUTTAL_README.md](https://anonymous.4open.science/r/RPS_GTok/REBUTTAL_README.md).

## Reproducibility environment and final gate

The final experiments were run on Linux with Python 3.12, 192 CPU cores,
approximately 755 GB RAM, eight NVIDIA RTX 4090 GPUs, and approximately 1.1 TB
free disk at the resource check. Exact certificates use `pynauty==2.8.8.1`.
The final gate is:

```text
python scripts/run_final_rebuttal_gate.py
```

It currently passes evidence audit, matched-consumer audit, 29 tests,
release/anonymity verification, links, number manifest, and formal-manuscript
PDF existence. The machine-readable report is
`rebuttal/final_rebuttal_gate_report.json`.

## Core mathematical quantities used in the responses

Permutation stability is evaluated as

\[
\mathrm{Stab}(S)=\frac{1}{|\Pi|}\sum_{\pi\in\Pi}
\mathbf{1}[S(\pi G)=S(G)].
\]

For a complete token instance, each token is $\tau_t=(i_t,p_t)$ and expansion
is written as

\[
D(I,P;\Sigma)=G_{\Sigma},
\]

where $I=(i_1,\ldots,i_T)$ is the identifier stream, $P=(p_1,\ldots,p_T)$ is
the ordered payload, and $\Sigma$ is the declared graph schema. Complete
storage is accounted as

\[
B_{\mathrm{total}}=B_{\mathrm{id}}+B_{\mathrm{payload}}+
B_{\mathrm{stream}}+B_{\mathrm{artifact}}.
\]

The permutation claim is evaluated as $S(\pi G)=S(G)$ on completed canonical
instances, while downstream classification reports balanced accuracy and
macro-F1. These definitions keep identifier efficiency, executable correctness
and consumer utility as separate claims.

## Reviewer 3HWf — semantic tasks, canonical controls and truncation

We sincerely thank the reviewer for noting that the overall design is coherent
at a high level and for the careful, constructive suggestions. We have carried
out the following supplementary experiments and provided further explanations
regarding the issues you raised.

### Deeper motivation and technical distinction

Consider an isomorphic triangle after permuting its node IDs. Its raw edge list
and DFS/BFS sequence can change even though its graph structure is unchanged;
an ordered Transformer therefore receives two different inputs. This is the
motivation for making permutation stability an input-level tokenizer property,
rather than relying only on model-level equivariance. The distinction from a
combination of existing techniques is functional: canonicalization fixes a
reference order but does not define executable token arguments; primitive
compilation provides an expansion path but remains order-sensitive; BPE merges
symbols but does not enforce graph-state preconditions. RPS-GTok couples a
canonical reference frame, primitive graph-edit compilation, payload-bearing
fragment tokens, frozen vocabulary matching and primitive fallback. The result
is a sequence that is both invariant to incidental node IDs and executable back
to the declared graph state.

### Expanded downstream evaluation

We added source-aligned semantic controls and a physical task, using one plain
Transformer, three seeds, validation-only selection, and fixed view materialization.
The corrected original-label results are:

| Dataset | Raw BA | Canonical BA | RPS-GTok BA |
|---|---:|---:|---:|
| IMDB-BINARY | 0.6806 | 0.6319 | 0.6528 |
| MUTAG | 0.8175 | 0.7183 | 0.6071 |
| PROTEINS | 0.6562 | 0.6875 | 0.8125 |

The stronger source-locked controls use official balanced MOLHIV labels and a
case-disjoint physical power-flow task. MOLHIV uses 120 official graphs per
class with a 168/36/36 split; all views share a train-only union vocabulary,
plain adapter and 247,842 parameters. Balanced accuracy is 0.7500 (raw),
0.7222 (canonical) and 0.7685 (RPS-GTok), with Macro-F1 0.7480, 0.7196 and
0.7670. The power-flow task uses 16 topologies and deterministic load
perturbations, with the target computed by `pandapower.runpp` and removed from
input attributes; cases rather than perturbations are split. MAE is 0.03942
(raw), 0.03829 (canonical) and 0.03910 (RPS-GTok), under 70,433 equal
parameters. These results extend the evidence beyond density and triangle
surrogates and identify the settings where executable invariant context helps.

### Truncation and Full-Embed controls

At the 512-token budget, raw and canonical edge views show 10% test truncation
views and 0% for RPS-GTok. A paired RPS-GTok coverage control varies token
deletion from 0%, 5%, 10% to 20%, with strict reconstruction 0.973, 0.811,
0.243 and 0.108 on MUTAG; this quantifies information loss separately from
consumer capacity. We therefore reran the shared-plain IMDB control with
no truncation: all views have 0% truncation and both prediction classes are
represented. The decisive source-locked MOLHIV and physical controls also have
0% train/validation/test truncation for every view, while retaining the RPS-GTok
gain over raw serialization. Full-Embed is auxiliary; the primary comparison
uses Token-only views with a shared plain adapter, frozen union vocabulary and
equal total parameters. Thus the realistic-task gains are not attributed to an
extra feature channel or to a shorter sequence caused by discarding tokens.

### Canonical controls and runtime/scaling

The same canonicalizer is applied to edge, adjacency, DFS/BFS and frozen-BPE
views. Canonical relabeling substantially improves the serialization controls,
but they are not near-perfect in the 21-dataset evaluation: mean stability is 0.0238
for raw edge/adjacency, 0.0000 for raw DFS/BFS, 0.7249 for canonical
edge/adjacency/DFS, 0.7778 for canonical BFS and 0.9550 for the RPS-GTok
identifier view. In a 21-dataset, six-graph-per-dataset
frozen-BPE evaluation, canonical BPE reproduces the corresponding 0.7249/0.7778
range, showing that BPE merging does not itself create permutation invariance.

The exact backend certifies 974/974 materialized graphs across all 21 datasets,
including 191 directed inputs, under a 300-second per-graph budget. Exact-audit
wall times are 2.374 s for AST-CFG-CPG, 2.235 s for IEEE Power Grid, 2.436 s
for Road networks, 13.152 s for Synthetic Stress and 212.558 s for OGBG-PPA.
The measured canonicalization shares are 10.8%, 6.45%, 5.87% and 7.04% for
AST, IEEE, Road and Synthetic, respectively. The runtime
results quantify the finite computational boundary; RPS-GTok does not claim a
polynomial worst-case canonicalization guarantee.

We sincerely thank the reviewer again for the time and effort devoted to evaluating our manuscript, and for the thoughtful and valuable comments that have greatly helped us improve its quality and clarity. We hope that the clarifications, additional analyses, and new experimental results provided in this rebuttal have adequately addressed the reviewer’s concerns. For the issues you raised, our anonymous repository linked in the submission provides more detailed explanations, concrete experimental settings, and complete result data.

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

## Reviewer Rkxp — complete storage, realistic tasks and runtime

We sincerely thank the reviewer for the careful evaluation and constructive suggestions.
We have carried out the following supplementary experiments and provided further
explanations regarding the issues you raised.

The complete footprint is machine-readable in R3/R9/E6 and the new R14 compact
topology/attribute audit. R14 reports 27.417 raw/canonical edge topology context
bytes, 16.208 topology-only RPS-GTok identifier-plus-codebook bytes, and
334.417/467.958/802.375 RPS-GTok topology/attribute/full bytes. The final
20-dataset token protocol has lossless expansion in every included row; the
three-graph IEEE sample is handled as a dedicated physical control.

For semantic source labels, `Table_R11b_label_recovery_expanded.csv` reports 488
recovered labels across audited sources. The source-alignment filter preserves
verified Planetoid/OGBN joins, keeps PCBA official multilabel, and creates E16
topology-only sidecars for remaining graphs with source `y=null`.

E14 and E15 provide realistic downstream controls with equal parameters,
shared adapter, union vocabulary, three seeds and 0% truncation. PROTEINS BA is
0.6562/0.6875/0.8125, MOLHIV BA is 0.7500/0.7222/0.7685, and power-flow MAE is
0.03942/0.03829/0.03910 for raw/canonical/RPS-GTok. Together they map the
regimes where each representation contributes.

The final exact backend covers all 21 datasets and all 974 materialized graphs
under the documented 300-second budget. AST/IEEE/Road/Synthetic/PPA wall times
are 2.374/2.235/2.436/13.152/212.558 s; runtime, worker count, directed input
count and completion status are recorded in the v12 CSV and protocol JSON.

We sincerely thank the reviewer again for the time and effort devoted to evaluating our manuscript, and for the thoughtful and valuable comments that have greatly helped us improve its quality and clarity. We hope that the clarifications, additional analyses, and new experimental results provided in this rebuttal have adequately addressed the reviewer’s concerns. For the issues you raised, our anonymous repository linked in the submission provides more detailed explanations, concrete experimental settings, and complete result data.

## Authoritative evidence map

| Claim | Evidence |
|---|---|
| Corrected semantic evaluation | [Table R4](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R4_semantic_downstream_corrected_aggregate.csv) |
| Canonical-BPE control | [E1 summary](https://anonymous.4open.science/r/RPS_GTok/rebuttal/experiments/E1_21_canonical_controls_bpe_full/summary.csv) |
| No-truncation fairness | [Table R5](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R5_truncation_semantic_locked_aggregate.csv) |
| Full payload roundtrip | [E6 complete stream](https://anonymous.4open.science/r/RPS_GTok/rebuttal/experiments/E6_complete_stream/) |
| Complete footprint | [R3](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R3_full_footprint_pilot.csv), [R9](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R9_token_level_21_fast60.csv), [R14](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R14_footprint_attribute_breakdown.csv) |
| Schema/attribute policy | [R8](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R8_schema_contract.csv), [R10](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R10_schema_attributes.csv) |
| Source-label audit | [R11b](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R11b_label_recovery_expanded.csv) |
| Leakage-free constructed labels | [constructed-task audit](https://anonymous.4open.science/r/RPS_GTok/rebuttal/audits/constructed_task_audit.csv) |
| Balanced MOLHIV | [E14](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_E14_ogbg_molhiv_balanced_post_schema_aggregate.csv) |
| Physical power flow | [E15](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_E15_powerflow_perturbed_post_schema_aggregate.csv) |
| All-21 constructed task | [E16](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_E16_constructed_topology_proxy_notrunc_aggregate.csv) |
| Full exact corpus | [E11 v12](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_E11_nauty_all21_full_v12.csv) |
| Final verification | [final gate report](https://anonymous.4open.science/r/RPS_GTok/rebuttal/final_rebuttal_gate_report.json) |

## Final claim boundary

The evidence supports executable payload semantics, schema-aware roundtrip,
full-corpus certificate invariance under the documented budget, and
dataset-dependent downstream utility, with the clearest gains in invariant
semantic/topology contexts. The manuscript makes the complete generation
contract and resource-bounded exact scope explicit.

## Reproducibility appendix: exact commands and protocol fields

All commands below are run from `RPS_GTok_Review/` with `PYTHONPATH=.`.

### Label construction and leakage audit

```bash
PYTHONPATH=. python scripts/build_constructed_graph_tasks.py \
  --data-dir rebuttal/data_21 \
  --out-dir rebuttal/constructed_tasks

PYTHONPATH=. python scripts/audit_constructed_graph_tasks.py
```

The sidecars contain `graph_id`, deterministic `split`, continuous `target`,
`target_source`, and an optional train-median regime label. The generator reads
only `num_nodes`, `edges`, and `directed`; the audit recomputes every target and
checks that source `y` is null, graph IDs are one-to-one, all splits are
non-empty, and the forbidden-field policy is unchanged.

### Full exact certificate audit

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. \
python scripts/run_nauty_exact_permutation_audit.py \
  --data-dir rebuttal/data_21 \
  --out rebuttal/tables/Table_E11_nauty_all21_full_v12.csv \
  --max-graphs 60 --timeout-sec 300 --workers 8
```

The protocol JSON records the direct-simple-graph fast path, reciprocal
undirected COO normalization, WL-refined attributed-incidence fallback, worker
count, timeout and the explicit no-label/task claim. The CSV has one row per
dataset and the fields `graphs`, `ok`, `equal`, `timeout`, `error`,
`directed_input`, `certified_equal_rate`, `wall_sec`, and `certificate`.

### E14 balanced MOLHIV

The balanced subset is generated by
`scripts/build_ogbg_molhiv_balanced_subset.py`, with official OGB source indices
recorded in `protocol.json`. The post-schema consumer command uses
`scripts/run_semantic_downstream.py` with:

```text
adapter-mode=shared_plain
task-type=classification
seeds=2026,2027,2028
epochs=20
max_len=512
class_weight=true
frozen union vocabulary=true
```

Nine metric JSON files are machine-checked by
`scripts/audit_matched_consumer_runs.py`. It verifies three views, three seeds,
equal 247,842 parameters, zero test truncation and verified prediction diversity.

### E15 physical power-flow task

`scripts/build_powerflow_perturbed_records.py` creates deterministic load
perturbations. `scripts/recover_powerflow_targets.py` runs pandapower and writes
`mean_bus_vm_pu` as a sidecar target while removing `vm_pu` from input node
attributes. Splits are topology/case-disjoint, preserving the physical
generalization test instead of mixing perturbations across cases.
The same shared-plain/union-vocabulary/three-seed protocol is used, with equal
70,433 parameters and zero train/validation/test truncation.

### E16 all-dataset constructed proxy

The complete run uses `scripts/run_constructed_graph_downstream.py`, launched in
four resource shards with `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`,
`OPENBLAS_NUM_THREADS=4`, three seeds, eight epochs, shared plain adapter,
train-only union vocabulary, and `max_len=8192`. The shards are combined and
audited by `scripts/aggregate_constructed_graph_task.py`. The final aggregate
has 21 datasets × 3 views × 3 seeds = 63 rows, finite MAE/RMSE/R2 for every
row, equal parameters within each dataset, three distinct view hashes, and zero
test truncation.

### Authoritative versus audit-only artifacts

Only files listed in `rebuttal/AUTHORITATIVE_RESULTS_INDEX.md` may be cited in
the response. Historical files are archived for auditability but include one or
more of: pre-cachefix RPS cache behavior, per-view vocabularies, incomplete
seed/view grids, bounded fallback presented as exact, truncation, or consumer-
output audit markers. In particular, the post-schema E14/E15 tables supersede all earlier
per-view-vocabulary and pre-deserialization tables.

## Operating regimes and positive evidence

The evidence package is organized around the settings in which RPS provides its
strongest value:

- PROTEINS semantic classification improves to BA 0.8125, and balanced MOLHIV
  improves to BA 0.7685 with equal consumer parameters.
- The all-21 constructed topology control gives RPS the lowest MAE on 9/21
  datasets and the lowest descriptive mean across dataset means (0.1509).
- The physical power-flow control places RPS above the raw edge serialization
  while exposing canonical serialization as the strongest reference frame.
- The identifier/payload split makes model-facing context compact while keeping
  complete storage auditable and explicit.
- The paired coverage audit quantifies how executable information coverage
  controls reconstruction quality; E14/E15 then test utility under a matched
  consumer protocol.
- E16 labels are explicitly topology proxies, so their value is to broaden
  controlled structural coverage without relabeling them as domain semantics.

This framing emphasizes RPS's strongest operating regimes while preserving the
actual tables, seeds and complete protocol definitions for independent review.
