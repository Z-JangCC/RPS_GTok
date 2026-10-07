# Detailed anonymous response to Reviewer 3HWf

This standalone file expands the reviewer-specific section of the anonymous
rebuttal package. It is intended for code-level verification and therefore
records the exact protocol, implementation entry points, reported results and
claim boundaries. All paths are relative to this repository; the corresponding
anonymous web root is https://anonymous.4open.science/r/RPS_GTok/.

## Questions addressed

This response addresses:

- deeper motivation and novelty beyond combining canonicalization, primitives and BPE.
- semantic and physical downstream evaluation beyond topology-surrogate targets.
- truncation controls, Full-Embed fairness, and canonicalized baseline comparisons.
- runtime, memory/scaling behavior and the cost of canonical relabeling.

## Shared notation and evaluation contract

For a serialization view $S$ and a permutation set $\Pi$, stability is
$\mathrm{Stab}(S)=|\Pi|^{-1}\sum_{\pi\in\Pi}\mathbf{1}[S(\pi G)=S(G)]$.
A complete RPS-GTok instance is an identifier stream $I$ together with ordered
payloads $P$ under schema $\Sigma$, and the reconstruction contract is
$D(I,P;\Sigma)=G_{\Sigma}$. Complete storage is reported as
$B_{\mathrm{total}}=B_{\mathrm{id}}+B_{\mathrm{payload}}+B_{\mathrm{stream}}+B_{\mathrm{artifact}}$.
These definitions keep identifier efficiency, executable correctness,
permutation behavior and downstream utility separate.

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

### Reproducibility details and interpretation

The semantic controls are materialized before training, so no view is generated
from the target label. The source-aligned label recovery and eligibility checks
are recorded in [`rebuttal/audits/label_recovery_audit.csv`](https://anonymous.4open.science/r/RPS_GTok/rebuttal/audits/label_recovery_audit.csv)
and [`rebuttal/tables/Table_R11b_label_recovery_expanded.csv`](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R11b_label_recovery_expanded.csv).
For MOLHIV, the balanced subset, source indices, split, and schema are recorded
in the experiment protocol; for PowerFlow, `pandapower.runpp` generates the
target after input attributes containing the target quantity are removed. The
case-disjoint split prevents perturbations from the same physical topology from
appearing in both training and test sets.

The fixed-consumer audit verifies one train-only union vocabulary, one shared
plain adapter, three seeds, validation-only model selection, finite and
non-collapsed test predictions, and zero truncation. The machine-readable gate
is [`rebuttal/audits/matched_consumer_runs.json`](https://anonymous.4open.science/r/RPS_GTok/rebuttal/audits/matched_consumer_runs.json).
The complete-stream audit separately verifies identifier grammar, payload
alignment and raw-token round-trip; it is not conflated with downstream
classification accuracy.

The canonical-baseline experiment is a matched isolation: the same canonicalizer,
permutation set, graph schema and frozen-BPE procedure are used for edge,
adjacency, DFS/BFS and BPE. Thus the stability comparison does not give only
RPS-GTok access to relabeling. Conversely, the downstream experiments use the
same Transformer consumer and parameter count, so the results test the
representation interface rather than a larger consumer. The all-21 constructed
task is retained as a structural coverage control and is explicitly labeled
`constructed_topology_proxy` in [`Table E16`](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_E16_constructed_topology_proxy_notrunc_aggregate.csv),
not presented as a semantic-label replacement.

The principal commands are:

```bash
PYTHONPATH=. python scripts/audit_matched_consumer_runs.py
PYTHONPATH=. python scripts/audit_constructed_graph_tasks.py
PYTHONPATH=. python scripts/run_footprint_attribute_breakdown.py
PYTHONPATH=. python scripts/run_21_canonical_controls.py \
  --data-dir rebuttal/data_21 \
  --out rebuttal/experiments/E1_21_canonical_controls_bpe_full \
  --graphs 6 --permutations 3
```

Together, the controls establish the intended conclusion: RPS-GTok is most
useful when the downstream consumer must operate on a stable, executable graph
sequence and when topology must be retained without silently discarding the
payload needed for expansion. This is the central operating regime targeted by
the proposed graph-to-sequence interface.

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
