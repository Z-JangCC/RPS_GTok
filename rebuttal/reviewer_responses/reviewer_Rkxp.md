# Detailed anonymous response to Reviewer Rkxp

This standalone file expands the reviewer-specific section of the anonymous
rebuttal package. It is intended for code-level verification and therefore
records the exact protocol, implementation entry points, reported results and
claim boundaries. All paths are relative to this repository; the corresponding
anonymous web root is https://anonymous.4open.science/r/RPS_GTok/.

## Questions addressed

This response addresses:

- whether token cost includes reconstruction payload and actual complete storage.
- realistic graph-learning tasks beyond topology-statistics prediction.
- runtime and scalability of canonical relabeling on the released corpus.

## Shared notation and evaluation contract

For a serialization view $S$ and a permutation set $\Pi$, stability is
$\mathrm{Stab}(S)=|\Pi|^{-1}\sum_{\pi\in\Pi}\mathbf{1}[S(\pi G)=S(G)]$.
A complete RPS-GTok instance is an identifier stream $I$ together with ordered
payloads $P$ under schema $\Sigma$, and the reconstruction contract is
$D(I,P;\Sigma)=G_{\Sigma}$. Complete storage is reported as
$B_{\mathrm{total}}=B_{\mathrm{id}}+B_{\mathrm{payload}}+B_{\mathrm{stream}}+B_{\mathrm{artifact}}$.
These definitions keep identifier efficiency, executable correctness,
permutation behavior and downstream utility separate.

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

### Storage protocol and payload accounting

The identifier-only values are deliberately separated from the self-contained
representation. For each graph, the compact audit records packed identifier
bytes, the amortized codebook/rulebook bytes, the executable local arguments,
the attribute stream and the fixed schema/artifact component. Thus the
topology-only value of 16.208 bytes is a model-facing context measure, whereas
the 334.417-byte topology component includes the arguments needed to execute
the graph program. The 467.958-byte attribute component is shown separately
because node/edge attributes are semantic information rather than an invisible
tokenizer overhead; the complete 802.375-byte value is their sum under the
declared protocol. Raw and canonical edge use the same packed-varint topology
protocol (27.417 bytes/graph), making the topology-only comparison direct.

The complete-stream checks in [`rebuttal/experiments/E6_complete_stream/`](https://anonymous.4open.science/r/RPS_GTok/rebuttal/experiments/E6_complete_stream/)
verify token grammar, payload alignment and raw-token round-trip on MUTAG and
IMDB. The reconstruction statement is therefore about a complete `(I, P,
Sigma)` instance, not about an identifier-only autoregressive sample. The
schema and continuous-feature policy are listed in
[`Table R8`](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R8_schema_contract.csv)
and [`Table R10`](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R10_schema_attributes.csv).

### Label provenance and task construction

The source-label audit distinguishes verified graph-level labels from structural
sidecars. Verified TU/OGB/Planetoid joins are retained only when graph IDs and
split membership are one-to-one; graphs without verifiable graph-level labels
use the explicitly named `constructed_topology_proxy` sidecar, generated from
`num_nodes`, edges and direction only. The leakage check recomputes targets,
checks graph-ID alignment, enforces train-only thresholds and verifies that
source `y` and forbidden attribute fields were not read. The detailed records
are in [`Table R11b`](https://anonymous.4open.science/r/RPS_GTok/rebuttal/tables/Table_R11b_label_recovery_expanded.csv)
and [`constructed_task_audit.csv`](https://anonymous.4open.science/r/RPS_GTok/rebuttal/audits/constructed_task_audit.csv).

E14 uses balanced official MOLHIV labels, a train-only union vocabulary, a
shared plain adapter, three seeds and a 168/36/36 split. E15 uses deterministic
load perturbations and `pandapower.runpp`; target quantities are removed from
input attributes and cases, rather than perturbations, define the split. The
matched-consumer audit checks 247,842 parameters for E14, 70,433 for E15,
complete 3-view/3-seed grids, finite outputs, prediction diversity and zero
truncation. This protocol is the basis for interpreting the downstream numbers
as tokenizer-interface evidence rather than as a change in model capacity.

### Runtime protocol and commands

The exact backend records per-graph completion, direct-simple-graph versus
WL-refined attributed-incidence path, directed role gadgets, timeout and wall
time. The runtime table additionally records total tokenization time, fitting
time, encoding time and the canonicalization fraction for each of the 21
datasets. This makes the expensive highly symmetric cases visible instead of
reporting only an aggregate average. The complete exact command is:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. \
python scripts/run_nauty_exact_permutation_audit.py \
  --data-dir rebuttal/data_21 \
  --out rebuttal/tables/Table_E11_nauty_all21_full_v12.csv \
  --max-graphs 60 --timeout-sec 300 --workers 8
```

The dataset-level timing entry point is
`scripts/run_dataset_runtime_audit.py`; its summary is
[`E2_runtime_dataset_fast2/summary.csv`](https://anonymous.4open.science/r/RPS_GTok/rebuttal/experiments/E2_runtime_dataset_fast2/summary.csv).
The release gate combines these checks with the matched-consumer and leakage
audits through `scripts/run_final_rebuttal_gate.py`.

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
