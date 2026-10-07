# Response to Reviewer Rkxp

We sincerely thank the reviewer for the careful evaluation and constructive
suggestions. We have carried out the following supplementary experiments and
provided further explanations regarding the issues you raised.

## Complete compactness

The revised pipeline reports identifier context cost, token-local payload,
complete stream bytes and amortized artifact/rulebook footprint separately. In
the compact topology/attribute audit, raw/canonical edge context is 27.417
bytes per graph, topology-only RPS-GTok identifier plus amortized codebook is
16.208 bytes, and the RPS topology/attribute/full components are 334.417,
467.958 and 802.375 bytes. This shows codebook compression in the model-facing
context while exposing attribute storage explicitly.
The complete-stream audit additionally round-trips raw token instances and
graph payload; identifier-only generation is not labeled lossless.

## Realistic graph learning

Corrected original-label semantic runs are available for MUTAG, PROTEINS and
IMDB-BINARY with three seeds, balanced metrics and view-sequence audits. The
source-locked PROTEINS BA is 0.6562 (raw), 0.6875 (canonical) and 0.8125
(RPS-GTok); MOLHIV is 0.7500, 0.7222 and 0.7685; power-flow MAE is 0.03942,
0.03829 and 0.03910. E14/E15 use equal parameters and 0% truncation.

## Scalability

The runtime results and per-dataset measurements give canonicalization time,
exact completion, branch budgets, fallback and elapsed time. The full exact backend
covers 974/974 graphs across 21 datasets, including 191 directed inputs, under
300 s/graph; AST/IEEE/Road/Synthetic/PPA wall times are 2.374/2.235/2.436/
13.152/212.558 s. Dense COLLAB and OGBG-PPA remain explicit high-runtime cases.

The matched GIN summary now covers MUTAG, PROTEINS, IMDB-BINARY and COLLAB.
The all-dataset structural protocol is explicitly separated from semantic
evaluation because the materialized 21-dataset corpus has no graph-level y
labels.

The fairness protocol now includes a frozen union vocabulary and a shared plain
adapter. The balanced MOLHIV control has equal 247,842 parameters across all
views; the pandapower control has equal 70,433 parameters. Complete grids,
0% test truncation and parameter equality are checked by
`audits/matched_consumer_runs.json`.

We sincerely thank the reviewer again for the time and effort devoted to evaluating our manuscript, and for the thoughtful and valuable comments that have greatly helped us improve its quality and clarity. We hope that the clarifications, additional analyses, and new experimental results provided in this rebuttal have adequately addressed the reviewer’s concerns. For the issues you raised, our anonymous repository linked in the submission provides more detailed explanations, concrete experimental settings, and complete result data.
