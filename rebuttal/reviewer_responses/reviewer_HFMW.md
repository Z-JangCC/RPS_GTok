# Response to Reviewer HFMW

We sincerely thank the reviewer for identifying graph representations consumed
by sequence models as a well-motivated problem, and for the careful suggestions.
We have carried out the following supplementary experiments and provided further
explanations regarding the issues you raised.

## Canonicalized reversible baselines

We apply the same canonicalizer to edge, adjacency, DFS/BFS and frozen-BPE
views. Mean stability is 0.0238 for raw edge/adjacency, 0.0000 for raw DFS/BFS,
0.7249 for canonical edge/adjacency/DFS, 0.7778 for canonical BFS and 0.9550
for the RPS-GTok identifier view. Canonical BPE reproduces the canonical range.
This directly tests the reviewer’s stronger baseline: canonical serialization
does recover reconstruction plus improved stability, while RPS-GTok adds
executable fragments, payloads and fallback.

## Complete reconstructable footprint

We report
$B_{total}=B_{id}+B_{payload}+B_{stream}+B_{artifact}$ rather than only
identifier counts. In the compact 24-graph audit, raw/canonical edge topology
context uses 27.417 bytes per graph, while topology-only RPS-GTok identifier
plus amortized codebook uses 16.208 bytes (40.9% lower). The executable RPS
topology component is 334.417 bytes; with attributes, 467.958 bytes are
attribute payload and 802.375 bytes are complete. This separates compact model
context from auditable expansion storage and makes attribute overhead explicit.

## Realistic downstream tasks

We added source-aligned PROTEINS and MOLHIV classification and a case-disjoint
pandapower load-perturbation task. PROTEINS BA is 0.6562 (raw), 0.6875
(canonical) and 0.8125 (RPS-GTok); MOLHIV is 0.7500, 0.7222 and 0.7685; power-
flow MAE is 0.03942, 0.03829 and 0.03910. E14/E15 use shared plain adapters,
union vocabulary, three seeds, equal parameters and 0% truncation.

## Canonicalization scalability

The exact backend covers 974/974 graphs across all 21 datasets, including 191
directed inputs, under 300 seconds per graph. Exact-audit wall times are 2.374 s
(AST), 2.235 s (IEEE), 2.436 s (Road), 13.152 s (Synthetic) and 212.558 s
(OGBG-PPA). The bounded per-dataset audit records total time and canonicalization
fraction for every dataset; the environment is Linux x86-64/Python 3.12.2 with
eight isolated workers. No polynomial worst-case canonicalization guarantee is
claimed.

We sincerely thank the reviewer again for the time and effort devoted to evaluating our manuscript, and for the thoughtful and valuable comments that have greatly helped us improve its quality and clarity. We hope that the clarifications, additional analyses, and new experimental results provided in this rebuttal have adequately addressed the reviewer’s concerns. For the issues you raised, our anonymous repository linked in the submission provides more detailed explanations, concrete experimental settings, and complete result data.
