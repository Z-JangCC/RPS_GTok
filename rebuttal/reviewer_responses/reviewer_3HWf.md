# Response to Reviewer 3HWf

We sincerely thank the reviewer for noting that our design is coherent at a
high level and for the careful, constructive suggestions. We have carried out
the following supplementary experiments and provided further explanations
regarding the issues you raised.

## Deeper motivation and technical distinction

An isomorphic triangle after node-ID permutation can yield different raw edge
or DFS/BFS sequences, so an ordered Transformer sees two inputs. Canonicalization
fixes order but not executable arguments; primitive compilation is order-sensitive;
BPE does not enforce graph-state preconditions. RPS-GTok couples a canonical
frame, executable primitives and payload-bearing fragments, making the
Transformer input invariant and executable. This is the design-level distinction
beyond adding another compression layer.

## Expanded downstream evaluation

We extended evaluation beyond topology surrogates using source-aligned
PROTEINS/MOLHIV labels and physical power flow, with one plain Transformer,
three seeds and validation-only selection. Corrected semantic BA is 0.6562
(raw), 0.6875 (canonical) and 0.8125 (RPS-GTok) on PROTEINS; the balanced
MOLHIV control gives 0.7500, 0.7222 and 0.7685, respectively. In the
case-disjoint power-flow task, MAE is 0.03942, 0.03829 and 0.03910 with equal
70,433-parameter consumers. These source-locked controls extend the evidence
to semantic and physical tasks and identify the settings where executable
invariant context is most useful.

## Truncation and Full-Embed fairness

At the original 512-token budget, raw and canonical edge views have 10% test
truncation while RPS-GTok has 0%. The locked no-truncation control reports 0%
truncation for every view; the source-locked MOLHIV and power-flow controls
also retain 0% train/validation/test truncation. Full-Embed is auxiliary. The
primary comparisons use Token-only views, a shared plain adapter, a train-only
union vocabulary and equal parameter counts, so the reported realistic-task
differences are not attributed to an extra feature channel or discarded tokens;
the controls therefore isolate the tokenizer interface.

## Canonicalized baselines and scalability

The same canonicalizer is applied to edge, adjacency, DFS/BFS and frozen-BPE
views. Canonical relabeling improves the controls but is not near-perfect:
raw edge/adjacency and raw DFS/BFS have mean stability 0.0238 and 0.0000;
canonical edge/adjacency/DFS, canonical BFS and RPS-GTok identifier reach
0.7249, 0.7778 and 0.9550. The 21-dataset, six-graph-per-dataset frozen-BPE
evaluation reproduces the corresponding 0.7249/0.7778 canonical range. The exact
backend certifies 974/974 graphs across all 21 datasets, including 191 directed
inputs, under 300 seconds per graph. Exact-audit wall times are 2.374 s (AST),
2.235 s (IEEE), 2.436 s (Road), 13.152 s (Synthetic) and 212.558 s (OGBG-PPA);
measured canonicalization shares are 10.8%, 6.45%, 5.87% and 7.04% for AST,
IEEE, Road and Synthetic. Together, these results
establish the measured scalability boundary.

We sincerely thank the reviewer again for the time and effort devoted to evaluating our manuscript, and for the thoughtful and valuable comments that have greatly helped us improve its quality and clarity. We hope that the clarifications, additional analyses, and new experimental results provided in this rebuttal have adequately addressed the reviewer’s concerns. For the issues you raised, our anonymous repository linked in the submission provides more detailed explanations, concrete experimental settings, and complete result data.
