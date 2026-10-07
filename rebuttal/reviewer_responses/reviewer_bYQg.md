# Response to Reviewer bYQg

We sincerely thank the reviewer for the careful evaluation and technically
precise questions. We have carried out the following supplementary experiments
and provided further explanations regarding the issues you raised.

## Canonical form and computational boundary

We agree that the two properties define a canonical reference usable for
isomorphism decisions. RPS-GTok does not claim a new polynomial-time
canonization algorithm; its contribution is the executable payload-aware
contract layered on that reference. Individualization-refinement has worst-case
$O(n!\,p(n,m,q))$. Primitive compilation and fragment selection have
$O(n+m_e+n_a)$ and $O(c(G)\log c(G))$ bounds in the paper.

## Runtime and environment

The exact backend uses a direct simple-graph path and a WL-refined attributed
incidence path, with directed role gadgets. The environment is Linux x86-64,
Python 3.12.2, eight isolated workers and a 300-second per-graph budget. The
reproducible entry points are `scripts/run_dataset_runtime_audit.py` and
`scripts/run_nauty_exact_permutation_audit.py`.

The per-dataset runtime measurements give total tokenization time and canonicalization
fraction: AST-CFG-CPG 4.938 s/10.78%, CIFAR10-Superpixels 99.021/1.45%, CiteSeer
41.168/1.19%, COLLAB 129.989/1.12%, Cora 24.919/1.52%, ENZYMES 2.839/5.27%,
FB15k-237 67.574/0.69%, IEEE 0.385/6.45%, IMDB 31.443/0.61%, MNIST 59.569/1.42%,
MUTAG 0.396/5.64%, Code2 3.274/12.07%, MOLHIV 0.410/10.15%, MOLPCBA 0.495/6.60%,
OGBN-Arxiv 194.531/0.43%, PROTEINS 4.867/5.62%, PubMed 57.995/0.72%, Road
3.948/5.87%, Synthetic 0.217/7.04% and WN18RR 13.700/0.68%.

## Exact versus pruned ordering

Exact branches exhaustively certify the selected order within budget. Bounded or
pruned branches are labeled fallback and Eq. (34) is not asserted for them. No
released v12 exact certificate fails Eq. (34): all 974 original/permuted pairs
agree.

## Payload generation and complete storage

An identifier-only autoregressive sequence cannot infer the payload $\alpha$.
The reconstructability claim is therefore for complete $(I,P)$ instances:
$D(I,P;\Sigma)=G_{\Sigma}$. E6 grammar, payload and raw-token roundtrips are
1.0 on MUTAG and IMDB; mean identifier/complete-stream token counts are
8.2/42.6 and 28.5/138.15. Since argument serialization is grammar-dependent,
there is no unique Tok./edge or Tok./node after adding $\alpha$; we report the
reproducible complete bytes separately.

## Schema and continuous features

R8/R10 provide one manifest per dataset. COLLAB and IMDB retain topology;
Cora/ENZYMES/MUTAG/PROTEINS retain topology and node types; FB15k-237,
Code2, OGBN-Arxiv and WN18RR use directed typed schemas; the remaining manifests
retain their declared node/edge types and attributes. All released manifests use
raw continuous mode. The 100% reconstruction figure is against this retained
representation, with loops, duplicates, directions and attributes audited.

## Truncation and G2PT/BPE attribution

At 512 tokens, raw and canonical edge views truncate 10% of IMDB test graphs and
RPS-GTok truncates 0%; COLLAB is 50% versus 0%, while MUTAG and PROTEINS are 0%
for all corrected views. Locked no-truncation and E14/E15 controls set every
view to 0%. The final downstream claims therefore do not rely on an uncontrolled
512-token attribution. The anonymous repository contains the corrected per-view
truncation results and no-truncation comparison; original G2PT/BPE rows are not
extrapolated beyond that matched evidence.

We sincerely thank the reviewer again for the time and effort devoted to evaluating our manuscript, and for the thoughtful and valuable comments that have greatly helped us improve its quality and clarity. We hope that the clarifications, additional analyses, and new experimental results provided in this rebuttal have adequately addressed the reviewer’s concerns. For the issues you raised, our anonymous repository linked in the submission provides more detailed explanations, concrete experimental settings, and complete result data.
