# Reviewer-response matrix

| Reviewer | Issue | Type | Severity | Existing evidence | New experiment/analysis | Response strategy |
|---|---|---|---|---|---|---|
| 3HWf | Topology-only downstream tasks | Experimental request | Major | R4/R12 | E14 balanced MOLHIV, E15 power flow, E16 proxy | Provide realistic gains and explicitly retain task dependence |
| 3HWf | 512-token truncation | Fairness | Major | R5 locked table | Shared plain no-truncation control | Report 0% truncation and calibrated operating regime |
| 3HWf | Per-method truncation rates under a common budget | Experimental request | Major | R5 corrected/locked tables | Budget-locked rerun | Report each view's truncation rate before interpreting downstream scores |
| 3HWf | Full-Embed vs token-only | Fairness | Major | R4 protocols | Shared plain union-vocabulary protocol | Make Full-Embed auxiliary only |
| 3HWf | Canonicalized baselines, including BPE | Baseline | Major | R1b/E1 plus BPE pilot | Same canonicalizer and frozen BPE rules for edge/adjacency/DFS/BFS | Show canonical and canonical-BPE controls reproduce the reference-frame range |
| 3HWf | Runtime/scaling | Experimental request | Major | R2/E2 | v12 exact backend and runtime CSVs | Report finite budget and no polynomial claim |
| 8tZd | Technical novelty | Novelty | Major | Method spec | Mechanism-level distinction | Executable payload-aware contract, not new GI algorithm |
| 8tZd | Reconstruction implies downstream gain | Theoretical/causal | Major | E5/R7 | Paired deletion control | State association only; no causal claim |
| 8tZd | Parameter/objective artifacts | Fairness | Major | matched audit | E14/E15 equal parameter/union vocabulary | Machine-check parity |
| 8tZd | Foundation baseline | Baseline | Major | E8 | Matched GIN reference | Report model-level reference without tokenizer attribution |
| bYQg | GI/canonical-form hardness | Theory | Major | R2/E2 | v12 full exact audit | Agree, scope complexity and empirical budget |
| bYQg | Identifier-only generation lacks payload | Correctness | Major | E6 | Complete-stream control | Narrow generation claim to complete instances |
| bYQg | Schema/continuous attributes | Reproducibility | Major | R8/R10 | Leakage/schema tests | Enumerate retained fields and policy |
| bYQg | Eq. 32 literal equality | Formal scope | Major | E6/schema audit | Claim graph-equivalent expansion only | Narrow theorem wording |
| bYQg | Cache/projection defects | Correctness | Major | E1 cachefix | Content fingerprint and cyclomatic anchor fixes | Exclude all pre-fix numbers |
| bYQg | Token/reconstruction metric populations | Reproducibility | Minor | Corrected metric builders | Population fields and locked aggregations | State the graph population for every reported metric |
| bYQg | Vocabulary sensitivity and macro mining | Ablation/reproducibility | Minor | R6 and tokenizer configuration | Frozen rules, minimum count 2, bounded rule sets | Explain active-code saturation and mining policy |
| bYQg | Definitions, algorithm order and figure symbols | Presentation | Minor | Method specification | Terminology/order cross-check | Define graph editing before use and map symbols to the eight families |
| bYQg | No-truncation comparison for Tables 3/4 | Experimental request | Major | R5 locked table | Common-budget semantic rerun | Report per-view truncation and use the locked no-truncation control |
| HFMW | Canonicalized standard baselines | Baseline | Major | R1b | Same canonical reference frame | Isolate tokenizer contribution |
| HFMW | Payload omitted from compactness | Metric | Major | R3/R9/E6 | Complete footprint | Report ID, payload, stream and artifacts separately |
| HFMW | Semantic generalization | Experimental request | Major | R4/R11b | E14/E15/E16 | Real tasks plus explicit proxy scope |
| HFMW | Scalability | Experimental request | Major | R2 | v12 direct/WL backend | Full materialized coverage at finite budget |
| Rkxp | Complete storage overhead | Metric | Major | R3/R9 | E6 complete stream | Give byte-level storage, not only Tok./edge |
| Rkxp | Realistic graph tasks | Experimental request | Major | R11b | E14/E15 | Fair realistic controls and complementary operating regimes |
| Rkxp | Canonicalization runtime | Experimental request | Major | R2/E2 | v12 protocol and wall times | Report workers, budget, directed count and completion |
| All | Unsupported graph labels | Data integrity | Major | R11b | Constructed E16 sidecars | No imputation; source y null; leakage audit |

No reviewer concern is silently omitted. Unsupported universal claims are
explicitly narrowed rather than defended with unsupported evidence.
