# Reviewer issue matrix and final evidence gates

This matrix is the source of truth for rebuttal completeness. “Evidence
available” means an executable artifact exists; it does not mean a broad claim
is licensed until the stated gate passes.

| Reviewer | Issue | Required response/evidence | Current state |
|---|---|---|---|
| 3HWf/HFMW/bYQg/Rkxp | canonicalized edge/adjacency/DFS/BFS/BPE controls | same canonicalizer, exact/fallback audit, permutation stability and reconstruction | E1 controls available; full 21-dataset exact certification remains conditional |
| bYQg | both properties imply a canonical form/GI hardness | explicitly acknowledge canonical-form reduction; give bounded-search worst-case and conditional guarantee | implementation and E2 pilot available; manuscript must not call it polynomial |
| bYQg/Rkxp | runtime/scaling/memory of canonicalization | per-dataset wall time, canonicalization fraction, search nodes, timeout/fallback, environment | E2 synthetic plus resumable 21-dataset audit (`E2_runtime_dataset_fast`); IEEE no-test sample excluded |
| bYQg | Eq. 34 under pruning | exact_completed/fallback fields and permutation failure table | implemented; fallback is never reported as exact |
| bYQg | identifier-only AR cannot recover payload | complete-stream control, payload round-trip, constrained grammar boundary | E6 complete-stream audit available; identifier-only generation claim is removed |
| bYQg | schema and continuous attributes unspecified | dataset schema manifest, quantizer/fallback policy, original-vs-quantized fidelity | schema tests and full Table R10 attribute/task audit available; remaining limitation is prescribed-policy scope |
| all datasets | converted 21-dataset JSONL has no graph-level `y` labels | source-index/topology recovery, task-availability audit, and explicit quarantine of ambiguous joins | R11b recovers audited TU/GNNBenchmark/OGB sources; PCBA stays multilabel; structural audit remains for unlabeled/unsupported records |
| evidence hygiene | historical runs can mix pre-cachefix, pilot, fallback, and authoritative results | explicit authoritative-results index and failure log | `AUTHORITATIVE_RESULTS_INDEX.md` added; audit scripts continue to reject duplicates/non-finite/collapsed rows |
| 3HWf/8tZd | raw serialization may be too easy for simple statistics | permutation-shift topology regression and permutation matching use the same Transformer and directly test invariant topology transfer | E10 cache-fixed MUTAG pilot complete; E11 full run is diagnostic; exact-only full-corpus subset is too small under bounded budget |
| bYQg/3HWf | 512 truncation confounds results | per-method truncation plus common no-truncation subset | locked shared-plain 60/20/20 no-truncation control complete; RPS negative result retained |
| Rkxp/bYQg | token cost excludes payload/rulebook | identifier, payload, complete stream, artifact amortization and bits/edge | E3/R9 available; zero-test datasets excluded |
| 3HWf/HFMW/Rkxp | downstream only topology surrogates | semantic labels, realistic graph tasks, fixed splits and class-balanced metrics | corrected TU controls, balanced MOLHIV E14, source-label audit R11b, and matched GIN references; negative/neutral results retained |
| 8tZd | reconstruction/downstream relationship | randomized coverage ablation, joint reconstruction and downstream metrics, uncertainty | E5 synthetic plus corrected MUTAG/PROTEINS paired diagnostics complete; downstream collapse means no causal claim is made |
| 8tZd | gains may be parameter/objective artifacts | frozen union vocabulary, shared adapter, equal parameter counts, same seeds/epochs, validation-only selection | E14/E15 union-vocabulary shared-plain audits pass; parameter counts and zero truncation are machine-checked |
| 8tZd | recent graph foundation baselines | matched GIN reference under same split, class weighting, validation selection and three seeds | E8 GIN complete for MUTAG/PROTEINS/IMDB-BINARY/COLLAB; GraphSAGE/GraphGPS remains optional |
| 8tZd | technical novelty | distinguish executable fragment semantics, fallback coverage, payload contract from canonicalization | method specification and controls available |
| bYQg | vocabulary mining underspecified | frequency thresholds, mining algorithm, frozen codebook and sensitivity sweep | implementation/table available |
| bYQg | metric population inconsistencies | one schema for Tok./edge, Recon., truncation and population fields | corrected table builders and audit available |
| bYQg | Eq. 32 literal sequence equality | either prove under a fixed canonical emission order or narrow to graph-equivalent expansion | formal boundary must use graph-equivalence wording unless proof artifact passes |

## Non-negotiable completion rule

The rebuttal is complete only for the explicitly scoped authoritative evidence
above; unsupported broader runs must be labeled as unavailable or diagnostic,
not silently treated as completed. If RPS-GTok++ is not statistically best
under the locked primary metric, report it as competitive and preserve the
negative result; do not tune on the test set or remove an unfavorable dataset.
