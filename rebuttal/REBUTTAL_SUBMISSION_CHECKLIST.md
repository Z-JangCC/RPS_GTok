# KDD27 rebuttal submission checklist

## Files to cite

- Formal manuscript PDF: `KDD_paper/graph_tokenizer_kdd_package/acm_submission/main_four.pdf`
- Reviewer comments: `KDD_Review.txt`
- Point-by-point draft: `rebuttal/FINAL_REBUTTAL_RESPONSE_DRAFT.md`
- Evidence index: `rebuttal/AUTHORITATIVE_RESULTS_INDEX.md`
- Full audit/plan: `rebuttal/CURRENT_EVIDENCE_AUDIT_AND_PLAN_20261004.md`
- Citation-number manifest: `rebuttal/rebuttal_number_manifest.json`

## Reviewer coverage

| Concern | Cite | Status |
|---|---|---|
| Canonicalized edge/adjacency/DFS/BFS controls | Table R1b; Table E1 | complete, exact/fallback separated |
| Canonical-form/GI complexity | main appendix; Table R2; E2 runtime audit | bounded/input-sensitive claim only |
| Runtime/scaling | Table R2; E2 runtime summaries | complete bounded audit; stress outliers explicit |
| Pruned IR and Eq. 34 | exact/fallback fields; E1/E2 | exact guarantee conditional only |
| Payload generation gap | E6 complete stream; payload.py | complete-stream control; identifier-only lossless generation not claimed |
| Schema/continuous attributes | Tables R8/R10; schema tests | retained fields and policy audited |
| 512 truncation | Table R5 locked | complete shared-plain 60/20/20 control |
| Semantic downstream | Table R4 corrected; Table E8 GIN | MUTAG/PROTEINS/IMDB/COLLAB pilot, dataset-dependent |
| Reconstruction/downstream relationship | E5 paired diagnostic | negative/diagnostic; no causal claim |
| Parameter/objective confounding | E7 fair grid; locked shared-plain | complete for authoritative controls |
| Foundation baseline | Table E8 | matched GIN complete for four labeled datasets |
| Complete footprint | Tables R3/R9; E6 | identifiers, payload, complete stream, artifact bytes |
| 21-dataset scope | Tables R11/R12 | labels audited; 16 structural datasets evaluable; 5 undersized excluded |

## Claims allowed in the response

- RPS-GTok defines executable primitive-fragment graph tokens with explicit
  expansion payloads and primitive fallback.
- Complete emitted instances are reconstructable under the declared schema.
- Permutation stability is conditional on exact canonicalization completion;
  bounded fallback is reported separately.
- Complete footprint is distinct from identifier-only context cost.
- RPS utility is dataset/task-dependent and can be positive on selected tasks.

## Claims prohibited

- Universal or statistically significant superiority over every baseline.
- Causal claim that reconstruction improves downstream performance.
- Automatic lossless decoding of unconstrained identifier-only AR output.
- Polynomial-time or universally exact canonicalization.
- Treating the unlabeled 21-dataset materialized corpus as semantic labels.
- Citing pre-cachefix permutation results.
- Treating fallback/timeout rows as exact certification.

## Final commands

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/audit_all_rebuttal_results.py
PYTHONDONTWRITEBYTECODE=1 pytest -q
PYTHONDONTWRITEBYTECODE=1 python scripts/verify_release.py
cd KDD_paper/graph_tokenizer_kdd_package/acm_submission
latexmk -pdf -interaction=nonstopmode -halt-on-error main_four.tex
```

All commands must pass immediately before submission. Historical diagnostic
directories must not be copied into the main result tables.

The combined gate is also available as:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/run_final_rebuttal_gate.py
```

Its machine-readable result is `rebuttal/final_rebuttal_gate_report.json`.
