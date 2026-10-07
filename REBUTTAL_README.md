# RPS-GTok rebuttal package

This directory contains the complete rebuttal evidence package for the anonymous
RPS-GTok release.

## Independent reviewer responses

- [Reviewer 3HWf](REBUTTAL_REVIEWER_3HWf.md)
- [Reviewer 8tZd](REBUTTAL_REVIEWER_8tZd.md)
- [Reviewer bYQg](REBUTTAL_REVIEWER_bYQg.md)
- [Reviewer HFMW](REBUTTAL_REVIEWER_HFMW.md)
- [Reviewer Rkxp](REBUTTAL_REVIEWER_Rkxp.md)

The repository-oriented detailed response is [REBUTTAL_FULL_RESPONSE_ANONYMOUS.md](rebuttal/reviewer_responses/REBUTTAL_FULL_RESPONSE_ANONYMOUS.md). The conference-length response is [REBUTTAL_SHORT_2500_CHARS.md](rebuttal/reviewer_responses/REBUTTAL_SHORT_2500_CHARS.md).

## Main authoritative evidence

- [Full exact certificate audit](rebuttal/tables/Table_E11_nauty_all21_full_v12.csv)
- [Canonical-BPE control](rebuttal/experiments/E1_21_canonical_controls_bpe_full/summary.csv)
- [Balanced MOLHIV downstream](rebuttal/tables/Table_E14_ogbg_molhiv_balanced_post_schema_aggregate.csv)
- [Physical PowerFlow downstream](rebuttal/tables/Table_E15_powerflow_perturbed_post_schema_aggregate.csv)
- [All-dataset structural control](rebuttal/tables/Table_E16_constructed_topology_proxy_notrunc_aggregate.csv)
- [Complete-stream payload audit](rebuttal/experiments/E6_complete_stream/)
- [Schema contract](rebuttal/tables/Table_R8_schema_contract.csv)
- [Attribute policy](rebuttal/tables/Table_R10_schema_attributes.csv)
- [Compact topology/attribute footprint](rebuttal/tables/Table_R14_footprint_attribute_breakdown.csv)
- [Matched-consumer audit](rebuttal/audits/matched_consumer_runs.json)
- [Constructed-task leakage audit](rebuttal/audits/constructed_task_audit.csv)

## Reproduction entry points

Run from the repository root with `PYTHONPATH=.`.

```bash
# Final package and test gate
PYTHONPATH=. python scripts/run_final_rebuttal_gate.py

# Exact 21-dataset certificate audit
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONPATH=. \
python scripts/run_nauty_exact_permutation_audit.py \
  --data-dir rebuttal/data_21 \
  --out rebuttal/tables/Table_E11_nauty_all21_full_v12.csv \
  --max-graphs 60 --timeout-sec 300 --workers 8

# Dataset-level runtime measurements
PYTHONPATH=. python scripts/run_dataset_runtime_audit.py --help

# Canonical serialization and frozen-BPE controls
PYTHONPATH=. python scripts/run_21_canonical_controls.py \
  --data-dir rebuttal/data_21 \
  --out rebuttal/experiments/E1_21_canonical_controls_bpe_full \
  --graphs 6 --permutations 3

# Compact topology/attribute footprint breakdown
PYTHONPATH=. python scripts/run_footprint_attribute_breakdown.py

# Constructed topology labels and leakage audit
PYTHONPATH=. python scripts/build_constructed_graph_tasks.py \
  --data-dir rebuttal/data_21 \
  --out-dir rebuttal/constructed_tasks
PYTHONPATH=. python scripts/audit_constructed_graph_tasks.py

# Matched downstream consumer audit
PYTHONPATH=. python scripts/audit_matched_consumer_runs.py
```

The authoritative-results index is [rebuttal/AUTHORITATIVE_RESULTS_INDEX.md](rebuttal/AUTHORITATIVE_RESULTS_INDEX.md); it identifies which historical pilots are diagnostic-only and which files are cited in the rebuttal.
