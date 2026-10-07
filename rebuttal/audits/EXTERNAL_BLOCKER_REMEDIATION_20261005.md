# External blocker remediation audit (2026-10-05)

## Blocker A — missing semantic labels

The materialized `rebuttal/data_21/*.jsonl` files are schema records with
`y=null`; this is verified for all 21 files.  A label must not be inferred from
graph size, metadata, or the structural regression target.

We added `scripts/recover_supervised_labels.py`. It joins stable graph indices
to locally retained TU, GNNBenchmark, OGB, Planetoid, and OGBN source caches,
and requires a source shape/topology check before writing a labeled copy. The
expanded status is in `tables/Table_R11b_label_recovery_expanded.csv`:

| dataset | sampled records | exact labels recovered | release status |
|---|---:|---:|---|
| MUTAG | 60 | 60 | labeled copy written |
| PROTEINS | 57 | 57 | labeled copy written |
| IMDB-BINARY | 60 | 60 | labeled copy written |
| ENZYMES | 60 | 60 | labeled copy written |
| CIFAR10 / MNIST | 60 / 60 | 60 / 60 | GNNBenchmark labels recovered |
| OGBG-MOLHIV / MOLPCBA / PPA | 60 / 60 / 8 | all | official OGB targets recovered; PCBA remains multilabel |
| IEEE Power Grid | 3 | 3 | `pandapower.runpp` mean-voltage regression target |
| COLLAB | 60 | 40 | not released (partial topology match) |
| remaining datasets | — | 0 or partial | no fully verified semantic join |

The exact machine-readable record is
`audits/label_recovery_audit.csv`; labeled copies are under
`data_21_labeled/`.  Partial joins are deliberately not used for a semantic
table. Thus the missing-label blocker is removed for the listed cached sources,
while single-class prefixes, multilabel PCBA, and ambiguous Planetoid/OGBN ego
joins remain
explicitly marked rather than hidden.

`audits/recovered_input_leakage.csv` reports zero forbidden target fields in
all recovered records. In particular, PCBA keeps an observed-mask vector and
is not converted into a hidden scalar label.

## Blocker B — exact permutation/canonicalization scalability

`pynauty` is now installed in the execution environment.  We added a reusable
`rps_gtok_consumption.exact_backend` and
`scripts/run_nauty_exact_permutation_audit.py`. The backend builds a degree-
refined colored incidence graph (node/edge attributes, parallel edges and
self-loops are retained); directed edges use source/destination role gadgets.
It uses nauty's
canonical certificate and isolates each graph in a hard-timeout worker. The
semantic TU audit (`tables/Table_E11_nauty_exact_semantic.csv`) certifies
equality for every completed pair:

| dataset | tested | completed | timeouts | certified equality |
|---|---:|---:|---:|---:|
| MUTAG | 60 | 60 | 0 | 1.000 |
| PROTEINS | 57 | 55 | 2 | 1.000 |
| IMDB-BINARY | 60 | 58 | 2 | 1.000 |
| ENZYMES | 60 | 60 | 0 | 1.000 |

Timeouts remain explicitly reported for difficult symmetric graphs; they are
not counted as exact.  The full 21-dataset corpus therefore still cannot be
claimed to have 100% exact canonicalization: several files contain very large
graphs or directed records.  The exact evidence is now a certified subset with
per-graph timeout accounting, which directly repairs the earlier claim scope.

## Consequence for the rebuttal

The fully recovered files may support supervised downstream reruns using
`data_21_labeled`, subject to the eligibility column in Table R11b. The
rebuttal should cite the source/task policy explicitly; no unlabeled 21-dataset
semantic aggregate or full-corpus 100%-exact claim is admissible.

The higher-budget full-corpus audit
(`tables/Table_E11_nauty_all21_full_v12.csv`) processes all 974 materialized
graphs in all 21 datasets with a direct simple-graph path plus a WL-refined
incidence fallback, eight spawn-isolated workers and a uniform 300-second
per-graph budget. It records 974 completed pairs (all certificate equality
1.000), zero timeouts, zero errors and 191 directed inputs. This is complete
exact coverage of the materialized corpus under the documented resource
budget; it does not assert a polynomial worst-case guarantee for arbitrary
graphs.
