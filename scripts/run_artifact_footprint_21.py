"""Measure dataset-level vocabulary/rulebook footprint for 21 datasets."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gptok2.data.io import load_records
from gptok2_tokenizer import GPTok2Tokenizer
from scripts.run_21_token_eval import DATASETS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="rebuttal/data_21")
    parser.add_argument("--out", default="rebuttal/experiments/E3_artifact_footprint_21")
    parser.add_argument("--train-graphs", type=int, default=6)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASETS:
        records = load_records(Path(args.data_dir) / f"{dataset}.jsonl")[: args.train_graphs]
        tokenizer = GPTok2Tokenizer(
            {
                "patch": {
                    "sparse_edge_patches": True,
                    "sparse_density_threshold": 1.0,
                    "sparse_clustering_threshold": 1.0,
                    "max_patches_per_graph": 128,
                    "max_cycle_signatures": 16,
                    "max_triangle_signatures": 64,
                },
                "program": {"noncanonical_order": True, "max_global_links": 64},
                "canonicalization": {"max_search_nodes": 100, "exact": True},
                "compact_entropy": {"max_macros": 16, "min_macro_count": 1000, "max_bpe_merges": 16, "min_bpe_count": 1000},
                "motif_macro": {"max_structural_macros": 16, "min_structural_count": 1000, "max_merge_schemas": 16, "min_merge_schema_count": 1000, "max_code_schemas": 16, "min_code_schema_count": 1000},
            }
        ).fit(records)
        edge_total = sum(int(record.edge_index.shape[1]) for record in records)
        artifact_bytes = tokenizer.artifact_footprint_bytes()
        rows.append(
            {
                "dataset": dataset,
                "train_graphs": len(records),
                "train_edges": edge_total,
                "artifact_bytes": artifact_bytes,
                "amortized_artifact_bytes_per_graph": artifact_bytes / max(1, len(records)),
                "amortized_artifact_bytes_per_edge": artifact_bytes / max(1, edge_total),
                "codebook_size": len(tokenizer.codebook.codes) if tokenizer.codebook is not None else 0,
                "interface_vocab_size": len(tokenizer.codebook.interfaces) if tokenizer.codebook is not None else 0,
            }
        )
    keys = sorted({key for row in rows for key in row})
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
