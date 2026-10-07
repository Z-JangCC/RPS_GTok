"""Evaluate raw and canonicalized serialization controls across 21 datasets."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch

from gptok2.data.io import load_records
from gptok2.data.schema import GraphRecord
from gptok2.canonical import canonical_record
from gptok2.data.schema_spec import SchemaSpec
from gptok2_tokenizer import GPTok2Tokenizer
from rps_gtok_consumption.views import BPE_BASE_VIEWS, TokenBPE, TokenViewBuilder
from scripts.run_21_token_eval import DATASETS


VIEWS = [
    "edge_list",
    "canonical_edge_list",
    "adjacency_list",
    "canonical_adjacency_list",
    "dfs_order",
    "canonical_dfs_order",
    "bfs_order",
    "canonical_bfs_order",
    "edge_list_bpe",
    "canonical_edge_list_bpe",
    "canonical_adjacency_list_bpe",
    "canonical_dfs_order_bpe",
    "canonical_bfs_order_bpe",
    "rps_gtok_identifier",
]
BPE_VIEWS = [
    "edge_list_bpe",
    "canonical_edge_list_bpe",
    "canonical_adjacency_list_bpe",
    "canonical_dfs_order_bpe",
    "canonical_bfs_order_bpe",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="rebuttal/data_21")
    parser.add_argument("--out", default="rebuttal/experiments/E1_21_canonical_controls")
    parser.add_argument("--graphs", type=int, default=6)
    parser.add_argument("--permutations", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASETS:
        records = load_records(Path(args.data_dir) / f"{dataset}.jsonl")[: args.graphs]
        if not records:
            continue
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
        builder = TokenViewBuilder(tokenizer)
        schema = SchemaSpec.from_config(tokenizer.config)

        def base_views(record: GraphRecord) -> dict[str, list[str]]:
            raw = {
                view: builder.build(record, view)
                for view in ("edge_list", "adjacency_list", "dfs_order", "bfs_order")
            }
            canonical = canonical_record(
                record,
                schema,
                exact=True,
                max_search_nodes=int(tokenizer.config.get("canonicalization", {}).get("max_search_nodes", 100000)),
            )
            raw.update(
                {
                    f"canonical_{view}": builder.build(canonical, view)
                    for view in ("edge_list", "adjacency_list", "dfs_order", "bfs_order")
                }
            )
            return raw

        base_cache = {record.graph_id: base_views(record) for record in records}
        bpe_models = {
            view: TokenBPE().fit(
                [base_cache[record.graph_id][BPE_BASE_VIEWS[view]] for record in records],
                max_merges=builder.bpe_merges,
                min_freq=builder.bpe_min_freq,
            )
            for view in BPE_VIEWS
        }

        def build_view(record: GraphRecord, view: str, cached_bases: dict[str, list[str]]) -> list[str]:
            if view in bpe_models:
                return bpe_models[view].encode(cached_bases[BPE_BASE_VIEWS[view]])
            if view in cached_bases:
                return cached_bases[view]
            return builder.build(record, view)

        for record in records:
            original_bases = base_cache[record.graph_id]
            permuted_records = {
                seed: permute_record(record, seed) for seed in range(args.permutations)
            }
            permuted_bases = {
                seed: base_views(permuted_records[seed]) for seed in range(args.permutations)
            }
            for view in VIEWS:
                base = build_view(record, view, original_bases)
                matches = 0
                for seed in range(args.permutations):
                    candidate = build_view(permuted_records[seed], view, permuted_bases[seed])
                    matches += int(candidate == base)
                rows.append(
                    {
                        "dataset": dataset,
                        "graph_id": record.graph_id,
                        "view": view,
                        "tokens": len(base),
                        "permutation_stability": matches / max(1, args.permutations),
                    }
                )
    write_csv(out / "per_graph.csv", rows)
    summary = []
    for dataset in DATASETS:
        for view in VIEWS:
            group = [row for row in rows if row["dataset"] == dataset and row["view"] == view]
            if group:
                summary.append(
                    {
                        "dataset": dataset,
                        "view": view,
                        "graphs": len(group),
                        "mean_tokens": sum(row["tokens"] for row in group) / len(group),
                        "mean_permutation_stability": sum(row["permutation_stability"] for row in group) / len(group),
                    }
                )
    write_csv(out / "summary.csv", summary)
    print(f"wrote {len(rows)} rows")


def permute_record(record: GraphRecord, seed: int) -> GraphRecord:
    nodes = list(range(record.num_nodes))
    random.Random(seed).shuffle(nodes)
    mapping = {old: new for new, old in enumerate(nodes)}
    edges = [(mapping[int(u)], mapping[int(v)]) for u, v in record.edge_index.t().tolist()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty(2, 0, dtype=torch.long)

    def reorder(value):
        if value is None:
            return None
        return value[nodes].clone() if value.ndim > 0 and value.shape[0] == record.num_nodes else value.clone()

    return GraphRecord(
        graph_id=record.graph_id + f"_perm_{seed}",
        num_nodes=record.num_nodes,
        edge_index=edge_index,
        node_type=reorder(record.node_type),
        edge_type=record.edge_type.clone() if record.edge_type is not None else None,
        node_attr=reorder(record.node_attr),
        edge_attr=record.edge_attr.clone() if record.edge_attr is not None else None,
        y=record.y.clone() if record.y is not None else None,
        metadata=dict(record.metadata),
        directed=record.directed,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
