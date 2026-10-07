"""Run a lightweight E1 canonicalization-isolation audit."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import networkx as nx
import torch

from gptok2.data.schema import GraphRecord, graph_to_record
from gptok2.data.synthetic import generate_synthetic_graphs
from gptok2_tokenizer import GPTok2Tokenizer
from gptok2_tokenizer.payload import footprint_stats
from rps_gtok_consumption.views import TokenViewBuilder


METHODS = [
    "edge_list",
    "canonical_edge_list",
    "adjacency_list",
    "canonical_adjacency_list",
    "dfs_order",
    "canonical_dfs_order",
    "bfs_order",
    "canonical_bfs_order",
    "rps_gtok_identifier",
    "rps_gtok_full",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run canonicalization isolation audit.")
    parser.add_argument("--out", default="rebuttal/experiments/E1_canonicalized_baselines")
    parser.add_argument("--graphs", type=int, default=24)
    parser.add_argument("--permutations", type=int, default=6)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "run": {"seed": 2026},
        "data": {
            "synthetic": {
                "num_graphs": int(args.graphs),
                "num_nodes_min": 6,
                "num_nodes_max": 14,
                "families": ["cycle", "star", "tree", "motif_mix"],
            }
        },
    }
    records = generate_synthetic_graphs(config, seed=2026)
    tokenizer = GPTok2Tokenizer().fit(records)
    builder = TokenViewBuilder(tokenizer, seed=2026)
    rows = []
    for record in records:
        base_tokens = {method: builder.build(record, method) for method in METHODS}
        for method in METHODS:
            stable = 0
            lengths = []
            footprint = []
            for seed in range(int(args.permutations)):
                permuted = permute_record(record, seed)
                if method in {"rps_gtok_full", "rps_gtok_identifier"}:
                    encoded = tokenizer.encode(permuted, mode="motif_hybrid")
                    tokens = encoded.tokens if method == "rps_gtok_full" else encoded.identifier_tokens
                    stats = {
                        "identifier_bytes": encoded.identifier_bytes,
                        "payload_bytes": encoded.payload_bytes,
                        "complete_token_bytes": encoded.complete_token_bytes,
                    }
                else:
                    tokens = builder.build(permuted, method)
                    stats = footprint_stats(tokens)
                stable += int(tokens == base_tokens[method])
                lengths.append(len(tokens))
                footprint.append(stats)
            rows.append(
                {
                    "graph_id": record.graph_id,
                    "method": method,
                    "permutations": int(args.permutations),
                    "stability_rate": stable / max(1, int(args.permutations)),
                    "base_length": len(base_tokens[method]),
                    "mean_permuted_length": sum(lengths) / max(1, len(lengths)),
                    "mean_identifier_bytes": sum(x["identifier_bytes"] for x in footprint) / max(1, len(footprint)),
                    "mean_payload_bytes": sum(x["payload_bytes"] for x in footprint) / max(1, len(footprint)),
                    "mean_complete_token_bytes": sum(x["complete_token_bytes"] for x in footprint) / max(1, len(footprint)),
                }
            )
    write_csv(out / "per_graph.csv", rows)
    summary = summarize(rows)
    write_csv(out / "summary.csv", summary)
    print(f"wrote {len(rows)} rows to {out}")


def permute_record(record, seed: int):
    nodes = list(range(record.num_nodes))
    random.Random(seed).shuffle(nodes)
    mapping = {old: new for new, old in enumerate(nodes)}
    edges = [
        (mapping[int(u)], mapping[int(v)])
        for u, v in record.edge_index.t().tolist()
    ] if record.edge_index.numel() else []
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty(2, 0, dtype=torch.long)
    )
    def reorder(value):
        if value is None:
            return None
        if value.ndim > 0 and value.shape[0] == record.num_nodes:
            return value[nodes].clone()
        return value.clone()

    return GraphRecord(
        record.graph_id,
        record.num_nodes,
        edge_index,
        node_type=reorder(record.node_type),
        edge_type=record.edge_type.clone() if record.edge_type is not None else None,
        node_attr=reorder(record.node_attr),
        edge_attr=record.edge_attr.clone() if record.edge_attr is not None else None,
        y=record.y.clone() if record.y is not None else None,
        metadata=dict(record.metadata),
        directed=record.directed,
    )


def summarize(rows: list[dict]) -> list[dict]:
    methods = sorted({row["method"] for row in rows})
    out = []
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        out.append(
            {
                "method": method,
                "graphs": len(group),
                "mean_stability_rate": sum(x["stability_rate"] for x in group) / max(1, len(group)),
                "mean_base_length": sum(x["base_length"] for x in group) / max(1, len(group)),
                "mean_identifier_bytes": sum(x["mean_identifier_bytes"] for x in group) / max(1, len(group)),
                "mean_payload_bytes": sum(x["mean_payload_bytes"] for x in group) / max(1, len(group)),
                "mean_complete_token_bytes": sum(x["mean_complete_token_bytes"] for x in group) / max(1, len(group)),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
