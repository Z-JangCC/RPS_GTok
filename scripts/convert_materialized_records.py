"""Convert graph_tokenizer_v2 materialized records to GraphRecord JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from gptok2.data.schema import GraphRecord, record_to_dict
from gptok2.data.io import save_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert materialized schema-v2 records.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-graphs", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=256)
    args = parser.parse_args()

    source = Path(args.input)
    rows = [json.loads(line) for line in source.open("r", encoding="utf-8") if line.strip()]
    if args.max_graphs > 0:
        rows = rows[: args.max_graphs]
    node_types = sorted({str(node.get("node_type", "")) for row in rows for node in row.get("nodes", [])})
    edge_types = sorted({str(edge.get("edge_type", "")) for row in rows for edge in row.get("edges", [])})
    node_type_map = {value: idx for idx, value in enumerate(node_types)}
    edge_type_map = {value: idx for idx, value in enumerate(edge_types)}
    node_attr_keys = sorted(
        {
            str(key)
            for row in rows
            for node in row.get("nodes", [])
            for key, value in dict(node.get("attrs", {})).items()
            if _numeric(value)
        }
    )
    edge_attr_keys = sorted(
        {
            str(key)
            for row in rows
            for edge in row.get("edges", [])
            for key, value in dict(edge.get("attrs", {})).items()
            if _numeric(value)
        }
    )

    records = []
    skipped = 0
    for row in rows:
        nodes = row.get("nodes", [])
        if len(nodes) == 0 or len(nodes) > int(args.max_nodes):
            skipped += 1
            continue
        node_map = {str(node.get("node_id")): idx for idx, node in enumerate(nodes)}
        edges = []
        edge_type_values = []
        edge_attr_values = []
        directed = bool(any(bool(edge.get("directed", False)) for edge in row.get("edges", [])))
        for edge in row.get("edges", []):
            source_id = str(edge.get("source"))
            target_id = str(edge.get("target"))
            if source_id not in node_map or target_id not in node_map:
                continue
            edges.append((node_map[source_id], node_map[target_id]))
            edge_type_values.append(edge_type_map.get(str(edge.get("edge_type", "")), 0))
            edge_attr_values.append([float(edge.get("attrs", {}).get(key, 0.0)) for key in edge_attr_keys])

        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.empty(2, 0, dtype=torch.long)
        )
        node_type = torch.tensor(
            [node_type_map.get(str(node.get("node_type", "")), 0) for node in nodes],
            dtype=torch.long,
        )
        node_attr = (
            torch.tensor(
                [[float(node.get("attrs", {}).get(key, 0.0)) for key in node_attr_keys] for node in nodes],
                dtype=torch.float32,
            )
            if node_attr_keys
            else None
        )
        edge_type = torch.tensor(edge_type_values, dtype=torch.long) if edge_type_values else None
        edge_attr = torch.tensor(edge_attr_values, dtype=torch.float32) if edge_attr_values else None
        records.append(
            GraphRecord(
                graph_id=str(row.get("graph_id", f"materialized_{len(records):06d}")),
                num_nodes=len(nodes),
                edge_index=edge_index,
                node_type=node_type,
                edge_type=edge_type,
                node_attr=node_attr,
                edge_attr=edge_attr,
                metadata={
                    "domain": row.get("domain"),
                    "graph_type": row.get("graph_type"),
                    "source_schema": row.get("schema"),
                    "source_attrs": row.get("attrs", {}),
                    "node_type_vocab": node_types,
                    "edge_type_vocab": edge_types,
                    "node_attr_keys": node_attr_keys,
                    "edge_attr_keys": edge_attr_keys,
                    "filtered_over_max_nodes": skipped,
                },
                directed=directed,
            )
        )
    save_records(args.output, records)
    print(json.dumps({"input": str(source), "output": args.output, "records": len(records), "skipped": skipped}, indent=2))


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


if __name__ == "__main__":
    main()
