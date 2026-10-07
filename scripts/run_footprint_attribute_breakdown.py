"""Audit topology-only codebook bytes and attribute/topology byte components.

The compact protocol uses unsigned-varint IDs and length-prefixed UTF-8 payload
fields. It reports model-facing codebook context separately from the complete
payload-bearing stream so that the two claims are not conflated.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from gptok2.data.synthetic import generate_synthetic_graphs
from gptok2_tokenizer import GPTok2Tokenizer
from gptok2_tokenizer.payload import token_instances


def varint_bytes(value: int) -> int:
    value = int(value)
    size = 1
    while value >= 128:
        value //= 128
        size += 1
    return size


def json_bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def config(retain_attributes: bool) -> dict:
    return {
        "schema": {
            "name": "topology_with_attributes" if retain_attributes else "topology_only",
            "retain_node_type": True,
            "retain_edge_type": retain_attributes,
            "retain_node_attr": retain_attributes,
            "retain_edge_attr": retain_attributes,
            "continuous_mode": "raw",
            "max_search_nodes": 100,
        },
        "patch": {"sparse_edge_patches": True, "sparse_density_threshold": 1.0, "sparse_clustering_threshold": 1.0, "max_patches_per_graph": 128, "max_cycle_signatures": 16, "max_triangle_signatures": 64},
        "program": {"noncanonical_order": True, "max_global_links": 64},
        "canonicalization": {"max_search_nodes": 100, "exact": True},
        "compact_entropy": {"max_macros": 16, "min_macro_count": 1000, "max_bpe_merges": 16, "min_bpe_count": 1000},
        "motif_macro": {"max_structural_macros": 16, "min_structural_count": 1000, "max_merge_schemas": 16, "min_merge_schema_count": 1000, "max_code_schemas": 16, "min_code_schema_count": 1000},
    }


def graph_payload_parts(payload: dict) -> tuple[int, int]:
    topology = {k: payload.get(k) for k in ("num_nodes", "node_type", "edge_type", "directed") if k in payload}
    attributes = {k: payload.get(k) for k in ("node_attr", "edge_attr") if k in payload}
    return json_bytes(topology), json_bytes(attributes) if attributes else 0


def packed_stats(tokenizer: GPTok2Tokenizer, records: list) -> dict[str, float]:
    encoded = [tokenizer.encode(record, mode="motif_hybrid") for record in records]
    identifiers = sorted({item.identifier for value in encoded for item in token_instances(value.tokens)})
    codebook = sum(varint_bytes(index) + len(identifier.encode("utf-8")) for index, identifier in enumerate(identifiers))
    index_map = {identifier: index for index, identifier in enumerate(identifiers)}
    id_stream = []
    token_payload = []
    topology_payload = []
    attribute_payload = []
    for value in encoded:
        instances = token_instances(value.tokens)
        id_stream.append(sum(varint_bytes(index_map[item.identifier]) for item in instances))
        token_payload.append(sum(varint_bytes(len(field.encode("utf-8"))) + len(field.encode("utf-8")) for item in instances for field in item.payload))
        topo, attr = graph_payload_parts(value.payload)
        topology_payload.append(topo)
        attribute_payload.append(attr)
    return {
        "codebook_bytes": float(codebook),
        "identifier_codebook_bytes": mean(id_stream) + codebook / max(1, len(records)),
        "token_payload_bytes": mean(token_payload),
        "topology_payload_bytes": mean(topology_payload),
        "attribute_bytes": mean(attribute_payload),
        "topology_complete_bytes": mean(id_stream) + codebook / max(1, len(records)) + mean(token_payload) + mean(topology_payload),
        "full_complete_bytes": mean(id_stream) + codebook / max(1, len(records)) + mean(token_payload) + mean(topology_payload) + mean(attribute_payload),
    }


def edge_bytes(records: list) -> float:
    values = []
    for record in records:
        total = 4
        for u, v in record.edge_index.t().tolist():
            total += varint_bytes(int(u)) + varint_bytes(int(v))
        values.append(total)
    return mean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="rebuttal/tables/Table_R14_footprint_attribute_breakdown.csv")
    args = parser.parse_args()
    records = generate_synthetic_graphs({"data": {"synthetic": {"num_graphs": 24, "num_nodes_min": 6, "num_nodes_max": 14, "families": ["cycle", "star", "tree", "motif_mix"]}}}, seed=2026)
    no_attr = packed_stats(GPTok2Tokenizer(config(False)).fit(records), records)
    with_attr = packed_stats(GPTok2Tokenizer(config(True)).fit(records), records)
    rows = [
        {"setting": "Raw edge / canonical edge topology bytes", "bytes_per_graph": f"{edge_bytes(records):.3f}", "note": "same packed varint protocol"},
        {"setting": "RPS-GTok topology-only identifier + amortized codebook", "bytes_per_graph": f"{no_attr['identifier_codebook_bytes']:.3f}", "note": "model-facing compact context"},
        {"setting": "RPS-GTok topology-only complete stream", "bytes_per_graph": f"{no_attr['topology_complete_bytes']:.3f}", "note": "includes executable local arguments"},
        {"setting": "RPS-GTok with attributes: topology component", "bytes_per_graph": f"{with_attr['topology_complete_bytes']:.3f}", "note": "identifier/codebook + token payload + topology schema"},
        {"setting": "RPS-GTok with attributes: attribute component", "bytes_per_graph": f"{with_attr['attribute_bytes']:.3f}", "note": "raw node/edge attribute payload"},
        {"setting": "RPS-GTok with attributes: full complete stream", "bytes_per_graph": f"{with_attr['full_complete_bytes']:.3f}", "note": "topology + attributes"},
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["setting", "bytes_per_graph", "note"])
        writer.writeheader(); writer.writerows(rows)
    (out.with_suffix(".protocol.json")).write_text(json.dumps({"graphs": 24, "seed": 2026, "encoding": "unsigned-varint IDs and length-prefixed UTF-8 fields", "attributes": "raw", "codebook_amortized": True}, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
