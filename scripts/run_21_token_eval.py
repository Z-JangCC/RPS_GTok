"""Run the common token-level protocol on materialized 21-dataset samples."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch

from gptok2.data.io import load_records
from gptok2.data.schema import GraphRecord
from gptok2.metrics.evaluate import structure_fidelity
from gptok2_tokenizer import GPTok2Tokenizer


DATASETS = [
    "ast_cfg_cpg",
    "cifar10_sp",
    "citeseer",
    "collab",
    "cora",
    "enzymes",
    "fb15k_237",
    "ieee_power_grid",
    "imdb_binary",
    "mnist_sp",
    "mutag",
    "ogbg_code2",
    "ogbg_molhiv",
    "ogbg_molpcba",
    "ogbg_ppa",
    "ogbn_arxiv",
    "proteins",
    "pubmed",
    "road_networks",
    "synthetic_stress",
    "wn18rr",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 21-dataset token protocol.")
    parser.add_argument("--data-dir", default="rebuttal/data_21")
    parser.add_argument("--out", default="rebuttal/experiments/token_level_21")
    parser.add_argument("--max-graphs", type=int, default=60)
    parser.add_argument("--max-test", type=int, default=12)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_graph: list[dict] = []
    statuses: list[dict] = []
    summaries: list[dict] = []
    for name in DATASETS:
        path = Path(args.data_dir) / f"{name}.jsonl"
        if not path.exists():
            statuses.append({"dataset": name, "status": "missing", "reason": str(path)})
            continue
        start = time.perf_counter()
        try:
            records = load_records(path)[: int(args.max_graphs)]
            if len(records) < 3:
                statuses.append({"dataset": name, "status": "too_small", "records": len(records)})
                continue
            n_train = max(1, int(round(0.7 * len(records))))
            n_val = max(1, int(round(0.15 * len(records))))
            train = records[:n_train]
            test = records[n_train + n_val : n_train + n_val + int(args.max_test)]
            if not test:
                statuses.append(
                    {
                        "dataset": name,
                        "status": "insufficient_test",
                        "records": len(records),
                        "train": len(train),
                        "test": 0,
                        "reason": "split leaves no held-out graph; excluded from metric aggregates",
                    }
                )
                continue
            tokenizer = GPTok2Tokenizer(
                {
                    "patch": {
                        "sparse_edge_patches": True,
                        "sparse_density_threshold": 1.0,
                        "sparse_clustering_threshold": 1.0,
                        "max_patches_per_graph": 12000,
                        "max_cycle_signatures": 32,
                        "max_triangle_signatures": 128,
                    },
                    "canonicalization": {"max_search_nodes": 100, "exact": True},
                    "compact_entropy": {
                        "max_macros": 32,
                        "min_macro_count": 1000,
                        "max_macro_len": 8,
                        "max_bpe_merges": 32,
                        "min_bpe_count": 1000,
                    },
                    "motif_macro": {
                        "max_structural_macros": 32,
                        "min_structural_count": 1000,
                        "max_structural_len": 8,
                        "max_parameterized_span_len": 32,
                        "min_parameterized_emit_count": 1000,
                        "max_merge_schemas": 32,
                        "min_merge_schema_count": 1000,
                        "max_code_schemas": 32,
                        "min_code_schema_count": 1000,
                    },
                }
            ).fit(train)
            fit_sec = time.perf_counter() - start
            artifact_bytes = tokenizer.artifact_footprint_bytes()
            rows = []
            for record in test:
                encoded = tokenizer.encode(record, mode="motif_hybrid")
                permutation_trials = 3
                permutation_matches = 0
                exact_flags = []
                for permutation_seed in range(permutation_trials):
                    permuted = permute_record(record, permutation_seed)
                    permuted_encoded = tokenizer.encode(permuted, mode="motif_hybrid")
                    permutation_matches += int(permuted_encoded.tokens == encoded.tokens)
                    audit = permuted_encoded.payload.get("metadata", {}).get("canonicalization", {})
                    exact_flags.append(bool(audit.get("exact_completed", False)))
                row = {
                    "dataset": name,
                    "graph_id": record.graph_id,
                    "num_nodes": record.num_nodes,
                    "num_edges": int(record.edge_index.shape[1]) if record.edge_index.ndim == 2 else 0,
                    "tokens": encoded.token_count,
                    "original_tokens": encoded.original_token_count,
                    "id_tokens_per_edge": encoded.token_count / max(1, int(record.edge_index.shape[1])),
                    "identifier_bytes": encoded.identifier_bytes,
                    "payload_bytes": encoded.payload_bytes,
                    "complete_token_bytes": encoded.complete_token_bytes,
                    "artifact_bytes": artifact_bytes,
                    "bits_per_edge": encoded.bits_per_edge,
                    "lossless_expand_match": encoded.lossless_expand_match,
                    "permutation_trials": permutation_trials,
                    "permutation_stability": permutation_matches / max(1, permutation_trials),
                    "canonical_exact_completion": sum(exact_flags) / max(1, len(exact_flags)),
                    "codebook_size": len(tokenizer.codebook) if tokenizer.codebook is not None else 0,
                    "node_coverage": encoded.program_metadata.get("node_coverage", 0.0),
                    "edge_coverage": encoded.program_metadata.get("edge_coverage", 0.0),
                    "fit_sec": fit_sec,
                }
                try:
                    reconstructed = tokenizer.decode(encoded)
                    fidelity = structure_fidelity(record, reconstructed)
                    row.update(
                        {
                            "strict_topology_exact": fidelity["strict_topology_exact"],
                            "strict_audit_skipped": fidelity["strict_topology_audit_skipped"],
                            "edge_f1": fidelity["edge_f1"],
                            "wl_hash_match": fidelity["wl_hash_match"],
                        }
                    )
                except Exception as exc:
                    row.update({"strict_topology_exact": 0.0, "strict_audit_skipped": 1.0, "error": repr(exc)})
                rows.append(row)
            per_graph.extend(rows)
            summaries.append(summarize(name, rows, fit_sec))
            statuses.append(
                {
                    "dataset": name,
                    "status": "ok",
                    "records": len(records),
                    "train": len(train),
                    "test": len(test),
                    "elapsed_sec": time.perf_counter() - start,
                }
            )
        except Exception as exc:
            statuses.append({"dataset": name, "status": "failed", "reason": repr(exc)})
    write_csv(out / "per_graph.csv", per_graph)
    write_csv(out / "summary.csv", summaries)
    write_csv(out / "dataset_status.csv", statuses)
    print(json.dumps({
        "datasets": len(DATASETS),
        "ok": sum(x.get("status") == "ok" for x in statuses),
        "excluded": sum(x.get("status") != "ok" for x in statuses),
        "out": str(out),
    }, indent=2))


def summarize(dataset: str, rows: list[dict], fit_sec: float) -> dict:
    numeric = [
        "num_nodes",
        "num_edges",
        "tokens",
        "original_tokens",
        "id_tokens_per_edge",
        "identifier_bytes",
        "payload_bytes",
        "complete_token_bytes",
        "artifact_bytes",
        "bits_per_edge",
        "lossless_expand_match",
        "strict_topology_exact",
        "strict_audit_skipped",
        "edge_f1",
        "wl_hash_match",
        "permutation_stability",
        "canonical_exact_completion",
    ]
    out = {"dataset": dataset, "graphs": len(rows), "fit_sec": fit_sec}
    for key in numeric:
        values = [float(row[key]) for row in rows if key in row]
        out[f"mean_{key}"] = sum(values) / max(1, len(values))
    return out


def permute_record(record: GraphRecord, seed: int) -> GraphRecord:
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
        return value[nodes].clone() if value is not None and value.ndim > 0 and value.shape[0] == record.num_nodes else value.clone() if value is not None else None
    return GraphRecord(
        graph_id=record.graph_id,
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
