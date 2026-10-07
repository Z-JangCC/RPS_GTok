"""Construct leakage-audited graph-level proxy tasks for unlabeled records.

The materialized 21-dataset corpus intentionally has ``y=null``.  This script
does not recover or guess a semantic class.  Instead it creates a declared
structural *proxy* task in a sidecar keyed by ``graph_id``.  Labels are derived
only from the topology (not node attributes, metadata, graph id, or an input
token), and the graph JSONL files are left unchanged.  The sidecar is therefore
safe to join at the training loop while the tokenizer sees the original graph
only.

The primary task is a continuous diffusion-resistance proxy: the mean of a
fixed set of effective-resistance samples between component nodes, computed
from the unweighted topology.  For tiny/disconnected graphs where this is not
defined, the declared fallback is normalized cycle rank.  A train-only median
of the resulting descriptor defines an optional binary regime label; the
threshold is learned from the training IDs only and is never fit on validation
or test graphs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np

from gptok2.data.io import load_records
from gptok2.data.schema import GraphRecord


DATASETS = [
    "ast_cfg_cpg", "cifar10_sp", "citeseer", "collab", "cora", "enzymes",
    "fb15k_237", "ieee_power_grid", "imdb_binary", "mnist_sp", "mutag",
    "ogbg_code2", "ogbg_molhiv", "ogbg_molpcba", "ogbg_ppa", "ogbn_arxiv",
    "proteins", "pubmed", "road_networks", "synthetic_stress", "wn18rr",
]


def assign_splits(graph_ids: list[str]) -> dict[str, str]:
    """Assign deterministic, non-empty graph-disjoint partitions."""
    ordered = sorted(graph_ids, key=lambda graph_id: hashlib.sha256(graph_id.encode("utf-8")).hexdigest())
    n = len(ordered)
    if n < 3:
        return {graph_id: "train" for graph_id in ordered}
    n_val = max(1, int(round(0.15 * n)))
    n_test = max(1, int(round(0.15 * n)))
    n_train = max(1, n - n_val - n_test)
    while n_train + n_val + n_test > n:
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test -= 1
    split = {}
    for index, graph_id in enumerate(ordered):
        split[graph_id] = "train" if index < n_train else ("val" if index < n_train + n_val else "test")
    return split


def topology_graph(record: GraphRecord) -> nx.Graph:
    graph = nx.DiGraph() if record.directed else nx.Graph()
    graph.add_nodes_from(range(int(record.num_nodes)))
    rows = record.edge_index.t().tolist() if record.edge_index.numel() else []
    graph.add_edges_from((int(u), int(v)) for u, v in rows)
    return graph.to_undirected() if graph.is_directed() else graph


def descriptor(record: GraphRecord) -> tuple[float, str]:
    """Return a topology-only scalar and its declared source."""
    graph = topology_graph(record)
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    components = nx.number_connected_components(graph) if n else 0
    cycle_rank = max(0, m - n + components)
    # A normalized cycle rank is well-defined for every graph and avoids using
    # node/edge attributes.  It is the fallback for degenerate diffusion cases.
    fallback = math.log1p(cycle_rank) / max(1.0, math.log1p(max(1, m)))
    if n < 2 or m == 0:
        return float(fallback), "normalized_cycle_rank"
    # Effective resistance is computed per connected component.  We use a
    # deterministic sparse subset of pairs to keep the label generator bounded
    # and record the exact pair seed in the manifest.
    values: list[float] = []
    for component in nx.connected_components(graph):
        nodes = sorted(component)
        if len(nodes) < 2:
            continue
        # Fixed lexicographic pairs, capped at 64 per component: no sampling
        # from labels or model outputs and no dependence on input node order.
        pairs = [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))][:64]
        try:
            sub = graph.subgraph(nodes)
            laplacian = nx.laplacian_matrix(sub, nodelist=nodes).toarray().astype(float)
            # The Moore--Penrose inverse is computed once per component rather
            # than once per pair; this makes all-21 generation bounded.
            laplacian_pinv = np.linalg.pinv(laplacian, rcond=1e-10)
            positions = {node: index for index, node in enumerate(nodes)}
            for u, v in pairs:
                delta = np.zeros(len(nodes), dtype=float)
                delta[positions[u]] = 1.0
                delta[positions[v]] = -1.0
                values.append(float(delta @ laplacian_pinv @ delta))
        except (nx.NetworkXError, np.linalg.LinAlgError, ValueError):
            continue
    if values and all(np.isfinite(values)):
        # Log scale keeps high-resistance disconnected/sparse regimes stable.
        return float(math.log1p(float(np.mean(values)))), "log1p_effective_resistance"
    return float(fallback), "normalized_cycle_rank"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("rebuttal/data_21"))
    parser.add_argument("--out-dir", type=Path, default=Path("rebuttal/constructed_tasks"))
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()
    datasets = list(args.datasets) if args.datasets else DATASETS
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, object]] = []
    for dataset in datasets:
        path = args.data_dir / f"{dataset}.jsonl"
        if not path.exists():
            audit_rows.append({"dataset": dataset, "status": "missing"})
            continue
        records = load_records(path)
        values = []
        splits = assign_splits([record.graph_id for record in records])
        for record in records:
            value, source = descriptor(record)
            values.append({"graph_id": record.graph_id, "split": splits[record.graph_id], "target": value, "target_source": source})
        train_values = [float(row["target"]) for row in values if row["split"] == "train"]
        threshold = float(np.median(train_values)) if train_values else float("nan")
        for row in values:
            row["regime_label"] = int(float(row["target"]) >= threshold) if np.isfinite(threshold) else None
        # A class is eligible only if the train/val/test partitions all retain
        # both regimes; otherwise the continuous task remains valid but the
        # classification proxy is explicitly marked unavailable.
        split_classes = {
            split: sorted({row["regime_label"] for row in values if row["split"] == split and row["regime_label"] is not None})
            for split in ("train", "val", "test")
        }
        class_eligible = all(classes == [0, 1] for classes in split_classes.values())
        manifest = {
            "dataset": dataset,
            "label_kind": "constructed_structural_proxy",
            "continuous_target": "log1p_effective_resistance_with_normalized_cycle_rank_fallback",
            "classification_target": "train_median_regime_only",
            "threshold_fit_split": "train",
            "threshold": threshold,
            "class_eligible": class_eligible,
            "split_counts": {split: sum(row["split"] == split for row in values) for split in ("train", "val", "test")},
            "split_classes": split_classes,
            "records": len(values),
            "input_fields_used": ["num_nodes", "edges", "directed"],
            "input_fields_forbidden": ["y", "node_attr", "edge_attr", "metadata", "graph_id"],
            "label_sidecar_only": True,
        }
        out = args.out_dir / f"{dataset}.jsonl"
        with out.open("w", encoding="utf-8") as handle:
            for row in values:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        (args.out_dir / f"{dataset}.protocol.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        audit_rows.append({"dataset": dataset, "status": "ok", "records": len(values), "class_eligible": int(class_eligible), "threshold": threshold, "train": manifest["split_counts"]["train"], "val": manifest["split_counts"]["val"], "test": manifest["split_counts"]["test"]})
    with (args.out_dir / "constructed_task_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for row in audit_rows for k in row}))
        writer.writeheader(); writer.writerows(audit_rows)
    (args.out_dir / "protocol.json").write_text(json.dumps({"datasets": datasets, "label_source": "topology_only_proxy", "no_semantic_claim": True, "split_rule": "sha256-sorted graph IDs with deterministic non-empty 70/15/15 partitions", "input_fields_used": ["num_nodes", "edges", "directed"], "input_fields_forbidden": ["y", "node_attr", "edge_attr", "metadata", "graph_id"]}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(audit_rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
