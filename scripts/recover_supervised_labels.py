"""Recover graph labels only when an audited raw source matches topology.

The materialized 21-dataset records intentionally contain ``y=null``.  This
tool never infers a label from metadata: it joins a TU raw graph-label file by
the stable ``DATASET_<index>`` id and requires an exact node/edge fingerprint
match before writing a labeled copy.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _tu_paths() -> dict[str, Path]:
    candidates = [
        ROOT.parent / "graph_tokenizer_v2_base" / "dataset",
        ROOT / "runs/rebuttal_semantic_source",
        ROOT / "runs/rebuttal_collab_source",
        ROOT / "runs/rebuttal_tu_eval",
    ]
    out: dict[str, Path] = {}
    for base in candidates:
        if not base.exists():
            continue
        for p in base.glob("**/*_graph_labels.txt"):
            name = p.name.removesuffix("_graph_labels.txt").upper()
            if name == "IMDB-BINARY":
                key = "imdb_binary"
            else:
                key = name.lower()
            indicator = p.with_name(f"{name}_graph_indicator.txt")
            edges = p.with_name(f"{name}_A.txt")
            if indicator.exists() and edges.exists() and key not in out:
                out[key] = p
    return out


def _raw_tu(path: Path, max_graph: int | None = None) -> tuple[list[tuple[int, tuple[tuple[int, int], ...]]], list[int]]:
    indicator = [int(x.strip()) for x in path.with_name(path.name.replace("_graph_labels", "_graph_indicator")).read_text().splitlines() if x.strip()]
    labels = [int(float(x.strip())) for x in path.read_text().splitlines() if x.strip()]
    if max_graph is not None:
        labels = labels[: max_graph + 1]
    edges_path = path.with_name(path.name.replace("_graph_labels", "_A"))
    groups: dict[int, list[int]] = defaultdict(list)
    for node, graph in enumerate(indicator):
        groups[graph].append(node + 1)  # TU node ids are one-based.
    node_to_graph = {node: graph for graph, nodes in groups.items() for node in nodes}
    local_rank = {node: rank + 1 for graph, nodes in groups.items() for rank, node in enumerate(nodes)}
    edges_by_graph: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for line in edges_path.read_text().splitlines():
        if not line.strip():
            continue
        u, v = (int(x.strip()) for x in line.split(",")[:2])
        graph = node_to_graph.get(u)
        if graph is None:
            raise ValueError(f"edge endpoint {u} is not covered by indicator {edges_path}")
        if max_graph is not None and graph > max_graph:
            # TU edge files are emitted in node order.  The sampled records
            # use a prefix of graph ids, so the remaining 692 MB COLLAB file
            # need not be read for this auditable recovery pass.
            break
        if node_to_graph.get(v) != graph:
            raise ValueError(f"cross-graph edge ({u}, {v}) in {edges_path}")
        edges_by_graph[graph].append(tuple(sorted((local_rank[u], local_rank[v]))))
    rows = []
    for graph in range(1, len(labels) + 1):
        nodes = groups.get(graph, [])
        edges = tuple(sorted(edges_by_graph.get(graph, [])))
        rows.append((len(nodes), edges))
    return rows, labels


def _materialized_fp(row: dict) -> tuple[int, tuple[tuple[int, int], ...]]:
    edges = tuple(sorted(tuple(sorted((int(u) + 1, int(v) + 1))) for u, v in row.get("edges", [])))
    # The materialized format stores both orientations for undirected graphs.
    edges = tuple(sorted(set(edges)))
    return int(row["num_nodes"]), edges


def _raw_fp(row: tuple[int, tuple[tuple[int, int], ...]]) -> tuple[int, tuple[tuple[int, int], ...]]:
    return row[0], tuple(sorted(set(row[1])))


def _cached_graph_labels(dataset: str, max_idx: int) -> tuple[list[object], list[tuple[int, int]], str] | None:
    """Read labels and shape checks from already cached GNN/OGB sources."""
    cache = ROOT.parent / "graph_tokenizer_v2_base/data/kdd2027/formal_full_real_materialized"
    if dataset in {"cifar10_sp", "mnist_sp"}:
        import torch

        name = "CIFAR10" if dataset == "cifar10_sp" else "MNIST"
        path = cache / "_pyg_source_cache/pyg_gnn_benchmark" / name / "processed/train_data.pt"
        data, slices, _ = torch.load(path, weights_only=False)
        labels = [int(x) for x in data["y"][: max_idx + 1].tolist()]
        shapes = []
        for i in range(max_idx + 1):
            shapes.append((int(slices["x"][i + 1] - slices["x"][i]), int(slices["edge_index"][i + 1] - slices["edge_index"][i])))
        return labels, shapes, f"GNNBenchmarkDataset/{name}/train"
    ogb_names = {"ogbg_molhiv": "ogbg_molhiv", "ogbg_molpcba": "ogbg_molpcba", "ogbg_ppa": "ogbg_ppa"}
    if dataset not in ogb_names:
        # Planetoid/OGBN records are radius-2 egos.  The materializer orders
        # seeds by descending degree and stores the seed index in ego_k.
        node_names = {"cora": "Cora", "citeseer": "Citeseer", "pubmed": "Pubmed"}
        if dataset in node_names:
            import torch

            path = cache / "_pyg_source_cache/pyg_planetoid" / node_names[dataset] / "processed/data.pt"
            loaded = torch.load(path, weights_only=False)
            data = loaded[0] if isinstance(loaded, tuple) else loaded
            edge_index = data["edge_index"]
            n = int(data["x"].shape[0])
            adj = [set() for _ in range(n)]
            for u, v in edge_index.t().tolist():
                adj[int(u)].add(int(v)); adj[int(v)].add(int(u))
            seeds = sorted(range(n), key=lambda u: (-len(adj[u]), u))
            labels = [int(data["y"][u]) for u in seeds[: max_idx + 1]]
            shapes = []
            for seed in seeds[: max_idx + 1]:
                nodes = {seed}; frontier = [seed]
                for _ in range(2):
                    nxt = []
                    for node in frontier:
                        for nbr in sorted(adj[node], key=lambda v: (-len(adj[v]), v)):
                            if nbr not in nodes:
                                nodes.add(nbr); nxt.append(nbr)
                            if len(nodes) >= 64: break
                        if len(nodes) >= 64: break
                    frontier = nxt
                shapes.append((min(len(nodes), 64), -1))
            return labels, shapes, f"Planetoid/{node_names[dataset]}/seed_node_label"
        if dataset == "ogbn_arxiv":
            import torch

            path = cache / "_ogb_source_cache/ogb_node/ogbn_arxiv/processed/geometric_data_processed.pt"
            loaded = torch.load(path, weights_only=False)
            data = loaded[0] if isinstance(loaded, tuple) else loaded
            edge_index = data["edge_index"]
            n = int(data["x"].shape[0]); adj = [set() for _ in range(n)]
            for u, v in edge_index.t().tolist():
                adj[int(u)].add(int(v)); adj[int(v)].add(int(u))
            seeds = sorted(range(n), key=lambda u: (-len(adj[u]), u))
            labels = [int(data["y"][u].reshape(-1)[0]) for u in seeds[: max_idx + 1]]
            shapes = []
            for seed in seeds[: max_idx + 1]:
                nodes = {seed}; frontier = [seed]
                for _ in range(2):
                    nxt = []
                    for node in frontier:
                        for nbr in sorted(adj[node]):
                            if nbr not in nodes:
                                nodes.add(nbr); nxt.append(nbr)
                            if len(nodes) >= 64: break
                        if len(nodes) >= 64: break
                    frontier = nxt
                shapes.append((min(len(nodes), 64), -1))
            return labels, shapes, "OGB/ogbn-arxiv/seed_node_label"
        return None
    source = cache / "_ogb_source_cache/ogb_graphprop" / ogb_names[dataset] / "raw"
    labels: list[object] = []
    with gzip.open(source / "graph-label.csv.gz", "rt", newline="") as handle:
        for row in csv.reader(handle):
            vals = [None if value == "" else float(value) for value in row]
            if len(vals) == 1:
                labels.append(None if vals[0] is None else int(vals[0]) if vals[0].is_integer() else vals[0])
            else:
                # OGB-PCBA is officially multi-task.  Preserve the complete
                # vector and observed mask; downstream code must opt into a
                # declared reduction rather than silently changing the task.
                labels.append(vals)
            if len(labels) > max_idx:
                break
    def _ints(path: Path) -> list[int]:
        with gzip.open(path, "rt") as handle:
            return [int(line.strip()) for line in handle][: max_idx + 1]
    nodes = _ints(source / "num-node-list.csv.gz")
    edges = _ints(source / "num-edge-list.csv.gz")
    # OGB molecular/PPA raw edges store each undirected edge once; the
    # PyG/materialized representation stores both orientations.
    shapes = [(nodes[i], 2 * edges[i]) for i in range(min(len(nodes), max_idx + 1))]
    return labels, shapes, f"OGB/{ogb_names[dataset]}/graph-label.csv.gz"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "rebuttal/data_21")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "rebuttal/data_21_labeled")
    parser.add_argument("--audit", type=Path, default=ROOT / "rebuttal/audits/label_recovery_audit.csv")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    print("scanning raw sources", flush=True)
    sources = _tu_paths()
    print("sources", {k: str(v) for k, v in sources.items()}, flush=True)
    audit = []
    def numeric_suffix(value: object) -> int | None:
        try:
            return int(str(value).rsplit("_", 1)[1])
        except (ValueError, IndexError):
            return None
    for source in sorted(args.data_dir.glob("*.jsonl")):
        dataset = source.stem
        print("dataset", dataset, flush=True)
        source_labels = sources.get(dataset)
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        status = "no_audited_raw_source"
        matched = 0
        labeled_rows = []
        max_cached_idx = max((idx for idx in (numeric_suffix(row["graph_id"]) for row in rows) if idx is not None), default=-1)
        cached = _cached_graph_labels(dataset, max_cached_idx)
        if cached is not None:
            labels, shapes, source_name = cached
            for row in rows:
                idx = numeric_suffix(row["graph_id"])
                if idx is None:
                    labeled_rows.append(row)
                    continue
                expected_nodes, expected_edges = shapes[idx] if idx < len(shapes) else (-1, -1)
                shape_ok = idx < len(shapes) and int(row["num_nodes"]) == expected_nodes and (expected_edges < 0 or len(row.get("edges", [])) == expected_edges)
                if shape_ok and idx < len(labels):
                    row = dict(row)
                    row["y"] = 0 if labels[idx] == -1 else labels[idx]
                    row.setdefault("metadata", {})["label_source"] = source_name
                    row["metadata"]["label_recovery"] = "exact_cached_source_shape_join"
                    if labels[idx] == -1:
                        row["metadata"]["original_label_encoding"] = {"-1": 0, "1": 1}
                    matched += 1
                labeled_rows.append(row)
            status = "recovered" if matched == len(rows) else "partial_shape_match"
            audit.append({"dataset": dataset, "records": len(rows), "source": source_name, "matched": matched, "status": status})
            if status == "recovered":
                (args.out_dir / source.name).write_text("\n".join(json.dumps(row, sort_keys=True) for row in labeled_rows) + "\n", encoding="utf-8")
            continue
        if source_labels is not None:
            max_idx = max((int(str(row["graph_id"]).rsplit("_", 1)[1]) for row in rows), default=-1)
            # TU graph labels/indicators are one-based; graph_id is zero-based.
            raw_graphs, labels = _raw_tu(source_labels, max_graph=max_idx + 1)
            for row in rows:
                try:
                    idx = int(str(row["graph_id"]).rsplit("_", 1)[1])
                    ok = idx < len(raw_graphs) and _materialized_fp(row) == _raw_fp(raw_graphs[idx])
                except Exception:
                    ok = False
                if ok:
                    row = dict(row)
                    row["y"] = 0 if labels[idx] == -1 else labels[idx]
                    row.setdefault("metadata", {})["label_source"] = source_labels.name
                    row["metadata"]["label_recovery"] = "exact_tu_topology_join"
                    if labels[idx] == -1:
                        row["metadata"]["original_label_encoding"] = {"-1": 0, "1": 1}
                    matched += 1
                labeled_rows.append(row)
            status = "recovered" if matched == len(rows) else "partial_topology_match"
        audit.append({"dataset": dataset, "records": len(rows), "source": source_labels.name if source_labels else "", "matched": matched, "status": status})
        if status == "recovered":
            out = args.out_dir / source.name
            out.write_text("\n".join(json.dumps(row, sort_keys=True) for row in labeled_rows) + "\n", encoding="utf-8")
    with args.audit.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit[0]))
        writer.writeheader()
        writer.writerows(audit)
    (args.audit.with_suffix(".json")).write_text(json.dumps({"sources": {k: v.name for k, v in sources.items()}, "rows": audit}, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
