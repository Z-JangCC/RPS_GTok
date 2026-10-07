"""Generic graph schema used by gptok2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import torch


@dataclass
class GraphRecord:
    graph_id: str
    num_nodes: int
    edge_index: torch.LongTensor
    node_type: torch.LongTensor | None = None
    edge_type: torch.LongTensor | None = None
    node_attr: torch.FloatTensor | None = None
    edge_attr: torch.FloatTensor | None = None
    y: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    directed: bool = False

    def clone(self) -> "GraphRecord":
        def c(x):
            return x.clone() if torch.is_tensor(x) else x

        return GraphRecord(
            self.graph_id,
            int(self.num_nodes),
            c(self.edge_index).long(),
            c(self.node_type),
            c(self.edge_type),
            c(self.node_attr),
            c(self.edge_attr),
            c(self.y),
            dict(self.metadata),
            bool(self.directed),
        )


def edge_rows(record: GraphRecord) -> list[tuple[int, int, int]]:
    """Return raw edge rows, preserving loops, duplicates and row identity."""
    if record.edge_index.numel() == 0:
        return []
    return [(int(u), int(v), idx) for idx, (u, v) in enumerate(record.edge_index.t().tolist())]


def edge_set(
    record: GraphRecord,
    undirected: bool = True,
    *,
    include_self_loops: bool = False,
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for u, v, _ in edge_rows(record):
        if u == v and not include_self_loops:
            continue
        edges.add(tuple(sorted((u, v))) if undirected and not record.directed else (u, v))
    return edges


def make_edge_index(
    edges,
    directed: bool = False,
    *,
    preserve_parallel: bool = True,
    include_self_loops: bool = True,
) -> torch.LongTensor:
    rows = [
        (int(u), int(v)) if directed else tuple(sorted((int(u), int(v))))
        for u, v in edges
        if include_self_loops or int(u) != int(v)
    ]
    if not preserve_parallel:
        rows = sorted(set(rows))
    else:
        rows = sorted(rows)
    return torch.tensor(rows, dtype=torch.long).t().contiguous() if rows else torch.empty(2, 0, dtype=torch.long)


def to_networkx(record: GraphRecord) -> nx.Graph:
    rows = edge_rows(record)
    duplicate_edges = len({(u, v) for u, v, _ in rows}) < len(rows)
    if duplicate_edges:
        graph = nx.MultiDiGraph() if record.directed else nx.MultiGraph()
    else:
        graph = nx.DiGraph() if record.directed else nx.Graph()
    graph.add_nodes_from(range(int(record.num_nodes)))
    for u, v, idx in rows:
        attrs = {}
        if record.edge_type is not None and idx < len(record.edge_type):
            attrs["edge_type"] = int(record.edge_type[idx])
        if record.edge_attr is not None and idx < len(record.edge_attr):
            attrs["edge_attr"] = record.edge_attr[idx].detach().cpu().tolist()
        if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
            graph.add_edge(u, v, key=idx, **attrs)
        else:
            graph.add_edge(u, v, **attrs)
    if record.node_type is not None:
        nx.set_node_attributes(graph, {i: int(record.node_type[i]) for i in range(record.num_nodes)}, "node_type")
    if record.node_attr is not None:
        nx.set_node_attributes(
            graph,
            {i: record.node_attr[i].detach().cpu().tolist() for i in range(record.num_nodes)},
            "node_attr",
        )
    return graph


def graph_to_record(graph: nx.Graph, graph_id: str, family: str = "unknown", node_types: int = 8) -> GraphRecord:
    graph = nx.convert_node_labels_to_integers(graph)
    directed = graph.is_directed()
    undirected_graph = graph.to_undirected() if directed else graph
    n = graph.number_of_nodes()
    edge_type_rows = []
    edge_attr_rows = []
    if isinstance(graph, (nx.MultiGraph, nx.MultiDiGraph)):
        raw_edges = list(graph.edges(keys=True, data=True))
        raw_edges.sort(key=lambda row: (int(row[0]), int(row[1]), int(row[2])))
        edges = [(int(u), int(v)) for u, v, _, _ in raw_edges]
        edge_type_rows = [data.get("edge_type") for _, _, _, data in raw_edges]
        edge_attr_rows = [data.get("edge_attr") for _, _, _, data in raw_edges]
    else:
        raw_edges = list(graph.edges(data=True))
        raw_edges.sort(key=lambda row: (int(row[0]), int(row[1])))
        edges = [(int(u), int(v)) for u, v, _ in raw_edges]
        edge_type_rows = [data.get("edge_type") for _, _, data in raw_edges]
        edge_attr_rows = [data.get("edge_attr") for _, _, data in raw_edges]
    edge_index = make_edge_index(edges, directed=directed, preserve_parallel=True, include_self_loops=True)
    deg = [undirected_graph.degree(i) for i in range(n)]
    node_type_values = [graph.nodes[i].get("node_type", (deg[i] + i) % max(node_types, 1)) for i in range(n)]
    node_type = torch.tensor(node_type_values, dtype=torch.long)
    node_attr_values = [graph.nodes[i].get("node_attr") for i in range(n)]
    node_attr = (
        torch.tensor(node_attr_values, dtype=torch.float32)
        if all(value is not None for value in node_attr_values)
        else torch.zeros(n, 4, dtype=torch.float32)
    )
    if n and not all(value is not None for value in node_attr_values):
        dmax = max(max(deg), 1)
        node_attr[:, 0] = torch.tensor([d / dmax for d in deg], dtype=torch.float32)
        node_attr[:, 1] = torch.tensor([float(nx.clustering(undirected_graph, i)) for i in range(n)], dtype=torch.float32)
        node_attr[:, 2] = torch.tensor([float(i) / max(n - 1, 1) for i in range(n)], dtype=torch.float32)
        node_attr[:, 3] = 1.0
    y = torch.tensor(_family_label(family), dtype=torch.long)
    simple_for_stats = nx.Graph(undirected_graph)
    metadata = {"family": family, "triangle_count": int(sum(nx.triangles(simple_for_stats).values()) // 3)}
    edge_type = (
        torch.tensor(edge_type_rows, dtype=torch.long)
        if edge_type_rows and all(value is not None for value in edge_type_rows)
        else None
    )
    edge_attr = (
        torch.tensor(edge_attr_rows, dtype=torch.float32)
        if edge_attr_rows and all(value is not None for value in edge_attr_rows)
        else None
    )
    return GraphRecord(
        graph_id,
        n,
        edge_index,
        node_type=node_type,
        edge_type=edge_type,
        node_attr=node_attr,
        edge_attr=edge_attr,
        y=y,
        metadata=metadata,
        directed=directed,
    )


def record_to_dict(record: GraphRecord) -> dict:
    def t(x):
        return x.detach().cpu().tolist() if torch.is_tensor(x) else x

    return {
        "graph_id": record.graph_id,
        "num_nodes": int(record.num_nodes),
        "edges": record.edge_index.t().tolist() if record.edge_index.numel() else [],
        "node_type": t(record.node_type),
        "edge_type": t(record.edge_type),
        "node_attr": t(record.node_attr),
        "edge_attr": t(record.edge_attr),
        "y": t(record.y),
        "metadata": record.metadata,
        "directed": record.directed,
    }


def record_from_dict(row: dict) -> GraphRecord:
    directed = bool(row.get("directed", False))
    # edge_type/edge_attr are aligned by COO row. Deserialization must not
    # reorder only the endpoints (or collapse their undirected orientation).
    # Canonicalization, if requested, sorts endpoints and attributes together.
    raw_edges = row.get("edges", [])
    edge_index = (
        torch.tensor(raw_edges, dtype=torch.long).reshape(-1, 2).t().contiguous()
        if raw_edges else torch.empty(2, 0, dtype=torch.long)
    )
    node_type = torch.tensor(row["node_type"], dtype=torch.long) if row.get("node_type") is not None else None
    edge_type = torch.tensor(row["edge_type"], dtype=torch.long) if row.get("edge_type") is not None else None
    node_attr = torch.tensor(row["node_attr"], dtype=torch.float32) if row.get("node_attr") is not None else None
    edge_attr = torch.tensor(row["edge_attr"], dtype=torch.float32) if row.get("edge_attr") is not None else None
    raw_y = row.get("y")
    if raw_y is None:
        y = None
    else:
        y_tensor = torch.tensor(raw_y)
        y = y_tensor.long() if not torch.is_floating_point(y_tensor) else y_tensor.float()
    return GraphRecord(
        str(row["graph_id"]),
        int(row["num_nodes"]),
        edge_index,
        node_type=node_type,
        edge_type=edge_type,
        node_attr=node_attr,
        edge_attr=edge_attr,
        y=y,
        metadata=dict(row.get("metadata", {})),
        directed=directed,
    )


def _family_label(family: str) -> int:
    families = ["er", "ba", "grid", "tree", "cycle", "star", "clique_chain", "motif_mix"]
    return families.index(family) if family in families else 0

