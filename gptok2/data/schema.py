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


def edge_set(record: GraphRecord, undirected: bool = True) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    if record.edge_index.numel() == 0:
        return edges
    for u, v in record.edge_index.t().tolist():
        u, v = int(u), int(v)
        if u == v:
            continue
        edges.add(tuple(sorted((u, v))) if undirected and not record.directed else (u, v))
    return edges


def make_edge_index(edges, directed: bool = False) -> torch.LongTensor:
    rows = sorted(
        {(int(u), int(v)) if directed else tuple(sorted((int(u), int(v)))) for u, v in edges if int(u) != int(v)}
    )
    return torch.tensor(rows, dtype=torch.long).t().contiguous() if rows else torch.empty(2, 0, dtype=torch.long)


def to_networkx(record: GraphRecord) -> nx.Graph:
    graph = nx.DiGraph() if record.directed else nx.Graph()
    graph.add_nodes_from(range(int(record.num_nodes)))
    graph.add_edges_from(edge_set(record, undirected=not record.directed))
    if record.node_type is not None:
        nx.set_node_attributes(graph, {i: int(record.node_type[i]) for i in range(record.num_nodes)}, "node_type")
    return graph


def graph_to_record(graph: nx.Graph, graph_id: str, family: str = "unknown", node_types: int = 8) -> GraphRecord:
    graph = nx.convert_node_labels_to_integers(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    directed = graph.is_directed()
    undirected_graph = graph.to_undirected() if directed else graph
    n = graph.number_of_nodes()
    edges = sorted((int(u), int(v)) for u, v in graph.edges())
    edge_index = make_edge_index(edges, directed=directed)
    deg = [undirected_graph.degree(i) for i in range(n)]
    node_type = torch.tensor([(deg[i] + i) % max(node_types, 1) for i in range(n)], dtype=torch.long)
    node_attr = torch.zeros(n, 4, dtype=torch.float32)
    if n:
        dmax = max(max(deg), 1)
        node_attr[:, 0] = torch.tensor([d / dmax for d in deg], dtype=torch.float32)
        node_attr[:, 1] = torch.tensor([float(nx.clustering(undirected_graph, i)) for i in range(n)], dtype=torch.float32)
        node_attr[:, 2] = torch.tensor([float(i) / max(n - 1, 1) for i in range(n)], dtype=torch.float32)
        node_attr[:, 3] = 1.0
    y = torch.tensor(_family_label(family), dtype=torch.long)
    metadata = {"family": family, "triangle_count": int(sum(nx.triangles(undirected_graph).values()) // 3)}
    return GraphRecord(graph_id, n, edge_index, node_type=node_type, node_attr=node_attr, y=y, metadata=metadata, directed=directed)


def record_to_dict(record: GraphRecord) -> dict:
    def t(x):
        return x.detach().cpu().tolist() if torch.is_tensor(x) else x

    return {
        "graph_id": record.graph_id,
        "num_nodes": int(record.num_nodes),
        "edges": record.edge_index.t().tolist() if record.edge_index.numel() else [],
        "node_type": t(record.node_type),
        "node_attr": t(record.node_attr),
        "y": t(record.y),
        "metadata": record.metadata,
        "directed": record.directed,
    }


def record_from_dict(row: dict) -> GraphRecord:
    directed = bool(row.get("directed", False))
    edge_index = make_edge_index(row.get("edges", []), directed=directed)
    node_type = torch.tensor(row["node_type"], dtype=torch.long) if row.get("node_type") is not None else None
    node_attr = torch.tensor(row["node_attr"], dtype=torch.float32) if row.get("node_attr") is not None else None
    y = torch.tensor(row["y"], dtype=torch.long) if row.get("y") is not None else None
    return GraphRecord(str(row["graph_id"]), int(row["num_nodes"]), edge_index, node_type=node_type, node_attr=node_attr, y=y, metadata=dict(row.get("metadata", {})), directed=directed)


def _family_label(family: str) -> int:
    families = ["er", "ba", "grid", "tree", "cycle", "star", "clique_chain", "motif_mix"]
    return families.index(family) if family in families else 0

