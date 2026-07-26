"""Synthetic graph suite for GPTok experiments."""

from __future__ import annotations

import math
import random

import networkx as nx

from gptok2.data.schema import GraphRecord, graph_to_record


def generate_synthetic_graphs(config: dict, seed: int = 42, ood: bool = False) -> list[GraphRecord]:
    cfg = config.get("data", {}).get("synthetic", {})
    rng = random.Random(seed + (1009 if ood else 0))
    families = list(cfg.get("families", ["er", "ba", "grid", "tree", "cycle", "star", "clique_chain", "motif_mix"]))
    n_graphs = int(cfg.get("num_graphs", 120))
    n_min = int(cfg.get("num_nodes_min", 12))
    n_max = int(cfg.get("num_nodes_max", 48))
    if ood:
        n_min = max(n_max, 2 * n_min)
        n_max = max(n_min + 1, 2 * n_max)
    p = float(cfg.get("edge_prob", 0.08)) * (1.6 if ood else 1.0)
    out: list[GraphRecord] = []
    for i in range(n_graphs):
        family = families[i % len(families)]
        n = rng.randint(n_min, n_max)
        graph = _make_graph(family, n, p, rng)
        out.append(graph_to_record(graph, f"{'ood' if ood else 'g'}_{i:05d}", family, int(cfg.get("num_node_types", 8))))
    return out


def split_records(records: list[GraphRecord], config: dict) -> dict[str, list[GraphRecord]]:
    split = config.get("data", {}).get("split", {})
    train = float(split.get("train_ratio", 0.7))
    val = float(split.get("val_ratio", 0.1))
    n = len(records)
    n_train = int(round(train * n))
    n_val = int(round(val * n))
    return {"train": records[:n_train], "val": records[n_train : n_train + n_val], "test": records[n_train + n_val :]}


def _make_graph(family: str, n: int, p: float, rng: random.Random) -> nx.Graph:
    seed = rng.randint(0, 10**9)
    if family == "ba":
        return nx.barabasi_albert_graph(n, max(1, min(4, n - 1)), seed=seed)
    if family == "grid":
        rows = max(2, int(math.sqrt(n)))
        cols = max(2, math.ceil(n / rows))
        graph = nx.convert_node_labels_to_integers(nx.grid_2d_graph(rows, cols))
        return graph.subgraph(range(n)).copy()
    if family == "tree":
        if hasattr(nx, "random_labeled_tree"):
            return nx.random_labeled_tree(n, seed=seed)
        return nx.random_tree(n, seed=seed)
    if family == "cycle":
        graph = nx.cycle_graph(n)
        for _ in range(max(1, n // 5)):
            u = rng.randrange(n)
            graph.add_edge(u, (u + rng.randint(2, max(2, n // 2))) % n)
        return graph
    if family == "star":
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        block = max(4, n // 4)
        for hub in range(0, n, block):
            for leaf in range(hub + 1, min(n, hub + block)):
                graph.add_edge(hub, leaf)
        return _connect(graph)
    if family == "clique_chain":
        return _clique_chain(n, rng)
    if family == "motif_mix":
        return _motif_mix(n, rng)
    return _connect(nx.erdos_renyi_graph(n, p, seed=seed))


def _connect(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph
    comps = [list(c) for c in nx.connected_components(graph)]
    for a, b in zip(comps, comps[1:]):
        graph.add_edge(a[0], b[0])
    return graph


def _clique_chain(n: int, rng: random.Random) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    cursor, prev = 0, None
    while cursor < n:
        size = min(rng.choice([3, 4, 5]), n - cursor)
        nodes = list(range(cursor, cursor + size))
        graph.add_edges_from((u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :])
        if prev is not None:
            graph.add_edge(prev, nodes[0])
        prev = nodes[-1]
        cursor += size
    return graph


def _motif_mix(n: int, rng: random.Random) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    cursor, prev = 0, None
    while cursor < n:
        size = min(rng.choice([3, 4, 5, 6]), n - cursor)
        nodes = list(range(cursor, cursor + size))
        kind = rng.choice(["path", "cycle", "star", "clique"])
        if kind == "cycle" and size >= 3:
            graph.add_edges_from((nodes[i], nodes[(i + 1) % size]) for i in range(size))
        elif kind == "star":
            graph.add_edges_from((nodes[0], v) for v in nodes[1:])
        elif kind == "clique" and size <= 5:
            graph.add_edges_from((u, v) for i, u in enumerate(nodes) for v in nodes[i + 1 :])
        else:
            graph.add_edges_from((nodes[i], nodes[i + 1]) for i in range(size - 1))
        if prev is not None:
            graph.add_edge(prev, nodes[0])
        prev = nodes[-1]
        cursor += size
    return graph

