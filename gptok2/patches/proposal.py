"""Port-aware overlapping seed-grow-boundary-prune patch proposal."""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

import networkx as nx

from gptok2.data.schema import GraphRecord, to_networkx
from gptok2.patches.types import Anchor, Patch, Port


@dataclass
class PatchProposalConfig:
    min_size: int = 2
    max_size: int = 7
    max_ports: int = 4
    max_patches_per_graph: int = 64
    overlap_budget: int = 2
    learned_port_subtypes: int = 8
    max_port_capacity: int = 8
    grow_rounds: int = 4
    sparse_edge_patches: bool = False
    sparse_density_threshold: float = 0.10
    sparse_clustering_threshold: float = 0.08
    lambda_in: float = 1.0
    lambda_out: float = 0.8
    lambda_ports: float = 0.7
    lambda_cut: float = 0.5
    lambda_motif: float = 0.8

    @staticmethod
    def from_config(config: dict) -> "PatchProposalConfig":
        row = config.get("patch", {})
        return PatchProposalConfig(
            min_size=int(row.get("min_size", 2)),
            max_size=int(row.get("max_size", 7)),
            max_ports=int(row.get("max_ports", 4)),
            max_patches_per_graph=int(row.get("max_patches_per_graph", 64)),
            overlap_budget=int(row.get("overlap_budget", 2)),
            learned_port_subtypes=int(row.get("learned_port_subtypes", 8)),
            max_port_capacity=int(row.get("max_port_capacity", 8)),
            grow_rounds=int(row.get("grow_rounds", 4)),
            sparse_edge_patches=bool(row.get("sparse_edge_patches", False)),
            sparse_density_threshold=float(row.get("sparse_density_threshold", 0.10)),
            sparse_clustering_threshold=float(row.get("sparse_clustering_threshold", 0.08)),
            lambda_in=float(row.get("lambda_in", 1.0)),
            lambda_out=float(row.get("lambda_out", 0.8)),
            lambda_ports=float(row.get("lambda_ports", 0.7)),
            lambda_cut=float(row.get("lambda_cut", 0.5)),
            lambda_motif=float(row.get("lambda_motif", 0.8)),
        )


def propose_patches(record: GraphRecord, config: PatchProposalConfig | dict) -> list[Patch]:
    cfg = config if isinstance(config, PatchProposalConfig) else PatchProposalConfig.from_config(config)
    graph = to_networkx(record)
    if graph.number_of_nodes() == 0:
        return []
    if cfg.sparse_edge_patches and _should_use_edge_patches(graph, cfg):
        return _edge_cover_patches(record.graph_id, graph, cfg)
    cycles = _cycle_nodes(graph)
    triangles = _triangles(graph)
    bridges = {tuple(sorted(e)) for e in nx.bridges(graph)}
    articulations = set(nx.articulation_points(graph))
    seeds = _structural_seeds(graph, cfg, cycles, triangles, articulations)
    candidates: dict[tuple[int, ...], Patch] = {}
    for seed_index, seed in enumerate(seeds):
        nodes = _grow_seed(graph, set(seed), cfg, cycles, triangles, bridges)
        if len(nodes) < cfg.min_size:
            continue
        nodes = _prune(graph, nodes, cfg, cycles, triangles, bridges)
        if len(nodes) < cfg.min_size:
            continue
        patch = _make_patch(record.graph_id, graph, nodes, cfg, cycles, triangles, bridges, articulations, seed_index)
        key = tuple(sorted(patch.nodes))
        if key not in candidates or patch.score > candidates[key].score:
            candidates[key] = patch

    selected = _select_covering_patches(record.graph_id, graph, list(candidates.values()), cfg)
    return selected[: cfg.max_patches_per_graph]


def _should_use_edge_patches(graph: nx.Graph, cfg: PatchProposalConfig) -> bool:
    if graph.number_of_edges() == 0:
        return True
    density = nx.density(graph)
    clustering = nx.average_clustering(graph.to_undirected())
    return nx.is_forest(graph) or (density <= cfg.sparse_density_threshold and clustering <= cfg.sparse_clustering_threshold)


def _edge_cover_patches(graph_id: str, graph: nx.Graph, cfg: PatchProposalConfig) -> list[Patch]:
    cycles = _cycle_nodes(graph)
    triangles = _triangles(graph)
    bridges = {tuple(sorted(e)) for e in nx.bridges(graph)}
    articulations = set(nx.articulation_points(graph))
    patches = []
    for i, edge in enumerate(sorted(tuple(sorted(e)) for e in graph.edges())):
        patches.append(_make_patch(graph_id, graph, set(edge), cfg, cycles, triangles, bridges, articulations, i))
    covered = {node for p in patches for node in p.nodes}
    for node in sorted(set(graph.nodes()) - covered):
        patches.append(_make_patch(graph_id, graph, {node}, cfg, cycles, triangles, bridges, articulations, len(patches)))
    for i, p in enumerate(patches):
        p.patch_id = f"{p.graph_id}_edge_{i:04d}_{p.structural_hash[:8]}"
    return patches[: cfg.max_patches_per_graph]


def make_edge_patches_for_edges(record: GraphRecord, edges, config: PatchProposalConfig | dict, prefix: str = "edge_refine") -> list[Patch]:
    cfg = config if isinstance(config, PatchProposalConfig) else PatchProposalConfig.from_config(config)
    graph = to_networkx(record)
    cycles = _cycle_nodes(graph)
    triangles = _triangles(graph)
    bridges = {tuple(sorted(e)) for e in nx.bridges(graph)}
    articulations = set(nx.articulation_points(graph))
    patches = []
    for i, edge in enumerate(sorted({tuple(sorted((int(u), int(v)))) for u, v in edges})):
        if not graph.has_edge(*edge):
            continue
        patch = _make_patch(record.graph_id, graph, set(edge), cfg, cycles, triangles, bridges, articulations, i)
        patch.patch_id = f"{patch.graph_id}_{prefix}_{i:04d}_{patch.structural_hash[:8]}"
        patches.append(patch)
    return patches


def _structural_seeds(graph: nx.Graph, cfg: PatchProposalConfig, cycles, triangles, articulations) -> list[tuple[int, ...]]:
    seeds: list[tuple[int, ...]] = []
    seeds.extend(triangles)
    for cyc in sorted(cycles, key=lambda x: (len(x), x)):
        if 3 <= len(cyc) <= cfg.max_size:
            seeds.append(tuple(cyc))
    cliques = [tuple(c) for c in nx.find_cliques(graph) if 3 <= len(c) <= cfg.max_size]
    seeds.extend(sorted(cliques, key=lambda x: (-len(x), x)))
    for v, d in sorted(graph.degree, key=lambda x: (-x[1], x[0])):
        if d >= 3:
            seeds.append(tuple(sorted([v] + sorted(graph.neighbors(v))[: cfg.max_size - 1])))
    for e in sorted(graph.edges()):
        seeds.append(tuple(sorted(e)))
    for v in sorted(articulations):
        nb = sorted(graph.neighbors(v))
        seeds.append(tuple([v] + nb[: max(1, cfg.max_size - 1)]))
    for v in sorted(graph.nodes()):
        seeds.append((v,))
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    for seed in seeds:
        s = tuple(sorted(set(seed)))
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _grow_seed(graph, nodes: set[int], cfg: PatchProposalConfig, cycles, triangles, bridges) -> set[int]:
    nodes = set(nodes)
    rounds = 0
    while len(nodes) < cfg.max_size and rounds < max(1, cfg.grow_rounds):
        rounds += 1
        frontier = sorted({u for v in nodes for u in graph.neighbors(v)} - nodes)
        if not frontier:
            break
        current = _patch_score(graph, nodes, cfg, cycles, triangles, bridges)
        best_node, best_delta = None, 0.0
        for u in frontier:
            cand = nodes | {u}
            if len(_boundary_nodes(graph, cand)) > cfg.max_ports:
                continue
            delta = _patch_score(graph, cand, cfg, cycles, triangles, bridges) - current
            if delta > best_delta + 1e-9 or (abs(delta - best_delta) <= 1e-9 and (best_node is None or u < best_node)):
                best_node, best_delta = u, delta
        if best_node is None:
            break
        nodes.add(best_node)
    return nodes


def _prune(graph, nodes: set[int], cfg: PatchProposalConfig, cycles, triangles, bridges) -> set[int]:
    nodes = set(nodes)
    while len(nodes) > cfg.min_size and (len(nodes) > cfg.max_size or len(_boundary_nodes(graph, nodes)) > cfg.max_ports):
        current = _patch_score(graph, nodes, cfg, cycles, triangles, bridges)
        violates_hard_limit = len(nodes) > cfg.max_size or len(_boundary_nodes(graph, nodes)) > cfg.max_ports
        best_remove, best_score = None, -1e18
        for v in sorted(nodes):
            cand = nodes - {v}
            if not cand or not nx.is_connected(graph.subgraph(cand)):
                continue
            score = _patch_score(graph, cand, cfg, cycles, triangles, bridges)
            if score > best_score or (score == best_score and (best_remove is None or v > best_remove)):
                best_remove, best_score = v, score
        if best_remove is None or (not violates_hard_limit and best_score < current - 1.0):
            break
        nodes.remove(best_remove)
    return nodes


def _make_patch(graph_id, graph, nodes, cfg, cycles, triangles, bridges, articulations, seed_index) -> Patch:
    nodes = tuple(sorted(nodes))
    local = {v: i for i, v in enumerate(nodes)}
    edges = tuple(sorted(tuple(sorted(e)) for e in graph.subgraph(nodes).edges()))
    ports = []
    for li, v in enumerate(nodes):
        outside = [u for u in graph.neighbors(v) if u not in local]
        if outside:
            role = _port_role(graph, v, set(nodes), cycles, bridges, articulations)
            mode = "CLOSE" if role == "cycle-anchor" else ("MERGE" if role in {"bridge-port", "module-boundary"} else "ATTACH")
            capacity = min(max(1, cfg.max_port_capacity), max(1, len(outside)))
            subtype = _stable_subtype((role, graph.degree(v), len(outside), li), cfg.learned_port_subtypes)
            ports.append(Port(len(ports), li, role, mode, capacity, subtype))
    anchors = []
    for li, v in enumerate(nodes):
        role = _port_role(graph, v, set(nodes), cycles, bridges, articulations)
        anchors.append(Anchor(len(anchors), "node", role, _stable_subtype((role, graph.degree(v), li), cfg.learned_port_subtypes), (li,)))
    edge_anchors = []
    for u, v in edges:
        edge_anchors.append(
            Anchor(
                len(edge_anchors),
                "edge",
                "cycle-anchor" if _edge_in_cycle((u, v), cycles) else "bridge-port" if tuple(sorted((u, v))) in bridges else "generic-port",
                _stable_subtype((graph.degree(u), graph.degree(v), local[u], local[v]), cfg.learned_port_subtypes),
                tuple(sorted((local[u], local[v]))),
            )
        )
    features = _features(graph, set(nodes), cycles, triangles, bridges)
    structural_hash = structural_hash_for_patch(len(nodes), tuple((local[u], local[v]) for u, v in edges), tuple(p.schema_key() for p in ports))
    patch_id = f"{graph_id}_p{seed_index:04d}_{structural_hash[:8]}"
    return Patch(patch_id, graph_id, nodes, edges, tuple(ports), tuple(anchors), tuple(edge_anchors), _patch_score(graph, set(nodes), cfg, cycles, triangles, bridges), structural_hash, features)


def _select_covering_patches(graph_id: str, graph, patches: list[Patch], cfg: PatchProposalConfig) -> list[Patch]:
    patches = sorted(patches, key=lambda p: (-p.score, p.structural_hash, p.patch_id))
    selected_edges: set[tuple[int, int]] = set()
    node_counts: Counter[int] = Counter()
    selected: list[Patch] = []
    all_edges = {tuple(sorted(e)) for e in graph.edges()}
    for patch in patches:
        patch_edges = {tuple(sorted(e)) for e in patch.edges}
        gain = len(patch_edges - selected_edges)
        max_uses = max(1, cfg.overlap_budget + 1)
        overlap_ok = all(node_counts[v] < max_uses for v in patch.nodes)
        if gain > 0 and overlap_ok:
            selected.append(patch)
            selected_edges |= patch_edges
            node_counts.update(patch.nodes)
        if selected_edges >= all_edges:
            break
    for edge in sorted(all_edges - selected_edges):
        nodes = set(edge)
        patch = _make_patch(graph_id, graph, nodes, cfg, _cycle_nodes(graph), set(), set(nx.bridges(graph)), set(nx.articulation_points(graph)), len(selected))
        selected.append(patch)
        selected_edges.add(edge)
        node_counts.update(nodes)
    covered_nodes = {node for patch in selected for node in patch.nodes}
    for node in sorted(set(graph.nodes()) - covered_nodes):
        patch = _make_patch(
            graph_id,
            graph,
            {node},
            cfg,
            _cycle_nodes(graph),
            set(),
            set(nx.bridges(graph)),
            set(nx.articulation_points(graph)),
            len(selected),
        )
        selected.append(patch)
    selected = sorted(selected, key=lambda p: (p.structural_hash, -len(p.edges), p.patch_id))
    for i, p in enumerate(selected):
        p.patch_id = f"{p.graph_id}_canon_{i:04d}_{p.structural_hash[:8]}"
    return selected


def structural_hash_for_patch(num_nodes: int, edges: tuple[tuple[int, int], ...], port_schema: tuple = ()) -> str:
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edges)
    wl = nx.weisfeiler_lehman_graph_hash(g)
    degree_pattern = ",".join(map(str, sorted(d for _, d in g.degree())))
    port_pattern = "|".join(map(str, sorted(port_schema)))
    raw = f"n={num_nodes};e={len(edges)};wl={wl};deg={degree_pattern};ports={port_pattern}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _patch_score(graph, nodes: set[int], cfg: PatchProposalConfig, cycles, triangles, bridges) -> float:
    if not nodes:
        return -1e9
    sub = graph.subgraph(nodes)
    internal = sub.number_of_edges()
    boundary = _boundary_nodes(graph, nodes)
    outgoing = sum(1 for u in nodes for v in graph.neighbors(u) if v not in nodes)
    possible = max(1, len(nodes) * (len(nodes) - 1) / 2)
    closed_motif = sum(1 for tri in triangles if set(tri) <= nodes)
    closed_motif += sum(1 for cyc in cycles if set(cyc) <= nodes and len(cyc) <= len(nodes))
    cut_motif = sum(1 for tri in triangles if set(tri) & nodes and not set(tri) <= nodes)
    cut_motif += sum(1 for cyc in cycles if set(cyc) & nodes and not set(cyc) <= nodes and len(cyc) <= cfg.max_size + 1)
    bridge_boundary_bonus = sum(1 for u in nodes for v in graph.neighbors(u) if v not in nodes and tuple(sorted((u, v))) in bridges)
    return (
        cfg.lambda_in * (internal / possible + internal)
        - cfg.lambda_out * outgoing
        - cfg.lambda_ports * len(boundary)
        - cfg.lambda_cut * cut_motif
        + cfg.lambda_motif * closed_motif
        + 0.25 * bridge_boundary_bonus
    )


def _boundary_nodes(graph, nodes: set[int]) -> set[int]:
    return {v for v in nodes if any(u not in nodes for u in graph.neighbors(v))}


def _cycle_nodes(graph) -> list[tuple[int, ...]]:
    basis = []
    for comp in nx.connected_components(graph):
        basis.extend(tuple(sorted(c)) for c in nx.cycle_basis(graph.subgraph(comp)))
    return sorted(set(basis), key=lambda x: (len(x), x))


def _triangles(graph) -> set[tuple[int, int, int]]:
    out = set()
    for clique in nx.enumerate_all_cliques(graph):
        if len(clique) > 3:
            break
        if len(clique) == 3:
            out.add(tuple(sorted(clique)))
    return out


def _edge_in_cycle(edge, cycles) -> bool:
    u, v = edge
    return any(u in cyc and v in cyc for cyc in cycles)


def _port_role(graph, v, nodes: set[int], cycles, bridges, articulations) -> str:
    inside_degree = sum(1 for u in graph.neighbors(v) if u in nodes)
    outside_degree = sum(1 for u in graph.neighbors(v) if u not in nodes)
    if any(v in cyc for cyc in cycles):
        return "cycle-anchor"
    if v in articulations or any(tuple(sorted((u, v))) in bridges for u in graph.neighbors(v)):
        return "bridge-port"
    if graph.degree(v) >= 4 or inside_degree >= 3:
        return "branch-port"
    if outside_degree >= 2:
        return "module-boundary"
    if inside_degree <= 1:
        return "chain-end"
    return "generic-port"


def _stable_subtype(values, modulo: int) -> int:
    if modulo <= 1:
        return 0
    raw = "|".join(map(str, values))
    return int(hashlib.sha1(raw.encode("utf-8")).hexdigest(), 16) % modulo


def _features(graph, nodes: set[int], cycles, triangles, bridges) -> dict[str, float]:
    sub = graph.subgraph(nodes)
    n = len(nodes)
    e = sub.number_of_edges()
    degrees = [d for _, d in sub.degree()]
    boundary = _boundary_nodes(graph, nodes)
    return {
        "num_nodes": float(n),
        "num_edges": float(e),
        "density": float(nx.density(sub)) if n > 1 else 0.0,
        "avg_degree": float(sum(degrees) / max(1, n)),
        "max_degree": float(max(degrees) if degrees else 0),
        "triangles": float(sum(1 for tri in triangles if set(tri) <= nodes)),
        "cycles": float(sum(1 for cyc in cycles if set(cyc) <= nodes)),
        "ports": float(len(boundary)),
        "bridge_cuts": float(sum(1 for u in nodes for v in graph.neighbors(u) if v not in nodes and tuple(sorted((u, v))) in bridges)),
    }


def patches_for_records(records: list[GraphRecord], config: dict) -> dict[str, list[Patch]]:
    cfg = PatchProposalConfig.from_config(config)
    return {record.graph_id: propose_patches(record, cfg) for record in records}

