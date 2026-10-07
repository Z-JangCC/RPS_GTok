"""Schema-aware attributed canonicalization for RPS-GTok.

The implementation deliberately exposes whether a canonical order was found
by exact individualization or by a deterministic non-certified fallback.  This
is needed for a truthful permutation-stability audit.
"""

from __future__ import annotations

import itertools
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import torch

try:  # Optional exact backend; the deterministic search remains the fallback.
    import pynauty  # type: ignore
except Exception:  # pragma: no cover - depends on the execution environment.
    pynauty = None

from gptok2.data.schema import GraphRecord
from gptok2.data.schema_spec import SchemaSpec


@dataclass
class CanonicalResult:
    record: GraphRecord
    old_to_new: dict[int, int]
    new_to_old: dict[int, int]
    refinement_rounds: int
    tie_groups: tuple[tuple[int, ...], ...]
    search_nodes: int
    search_branches: int
    exact_completed: bool
    fallback_used: bool
    elapsed_sec: float
    canonical_key: tuple
    reconstruction_target: str = "topology"
    backend: str = "bounded_search"

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.record.graph_id,
            "num_nodes": self.record.num_nodes,
            "num_edges": int(self.record.edge_index.shape[1]) if self.record.edge_index.ndim == 2 else 0,
            "old_to_new": {str(k): int(v) for k, v in self.old_to_new.items()},
            "new_to_old": {str(k): int(v) for k, v in self.new_to_old.items()},
            "refinement_rounds": int(self.refinement_rounds),
            "tie_groups": [list(g) for g in self.tie_groups],
            "search_nodes": int(self.search_nodes),
            "search_branches": int(self.search_branches),
            "exact_completed": bool(self.exact_completed),
            "fallback_used": bool(self.fallback_used),
            "elapsed_sec": float(self.elapsed_sec),
            "reconstruction_target": self.reconstruction_target,
            "backend": self.backend,
        }


def canonicalize(
    record: GraphRecord,
    schema: SchemaSpec | None = None,
    *,
    exact: bool = True,
    max_search_nodes: int | None = None,
    timeout_sec: float | None = None,
    exact_backend: str = "bounded_search",
) -> CanonicalResult:
    """Return a deterministic canonical copy and audit statistics.

    The routine supports the simple edge-index representation used by the
    release while preserving duplicate rows and self loops in the returned
    record.  Schema validation is explicit; unsupported fields are not silently
    treated as retained.
    """

    spec = schema or SchemaSpec()
    start = time.perf_counter()
    n = int(record.num_nodes)
    edges = _edge_rows(record)
    node_keys = [_node_key(record, i, spec) for i in range(n)]
    edge_keys = _edge_key_map(edges, record, spec)
    colors = _refine_colors(n, edges, node_keys, edge_keys, spec)
    groups = _groups_from_colors(colors)
    tie_groups = tuple(tuple(group) for group in groups if len(group) > 1)
    budget = int(max_search_nodes if max_search_nodes is not None else spec.max_search_nodes)
    deadline = (
        start + float(timeout_sec)
        if timeout_sec is not None
        else (start + float(spec.timeout_sec) if spec.timeout_sec is not None else None)
    )

    order_groups = [tuple(group) for group in groups]
    search_nodes = 0
    search_branches = 0
    best_order: tuple[int, ...] | None = None
    best_key: tuple | None = None
    exact_completed = True
    fallback_used = False

    backend_used = "bounded_search"
    if exact and exact_backend == "nauty":
        # Nauty supplies a canonical ordering for the colored incidence
        # expansion.  Unlike the bounded Python search this path scales to
        # large tie groups; failures are reported as fallback rather than
        # silently downgraded to exact.
        nauty_order = _nauty_order(record, edges, node_keys, edge_keys, spec)
        if nauty_order is not None:
            best_order = nauty_order
            best_key = _serialization_key(record, best_order, edges, spec, edge_keys)
            search_nodes = 1
            search_branches = 1
            backend_used = "pynauty_colored_incidence_v2"
        else:
            exact_completed = False
            fallback_used = True
            backend_used = "pynauty_unavailable_or_error"
    elif exact and tie_groups:
        for candidate in _candidate_orders(order_groups):
            search_nodes += 1
            search_branches += 1
            if search_nodes > budget or (deadline is not None and time.perf_counter() > deadline):
                exact_completed = False
                fallback_used = True
                break
            key = _serialization_key(record, candidate, edges, spec, edge_keys)
            if best_key is None or key < best_key:
                best_key = key
                best_order = candidate
    elif exact and not tie_groups:
        best_order = tuple(node for group in order_groups for node in group)
        best_key = _serialization_key(record, best_order, edges, spec, edge_keys)
    else:
        exact_completed = False
        fallback_used = True

    if best_order is None:
        fallback_used = True
        exact_completed = False
        best_order = tuple(
            sorted(
                range(n),
                key=lambda node: (colors[node], _node_key(record, node, spec), int(node)),
            )
        )
        best_key = _serialization_key(record, best_order, edges, spec, edge_keys)

    old_to_new = {old: new for new, old in enumerate(best_order)}
    new_to_old = {new: old for old, new in old_to_new.items()}
    canonical_record = _relabel_record(record, old_to_new, spec)
    elapsed = time.perf_counter() - start
    return CanonicalResult(
        record=canonical_record,
        old_to_new=old_to_new,
        new_to_old=new_to_old,
        refinement_rounds=_refinement_rounds(n, edges, node_keys, edge_keys, spec),
        tie_groups=tie_groups,
        search_nodes=search_nodes,
        search_branches=search_branches,
        exact_completed=bool(exact_completed),
        fallback_used=bool(fallback_used),
        elapsed_sec=elapsed,
        canonical_key=best_key or (),
        reconstruction_target=spec.reconstruction_target,
        backend=backend_used,
    )


def _candidate_orders(groups: list[tuple[int, ...]]) -> Iterable[tuple[int, ...]]:
    """Lazily enumerate group permutations without recursive stack growth."""
    if not groups:
        yield ()
        return
    iterators: list[Iterable[tuple[int, ...]] | None] = [None] * len(groups)
    chosen: list[tuple[int, ...] | None] = [None] * len(groups)
    level = 0
    while level >= 0:
        if level == len(groups):
            yield tuple(node for block in chosen if block is not None for node in block)
            level -= 1
            continue
        if iterators[level] is None:
            iterators[level] = iter(itertools.permutations(groups[level]))
        try:
            chosen[level] = next(iterators[level])  # type: ignore[arg-type]
            level += 1
            if level < len(groups):
                iterators[level] = None
        except StopIteration:
            iterators[level] = None
            chosen[level] = None
            level -= 1


def _nauty_order(
    record: GraphRecord,
    edges: list[tuple[int, int, int]],
    node_keys: list[tuple],
    edge_keys: dict[tuple[int, int], tuple],
    spec: SchemaSpec,
) -> tuple[int, ...] | None:
    """Return a certified order from nauty for a colored incidence graph.

    Edge and node attributes are represented as vertex colors in an incidence
    expansion.  This preserves parallel edges and self-loops, and prevents an
    edge gadget from being confused with an original node.  The returned
    ``canon_label`` order is the old-node sequence in canonical positions.
    """
    if pynauty is None:
        return None
    n = int(record.num_nodes)
    m = len(edges)
    directed = bool(spec.directed or record.directed)
    family = 3 if directed else 1
    total = n + family * m
    adjacency: dict[int, set[int]] = {i: set() for i in range(total)}
    in_degree = [0] * n
    out_degree = [0] * n
    for u_raw, v_raw, _ in edges:
        u, v = int(u_raw), int(v_raw)
        if directed:
            out_degree[u] += 1
            in_degree[v] += 1
        else:
            out_degree[u] += 1
            if v != u:
                out_degree[v] += 1
    color_keys: list[tuple] = [
        ("node", (out_degree[i], in_degree[i]) if directed else (out_degree[i],), key)
        for i, key in enumerate(node_keys)
    ]
    for eidx, (u, v, row_idx) in enumerate(edges):
        gadget = n + family * eidx
        endpoint = (u, v) if directed else tuple(sorted((u, v)))
        edge_type = (
            int(record.edge_type[row_idx])
            if spec.retain_edge_type and record.edge_type is not None and row_idx < len(record.edge_type)
            else None
        )
        edge_attr = (
            _tensor_row(record.edge_attr, row_idx)
            if spec.retain_edge_attr and record.edge_attr is not None and row_idx < len(record.edge_attr)
            else ()
        )
        edge_color = (edge_type, edge_attr)
        if directed:
            src, dst = gadget + 1, gadget + 2
            color_keys.extend([("edge", edge_color), ("src_role", edge_color), ("dst_role", edge_color)])
            for a, b in ((gadget, src), (src, int(u)), (gadget, dst), (dst, int(v))):
                adjacency[a].add(b); adjacency[b].add(a)
        else:
            color_keys.append(("edge", edge_color))
            adjacency[gadget].add(int(u)); adjacency[int(u)].add(gadget)
            if int(v) != int(u):
                adjacency[gadget].add(int(v)); adjacency[int(v)].add(gadget)
    partitions: dict[tuple, set[int]] = defaultdict(set)
    for vertex, key in enumerate(color_keys):
        partitions[key].add(vertex)
    try:
        graph = pynauty.Graph(
            number_of_vertices=total,
            directed=False,
            adjacency_dict={k: sorted(v) for k, v in adjacency.items()},
            vertex_coloring=[partitions[key] for key in sorted(partitions, key=repr)],
        )
        label = list(pynauty.canon_label(graph))
    except Exception:
        return None
    if len(label) != total or sorted(label) != list(range(total)):
        return None
    # pynauty returns the original vertex ids in canonical order (not a map).
    original = tuple(int(v) for v in label if int(v) < n)
    return original if len(original) == n else None


def canonical_record(
    record: GraphRecord,
    schema: SchemaSpec | None = None,
    **kwargs: Any,
) -> GraphRecord:
    return canonicalize(record, schema, **kwargs).record


def _edge_rows(record: GraphRecord) -> list[tuple[int, int, int]]:
    if record.edge_index.numel() == 0:
        return []
    return [(int(u), int(v), idx) for idx, (u, v) in enumerate(record.edge_index.t().tolist())]


def _node_key(record: GraphRecord, node: int, spec: SchemaSpec) -> tuple:
    node_type = int(record.node_type[node]) if spec.retain_node_type and record.node_type is not None else None
    attr = _tensor_row(record.node_attr, node) if spec.retain_node_attr else ()
    return (node_type, attr)


def _tensor_row(tensor: torch.Tensor | None, index: int) -> tuple:
    if tensor is None:
        return ()
    values = tensor[index].detach().cpu().reshape(-1).tolist()
    return tuple(values)


def _edge_key_map(
    edges: list[tuple[int, int, int]],
    record: GraphRecord,
    spec: SchemaSpec,
) -> dict[tuple[int, int], tuple]:
    grouped: dict[tuple[int, int], list[tuple]] = defaultdict(list)
    for u, v, idx in edges:
        endpoint = (u, v) if spec.directed or record.directed else tuple(sorted((u, v)))
        edge_type = (
            int(record.edge_type[idx])
            if spec.retain_edge_type and record.edge_type is not None and idx < len(record.edge_type)
            else None
        )
        attr = (
            _tensor_row(record.edge_attr, idx)
            if spec.retain_edge_attr and record.edge_attr is not None and idx < len(record.edge_attr)
            else ()
        )
        grouped[endpoint].append((edge_type, attr))
    return {key: tuple(sorted(values)) for key, values in grouped.items()}


def _refine_colors(
    n: int,
    edges: list[tuple[int, int, int]],
    node_keys: list[tuple],
    edge_keys: dict[tuple[int, int], tuple],
    spec: SchemaSpec,
) -> list[tuple]:
    neighbors = _neighbor_signatures(edge_keys, spec)
    colors = list(node_keys)
    rounds = max(1, min(n + 1, 64))
    for _ in range(rounds):
        updated = []
        for node in range(n):
            pairs = [(colors[other], signature) for other, signature in neighbors.get(node, [])]
            updated.append((colors[node], tuple(sorted(pairs, key=repr))))
        labels = {value: idx for idx, value in enumerate(sorted(set(updated), key=repr))}
        new_colors = [labels[value] for value in updated]
        if new_colors == colors:
            break
        colors = new_colors
    return [(colors[i], node_keys[i]) for i in range(n)]


def _refinement_rounds(
    n: int,
    edges: list[tuple[int, int, int]],
    node_keys: list[tuple],
    edge_keys: dict[tuple[int, int], tuple],
    spec: SchemaSpec,
) -> int:
    neighbors = _neighbor_signatures(edge_keys, spec)
    colors = list(node_keys)
    for iteration in range(1, max(1, min(n + 1, 64)) + 1):
        updated = []
        for node in range(n):
            pairs = [(colors[other], signature) for other, signature in neighbors.get(node, [])]
            updated.append((colors[node], tuple(sorted(pairs, key=repr))))
        labels = {value: idx for idx, value in enumerate(sorted(set(updated), key=repr))}
        new_colors = [labels[value] for value in updated]
        if new_colors == colors:
            return iteration
        colors = new_colors
    return max(1, min(n + 1, 64))


def _neighbor_signatures(
    edge_keys: dict[tuple[int, int], tuple],
    spec: SchemaSpec,
) -> dict[int, list[tuple[int, tuple]]]:
    neighbors: dict[int, list[tuple[int, tuple]]] = defaultdict(list)
    for (u, v), signature in edge_keys.items():
        neighbors[u].append((v, signature))
        if not spec.directed and u != v:
            neighbors[v].append((u, signature))
    return neighbors


def _groups_from_colors(colors: list[tuple]) -> list[list[int]]:
    grouped: dict[tuple, list[int]] = defaultdict(list)
    for node, color in enumerate(colors):
        grouped[color].append(node)
    return sorted(grouped.values(), key=lambda group: (repr(colors[group[0]]), group[0]))


def _serialization_key(
    record: GraphRecord,
    order: tuple[int, ...],
    edges: list[tuple[int, int, int]],
    spec: SchemaSpec,
    edge_keys: dict[tuple[int, int], tuple],
) -> tuple:
    rank = {node: idx for idx, node in enumerate(order)}
    nodes = tuple((_node_key(record, node, spec), rank[node]) for node in order)
    edge_rows = []
    for u, v, idx in edges:
        a, b = rank[u], rank[v]
        if not spec.directed and not record.directed:
            a, b = sorted((a, b))
        endpoint = (u, v) if spec.directed or record.directed else tuple(sorted((u, v)))
        edge_rows.append((a, b, edge_keys.get(endpoint, ())))
    return (nodes, tuple(sorted(edge_rows)))


def _relabel_record(record: GraphRecord, mapping: dict[int, int], spec: SchemaSpec) -> GraphRecord:
    edges = []
    for old_idx, (u, v) in enumerate(record.edge_index.t().tolist() if record.edge_index.numel() else []):
        edges.append((mapping[int(u)], mapping[int(v)], old_idx))
    order = sorted(range(int(record.num_nodes)), key=lambda old: mapping[old])

    def reorder(tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.ndim == 0 or tensor.shape[0] != len(order):
            return tensor.clone()
        return tensor[order].clone()

    def edge_sort_key(row: tuple[int, int, int]) -> tuple:
        u, v, old_idx = row
        a, b = (u, v) if spec.directed or record.directed else tuple(sorted((u, v)))
        edge_type = int(record.edge_type[old_idx]) if record.edge_type is not None else None
        edge_attr = _tensor_row(record.edge_attr, old_idx) if record.edge_attr is not None else ()
        return (a, b, edge_type, edge_attr)

    edges = sorted(edges, key=edge_sort_key)
    edge_rows = [(u, v) for u, v, _ in edges]
    edge_order = [old_idx for _, _, old_idx in edges]
    return GraphRecord(
        graph_id=record.graph_id,
        num_nodes=record.num_nodes,
        edge_index=torch.tensor(edge_rows, dtype=torch.long).t().contiguous()
        if edge_rows
        else torch.empty(2, 0, dtype=torch.long),
        node_type=reorder(record.node_type),
        edge_type=record.edge_type[edge_order].clone() if record.edge_type is not None else None,
        node_attr=reorder(record.node_attr),
        edge_attr=record.edge_attr[edge_order].clone() if record.edge_attr is not None else None,
        y=record.y.clone() if torch.is_tensor(record.y) else record.y,
        metadata=dict(record.metadata),
        directed=bool(record.directed),
    )
