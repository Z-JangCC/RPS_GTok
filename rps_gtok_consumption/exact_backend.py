"""Scalable exact isomorphism certificates for attributed multigraph records.

The backend reduces an attributed graph to a vertex-coloured incidence graph
and delegates canonical certification to :mod:`pynauty`.  Undirected edges use
one edge gadget.  Directed edges use an edge carrier plus source/destination
role gadgets, so reversing an edge cannot preserve the certificate.  Parallel
edges and self loops remain distinct because every input edge gets its own
gadget family and its attributes are part of its color.

This module deliberately returns a certificate rather than claiming that a
bounded Python individualisation search completed.  A caller can compare two
certificates for exact permutation equality and account for timeouts outside
this module.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from gptok2.data.schema import GraphRecord

try:  # optional dependency; callers should report unavailable explicitly
    import pynauty  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    pynauty = None


def _row(tensor: torch.Tensor | None, index: int) -> tuple[Any, ...]:
    if tensor is None or index >= int(tensor.shape[0]):
        return ()
    return tuple(tensor[index].detach().cpu().reshape(-1).tolist())


def _edge_color(record: GraphRecord, index: int) -> tuple[Any, ...]:
    edge_type = (
        int(record.edge_type[index])
        if record.edge_type is not None and index < len(record.edge_type)
        else None
    )
    return (edge_type, _row(record.edge_attr, index))


def _semantic_edge_rows(record: GraphRecord) -> list[tuple[int, int, int]]:
    """Normalize reciprocal COO rows for an undirected schema.

    Many graph loaders materialize an undirected edge twice (``u,v`` and
    ``v,u``).  We collapse only matching reverse rows with matching edge
    colors. Same-orientation parallel rows and attribute-mismatched rows are
    retained, so multigraph information is not silently discarded.
    """
    raw = (
        [(int(u), int(v), idx) for idx, (u, v) in enumerate(record.edge_index.t().tolist())]
        if record.edge_index.numel()
        else []
    )
    if record.directed:
        return raw
    groups = defaultdict(
        lambda: {"lo": [], "hi": []}
    )
    loops: list[tuple[int, int, int]] = []
    for u, v, idx in raw:
        color = _edge_color(record, idx)
        if u == v:
            loops.append((u, v, idx))
            continue
        lo, hi = sorted((u, v))
        orientation = "lo" if (u, v) == (lo, hi) else "hi"
        groups[(lo, hi, color)][orientation].append((u, v, idx))
    out = list(loops)
    for (lo, hi, _), parts in groups.items():
        left, right = parts["lo"], parts["hi"]
        paired = min(len(left), len(right))
        out.extend(left[:paired])
        out.extend(left[paired:])
        out.extend(right[paired:])
    return sorted(out, key=lambda row: row[2])


def _direct_graph_possible(record: GraphRecord, edges: list[tuple[int, int, int]]) -> bool:
    if any(u == v for u, v, _ in edges):
        return False
    endpoints = [(u, v) if record.directed else tuple(sorted((u, v))) for u, v, _ in edges]
    if len(endpoints) != len(set(endpoints)):
        return False
    return len({_edge_color(record, idx) for _, _, idx in edges}) <= 1


def _refined_node_colors(
    record: GraphRecord,
    edges: list[tuple[int, int, int]],
    directed: bool,
) -> list[tuple[Any, ...]]:
    """Compute a few exact 1-WL refinement rounds as vertex colors."""
    n = int(record.num_nodes)
    out_degree = [0] * n
    in_degree = [0] * n
    for u, v, _ in edges:
        if directed:
            out_degree[int(u)] += 1
            in_degree[int(v)] += 1
        else:
            out_degree[int(u)] += 1
            if int(v) != int(u):
                out_degree[int(v)] += 1
    colors: list[tuple[Any, ...]] = []
    for i in range(n):
        node_type = int(record.node_type[i]) if record.node_type is not None and i < len(record.node_type) else None
        degree_key = (out_degree[i], in_degree[i]) if directed else (out_degree[i],)
        colors.append(("node", degree_key, node_type, _row(record.node_attr, i)))
    for _ in range(3):
        neighborhoods: list[list[tuple[Any, ...]]] = [[] for _ in range(n)]
        for u, v, idx in edges:
            edge_color = _edge_color(record, idx)
            if directed:
                neighborhoods[int(u)].append(("out", edge_color, colors[int(v)]))
                neighborhoods[int(v)].append(("in", edge_color, colors[int(u)]))
            else:
                neighborhoods[int(u)].append(("adj", edge_color, colors[int(v)]))
                if int(v) != int(u):
                    neighborhoods[int(v)].append(("adj", edge_color, colors[int(u)]))
        colors = [
            ("wl", colors[i], tuple(sorted(neighborhoods[i], key=repr)))
            for i in range(n)
        ]
    return colors


def build_colored_incidence_graph(record: GraphRecord):
    """Build a pynauty graph and return it with the source vertex count.

    The returned graph is exact for the represented schema fields: node type
    and node attributes are colors, as are edge type and edge attributes.
    ``record.directed`` controls whether source/destination role gadgets are
    inserted.
    """
    if pynauty is None:
        raise RuntimeError("pynauty is not installed")
    n = int(record.num_nodes)
    raw_rows = (
        [(int(u), int(v), idx) for idx, (u, v) in enumerate(record.edge_index.t().tolist())]
        if record.edge_index.numel()
        else []
    )
    semantic_rows = _semantic_edge_rows(record)
    directed = bool(record.directed)
    use_direct = _direct_graph_possible(record, semantic_rows)
    degree_rows = semantic_rows
    edge_rows = [(u, v) for u, v, _ in degree_rows]
    node_colors = _refined_node_colors(record, degree_rows, directed)
    if use_direct:
        adjacency: dict[int, set[int]] = {i: set() for i in range(n)}
        for u, v in edge_rows:
            adjacency[int(u)].add(int(v))
            if not directed:
                adjacency[int(v)].add(int(u))
        partitions: dict[tuple[Any, ...], set[int]] = defaultdict(set)
        for vertex, color in enumerate(node_colors):
            partitions[color].add(vertex)
        graph = pynauty.Graph(
            number_of_vertices=n,
            directed=directed,
            adjacency_dict={k: sorted(v) for k, v in adjacency.items()},
            vertex_coloring=[partitions[key] for key in sorted(partitions, key=repr)],
        )
        return graph, n, len(edge_rows)
    # Attributed/multigraph records still use incidence gadgets, but retain the
    # schema-level reciprocal-row normalization. Same-orientation parallel
    # rows and reverse rows with different colors remain distinct in
    # ``_semantic_edge_rows``.
    # One carrier per undirected edge; carrier + two role vertices per directed
    # edge.  The role colors make orientation part of the certificate.
    family = 3 if directed else 1
    total = n + family * len(edge_rows)
    adjacency: dict[int, set[int]] = {i: set() for i in range(total)}
    colors: list[tuple[Any, ...]] = []
    colors.extend(node_colors)

    for j, (u_raw, v_raw) in enumerate(edge_rows):
        u, v = int(u_raw), int(v_raw)
        key = _edge_color(record, semantic_rows[j][2])
        carrier = n + family * j
        if directed:
            src = carrier + 1
            dst = carrier + 2
            colors.extend(
                [
                    ("edge", key),
                    ("src_role", key),
                    ("dst_role", key),
                ]
            )
            for a, b in ((carrier, src), (src, u), (carrier, dst), (dst, v)):
                adjacency[a].add(b)
                adjacency[b].add(a)
        else:
            colors.append(("edge", key))
            adjacency[carrier].add(u)
            adjacency[u].add(carrier)
            if v != u:
                adjacency[carrier].add(v)
                adjacency[v].add(carrier)

    partitions: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for vertex, color in enumerate(colors):
        partitions[color].add(vertex)
    ordered = [partitions[key] for key in sorted(partitions, key=repr)]
    graph = pynauty.Graph(
        number_of_vertices=total,
        directed=False,
        adjacency_dict={k: sorted(v) for k, v in adjacency.items()},
        vertex_coloring=ordered,
    )
    return graph, n, len(edge_rows)


def exact_certificate(record: GraphRecord) -> bytes:
    """Return a nauty certificate for ``record`` or raise a clear error."""
    graph, _, _ = build_colored_incidence_graph(record)
    return pynauty.certificate(graph)


def exact_canonical_order(record: GraphRecord) -> tuple[int, ...]:
    """Return one exact old-node order induced by nauty's canonical labels.

    ``pynauty.canon_label`` returns original vertex IDs in canonical order;
    filtering the auxiliary incidence vertices recovers a canonical node order.
    For graphs with automorphisms, multiple representative orders can be
    valid; callers requiring permutation-invariant equality must compare
    :func:`exact_certificate`, not the representative order tuple itself.
    """
    graph, n, _ = build_colored_incidence_graph(record)
    labels = list(pynauty.canon_label(graph))
    if len(labels) != int(graph.number_of_vertices):
        raise RuntimeError("pynauty returned an invalid canonical labeling")
    order = tuple(int(vertex) for vertex in labels if int(vertex) < n)
    if len(order) != n:
        raise RuntimeError("canonical labeling lost an original node")
    return order
