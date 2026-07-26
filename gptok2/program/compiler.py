"""Canonical graph-program compiler for gptok2."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from gptok2.data.schema import GraphRecord, edge_set
from gptok2.patches.proposal import make_edge_patches_for_edges
from gptok2.patches.types import Patch
from gptok2.program.actions import Action, GraphProgram
from gptok2.program.interpreter import InterpreterState, execute_action
from gptok2.vq.codebook import PrototypeVQCodebook


def compile_program(record: GraphRecord, patches: list[Patch], codebook: PrototypeVQCodebook, config: dict) -> GraphProgram:
    cfg = config.get("program", {})
    ref_window = int(cfg.get("ref_window", 16))
    use_program_layer = bool(cfg.get("program_layer", True))
    use_blocks = bool(cfg.get("begin_blocks", True)) and use_program_layer
    patches = _refine_nonexact_large_patches(record, patches, codebook, config)
    assignments = {p.patch_id: codebook.assign(p) for p in patches}
    ordered = list(patches) if cfg.get("noncanonical_order", False) else _canonical_frontier_order(record, patches, assignments)
    ref_window = max(1, ref_window)
    close_cycle_consumes_all = bool(cfg.get("close_cycle_consumes_all", False))
    state = InterpreterState()
    actions: list[Action] = []

    def emit_action(action: Action) -> None:
        actions.append(action)
        execute_action(state, action, codebook, ref_window, close_cycle_consumes_all)

    emit_action(Action("BEGIN_GRAPH"))
    runtime_node_anchor: dict[int, list[int]] = defaultdict(list)
    runtime_edge_anchor: dict[tuple[int, int], list[int]] = defaultdict(list)
    runtime_port: dict[int, list[int]] = defaultdict(list)
    edge_to_patches: dict[tuple[int, int], list[int]] = defaultdict(list)
    for pi, patch in enumerate(ordered):
        for e in patch.edges:
            edge_to_patches[tuple(sorted(e))].append(pi)
    covered_internal_edges = set(edge_to_patches)

    for pi, patch in enumerate(ordered):
        code_id = assignments[patch.patch_id]
        code = codebook.get(code_id)
        interface_id = codebook.assign_interface(patch) if codebook.factorized_interfaces else -1
        exact_code_match = _patch_matches_code_exactly(patch, code, factorized=codebook.factorized_interfaces)
        block_size = max(1, int(cfg.get("block_size", 8)))
        if use_blocks and pi % block_size == 0:
            emit_action(Action("BEGIN_BLOCK"))
        emit_action(Action("EMIT", (code_id,)))
        if interface_id >= 0 and cfg.get("emit_interface_actions", True):
            emit_action(Action("INTERFACE", (interface_id,)))
        if not use_program_layer:
            continue
        node_to_anchor = {patch.nodes[a.nodes[0]]: a.local_index for a in patch.anchors if a.nodes}
        node_to_port = {patch.nodes[p.anchor]: p.local_index for p in patch.ports if p.anchor < len(patch.nodes)}
        edge_to_anchor = {tuple(sorted((patch.nodes[a.nodes[0]], patch.nodes[a.nodes[1]]))): a.local_index for a in patch.edge_anchors if len(a.nodes) >= 2}

        merge_edges = []
        for edge, owners in edge_to_patches.items():
            if pi in owners and any(o < pi for o in owners) and edge in edge_to_anchor:
                prev_ref = _runtime_anchor_ref(runtime_edge_anchor.get(edge, []), state, ref_window, "edge")
                if prev_ref is not None:
                    merge_edges.append((edge_to_anchor[edge], prev_ref))
        allow_nonexact_merges = bool(cfg.get("allow_nonexact_merges", True))
        if not cfg.get("disable_merge_edge", False) and (exact_code_match or allow_nonexact_merges):
            for anchor_idx, ref in sorted(merge_edges):
                if anchor_idx < len(state.current_edge_anchors):
                    emit_action(Action("MERGE_EDGE", (anchor_idx, ref)))

        merge_nodes = []
        if (exact_code_match or allow_nonexact_merges) and not cfg.get("disable_merge_node", False):
            for node in patch.nodes:
                if node in runtime_node_anchor and node in node_to_anchor:
                    ref = _runtime_anchor_ref(
                        runtime_node_anchor.get(node, []),
                        state,
                        ref_window,
                        "node",
                        node_to_anchor[node] if cfg.get("enforce_anchor_role_consistency", True) else None,
                    )
                    if ref is not None:
                        merge_nodes.append((node_to_anchor[node], ref))
            for anchor_idx, ref in sorted(merge_nodes):
                if anchor_idx < len(state.current_anchors):
                    emit_action(Action("MERGE_NODE", (anchor_idx, ref)))

        attach_ops = []
        current_nodes = set(patch.nodes)
        for u, v in sorted(edge_set(record)):
            if tuple(sorted((u, v))) in covered_internal_edges:
                continue
            if (u in current_nodes) ^ (v in current_nodes):
                cu = u if u in current_nodes else v
                pv = v if u in current_nodes else u
                if cu in node_to_port and pv in runtime_port:
                    ref = _runtime_port_ref(runtime_port.get(pv, []), state, ref_window)
                    if ref is not None:
                        op = "CLOSE_CYCLE" if _is_cycle_like(patch, cu) and not cfg.get("disable_close_cycle", False) else "ATTACH"
                        attach_ops.append((op, node_to_port[cu], ref))
        for op, port_idx, ref in sorted(set(attach_ops), key=lambda x: (x[2], x[1], x[0])):
            if port_idx < len(state.current_ports):
                emit_action(Action(op, (port_idx, ref)))

        for node, anchor_idx in node_to_anchor.items():
            if anchor_idx < len(state.current_anchors):
                runtime_node_anchor[node].append(state.current_anchors[anchor_idx])
        for node, port_idx in node_to_port.items():
            if port_idx < len(state.current_ports):
                runtime_port[node].append(state.current_ports[port_idx])
        for edge, anchor_idx in edge_to_anchor.items():
            if anchor_idx < len(state.current_edge_anchors):
                runtime_edge_anchor[edge].append(state.current_edge_anchors[anchor_idx])
        if use_blocks and (pi % block_size == block_size - 1):
            emit_action(Action("END_BLOCK"))
    if use_blocks and actions[-1].op != "END_BLOCK":
        emit_action(Action("END_BLOCK"))
    if use_program_layer:
        actions.extend(
            _global_links_stateful(
                record,
                state,
                runtime_port,
                int(cfg.get("max_global_links", 8)),
                bool(cfg.get("global_link_all_cross_edges", True)),
                covered_internal_edges,
                codebook,
                ref_window,
                close_cycle_consumes_all,
            )
        )
    emit_action(Action("END_GRAPH"))
    emit_action(Action("STOP"))
    meta = {
        "num_patches": len(ordered),
        "num_codes": len(set(assignments.values())),
        "program_version": "gptok_opg_v1",
        "leakage_policy": "abstract_refs_only",
    }
    return GraphProgram(record.graph_id, actions, meta)


def _canonical_frontier_order(record: GraphRecord, patches: list[Patch], assignments: dict[str, int]) -> list[Patch]:
    if not patches:
        return []
    edges = edge_set(record)
    node_sets = [set(p.nodes) for p in patches]
    edge_sets = [{tuple(sorted(e)) for e in p.edges} for p in patches]
    patch_scores = []
    for i, patch in enumerate(patches):
        closed = len(edge_sets[i])
        ports = len(patch.ports)
        patch_scores.append((-(closed), ports, assignments[patch.patch_id], patch.structural_hash, patch.patch_id, i))
    start = min(range(len(patches)), key=lambda i: patch_scores[i])
    ordered = [start]
    remaining = set(range(len(patches))) - {start}
    emitted_nodes = set(node_sets[start])
    emitted_edges = set(edge_sets[start])
    while remaining:
        best = min(
            remaining,
            key=lambda j: (
                -_frontier_weight(j, node_sets, edge_sets, emitted_nodes, emitted_edges, edges),
                assignments[patches[j].patch_id],
                patches[j].structural_hash,
                -len(edge_sets[j]),
                len(patches[j].ports),
                patches[j].patch_id,
            ),
        )
        ordered.append(best)
        remaining.remove(best)
        emitted_nodes |= node_sets[best]
        emitted_edges |= edge_sets[best]
    return [patches[i] for i in ordered]


def _frontier_weight(index: int, node_sets, edge_sets, emitted_nodes, emitted_edges, graph_edges) -> int:
    nodes = node_sets[index]
    shared_nodes = len(nodes & emitted_nodes)
    shared_edges = len(edge_sets[index] & emitted_edges)
    cross_edges = sum(1 for u, v in graph_edges if (u in nodes and v in emitted_nodes) or (v in nodes and u in emitted_nodes))
    internal_gain = len(edge_sets[index] - emitted_edges)
    return 5 * shared_edges + 3 * shared_nodes + 2 * cross_edges + internal_gain


def _runtime_port_ref(candidates: list[int], state: InterpreterState, ref_window: int) -> int | None:
    recent = state.recent_open_ports(ref_window, exclude=set(state.current_ports))
    for ref, runtime_idx in enumerate(recent, start=1):
        if runtime_idx in candidates:
            return ref
    return None


def _runtime_any_port_ref(candidates: list[int], state: InterpreterState) -> int | None:
    recent = state.recent_ports(10**9)
    for ref, runtime_idx in enumerate(recent, start=1):
        if runtime_idx in candidates:
            return ref
    return None


def _runtime_anchor_ref(candidates: list[int], state: InterpreterState, ref_window: int, kind: str, current_local_index: int | None = None) -> int | None:
    exclude = set(state.current_edge_anchors if kind == "edge" else state.current_anchors)
    recent = state.recent_anchors(ref_window, kind, exclude=exclude)
    for ref, runtime_idx in enumerate(recent, start=1):
        if runtime_idx in candidates and _anchor_ref_consistent(state, runtime_idx, kind, current_local_index):
            return ref
    return None


def _anchor_ref_consistent(state: InterpreterState, runtime_idx: int, kind: str, current_local_index: int | None) -> bool:
    if current_local_index is None or kind != "node":
        return True
    if current_local_index < 0 or current_local_index >= len(state.current_anchors):
        return False
    current = state.anchors[state.current_anchors[current_local_index]]
    ref = state.anchors[runtime_idx]
    return current.role == ref.role or "generic-port" in {current.role, ref.role}


def _is_cycle_like(patch: Patch, node: int) -> bool:
    return any(patch.nodes[p.anchor] == node and p.role == "cycle-anchor" for p in patch.ports if p.anchor < len(patch.nodes))


def _global_links_stateful(record, state, runtime_port, max_global_links, all_cross_edges, covered_internal_edges, codebook, ref_window, close_cycle_consumes_all) -> list[Action]:
    actions: list[Action] = []
    for u, v in sorted(edge_set(record)):
        if tuple(sorted((u, v))) in covered_internal_edges:
            continue
        if len(actions) >= max_global_links:
            break
        ru = _runtime_any_port_ref(runtime_port.get(u, []), state)
        rv = _runtime_any_port_ref(runtime_port.get(v, []), state)
        if ru is None or rv is None or ru == rv:
            continue
        ports = state.recent_ports(10**9)
        pu = state.ports[ports[ru - 1]]
        pv = state.ports[ports[rv - 1]]
        if state.graph.has_edge(pu.node, pv.node):
            continue
        if not all_cross_edges and abs(ru - rv) <= ref_window:
            continue
        action = Action("GLOBAL_LINK", (ru, rv))
        actions.append(action)
        execute_action(state, action, codebook, ref_window, close_cycle_consumes_all)
    return actions


def _patch_matches_code_exactly(patch: Patch, code, factorized: bool = False) -> bool:
    local = {v: i for i, v in enumerate(patch.nodes)}
    patch_edges = tuple(sorted(tuple(sorted((local[u], local[v]))) for u, v in patch.edges))
    shape_exact = (
        len(patch.nodes) == code.prototype_num_nodes
        and patch_edges == tuple(sorted(tuple(sorted(e)) for e in code.prototype_edges))
    )
    if factorized:
        return shape_exact
    return (
        shape_exact
        and len(patch.ports) == len(code.port_schema)
        and len(patch.anchors) == len(code.anchor_schema)
        and len(patch.edge_anchors) == len(code.edge_anchor_schema)
    )


def _refine_nonexact_large_patches(record: GraphRecord, patches: list[Patch], codebook: PrototypeVQCodebook, config: dict) -> list[Patch]:
    cfg = config.get("program", {})
    if not cfg.get("fallback_nonexact_large_patches", True):
        return patches
    min_nodes = int(cfg.get("fallback_nonexact_min_nodes", 3))
    keep: list[Patch] = []
    fallback_edges: set[tuple[int, int]] = set()
    covered_by_keep: set[tuple[int, int]] = set()
    decisions = []
    for patch in patches:
        code = codebook.get(codebook.assign(patch))
        exact = _patch_matches_code_exactly(patch, code, factorized=codebook.factorized_interfaces)
        should_split = len(patch.nodes) >= min_nodes and not exact
        decisions.append((patch, should_split))
        if not should_split:
            keep.append(patch)
            covered_by_keep.update(tuple(sorted(e)) for e in patch.edges)
    for patch, should_split in decisions:
        if should_split:
            fallback_edges.update(tuple(sorted(e)) for e in patch.edges if tuple(sorted(e)) not in covered_by_keep)
    if fallback_edges:
        keep.extend(make_edge_patches_for_edges(record, fallback_edges, config, prefix="fallback"))
    keep = sorted({p.patch_id: p for p in keep}.values(), key=lambda p: (p.structural_hash, -len(p.edges), p.patch_id))
    return [
        replace(patch, patch_id=f"{patch.graph_id}_prog_{i:04d}_{patch.structural_hash[:8]}")
        for i, patch in enumerate(keep)
    ]

