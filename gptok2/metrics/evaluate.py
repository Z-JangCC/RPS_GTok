"""GPTok-IE intrinsic evaluation metrics."""

from __future__ import annotations

import math
from collections import Counter
from statistics import mean

import networkx as nx
import torch

from gptok2.data.schema import GraphRecord, edge_set, to_networkx
from gptok2.program.actions import GraphProgram
from gptok2.program.interpreter import InterpreterState
from gptok2.vq.codebook import PrototypeVQCodebook


def structure_fidelity(original: GraphRecord, reconstructed: GraphRecord) -> dict[str, float]:
    eo = edge_set(original)
    er = edge_set(reconstructed)
    go = _as_undirected(to_networkx(original))
    gr = _as_undirected(to_networkx(reconstructed))
    raw_tp = len(eo & er)
    raw_fp = len(er - eo)
    raw_fn = len(eo - er)
    raw_precision = raw_tp / max(1, raw_tp + raw_fp)
    raw_recall = raw_tp / max(1, raw_tp + raw_fn)
    raw_f1 = 2 * raw_precision * raw_recall / max(1e-12, raw_precision + raw_recall)
    mapped_er = _aligned_reconstructed_edges(go, gr)
    tp = len(eo & mapped_er)
    fp = len(mapped_er - eo)
    fn = len(eo - mapped_er)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    if raw_f1 > f1:
        precision, recall, f1 = raw_precision, raw_recall, raw_f1
        tp, fp, fn = raw_tp, raw_fp, raw_fn
    exact = float(_isomorphic(go, gr))
    tri_o = sum(nx.triangles(go).values()) // 3
    tri_r = sum(nx.triangles(gr).values()) // 3
    cyc_o = _cycle_count(go)
    cyc_r = _cycle_count(gr)
    return {
        "exact_reconstruction": exact,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "edge_fp": float(fp),
        "edge_fn": float(fn),
        "raw_edge_precision": raw_precision,
        "raw_edge_recall": raw_recall,
        "raw_edge_f1": raw_f1,
        "node_count_error": abs(original.num_nodes - reconstructed.num_nodes) / max(1, original.num_nodes),
        "edge_count_error": abs(len(eo) - len(er)) / max(1, len(eo)),
        "connected_component_error": abs(nx.number_connected_components(go) - nx.number_connected_components(gr)),
        "triangle_count_error": abs(tri_o - tri_r) / max(1, tri_o),
        "cycle_count_error": abs(cyc_o - cyc_r) / max(1, cyc_o),
        "degree_l1": _degree_l1(go, gr),
        "spectrum_distance": _spectrum_distance(go, gr),
        "graph_edit_approx": float(fp + fn + abs(original.num_nodes - reconstructed.num_nodes)),
        "motif_f1": _motif_f1(go, gr),
    }


def program_executability(program: GraphProgram, state: InterpreterState) -> dict[str, float]:
    total = max(1, state.executed_actions)
    dangling = sum(1 for p in state.ports if p.open)
    complete_lifecycle = state.graph_started and state.graph_ended and state.stopped and len(state.block_stack) == 0
    return {
        "executable_rate": float(state.illegal_actions == 0 and complete_lifecycle),
        "complete_lifecycle_rate": float(complete_lifecycle),
        "illegal_action_rate": state.illegal_actions / total,
        "runtime_error_rate": state.runtime_errors / total,
        "invalid_reference_rate": state.invalid_refs / total,
        "port_capacity_violation_rate": state.capacity_violations / total,
        "type_mismatch_rate": state.type_mismatches / total,
        "duplicate_edge_error": state.duplicate_edges / total,
        "dangling_port_rate": dangling / max(1, len(state.ports)),
        "block_consistency_rate": float(len(state.block_stack) == 0),
        "global_link_dependency": sum(1 for a in program.actions if a.op == "GLOBAL_LINK") / max(1, len(program.actions)),
    }


def compression_metrics(original: GraphRecord, program: GraphProgram, codebook_size: int, ref_window: int) -> dict[str, float]:
    actions = program.actions
    op_counts = Counter(a.op for a in actions)
    token_len = len(actions)
    bits = 0.0
    for a in actions:
        if a.op == "EMIT":
            bits += math.log2(max(2, codebook_size))
        elif a.op == "INTERFACE":
            bits += math.log2(max(2, ref_window))
        elif a.op in {"ATTACH", "MERGE_NODE", "MERGE_EDGE", "CLOSE_CYCLE"}:
            bits += math.log2(max(2, ref_window)) + 4.0
        elif a.op == "GLOBAL_LINK":
            bits += 2 * math.log2(max(2, token_len))
        else:
            bits += 1.0
    return {
        "program_tokens": float(token_len),
        "token_length_ratio_edges": token_len / max(1, len(edge_set(original))),
        "token_length_ratio_graph": token_len / max(1, original.num_nodes + len(edge_set(original))),
        "bits_per_edge": bits / max(1, len(edge_set(original))),
        "bits_per_node": bits / max(1, original.num_nodes),
        "global_link_rate": op_counts["GLOBAL_LINK"] / max(1, token_len),
        "attach_rate": op_counts["ATTACH"] / max(1, token_len),
        "merge_rate": (op_counts["MERGE_NODE"] + op_counts["MERGE_EDGE"]) / max(1, token_len),
        "close_cycle_rate": op_counts["CLOSE_CYCLE"] / max(1, token_len),
    }


def prototype_stability(codebook: PrototypeVQCodebook, patches, assignments: dict[str, int]) -> dict[str, float]:
    by_code: dict[int, list] = {}
    for patch in patches:
        by_code.setdefault(assignments[patch.patch_id], []).append(patch)
    variances = []
    port_cons = []
    for code_id, group in by_code.items():
        code = codebook.get(code_id)
        if not group:
            continue
        dists = [codebook.patch_distance(p, code) for p in group]
        variances.append(mean(dists))
        if codebook.factorized_interfaces:
            port_cons.append(1.0)
        else:
            port_cons.append(sum(1 for p in group if tuple(x.prototype_schema() for x in p.ports) == code.port_schema) / len(group))
    stats = codebook.usage_stats()
    stats.update(
        {
            "intra_code_structural_variance": float(mean(variances)) if variances else 0.0,
            "prototype_distance": float(mean(variances)) if variances else 0.0,
            "port_schema_consistency": float(mean(port_cons)) if port_cons else 0.0,
            "topk_prototype_agreement": float(mean(port_cons)) if port_cons else 0.0,
            "split_necessity_rate": float(sum(1 for v in variances if v > 2.5) / max(1, len(variances))),
            "merge_redundancy_rate": _redundancy_rate(codebook),
        }
    )
    return stats


def token_quality_metrics(codebook: PrototypeVQCodebook, patches, programs: list[GraphProgram]) -> dict[str, float]:
    patch_count = max(1, len(patches))
    edge_patches = sum(1 for p in patches if len(p.nodes) == 2 and len(p.edges) == 1)
    large_motif_patches = sum(1 for p in patches if len(p.nodes) >= 4 and len(p.edges) >= len(p.nodes))
    code_usage = [max(0, int(c.usage)) for c in codebook.codes]
    total_usage = max(1, sum(code_usage))
    visual_groups: dict[tuple, list] = {}
    for code in codebook.codes:
        sig = (int(code.prototype_num_nodes), tuple(sorted(tuple(sorted(e)) for e in code.prototype_edges)))
        visual_groups.setdefault(sig, []).append(code)
    duplicate_codes = sum(max(0, len(group) - 1) for group in visual_groups.values())
    duplicate_usage = sum(sum(max(0, int(c.usage)) for c in group) for group in visual_groups.values() if len(group) > 1)
    large_motif_usage = sum(
        max(0, int(c.usage))
        for c in codebook.codes
        if int(c.prototype_num_nodes) >= 4 and len(c.prototype_edges) >= int(c.prototype_num_nodes)
    )
    op_counts = Counter(a.op for p in programs for a in p.actions)
    program_lengths = [len(p.actions) for p in programs]
    sorted_usage = sorted(code_usage, reverse=True)
    return {
        "edge_patch_ratio": edge_patches / patch_count,
        "large_motif_patch_ratio": large_motif_patches / patch_count,
        "unique_visual_structures": float(len(visual_groups)),
        "visual_duplicate_ratio": duplicate_codes / max(1, len(codebook.codes)),
        "visual_duplicate_usage_share": duplicate_usage / total_usage,
        "top1_usage_share": (sorted_usage[0] if sorted_usage else 0) / total_usage,
        "top10_usage_share": sum(sorted_usage[:10]) / total_usage,
        "large_motif_token_usage_share": large_motif_usage / total_usage,
        "shape_vocab_size": float(len(codebook.codes)),
        "interface_vocab_size": float(len(codebook.interfaces)),
        "program_length_mean": float(mean(program_lengths)) if program_lengths else 0.0,
        "emit_action_share": op_counts["EMIT"] / max(1, sum(op_counts.values())),
        "interface_action_share": op_counts["INTERFACE"] / max(1, sum(op_counts.values())),
        "merge_node_action_share": op_counts["MERGE_NODE"] / max(1, sum(op_counts.values())),
    }


def port_merge_quality(programs: list[GraphProgram], states: list[InterpreterState]) -> dict[str, float]:
    total_ports = sum(len(s.ports) for s in states)
    used_ports = sum(sum(1 for p in s.ports if p.used > 0) for s in states)
    total_actions = sum(max(1, s.executed_actions) for s in states)
    merge_ops = sum(sum(1 for a in p.actions if a.op in {"MERGE_NODE", "MERGE_EDGE"}) for p in programs)
    wrong_merge = sum(s.type_mismatches for s in states)
    return {
        "port_utilization_rate": used_ports / max(1, total_ports),
        "port_overhead": 1.0 - used_ports / max(1, total_ports),
        "merge_precision": 1.0 - wrong_merge / max(1, merge_ops),
        "merge_recall": merge_ops / max(1, total_actions),
        "merge_f1": _f1(1.0 - wrong_merge / max(1, merge_ops), merge_ops / max(1, total_actions)),
        "wrong_merge_rate": wrong_merge / max(1, merge_ops),
        "cycle_closure_accuracy": sum(sum(1 for a in p.actions if a.op == "CLOSE_CYCLE") for p in programs) / max(1, total_actions),
    }


def aggregate(rows: list[dict[str, float]], prefix: str = "") -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({k for r in rows for k in r})
    return {prefix + k: float(mean(float(r.get(k, 0.0)) for r in rows)) for k in keys}


def _isomorphic(a: nx.Graph, b: nx.Graph) -> bool:
    if a.number_of_nodes() != b.number_of_nodes() or a.number_of_edges() != b.number_of_edges():
        return False
    if sorted(dict(a.degree()).values()) != sorted(dict(b.degree()).values()):
        return False
    return nx.weisfeiler_lehman_graph_hash(a) == nx.weisfeiler_lehman_graph_hash(b)


def _as_undirected(graph: nx.Graph) -> nx.Graph:
    return graph.to_undirected() if graph.is_directed() else graph


def _aligned_reconstructed_edges(original: nx.Graph, reconstructed: nx.Graph) -> set[tuple[int, int]]:
    if original.number_of_nodes() == 0 or reconstructed.number_of_nodes() == 0:
        return set()
    iso_mapping = _safe_isomorphism_mapping(original, reconstructed)
    if iso_mapping is not None:
        return {
            tuple(sorted((iso_mapping[u], iso_mapping[v])))
            for u, v in reconstructed.edges()
            if u in iso_mapping and v in iso_mapping
        }
    original_order = _canonical_node_order(original)
    recon_order = _canonical_node_order(reconstructed)
    mapping = {
        r: original_order[i] if i < len(original_order) else len(original_order) + i
        for i, r in enumerate(recon_order)
    }
    return {tuple(sorted((mapping[u], mapping[v]))) for u, v in reconstructed.edges() if u in mapping and v in mapping}


def _safe_isomorphism_mapping(original: nx.Graph, reconstructed: nx.Graph) -> dict[int, int] | None:
    if not _isomorphic(original, reconstructed):
        return None
    if nx.is_forest(original) and nx.is_forest(reconstructed):
        try:
            return _forest_isomorphism_mapping(original, reconstructed)
        except (nx.NetworkXException, RuntimeError, StopIteration):
            return None
    if original.number_of_nodes() <= 256 and original.number_of_edges() <= 4096:
        try:
            mapping = nx.vf2pp_isomorphism(original, reconstructed)
        except (nx.NetworkXException, RuntimeError, MemoryError):
            return None
        if mapping:
            return {recon_node: orig_node for orig_node, recon_node in mapping.items()}
    return None


def _forest_isomorphism_mapping(original: nx.Graph, reconstructed: nx.Graph) -> dict[int, int] | None:
    orig_components = [_component_subgraph(original, c) for c in nx.connected_components(original)]
    rec_components = [_component_subgraph(reconstructed, c) for c in nx.connected_components(reconstructed)]
    if len(orig_components) != len(rec_components):
        return None
    buckets: dict[tuple, list[nx.Graph]] = {}
    for comp in rec_components:
        buckets.setdefault(_component_signature(comp), []).append(comp)
    mapping: dict[int, int] = {}
    for orig_comp in orig_components:
        sig = _component_signature(orig_comp)
        if not buckets.get(sig):
            return None
        rec_comp = buckets[sig].pop()
        pairs = nx.algorithms.isomorphism.tree_isomorphism(orig_comp, rec_comp)
        if not pairs:
            return None
        mapping.update({rec_node: orig_node for orig_node, rec_node in pairs})
    return mapping


def _component_subgraph(graph: nx.Graph, nodes) -> nx.Graph:
    return graph.subgraph(nodes).copy()


def _component_signature(graph: nx.Graph) -> tuple:
    return (
        graph.number_of_nodes(),
        graph.number_of_edges(),
        tuple(sorted(dict(graph.degree()).values())),
        nx.weisfeiler_lehman_graph_hash(graph),
    )


def _canonical_node_order(graph: nx.Graph) -> list[int]:
    triangles = nx.triangles(graph) if not graph.is_directed() else nx.triangles(graph.to_undirected())
    clustering = nx.clustering(graph.to_undirected() if graph.is_directed() else graph)
    core = nx.core_number(graph.to_undirected() if graph.is_directed() else graph) if graph.number_of_nodes() else {}
    return sorted(
        graph.nodes(),
        key=lambda n: (
            -graph.degree(n),
            -triangles.get(n, 0),
            -core.get(n, 0),
            -round(float(clustering.get(n, 0.0)), 6),
            n,
        ),
    )


def _cycle_count(graph: nx.Graph) -> int:
    return len(nx.cycle_basis(graph)) if graph.number_of_nodes() else 0


def _degree_l1(a: nx.Graph, b: nx.Graph) -> float:
    da = Counter(dict(a.degree()).values())
    db = Counter(dict(b.degree()).values())
    keys = set(da) | set(db)
    return sum(abs(da[k] - db[k]) for k in keys) / max(1, a.number_of_nodes())


def _spectrum_distance(a: nx.Graph, b: nx.Graph, k: int = 10) -> float:
    def vals(g):
        if g.number_of_nodes() == 0:
            return torch.zeros(k)
        lap = nx.normalized_laplacian_matrix(g).toarray()
        eig = torch.linalg.eigvalsh(torch.tensor(lap, dtype=torch.float32))
        if eig.numel() < k:
            eig = torch.cat([eig, torch.zeros(k - eig.numel())])
        return eig[:k]

    return float(torch.mean(torch.abs(vals(a) - vals(b))).item())


def _motif_f1(a: nx.Graph, b: nx.Graph) -> float:
    ca = Counter({"triangles": sum(nx.triangles(a).values()) // 3, "cycles": _cycle_count(a), "stars": sum(1 for _, d in a.degree() if d >= 3)})
    cb = Counter({"triangles": sum(nx.triangles(b).values()) // 3, "cycles": _cycle_count(b), "stars": sum(1 for _, d in b.degree() if d >= 3)})
    tp = sum(min(ca[k], cb[k]) for k in ca)
    fp = sum(max(0, cb[k] - ca[k]) for k in ca)
    fn = sum(max(0, ca[k] - cb[k]) for k in ca)
    return _f1(tp / max(1, tp + fp), tp / max(1, tp + fn))


def _f1(p: float, r: float) -> float:
    return 2 * p * r / max(1e-12, p + r)


def _redundancy_rate(codebook: PrototypeVQCodebook) -> float:
    keys = [(c.structural_hash, c.port_schema) for c in codebook.codes]
    return 1.0 - len(set(keys)) / max(1, len(keys))

