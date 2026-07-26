from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import networkx as nx
import torch

from gptok2.data.schema import GraphRecord, edge_set, graph_to_record, to_networkx
from gptok2.patches.proposal import patches_for_records
from gptok2.program.compiler import compile_program
from gptok2.program.actions import Action, GraphProgram
from gptok2.program.interpreter import InterpreterState, execute_action, execute_program
from gptok2.utils.io import ensure_dir
from gptok2.vq.codebook import learn_codebook
from gptok2_compact import CompactCodec, EntropyModel
from gptok2_compact.codec import compact_symbol_bits, original_program_bits
from gptok2_motif_macro import MotifEntropyModel, MotifMacroCodec, motif_symbol_bits
from gptok2_tokenizer.api import DEFAULT_CONFIG, _deep_update


MODES = ["original", "compact", "entropy", "motif_macro", "motif_entropy", "motif_hybrid"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate component GPTok2 on whole Cora citation graph.")
    parser.add_argument("--out", default="runs/rps_gtok_cora_full_graph_eval")
    parser.add_argument("--root", default="runs/rps_gtok_representative_eval/CORA/pyg_cora")
    parser.add_argument("--variant", choices=["default", "high_fidelity", "both"], default="both")
    parser.add_argument("--save-artifact", action="store_true")
    args = parser.parse_args()

    out_root = ensure_dir(args.out)
    record = load_cora_full_graph(Path(args.root))
    variants = ["default", "high_fidelity"] if args.variant == "both" else [args.variant]

    all_rows: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    for variant in variants:
        variant_dir = ensure_dir(out_root / variant)
        try:
            rows, status = run_variant(record, variant, variant_dir, save_artifact=bool(args.save_artifact))
            all_rows.extend(rows)
            statuses.append(status)
            write_csv(variant_dir / "summary.csv", rows)
            (variant_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        except Exception as exc:
            status = {
                "dataset": "CORA_FULL_GRAPH",
                "variant": variant,
                "run_status": "failed",
                "reason": repr(exc),
            }
            statuses.append(status)
            (variant_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(f"[failed] {variant}: {exc!r}", flush=True)

    write_csv(out_root / "summary.csv", all_rows)
    write_csv(out_root / "status.csv", statuses)
    print(f"[done] Cora full graph evaluation: {out_root}", flush=True)


def load_cora_full_graph(root: Path) -> GraphRecord:
    from torch_geometric.datasets import Planetoid

    pyg = Planetoid(str(root), name="Cora")[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(int(pyg.num_nodes)))
    for u, v in pyg.edge_index.t().tolist():
        u, v = int(u), int(v)
        if u != v:
            graph.add_edge(u, v)
    graph.graph["graph_id"] = "cora_full_citation_graph"
    record = graph_to_record(graph, "cora_full_citation_graph", "cora_full")
    if torch.is_tensor(getattr(pyg, "y", None)):
        record.y = pyg.y.detach().cpu()
    return record


def config_for_variant(variant: str) -> dict[str, Any]:
    if variant == "default":
        return {}
    if variant == "high_fidelity":
        return {
            "patch": {
                "max_patches_per_graph": 12000,
                "sparse_edge_patches": True,
                "sparse_density_threshold": 1.0,
                "sparse_clustering_threshold": 1.0,
                "max_ports": 8,
                "max_port_capacity": 32,
            },
            "program": {
                "ref_window": 20000,
                "block_size": 64,
                "max_global_links": 20000,
                "global_link_all_cross_edges": True,
            },
            "compact_entropy": {
                "max_macros": 768,
                "max_bpe_merges": 768,
                "max_macro_len": 24,
                "min_macro_count": 2,
                "min_bpe_count": 2,
            },
            "motif_macro": {
                "max_parameterized_span_len": 512,
                "max_structural_len": 128,
                "max_structural_macros": 768,
                "max_merge_schemas": 768,
                "max_code_schemas": 768,
                "min_parameterized_emit_count": 2,
                "min_structural_count": 2,
                "min_merge_schema_count": 2,
                "min_code_schema_count": 2,
            },
        }
    raise ValueError(variant)


def run_variant(record: GraphRecord, variant: str, out_dir: Path, *, save_artifact: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    print(f"[variant] {variant}: staged final fit on Cora full graph", flush=True)
    config = _deep_update(DEFAULT_CONFIG, config_for_variant(variant))
    compact_cfg = config.get("compact_entropy", {})
    motif_cfg = config.get("motif_macro", {})

    start = time.time()
    patch_map = patches_for_records([record], config)
    patches = patch_map[record.graph_id]
    patch_sec = time.time() - start
    print(f"[stage] {variant}: patches={len(patches)} in {patch_sec:.1f}s", flush=True)

    start = time.time()
    codebook = learn_codebook(patches, config)
    codebook_sec = time.time() - start
    print(f"[stage] {variant}: codebook={len(codebook.codes)} in {codebook_sec:.1f}s", flush=True)

    start = time.time()
    if variant == "high_fidelity":
        program, compiled_state = compile_edge_cover_program(record, patches, codebook, config)
    else:
        program = compile_program(record, patches, codebook, config)
        compiled_state = None
    program_sec = time.time() - start
    print(f"[stage] {variant}: original program actions={len(program.actions)} in {program_sec:.1f}s", flush=True)

    original = program.to_tokens()

    start = time.time()
    compact_codec = CompactCodec(
        max_macros=int(compact_cfg.get("max_macros", 384)),
        min_macro_count=int(compact_cfg.get("min_macro_count", 3)),
        max_macro_len=int(compact_cfg.get("max_macro_len", 12)),
        max_bpe_merges=int(compact_cfg.get("max_bpe_merges", 384)),
        min_bpe_count=int(compact_cfg.get("min_bpe_count", 3)),
    ).fit([original])
    compact = compact_codec.encode(original)
    compact_vocab_size = max(2, len(set(compact)))
    entropy_model = EntropyModel().fit([compact], compact_codec.profile)
    compact_sec = time.time() - start
    print(f"[stage] {variant}: compact tokens={len(compact)} in {compact_sec:.1f}s", flush=True)

    start = time.time()
    motif_codec = MotifMacroCodec(
        codebook,
        max_structural_macros=int(motif_cfg.get("max_structural_macros", 384)),
        min_structural_count=int(motif_cfg.get("min_structural_count", 2)),
        max_structural_len=int(motif_cfg.get("max_structural_len", 24)),
        max_compact_macros=int(compact_cfg.get("max_macros", 384)),
        min_compact_macro_count=int(compact_cfg.get("min_macro_count", 3)),
        max_compact_macro_len=int(compact_cfg.get("max_macro_len", 12)),
        max_bpe_merges=int(compact_cfg.get("max_bpe_merges", 384)),
        min_bpe_count=int(compact_cfg.get("min_bpe_count", 3)),
        use_parameterized_macros=bool(motif_cfg.get("use_parameterized_macros", True)),
        min_parameterized_emit_count=int(motif_cfg.get("min_parameterized_emit_count", 3)),
        max_parameterized_span_len=int(motif_cfg.get("max_parameterized_span_len", 96)),
        max_merge_schemas=int(motif_cfg.get("max_merge_schemas", 256)),
        min_merge_schema_count=int(motif_cfg.get("min_merge_schema_count", 2)),
        max_code_schemas=int(motif_cfg.get("max_code_schemas", 256)),
        min_code_schema_count=int(motif_cfg.get("min_code_schema_count", 2)),
    ).fit([original])
    motif = motif_codec.encode(original)
    motif_vocab_size = max(2, len(set(motif)))
    motif_entropy_model = MotifEntropyModel().fit([motif], motif_codec.profile)
    hybrid = motif_codec.encode_hybrid(original, entropy_model, motif_entropy_model)
    hybrid_entropy_model = MotifEntropyModel().fit([hybrid], motif_codec.profile)
    motif_sec = time.time() - start
    print(f"[stage] {variant}: motif tokens={len(motif)}, hybrid={len(hybrid)} in {motif_sec:.1f}s", flush=True)

    fit_sec = patch_sec + codebook_sec + program_sec + compact_sec + motif_sec
    if save_artifact:
        write_json(
            out_dir / "final_artifact_summary.json",
            {
                "config": config,
                "codebook_size": len(codebook.codes),
                "compact_vocab_size": compact_vocab_size,
                "motif_vocab_size": motif_vocab_size,
                "patches": len(patches),
                "original_tokens": len(original),
            },
        )

    print(f"[variant] {variant}: execute original program", flush=True)
    start = time.time()
    if compiled_state is not None:
        recon = graph_to_record(compiled_state.graph, program.graph_id + "_recon", family="gptok_reconstruction")
        state = compiled_state
    else:
        recon, state = execute_program(program, codebook, config)
    execute_sec = time.time() - start
    fidelity = lightweight_fidelity(record, recon)

    rows = []
    for mode in MODES:
        print(f"[variant] {variant}: encode {mode}", flush=True)
        start = time.time()
        if mode == "original":
            tokens = original
            bits = original_program_bits(
                original,
                codebook_size=len(codebook.codes),
                ref_window=int(config.get("program", {}).get("ref_window", 768)),
            )
            expanded = original
        elif mode == "compact":
            tokens = compact_codec.encode(original)
            bits = compact_symbol_bits(tokens, compact_vocab_size)
            expanded = compact_codec.decode(tokens)
        elif mode == "entropy":
            tokens = compact_codec.encode(original)
            bits = entropy_model.bits(tokens, include_rulebook=False)
            expanded = compact_codec.decode(tokens)
        elif mode == "motif_macro":
            tokens = motif_codec.encode(original)
            bits = motif_symbol_bits(tokens, motif_vocab_size)
            expanded = motif_codec.decode(tokens)
        elif mode == "motif_entropy":
            tokens = motif_codec.encode(original)
            bits = motif_entropy_model.bits(tokens, include_rulebook=False)
            expanded = motif_codec.decode(tokens)
        elif mode == "motif_hybrid":
            tokens = motif_codec.encode_hybrid(original, entropy_model, motif_entropy_model)
            bits = hybrid_entropy_model.bits(tokens, include_rulebook=False)
            expanded = motif_codec.decode(tokens)
        else:
            raise AssertionError(mode)
        encode_sec = time.time() - start
        rows.append(
            {
                "dataset": "CORA_FULL_GRAPH",
                "variant": variant,
                "mode": mode,
                "num_nodes": record.num_nodes,
                "num_edges": len(edge_set(record)),
                "num_patches": len(patches),
                "codebook_size": len(codebook.codes),
                "original_tokens": len(original),
                "tokens": len(tokens),
                "token_reduction_vs_original": 1.0 - len(tokens) / max(1, len(original)),
                "bits": float(bits),
                "bits_per_edge": float(bits) / max(1, len(edge_set(record))),
                "lossless_expand_match": float(expanded == original),
                "fit_sec": fit_sec,
                "patch_sec": patch_sec,
                "codebook_sec": codebook_sec,
                "program_sec": program_sec,
                "compact_sec": compact_sec,
                "motif_sec": motif_sec,
                "compile_sec": program_sec,
                "execute_sec": execute_sec,
                "encode_sec": encode_sec,
                "executable": float(state.illegal_actions == 0 and state.runtime_errors == 0),
                "illegal_actions": state.illegal_actions,
                "runtime_errors": state.runtime_errors,
                **fidelity,
            }
        )
    status = {
        "dataset": "CORA_FULL_GRAPH",
        "variant": variant,
        "run_status": "loaded",
        "nodes": record.num_nodes,
        "edges": len(edge_set(record)),
        "patches": len(patches),
        "original_tokens": len(original),
        "fit_sec": fit_sec,
        "patch_sec": patch_sec,
        "codebook_sec": codebook_sec,
        "program_sec": program_sec,
        "compact_sec": compact_sec,
        "motif_sec": motif_sec,
        "compile_sec": program_sec,
        "execute_sec": execute_sec,
    }
    return rows, status


def compile_edge_cover_program(record: GraphRecord, patches, codebook, config: dict[str, Any]) -> tuple[GraphProgram, InterpreterState]:
    cfg = config.get("program", {})
    ref_window = int(cfg.get("ref_window", 20000))
    close_cycle_consumes_all = bool(cfg.get("close_cycle_consumes_all", False))
    block_size = max(1, int(cfg.get("block_size", 64)))
    state = InterpreterState()
    actions: list[Action] = []
    runtime_node_anchor: dict[int, list[int]] = {}

    def emit(action: Action) -> None:
        actions.append(action)
        execute_action(state, action, codebook, ref_window, close_cycle_consumes_all)

    def node_ref(candidates: list[int]) -> int | None:
        recent = state.recent_anchors(ref_window, "node", exclude=set(state.current_anchors))
        candidate_set = set(candidates)
        for ref, runtime_idx in enumerate(recent, start=1):
            if runtime_idx in candidate_set:
                return ref
        return None

    emit(Action("BEGIN_GRAPH"))
    ordered = sorted(patches, key=lambda p: (tuple(sorted(p.nodes)), p.patch_id))
    assignments = {p.patch_id: codebook.assign(p) for p in ordered}
    for idx, patch in enumerate(ordered):
        if idx % block_size == 0:
            if idx > 0:
                emit(Action("END_BLOCK"))
            emit(Action("BEGIN_BLOCK"))
        emit(Action("EMIT", (assignments[patch.patch_id],)))
        for local_idx, node in enumerate(patch.nodes):
            ref = node_ref(runtime_node_anchor.get(int(node), []))
            if ref is not None and local_idx < len(state.current_anchors):
                emit(Action("MERGE_NODE", (local_idx, ref)))
        for local_idx, node in enumerate(patch.nodes):
            if local_idx < len(state.current_anchors):
                runtime_node_anchor.setdefault(int(node), []).append(state.current_anchors[local_idx])
    if state.block_stack:
        emit(Action("END_BLOCK"))
    emit(Action("END_GRAPH"))
    emit(Action("STOP"))
    return GraphProgram(
        record.graph_id,
        actions,
        {
            "num_patches": len(ordered),
            "num_codes": len(set(assignments.values())),
            "program_version": "gptok_opg_v1_edge_cover_fast",
            "leakage_policy": "abstract_refs_only",
        },
    ), state


def lightweight_fidelity(original: GraphRecord, reconstructed: GraphRecord) -> dict[str, Any]:
    eo = edge_set(original)
    er = edge_set(reconstructed)
    raw_tp = len(eo & er)
    raw_fp = len(er - eo)
    raw_fn = len(eo - er)
    raw_precision = raw_tp / max(1, raw_tp + raw_fp)
    raw_recall = raw_tp / max(1, raw_tp + raw_fn)
    raw_f1 = 2 * raw_precision * raw_recall / max(1e-12, raw_precision + raw_recall)
    go = to_networkx(original).to_undirected()
    gr = to_networkx(reconstructed).to_undirected()
    same_counts = go.number_of_nodes() == gr.number_of_nodes() and go.number_of_edges() == gr.number_of_edges()
    same_degree = sorted(dict(go.degree()).values()) == sorted(dict(gr.degree()).values())
    try:
        wl_match = nx.weisfeiler_lehman_graph_hash(go) == nx.weisfeiler_lehman_graph_hash(gr)
    except Exception:
        wl_match = False
    return {
        "raw_edge_precision": raw_precision,
        "raw_edge_recall": raw_recall,
        "raw_edge_f1": raw_f1,
        "raw_edge_fp": raw_fp,
        "raw_edge_fn": raw_fn,
        "same_node_edge_counts": float(same_counts),
        "same_degree_sequence": float(same_degree),
        "wl_hash_match": float(bool(same_counts and same_degree and wl_match)),
        "original_components": nx.number_connected_components(go),
        "reconstructed_components": nx.number_connected_components(gr),
        "component_error": abs(nx.number_connected_components(go) - nx.number_connected_components(gr)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, row: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(row, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
