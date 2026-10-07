"""High-level API for GPTok2 original, compact, entropy, and motif-macro modes."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from gptok2.data.schema import GraphRecord, edge_set, graph_to_record
from gptok2.data.schema_spec import SchemaSpec
from gptok2.canonical import canonicalize
from gptok2.metrics.evaluate import structure_fidelity
from gptok2.patches.proposal import patches_for_records
from gptok2.program.actions import GraphProgram, parse_token
from gptok2.program.compiler import compile_program
from gptok2.program.interpreter import execute_program
from gptok2.vq.codebook import PrototypeVQCodebook, learn_codebook
from gptok2_compact import CompactCodec, CompactProfile, EntropyModel
from gptok2_compact.codec import compact_symbol_bits, original_program_bits
from gptok2_motif_macro import MotifEntropyModel
from gptok2_motif_macro import MotifMacroCodec
from gptok2_motif_macro import MotifMacroProfile
from gptok2_motif_macro import motif_symbol_bits
from gptok2_tokenizer.payload import (
    complete_stream_tokens,
    footprint_stats,
    serialize_complete_stream,
    token_instances,
    validate_complete_stream_tokens,
)


DEFAULT_CONFIG: dict[str, Any] = {
    "schema": {
        "name": "topology",
        "directed": False,
        "multiedge": False,
        "self_loop": False,
        "retain_node_type": False,
        "retain_edge_type": False,
        "retain_node_attr": False,
        "retain_edge_attr": False,
        "continuous_mode": "raw",
        "max_search_nodes": 100000,
    },
    "canonicalization": {
        "enabled": True,
        "exact": True,
        "max_search_nodes": 100000,
    },
    "patch": {
        "min_size": 2,
        "max_size": 9,
        "max_ports": 8,
        "max_patches_per_graph": 384,
        "overlap_budget": 2,
        "learned_port_subtypes": 16,
        "max_port_capacity": 12,
        "grow_rounds": 6,
        "max_cycle_signatures": 64,
        "max_triangle_signatures": 256,
        "sparse_edge_patches": False,
        "lambda_out": 0.25,
        "lambda_ports": 0.20,
        "lambda_cut": 0.20,
        "lambda_motif": 1.20,
    },
    "vq": {
        "num_codes": 4096,
        "fit_scope": "train",
        "min_usage": 1,
        "dead_code_reinit": True,
        "merge_near_duplicates": True,
        "factorize_interfaces": True,
    },
    "program": {
        "ref_window": 768,
        "begin_blocks": True,
        "block_size": 12,
        "max_global_links": 256,
        "allow_nonexact_merges": False,
        "fallback_nonexact_large_patches": True,
        "fallback_nonexact_min_nodes": 3,
        "close_cycle_consumes_all": False,
        "disable_merge_edge": True,
        "global_link_all_cross_edges": True,
        "emit_interface_actions": False,
    },
    "compact_entropy": {
        "max_macros": 384,
        "min_macro_count": 3,
        "max_macro_len": 12,
        "max_bpe_merges": 384,
        "min_bpe_count": 3,
    },
    "motif_macro": {
        "use_parameterized_macros": True,
        "min_parameterized_emit_count": 3,
        "max_parameterized_span_len": 96,
        "max_merge_schemas": 256,
        "min_merge_schema_count": 2,
        "max_code_schemas": 256,
        "min_code_schema_count": 2,
        "max_structural_macros": 384,
        "min_structural_count": 2,
        "max_structural_len": 24,
    },
}

_ENCODE_MODES = {
    "compact",
    "entropy",
    "motif_macro",
    "motif_entropy",
    "motif_hybrid",
}


@dataclass
class EncodedProgram:
    graph_id: str
    mode: str
    tokens: list[str]
    bits: float
    bits_per_edge: float
    original_token_count: int
    token_count: int
    lossless_expand_match: float
    identifier_bytes: float = 0.0
    payload_bytes: float = 0.0
    complete_token_bytes: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)
    token_instances_data: list[dict[str, Any]] = field(default_factory=list)
    program_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        stats = footprint_stats(self.tokens)
        self.identifier_bytes = float(stats["identifier_bytes"])
        graph_payload_bytes = len(
            json.dumps(self.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        self.payload_bytes = float(stats["payload_bytes"] + graph_payload_bytes)
        self.complete_token_bytes = float(
            len(serialize_complete_stream(self.tokens, self.payload))
        )
        self.token_instances_data = [item.to_dict() for item in token_instances(self.tokens)]

    @property
    def identifier_tokens(self) -> list[str]:
        return [item.identifier for item in token_instances(self.tokens)]

    @property
    def complete_stream_tokens(self) -> list[str]:
        """Self-contained identifier+payload stream for generation controls."""
        return complete_stream_tokens(self.tokens, self.payload)

    def validate_complete_stream(self) -> dict[str, Any]:
        return validate_complete_stream_tokens(self.complete_stream_tokens)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GPTok2Tokenizer:
    """Fit and run GPTok2 original/compact/entropy/motif-macro tokenizers.

    All compressed modes are reversible coding layers over the original GPTok2
    graph program. Decoding always expands back to the original GPTok2 action
    sequence and then uses the standard GPTok2 interpreter.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = _deep_update(DEFAULT_CONFIG, config or {})
        self.codebook: PrototypeVQCodebook | None = None
        self.compact_codec: CompactCodec | None = None
        self.entropy_model: EntropyModel | None = None
        self.motif_codec: MotifMacroCodec | None = None
        self.motif_entropy_model: MotifEntropyModel | None = None
        self.motif_hybrid_entropy_model: MotifEntropyModel | None = None
        self.compact_vocab_size: int = 2
        self.motif_vocab_size: int = 2
        self.motif_hybrid_vocab_size: int = 2
        self._fit_patch_map: dict[str, list] = {}

    @property
    def fitted(self) -> bool:
        return (
            self.codebook is not None
            and self.compact_codec is not None
            and self.entropy_model is not None
            and self.motif_codec is not None
            and self.motif_entropy_model is not None
            and self.motif_hybrid_entropy_model is not None
        )

    def fit(self, graphs: Iterable[GraphRecord | nx.Graph]) -> "GPTok2Tokenizer":
        records = [
            self._canonical_record(_as_record(g, f"graph_{i:05d}"))
            for i, g in enumerate(graphs)
        ]
        if not records:
            raise ValueError("GPTok2Tokenizer.fit requires at least one graph.")
        self._fit_patch_map = patches_for_records(records, self.config)
        train_patches = [p for patches in self._fit_patch_map.values() for p in patches]
        self.codebook = learn_codebook(train_patches, self.config)
        programs = [compile_program(r, self._fit_patch_map[r.graph_id], self.codebook, self.config) for r in records]
        compact_cfg = self.config.get("compact_entropy", {})
        self.compact_codec = CompactCodec(
            max_macros=int(compact_cfg.get("max_macros", 384)),
            min_macro_count=int(compact_cfg.get("min_macro_count", 3)),
            max_macro_len=int(compact_cfg.get("max_macro_len", 12)),
            max_bpe_merges=int(compact_cfg.get("max_bpe_merges", 384)),
            min_bpe_count=int(compact_cfg.get("min_bpe_count", 3)),
        ).fit([p.to_tokens() for p in programs])
        compact_sequences = [self.compact_codec.encode(p.to_tokens()) for p in programs]
        self.compact_vocab_size = max(2, len({tok for seq in compact_sequences for tok in seq}))
        self.entropy_model = EntropyModel().fit(compact_sequences, self.compact_codec.profile)
        motif_cfg = self.config.get("motif_macro", {})
        self.motif_codec = MotifMacroCodec(
            self.codebook,
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
        ).fit([p.to_tokens() for p in programs])
        motif_sequences = [self.motif_codec.encode(p.to_tokens()) for p in programs]
        self.motif_vocab_size = max(2, len({tok for seq in motif_sequences for tok in seq}))
        self.motif_entropy_model = MotifEntropyModel().fit(motif_sequences, self.motif_codec.profile)
        motif_hybrid_sequences = [
            self.motif_codec.encode_hybrid(p.to_tokens(), self.entropy_model, self.motif_entropy_model)
            for p in programs
        ]
        self.motif_hybrid_vocab_size = max(2, len({tok for seq in motif_hybrid_sequences for tok in seq}))
        self.motif_hybrid_entropy_model = MotifEntropyModel().fit(motif_hybrid_sequences, self.motif_codec.profile)
        self.codebook.freeze()
        return self

    def encode(self, graph: GraphRecord | nx.Graph, mode: str = "original") -> EncodedProgram:
        self._require_fit()
        record = self._canonical_record(_as_record(graph, "graph"))
        program = self._compile(record)
        original_tokens = program.to_tokens()
        edges = max(1, len(edge_set(record)))
        if mode == "original":
            bits = original_program_bits(
                original_tokens,
                codebook_size=len(self.codebook),  # type: ignore[arg-type]
                ref_window=int(self.config.get("program", {}).get("ref_window", 768)),
            )
            return EncodedProgram(
                record.graph_id, mode, original_tokens, bits, bits / edges,
                len(original_tokens), len(original_tokens), 1.0,
                payload=_record_payload(record),
                program_metadata=dict(program.metadata or {}),
            )
        if mode not in _ENCODE_MODES:
            raise ValueError("mode must be one of: " + ", ".join(sorted(_ENCODE_MODES | {"original"})))
        assert self.compact_codec is not None and self.entropy_model is not None
        assert self.motif_codec is not None and self.motif_entropy_model is not None and self.motif_hybrid_entropy_model is not None
        if mode in {"motif_macro", "motif_entropy", "motif_hybrid"}:
            if mode == "motif_hybrid":
                motif_tokens = self.motif_codec.encode_hybrid(original_tokens, self.entropy_model, self.motif_entropy_model)
                bits = self.motif_hybrid_entropy_model.bits(motif_tokens, include_rulebook=False)
            else:
                motif_tokens = self.motif_codec.encode(original_tokens)
                bits = (
                    motif_symbol_bits(motif_tokens, self.motif_vocab_size)
                    if mode == "motif_macro"
                    else self.motif_entropy_model.bits(motif_tokens, include_rulebook=False)
                )
            expanded = self.motif_codec.decode(motif_tokens)
            lossless = float(expanded == original_tokens)
            return EncodedProgram(
                record.graph_id, mode, motif_tokens, bits, bits / edges,
                len(original_tokens), len(motif_tokens), lossless,
                payload=_record_payload(record),
                program_metadata=dict(program.metadata or {}),
            )
        compact_tokens = self.compact_codec.encode(original_tokens)
        expanded = self.compact_codec.decode(compact_tokens)
        lossless = float(expanded == original_tokens)
        if mode == "compact":
            bits = compact_symbol_bits(compact_tokens, self.compact_vocab_size)
        else:
            bits = self.entropy_model.bits(compact_tokens, include_rulebook=False)
        return EncodedProgram(
            record.graph_id, mode, compact_tokens, bits, bits / edges,
            len(original_tokens), len(compact_tokens), lossless,
            payload=_record_payload(record),
            program_metadata=dict(program.metadata or {}),
        )

    def decode(self, encoded: EncodedProgram | dict[str, Any] | list[str], mode: str | None = None) -> GraphRecord:
        self._require_fit()
        if isinstance(encoded, EncodedProgram):
            tokens, mode = encoded.tokens, encoded.mode
            graph_id = encoded.graph_id
            payload = dict(encoded.payload)
        elif isinstance(encoded, dict):
            tokens = list(encoded["tokens"])
            mode = str(mode or encoded.get("mode", "original"))
            graph_id = str(encoded.get("graph_id", "decoded"))
            payload = dict(encoded.get("payload", {}))
        else:
            tokens = list(encoded)
            mode = str(mode or "original")
            graph_id = "decoded"
            payload = {}
        if mode in {"motif_macro", "motif_entropy", "motif_hybrid"}:
            assert self.motif_codec is not None
            tokens = self.motif_codec.decode(tokens)
        elif mode in {"compact", "entropy"}:
            assert self.compact_codec is not None
            tokens = self.compact_codec.decode(tokens)
        program = GraphProgram(graph_id, [parse_token(t) for t in tokens])
        recon, _ = execute_program(program, self.codebook, self.config)  # type: ignore[arg-type]
        _apply_record_payload(recon, payload)
        return recon

    def evaluate_reconstruction(self, graph: GraphRecord | nx.Graph, mode: str = "compact") -> dict[str, Any]:
        record = _as_record(graph, "graph")
        encoded = self.encode(record, mode=mode)
        recon = self.decode(encoded)
        metrics = structure_fidelity(record, recon)
        return {**encoded.to_dict(), **metrics}

    def save(self, path: str | Path) -> None:
        self._require_fit()
        assert self.codebook is not None and self.compact_codec is not None and self.entropy_model is not None
        assert self.motif_codec is not None and self.motif_entropy_model is not None and self.motif_hybrid_entropy_model is not None
        artifact = {
            "format": "rps_gtok_final_artifact_v1",
            "config": self.config,
            "codebook": self.codebook.to_dict(),
            "compact_profile": self.compact_codec.profile.to_dict(),
            "entropy_model": self.entropy_model.to_dict(),
            "compact_vocab_size": self.compact_vocab_size,
            "motif_profile": self.motif_codec.profile.to_dict(),
            "motif_entropy_model": self.motif_entropy_model.to_dict(),
            "motif_vocab_size": self.motif_vocab_size,
            "motif_hybrid_entropy_model": self.motif_hybrid_entropy_model.to_dict(),
            "motif_hybrid_vocab_size": self.motif_hybrid_vocab_size,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "GPTok2Tokenizer":
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = GPTok2Tokenizer(artifact.get("config", {}))
        tok.codebook = PrototypeVQCodebook.from_dict(artifact["codebook"])
        tok.compact_codec = CompactCodec()
        tok.compact_codec.profile = CompactProfile.from_dict(artifact["compact_profile"])
        tok.entropy_model = _entropy_from_dict(artifact["entropy_model"])
        tok.compact_vocab_size = int(artifact.get("compact_vocab_size", 2))
        tok.motif_codec = MotifMacroCodec(tok.codebook)
        tok.motif_codec.profile = MotifMacroProfile.from_dict(artifact["motif_profile"])
        tok.motif_entropy_model = _motif_entropy_from_dict(
            artifact["motif_entropy_model"],
            tok.motif_codec.profile,
            MotifEntropyModel,
        )
        tok.motif_vocab_size = int(artifact.get("motif_vocab_size", 2))
        tok.motif_hybrid_entropy_model = _motif_entropy_from_dict(
            artifact["motif_hybrid_entropy_model"],
            tok.motif_codec.profile,
            MotifEntropyModel,
        )
        tok.motif_hybrid_vocab_size = int(artifact.get("motif_hybrid_vocab_size", tok.motif_vocab_size))
        tok.codebook.freeze()
        return tok

    def _compile(self, record: GraphRecord) -> GraphProgram:
        assert self.codebook is not None
        patches = self._fit_patch_map.get(record.graph_id)
        if patches is None:
            patches = patches_for_records([record], self.config)[record.graph_id]
        return compile_program(record, patches, self.codebook, self.config)

    def _canonical_record(self, record: GraphRecord) -> GraphRecord:
        cfg = self.config.get("canonicalization", {})
        if not bool(cfg.get("enabled", True)):
            return record
        spec = SchemaSpec.from_config(self.config)
        result = canonicalize(
            record,
            spec,
            exact=bool(cfg.get("exact", True)),
            max_search_nodes=int(cfg.get("max_search_nodes", spec.max_search_nodes)),
            timeout_sec=cfg.get("timeout_sec"),
        )
        canonical = result.record
        canonical.metadata = {
            **canonical.metadata,
            "canonicalization": result.to_dict(),
            "reconstruction_target": spec.reconstruction_target,
        }
        return canonical

    def _require_fit(self) -> None:
        if not self.fitted:
            raise RuntimeError("GPTok2Tokenizer is not fitted. Call fit() or load() first.")

    def artifact_footprint_bytes(self) -> int:
        """Return deterministic serialized vocabulary/rulebook footprint."""
        self._require_fit()
        assert self.codebook is not None
        assert self.compact_codec is not None
        assert self.entropy_model is not None
        assert self.motif_codec is not None
        assert self.motif_entropy_model is not None
        assert self.motif_hybrid_entropy_model is not None
        artifact = {
            "config": self.config,
            "codebook": self.codebook.to_dict(),
            "compact_profile": self.compact_codec.profile.to_dict(),
            "entropy_model": self.entropy_model.to_dict(),
            "motif_profile": self.motif_codec.profile.to_dict(),
            "motif_entropy_model": self.motif_entropy_model.to_dict(),
            "motif_hybrid_entropy_model": self.motif_hybrid_entropy_model.to_dict(),
        }
        return len(
            json.dumps(
                artifact,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )


def _as_record(graph: GraphRecord | nx.Graph, fallback_id: str) -> GraphRecord:
    if isinstance(graph, GraphRecord):
        return graph
    if isinstance(graph, nx.Graph):
        gid = str(graph.graph.get("graph_id", fallback_id))
        return graph_to_record(graph, gid)
    raise TypeError(f"Unsupported graph object: {type(graph)!r}")


def _record_payload(record: GraphRecord) -> dict[str, Any]:
    def tensor_value(value):
        return value.detach().cpu().tolist() if hasattr(value, "detach") else value

    return {
        "num_nodes": int(record.num_nodes),
        "node_type": tensor_value(record.node_type),
        "edge_type": tensor_value(record.edge_type),
        "node_attr": tensor_value(record.node_attr),
        "edge_attr": tensor_value(record.edge_attr),
        "directed": bool(record.directed),
        "metadata": dict(record.metadata),
    }


def _apply_record_payload(record: GraphRecord, payload: dict[str, Any]) -> None:
    if not payload:
        return
    import torch

    if payload.get("node_type") is not None and len(payload["node_type"]) == record.num_nodes:
        record.node_type = torch.tensor(payload["node_type"], dtype=torch.long)
    if payload.get("node_attr") is not None and len(payload["node_attr"]) == record.num_nodes:
        record.node_attr = torch.tensor(payload["node_attr"], dtype=torch.float32)
    edge_count = int(record.edge_index.shape[1]) if record.edge_index.ndim == 2 else 0
    if payload.get("edge_type") is not None and len(payload["edge_type"]) == edge_count:
        record.edge_type = torch.tensor(payload["edge_type"], dtype=torch.long)
    if payload.get("edge_attr") is not None and len(payload["edge_attr"]) == edge_count:
        record.edge_attr = torch.tensor(payload["edge_attr"], dtype=torch.float32)
    if payload.get("metadata"):
        record.metadata.update(dict(payload["metadata"]))


def _entropy_from_dict(row: dict[str, Any]) -> EntropyModel:
    model = EntropyModel()
    model.token_counts = Counter({str(k): int(v) for k, v in row.get("token_counts", {}).items()})
    model.op_counts = Counter({str(k): int(v) for k, v in row.get("op_counts", {}).items()})
    model.symbol_counts = Counter({str(k): int(v) for k, v in row.get("symbol_counts", {}).items()})
    model.code_counts = Counter({str(k): int(v) for k, v in row.get("code_counts", {}).items()})
    model.local_counts = Counter({str(k): int(v) for k, v in row.get("local_counts", {}).items()})
    model.ref_delta_counts = Counter({str(k): int(v) for k, v in row.get("ref_delta_counts", {}).items()})
    model.count_counts = Counter({str(k): int(v) for k, v in row.get("count_counts", {}).items()})
    model.num_sequences = int(row.get("num_sequences", 0))
    model.rulebook_bits = float(row.get("rulebook_bits", 0.0))
    return model


def _motif_entropy_from_dict(row: dict[str, Any], profile: Any, model_cls: Any) -> Any:
    model = model_cls()
    model.token_counts = Counter({str(k): int(v) for k, v in row.get("token_counts", {}).items()})
    model.op_counts = Counter({str(k): int(v) for k, v in row.get("op_counts", {}).items()})
    model.category_counts = Counter({str(k): int(v) for k, v in row.get("category_counts", {}).items()})
    model.payload_counts = Counter({str(k): int(v) for k, v in row.get("payload_counts", {}).items()})
    model.length_counts = Counter({str(k): int(v) for k, v in row.get("length_counts", {}).items()})
    model.code_counts = Counter({str(k): int(v) for k, v in row.get("code_counts", {}).items()})
    model.code_schema_counts = Counter({str(k): int(v) for k, v in row.get("code_schema_counts", {}).items()})
    model.schema_counts = Counter({str(k): int(v) for k, v in row.get("schema_counts", {}).items()})
    model.glue_op_counts = Counter({str(k): int(v) for k, v in row.get("glue_op_counts", {}).items()})
    model.merge_family_counts = Counter({str(k): int(v) for k, v in row.get("merge_family_counts", {}).items()})
    model.merge_pair_count_counts = Counter({str(k): int(v) for k, v in row.get("merge_pair_count_counts", {}).items()})
    model.merge_local_start_counts = Counter({str(k): int(v) for k, v in row.get("merge_local_start_counts", {}).items()})
    model.merge_local_step_counts = Counter({str(k): int(v) for k, v in row.get("merge_local_step_counts", {}).items()})
    model.merge_ref_start_delta_counts = Counter({str(k): int(v) for k, v in row.get("merge_ref_start_delta_counts", {}).items()})
    model.merge_ref_step_counts = Counter({str(k): int(v) for k, v in row.get("merge_ref_step_counts", {}).items()})
    model.local_counts = Counter({str(k): int(v) for k, v in row.get("local_counts", {}).items()})
    model.ref_delta_counts = Counter({str(k): int(v) for k, v in row.get("ref_delta_counts", {}).items()})
    model.close_counts = Counter({str(k): int(v) for k, v in row.get("close_counts", {}).items()})
    model.merge_schema_rules = dict(profile.merge_schema_rules)
    model.code_schema_rules = dict(getattr(profile, "code_schema_rules", {}))
    model.num_sequences = int(row.get("num_sequences", 0))
    model.rulebook_bits = float(row.get("rulebook_bits", 0.0))
    return model


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out
