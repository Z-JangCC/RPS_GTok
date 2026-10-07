from __future__ import annotations

import random

import networkx as nx
import torch

from gptok2.canonical import canonicalize
from gptok2.data.schema import GraphRecord, graph_to_record, record_from_dict, record_to_dict, to_networkx
from gptok2.data.schema_spec import SchemaSpec
from gptok2_tokenizer.payload import (
    complete_stream_tokens,
    complete_stream_to_tokens,
    deserialize_complete_stream,
    footprint_stats,
    serialize_complete_stream,
    split_token,
    validate_complete_stream_tokens,
)
from gptok2_tokenizer import GPTok2Tokenizer


def test_exact_canonicalization_is_invariant_to_node_permutation() -> None:
    graph = nx.cycle_graph(6)
    spec = SchemaSpec(name="topology", retain_node_type=False, retain_node_attr=False)
    base = canonicalize(graph_to_record(graph, "g"), spec)
    for seed in range(5):
        nodes = list(graph.nodes())
        random.Random(seed).shuffle(nodes)
        permuted = nx.relabel_nodes(graph, {old: new for new, old in enumerate(nodes)}, copy=True)
        result = canonicalize(graph_to_record(permuted, "g"), spec)
        assert result.exact_completed
        assert result.canonical_key == base.canonical_key


def test_payload_and_identifier_are_separate() -> None:
    item = split_token("ATTACH(3,7)")
    assert item.identifier == "ATTACH"
    assert item.payload == ("3", "7")
    stats = footprint_stats(["BEGIN_GRAPH", "EMIT(4)", "ATTACH(3,7)", "END_GRAPH"])
    assert stats["identifier_token_count"] == 4
    assert stats["payload_bytes"] > 0
    assert stats["complete_token_bytes"] >= stats["payload_bytes"]


def test_schema_round_trip_preserves_loops_parallel_edges_and_attributes() -> None:
    record = GraphRecord(
        "edge_case",
        3,
        torch.tensor([[0, 0, 1], [0, 1, 2]], dtype=torch.long),
        node_type=torch.tensor([1, 2, 3], dtype=torch.long),
        edge_type=torch.tensor([4, 5, 6], dtype=torch.long),
        node_attr=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_attr=torch.tensor([[0.1], [0.2], [0.3]]),
        directed=True,
    )
    restored = record_from_dict(record_to_dict(record))
    assert restored.edge_index.tolist() == record.edge_index.tolist()
    assert torch.equal(restored.edge_type, record.edge_type)
    assert torch.equal(restored.node_attr, record.node_attr)
    assert torch.equal(restored.edge_attr, record.edge_attr)
    graph = to_networkx(restored)
    assert graph.is_directed()
    assert graph.number_of_edges() == 3
    assert graph.has_edge(0, 0)


def test_encoded_program_contains_complete_graph_payload() -> None:
    record = graph_to_record(nx.path_graph(4), "payload")
    tokenizer = GPTok2Tokenizer().fit([record])
    encoded = tokenizer.encode(record, mode="original")
    assert encoded.payload["num_nodes"] == 4
    assert encoded.payload_bytes > 0
    assert encoded.complete_token_bytes >= encoded.payload_bytes


def test_self_contained_payload_stream_is_deterministic() -> None:
    blob_a = serialize_complete_stream(["EMIT(1)", "ATTACH(0,2)"], {"num_nodes": 2})
    blob_b = serialize_complete_stream(["EMIT(1)", "ATTACH(0,2)"], {"num_nodes": 2})
    assert blob_a == blob_b
    assert deserialize_complete_stream(blob_a)["graph_payload"]["num_nodes"] == 2


def test_self_contained_generation_projection_contains_payload_and_validates() -> None:
    stream = complete_stream_tokens(["EMIT(1)", "ATTACH(0,2)"], {"num_nodes": 2})
    audit = validate_complete_stream_tokens(stream)
    assert audit["valid"]
    assert audit["payload_included"]
    tokens, payload = complete_stream_to_tokens(stream)
    assert tokens == ["EMIT(1)", "ATTACH(0,2)"]
    assert payload["num_nodes"] == 2
