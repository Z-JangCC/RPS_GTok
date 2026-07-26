from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx

COMPONENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMPONENT_ROOT))

from gptok2_tokenizer import GPTok2Tokenizer  # noqa: E402


def test_original_compact_entropy_motif_roundtrip(tmp_path):
    graphs = [nx.cycle_graph(6), nx.star_graph(5), nx.path_graph(7), nx.complete_graph(4)]
    for i, graph in enumerate(graphs):
        graph.graph["graph_id"] = f"smoke_{i}"
    tokenizer = GPTok2Tokenizer().fit(graphs)
    artifact = tmp_path / "tokenizer.json"
    tokenizer.save(artifact)
    loaded = GPTok2Tokenizer.load(artifact)
    modes = [
        "original",
        "compact",
        "entropy",
        "motif_macro",
        "motif_entropy",
        "motif_hybrid",
    ]
    for mode in modes:
        encoded = loaded.encode(graphs[0], mode=mode)
        assert encoded.tokens
        assert encoded.bits_per_edge > 0
        if mode != "original":
            assert encoded.lossless_expand_match == 1.0
            assert encoded.token_count <= encoded.original_token_count
        metrics = loaded.evaluate_reconstruction(graphs[0], mode=mode)
        assert metrics["edge_f1"] >= 0.99
        assert metrics["exact_reconstruction"] == 1.0
