"""Minimal Python API example for the standalone GPTok2 tokenizer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gptok2_tokenizer import GPTok2Tokenizer


def main() -> None:
    graphs = [
        nx.cycle_graph(6),
        nx.star_graph(5),
        nx.path_graph(8),
        nx.complete_graph(5),
    ]
    for i, graph in enumerate(graphs):
        graph.graph["graph_id"] = f"example_{i}"
    tokenizer = GPTok2Tokenizer().fit(graphs)
    out_dir = Path("runs/rps_gtok_example")
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")
    rows = []
    for graph in graphs:
        for mode in ["original", "compact", "entropy"]:
            rows.append(tokenizer.evaluate_reconstruction(graph, mode=mode))
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
