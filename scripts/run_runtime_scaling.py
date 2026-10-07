"""Measure canonicalization and end-to-end tokenizer timing on controlled graphs."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import networkx as nx

from gptok2.canonical import canonicalize
from gptok2.data.schema import graph_to_record
from gptok2.data.schema_spec import SchemaSpec
from gptok2_tokenizer import GPTok2Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RPS-GTok runtime scaling experiment.")
    parser.add_argument("--out", default="rebuttal/experiments/E2_runtime_scaling")
    parser.add_argument("--max-search-nodes", type=int, default=100_000)
    parser.add_argument("--include-tokenizer", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = SchemaSpec(name="runtime_topology", retain_node_type=False, retain_node_attr=False, max_search_nodes=args.max_search_nodes)
    rows = []
    for family, sizes in families().items():
        for n in sizes:
            graph = make_graph(family, n)
            record = graph_to_record(graph, f"{family}_{n}", family="runtime")
            t0 = time.perf_counter()
            canonical = canonicalize(record, spec, exact=True, max_search_nodes=args.max_search_nodes)
            canonical_sec = time.perf_counter() - t0
            fit_sec = 0.0
            encode_sec = 0.0
            token_count = 0
            if args.include_tokenizer:
                tokenizer = GPTok2Tokenizer({"schema": spec.to_dict(), "canonicalization": {"enabled": False}})
                t1 = time.perf_counter()
                tokenizer.fit([canonical.record])
                fit_sec = time.perf_counter() - t1
                t2 = time.perf_counter()
                encoded = tokenizer.encode(canonical.record, mode="original")
                encode_sec = time.perf_counter() - t2
                token_count = encoded.token_count
            rows.append(
                {
                    "family": family,
                    "nodes": n,
                    "edges": graph.number_of_edges(),
                    "canonicalization_sec": canonical_sec,
                    "fit_sec": fit_sec,
                    "encode_sec": encode_sec,
                    "total_sec": canonical_sec + fit_sec + encode_sec,
                    "canonical_fraction": canonical_sec / max(1e-12, canonical_sec + fit_sec + encode_sec),
                    "search_nodes": canonical.search_nodes,
                    "search_branches": canonical.search_branches,
                    "exact_completed": canonical.exact_completed,
                    "fallback_used": canonical.fallback_used,
                    "tie_group_count": len(canonical.tie_groups),
                    "max_tie_group": max((len(x) for x in canonical.tie_groups), default=1),
                    "token_count": token_count,
                }
            )
    write_csv(out / "scaling.csv", rows)
    print(f"wrote {len(rows)} rows to {out / 'scaling.csv'}")


def families() -> dict[str, list[int]]:
    return {
        "path": [8, 16, 32, 64, 128],
        "cycle": [8, 16, 24, 32],
        "regular": [8, 10, 12, 14],
        "complete": [4, 5, 6, 7, 8],
        "sparse_random": [8, 16, 32, 64, 128],
    }


def make_graph(family: str, n: int) -> nx.Graph:
    if family == "path":
        return nx.path_graph(n)
    if family == "cycle":
        return nx.cycle_graph(n)
    if family == "regular":
        degree = min(3, max(1, n - 1))
        if (n * degree) % 2:
            degree = max(1, degree - 1)
        return nx.random_regular_graph(degree, n, seed=n)
    if family == "complete":
        return nx.complete_graph(n)
    return nx.gnp_random_graph(n, min(0.2, 4.0 / max(1, n)), seed=n)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
