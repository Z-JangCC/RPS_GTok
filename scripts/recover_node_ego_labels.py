"""Recover Planetoid/OGBN ego labels by topology-hash matching.

The ego materializer can be regenerated from cached full graphs, but seed
ordering can differ after a PyG cache refresh.  We therefore match each
materialized ego to a source ego by node/edge counts, degree multiset and a
Weisfeiler--Lehman hash before copying the seed-node label.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
import torch


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT.parent / "graph_tokenizer_v2_base/data/kdd2027/formal_full_real_materialized"


def _source_egos(dataset: str):
    if dataset in {"cora", "citeseer", "pubmed"}:
        name = {"cora": "Cora", "citeseer": "Citeseer", "pubmed": "Pubmed"}[dataset]
        path = CACHE / "_pyg_source_cache/pyg_planetoid" / name / "processed/data.pt"
        loaded = torch.load(path, weights_only=False); data = loaded[0] if isinstance(loaded, tuple) else loaded
    elif dataset == "ogbn_arxiv":
        path = CACHE / "_ogb_source_cache/ogb_node/ogbn_arxiv/processed/geometric_data_processed.pt"
        loaded = torch.load(path, weights_only=False); data = loaded[0] if isinstance(loaded, tuple) else loaded
    else:
        raise ValueError(dataset)
    n = int(data["x"].shape[0]); adj = [set() for _ in range(n)]
    for u, v in data["edge_index"].t().tolist():
        adj[int(u)].add(int(v)); adj[int(v)].add(int(u))
    seeds = sorted(range(n), key=lambda u: (-len(adj[u]), u))
    out = []
    for seed in seeds[:60]:
        nodes = {seed}; ordered = [seed]; frontier = [seed]
        for _ in range(2):
            nxt = []
            for node in frontier:
                nbrs = sorted(adj[node], key=lambda v: (-len(adj[v]), v)) if dataset != "ogbn_arxiv" else sorted(adj[node])
                for nbr in nbrs:
                    if nbr not in nodes:
                        nodes.add(nbr); ordered.append(nbr); nxt.append(nbr)
                    if len(ordered) >= 64: break
                if len(ordered) >= 64: break
            frontier = nxt
        ordered = ordered[:64]; graph = nx.Graph(); graph.add_nodes_from(range(len(ordered)))
        inv = {node: i for i, node in enumerate(ordered)}
        for u in ordered:
            for v in adj[u]:
                if v in inv and inv[u] < inv[v]: graph.add_edge(inv[u], inv[v])
        key = (graph.number_of_nodes(), graph.number_of_edges(), tuple(sorted(dict(graph.degree()).values())), nx.weisfeiler_lehman_graph_hash(graph))
        out.append((key, int(data["y"][seed].reshape(-1)[0]), seed))
    return out


def _row_key(row: dict):
    graph = nx.Graph(); graph.add_nodes_from(range(int(row["num_nodes"])))
    graph.add_edges_from((int(u), int(v)) for u, v in row.get("edges", []))
    return (graph.number_of_nodes(), graph.number_of_edges(), tuple(sorted(dict(graph.degree()).values())), nx.weisfeiler_lehman_graph_hash(graph))


def main() -> None:
    out_dir = ROOT / "rebuttal/data_21_labeled"
    audit = []
    for dataset in ["cora", "citeseer", "pubmed", "ogbn_arxiv"]:
        source = ROOT / f"rebuttal/data_21/{dataset}.jsonl"
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        candidates = defaultdict(list)
        for key, label, seed in _source_egos(dataset): candidates[key].append((label, seed))
        matched = 0; ambiguous = 0; labeled = []
        for row in rows:
            hits = candidates.get(_row_key(row), [])
            if len(hits) == 1:
                row = dict(row); row["y"] = hits[0][0]
                row.setdefault("metadata", {})["label_source"] = f"cached_{dataset}_seed_node_label"
                row["metadata"]["label_recovery"] = "unique_ego_wl_hash_join"
                matched += 1
            elif len(hits) > 1:
                ambiguous += 1
            labeled.append(row)
        status = "recovered" if matched == len(rows) else "ambiguous_or_unmatched"
        if status == "recovered":
            (out_dir / source.name).write_text("\n".join(json.dumps(r, sort_keys=True) for r in labeled) + "\n", encoding="utf-8")
        audit.append({"dataset": dataset, "records": len(rows), "matched": matched, "ambiguous": ambiguous, "status": status})
    (ROOT / "rebuttal/audits/node_ego_label_recovery.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
