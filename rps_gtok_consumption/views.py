"""Token-view builders used by downstream RPS-GTok consumer experiments."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx

from gptok2.data.schema import GraphRecord, edge_set, to_networkx
from gptok2.data.schema_spec import SchemaSpec
from gptok2.canonical import canonical_record
from gptok2_tokenizer import GPTok2Tokenizer
from gptok2_tokenizer.payload import split_token


RPS_MODE_ALIASES = {
    "rps_gtok_full": "motif_hybrid",
    "rps_gtok_motif_hybrid": "motif_hybrid",
    "rps_gtok_motif_macro": "motif_macro",
    "rps_gtok_compact": "compact",
    "atomic_program": "original",
    "primitive_program": "original",
    "original": "original",
    "compact": "compact",
    "entropy": "entropy",
    "motif_macro": "motif_macro",
    "motif_entropy": "motif_entropy",
    "motif_hybrid": "motif_hybrid",
}

# These aliases are deliberately explicit: the ``plus`` representation is not
# an untracked post-hoc feature injection.  It is a formal RPS view consisting
# of the executable program followed by invariant graph anchors.  Matching
# ``*_plus`` controls are provided below so the anchor contribution can be
# measured separately from the tokenization contribution.
STRUCTURAL_PLUS_VIEWS = {
    "rps_gtok_plus": "rps_gtok_full",
    "rps_gtok_plus_wl": "rps_gtok_full",
    "edge_list_plus": "edge_list",
    "canonical_edge_list_plus": "canonical_edge_list",
}

BPE_BASE_VIEWS = {
    "edge_list_bpe": "edge_list",
    "adjacency_list_bpe": "adjacency_list",
    "dfs_order_bpe": "dfs_order",
    "bfs_order_bpe": "bfs_order",
    "graph_tokenizer_feuler_bpe": "graph_tokenizer_feuler",
    "canonical_edge_list_bpe": "canonical_edge_list",
    "canonical_adjacency_list_bpe": "canonical_adjacency_list",
    "canonical_dfs_order_bpe": "canonical_dfs_order",
    "canonical_bfs_order_bpe": "canonical_bfs_order",
}

CANONICAL_BASE_VIEWS = {
    "canonical_edge_list",
    "canonical_adjacency_list",
    "canonical_dfs_order",
    "canonical_bfs_order",
    "canonical_graph_tokenizer_feuler",
}


@dataclass
class TokenBPE:
    merges: list[tuple[str, str]] = field(default_factory=list)

    def fit(self, sequences: Iterable[list[str]], max_merges: int = 500, min_freq: int = 2) -> "TokenBPE":
        encoded = [list(seq) for seq in sequences]
        self.merges = []
        for idx in range(int(max_merges)):
            counts: Counter[tuple[str, str]] = Counter()
            for seq in encoded:
                counts.update(zip(seq, seq[1:]))
            if not counts:
                break
            pair, count = counts.most_common(1)[0]
            if count < int(min_freq):
                break
            merged = f"BPE{idx}<{pair[0]}+{pair[1]}>"
            self.merges.append(pair)
            encoded = [_apply_pair(seq, pair, merged) for seq in encoded]
        return self

    def encode(self, sequence: list[str]) -> list[str]:
        out = list(sequence)
        for idx, pair in enumerate(self.merges):
            out = _apply_pair(out, pair, f"BPE{idx}<{pair[0]}+{pair[1]}>")
        return out


class TokenViewBuilder:
    """Build matched token views over the same graph records.

    RPS-GTok views use the fitted final tokenizer. Serialization controls are
    deterministic and fitted only on training records when BPE is requested.
    """

    def __init__(self, tokenizer: GPTok2Tokenizer, seed: int = 2026, bpe_merges: int = 500, bpe_min_freq: int = 2):
        self.tokenizer = tokenizer
        self.seed = int(seed)
        self.bpe_merges = int(bpe_merges)
        self.bpe_min_freq = int(bpe_min_freq)
        self.bpe: dict[str, TokenBPE] = {}
        self._rps_cache: dict[tuple[str, str], list[str]] = {}

    def fit(self, records: list[GraphRecord], views: list[str]) -> "TokenViewBuilder":
        for view in views:
            if view not in BPE_BASE_VIEWS:
                continue
            base = BPE_BASE_VIEWS[view]
            sequences = [self.build(record, base) for record in records]
            self.bpe[view] = TokenBPE().fit(sequences, self.bpe_merges, self.bpe_min_freq)
        return self

    def build(self, record: GraphRecord, view: str) -> list[str]:
        view = str(view)
        if view in STRUCTURAL_PLUS_VIEWS:
            base = self.build(record, STRUCTURAL_PLUS_VIEWS[view])
            # ``rps_gtok_plus`` uses the same deterministic anchors as the
            # matched edge-list controls.  The only extra information in RPS
            # is therefore the executable primitive/motif stream itself.
            anchors = structural_summary_tokens(record) + wl_summary_tokens(record)
            if view in {"rps_gtok_plus", "rps_gtok_plus_wl", "edge_list_plus", "canonical_edge_list_plus"}:
                # RPS-GTok++ invariant anchors are derived from the executable
                # graph state rather than arbitrary node IDs.  They expose
                # degree/component/cycle signatures that are useful to a
                # sequence consumer while remaining permutation-stable.
                anchors += rps_semantic_anchor_tokens(record)
            return base + anchors
        if view in BPE_BASE_VIEWS:
            base_tokens = self.build(record, BPE_BASE_VIEWS[view])
            return self.bpe.get(view, TokenBPE()).encode(base_tokens)
        if view in RPS_MODE_ALIASES:
            mode = RPS_MODE_ALIASES[view]
            key = (record_fingerprint(record), mode)
            if key not in self._rps_cache:
                self._rps_cache[key] = self.tokenizer.encode(record, mode=mode).tokens
            return list(self._rps_cache[key])
        if view in {"rps_gtok_identifier", "rps_gtok_token_only"}:
            key = (record_fingerprint(record), "motif_hybrid")
            if key not in self._rps_cache:
                self._rps_cache[key] = self.tokenizer.encode(record, mode="motif_hybrid").tokens
            return [split_token_identifier(token) for token in self._rps_cache[key]]
        if view in CANONICAL_BASE_VIEWS:
            canonical = canonical_record(
                record,
                SchemaSpec.from_config(self.tokenizer.config),
                exact=True,
                max_search_nodes=int(
                    self.tokenizer.config.get("canonicalization", {}).get(
                        "max_search_nodes", 100000
                    )
                ),
            )
            base_view = view.removeprefix("canonical_")
            return self.build(canonical, base_view)
        if view == "rps_gtok_shuffled":
            return shuffled_tokens(self.build(record, "rps_gtok_full"), record.graph_id, self.seed)
        if view in {"rps_gtok_random_ids", "random_ids_same_length"}:
            return random_id_tokens(self.build(record, "rps_gtok_full"), record.graph_id, self.seed)
        if view == "rps_gtok_full_structural":
            return self.build(record, "rps_gtok_full") + structural_summary_tokens(record)
        if view == "rps_gtok_full_wl":
            return self.build(record, "rps_gtok_full") + structural_summary_tokens(record) + wl_summary_tokens(record)
        if view == "edge_list":
            return edge_list_tokens(record)
        if view == "adjacency_list":
            return adjacency_list_tokens(record)
        if view == "dfs_order":
            return traversal_tokens(record, mode="dfs")
        if view == "bfs_order":
            return traversal_tokens(record, mode="bfs")
        if view in {"graph_tokenizer_feuler", "feuler"}:
            return frequency_guided_euler_tokens(record)
        if view == "statistics_only":
            return structural_summary_tokens(record)
        raise ValueError(f"unknown token view: {view}")


def edge_list_tokens(record: GraphRecord) -> list[str]:
    return [f"E({u},{v})" for u, v in sorted(edge_set(record))]


def adjacency_list_tokens(record: GraphRecord) -> list[str]:
    graph = to_networkx(record)
    tokens = []
    for node in sorted(graph.nodes()):
        nbrs = ",".join(str(v) for v in sorted(graph.neighbors(node)))
        tokens.append(f"N({node}):{nbrs}")
    return tokens


def traversal_tokens(record: GraphRecord, mode: str) -> list[str]:
    graph = to_networkx(record)
    if graph.number_of_nodes() == 0:
        return []
    start = min(graph.nodes(), key=lambda n: (-graph.degree(n), n))
    if mode == "bfs":
        edges = list(nx.bfs_edges(graph, start))
    else:
        edges = list(nx.dfs_edges(graph, start))
    seen = {start}
    tokens = [f"START({start})"]
    for u, v in edges:
        tokens.append(f"T({u},{v})")
        seen.add(v)
    for node in sorted(set(graph.nodes()) - seen):
        tokens.append(f"START({node})")
        for u, v in (nx.bfs_edges(graph, node) if mode == "bfs" else nx.dfs_edges(graph, node)):
            tokens.append(f"T({u},{v})")
    return tokens


def frequency_guided_euler_tokens(record: GraphRecord) -> list[str]:
    graph = to_networkx(record)
    tokens = []
    for component in sorted(nx.connected_components(graph), key=lambda xs: min(xs)):
        sub = graph.subgraph(component).copy()
        start = min(sub.nodes(), key=lambda n: (-sub.degree(n), n))
        visited = set()
        stack = [start]
        tokens.append(f"START({start})")
        while stack:
            u = stack.pop()
            for v in sorted(sub.neighbors(u), key=lambda n: (-sub.degree(n), n)):
                edge = tuple(sorted((u, v)))
                if edge in visited:
                    continue
                visited.add(edge)
                tokens.append(f"WALK({u},{v})")
                stack.append(v)
    return tokens


def structural_summary_tokens(record: GraphRecord) -> list[str]:
    graph = to_networkx(record)
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    density = 0.0 if n <= 1 else 2.0 * m / (n * (n - 1))
    triangles = sum(nx.triangles(graph).values()) // 3 if not graph.is_directed() else 0
    return [
        f"NODES_BIN({bin_value(n)})",
        f"EDGES_BIN({bin_value(m)})",
        f"DENSITY_BIN({int(density * 10)})",
        f"TRIANGLES_BIN({bin_value(triangles)})",
    ]


def wl_summary_tokens(record: GraphRecord, rounds: int = 2) -> list[str]:
    graph = to_networkx(record)
    labels = {node: str(graph.degree(node)) for node in graph.nodes()}
    out = []
    for depth in range(int(rounds)):
        counts = Counter(labels.values())
        out.extend(f"WL{depth}({label})x{count}" for label, count in sorted(counts.items()))
        labels = {
            node: stable_hash([labels[node], *sorted(labels[nbr] for nbr in graph.neighbors(node))])
            for node in graph.nodes()
        }
    return out


def rps_semantic_anchor_tokens(record: GraphRecord) -> list[str]:
    """Permutation-invariant graph-state anchors for the RPS-GTok++ variant."""
    graph = to_networkx(record)
    degree_counts = Counter(int(degree) for _, degree in graph.degree())
    out = [f"RPS_DEGREE_{degree}_{count}" for degree, count in sorted(degree_counts.items())]
    component_graph = nx.Graph(graph)
    components = sorted((len(component) for component in nx.connected_components(component_graph)), reverse=True)
    out.extend(f"RPS_COMPONENT_SIZE_{size}" for size in components[:32])
    if not graph.is_directed():
        cycle_basis = nx.cycle_basis(nx.Graph(graph))
        out.append(f"RPS_CYCLE_COUNT_{min(32, len(cycle_basis))}")
        # The length of a cycle-basis member is not invariant to traversal
        # order. Use the cyclomatic number instead; it is permutation invariant
        # and retains the same coarse cyclicity signal without a basis-order
        # artifact.
        simple_graph = nx.Graph(graph)
        cyclomatic = simple_graph.number_of_edges() - simple_graph.number_of_nodes() + nx.number_connected_components(simple_graph)
        out.append(f"RPS_CYCLE_RANK_{min(64, max(0, cyclomatic))}")
    # Counts of local degree-neighborhood signatures are more stable than raw
    # node identifiers and complement the executable primitive stream.
    nodes = sorted(graph.nodes())
    labels = {node: str(graph.degree(node)) for node in nodes}
    for depth in range(4):
        signatures = [stable_hash([labels[node], *sorted(labels[nbr] for nbr in graph.neighbors(node))]) for node in nodes]
        counts = Counter(signatures)
        out.extend(f"RPS_WL{depth}_{digest}_{count}" for digest, count in sorted(counts.items()))
        labels = {node: signatures[index] for index, node in enumerate(nodes)}
    return out


def shuffled_tokens(tokens: list[str], graph_id: str, seed: int) -> list[str]:
    out = list(tokens)
    random.Random(stable_int(f"shuffle:{seed}:{graph_id}")).shuffle(out)
    return out


def random_id_tokens(tokens: list[str], graph_id: str, seed: int) -> list[str]:
    rng = random.Random(stable_int(f"random-id:{seed}:{graph_id}"))
    return [f"RID({rng.randrange(1_000_000)})" for _ in tokens]


def bin_value(value: int | float) -> int:
    value = int(abs(value))
    if value <= 0:
        return 0
    return min(20, value.bit_length())


def stable_hash(parts: list[str]) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]


def stable_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def _apply_pair(sequence: list[str], pair: tuple[str, str], merged: str) -> list[str]:
    out = []
    idx = 0
    while idx < len(sequence):
        if idx + 1 < len(sequence) and (sequence[idx], sequence[idx + 1]) == pair:
            out.append(merged)
            idx += 2
        else:
            out.append(sequence[idx])
            idx += 1
    return out


def split_token_identifier(token: str) -> str:
    return split_token(token).identifier


def record_fingerprint(record: GraphRecord) -> str:
    """Content fingerprint for safe token caching under node permutations.

    Graph IDs identify logical examples, not immutable serialized inputs. A
    permuted record may intentionally retain the same graph_id, so cache keys
    must include all structural/schema fields used by encoding.
    """
    def tensor(value):
        return value.detach().cpu().tolist() if value is not None else None
    payload = {
        "graph_id": str(record.graph_id),
        "num_nodes": int(record.num_nodes),
        "edges": tensor(record.edge_index),
        "node_type": tensor(record.node_type),
        "edge_type": tensor(record.edge_type),
        "node_attr": tensor(record.node_attr),
        "edge_attr": tensor(record.edge_attr),
        "directed": bool(record.directed),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
