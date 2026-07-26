"""Prototype-stable VQ-style graph codebook."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from gptok2.patches.types import Patch


@dataclass
class PrototypeCode:
    code_id: int
    structural_hash: str
    prototype_num_nodes: int
    prototype_edges: tuple[tuple[int, int], ...]
    port_schema: tuple[tuple, ...]
    anchor_schema: tuple[tuple, ...]
    edge_anchor_schema: tuple[tuple, ...]
    feature_mean: dict[str, float]
    usage: int = 0
    top_patch_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_id": self.code_id,
            "structural_hash": self.structural_hash,
            "prototype_num_nodes": self.prototype_num_nodes,
            "prototype_edges": [list(e) for e in self.prototype_edges],
            "port_schema": [list(x) for x in self.port_schema],
            "anchor_schema": [list(x) for x in self.anchor_schema],
            "edge_anchor_schema": [list(x) for x in self.edge_anchor_schema],
            "feature_mean": self.feature_mean,
            "usage": self.usage,
            "top_patch_ids": self.top_patch_ids,
        }

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "PrototypeCode":
        return PrototypeCode(
            int(row["code_id"]),
            str(row["structural_hash"]),
            int(row["prototype_num_nodes"]),
            tuple(tuple(sorted((int(u), int(v)))) for u, v in row.get("prototype_edges", [])),
            tuple(_freeze(x) for x in row.get("port_schema", [])),
            tuple(_freeze(x) for x in row.get("anchor_schema", [])),
            tuple(_freeze(x) for x in row.get("edge_anchor_schema", [])),
            {str(k): float(v) for k, v in row.get("feature_mean", {}).items()},
            int(row.get("usage", 0)),
            list(row.get("top_patch_ids", [])),
        )


@dataclass
class InterfaceSchema:
    interface_id: int
    port_schema: tuple[tuple, ...]
    anchor_schema: tuple[tuple, ...]
    edge_anchor_schema: tuple[tuple, ...]
    usage: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "interface_id": self.interface_id,
            "port_schema": [list(x) for x in self.port_schema],
            "anchor_schema": [list(x) for x in self.anchor_schema],
            "edge_anchor_schema": [list(x) for x in self.edge_anchor_schema],
            "usage": self.usage,
        }

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "InterfaceSchema":
        return InterfaceSchema(
            int(row["interface_id"]),
            tuple(_freeze(x) for x in row.get("port_schema", [])),
            tuple(_freeze(x) for x in row.get("anchor_schema", [])),
            tuple(_freeze(x) for x in row.get("edge_anchor_schema", [])),
            int(row.get("usage", 0)),
        )


class PrototypeVQCodebook:
    def __init__(
        self,
        codes: list[PrototypeCode],
        distance_weights: dict[str, float] | None = None,
        interfaces: list[InterfaceSchema] | None = None,
        factorized_interfaces: bool = False,
    ):
        self.codes = sorted(codes, key=lambda c: c.code_id)
        self.interfaces = sorted(interfaces or [], key=lambda s: s.interface_id)
        self.factorized_interfaces = bool(factorized_interfaces)
        self.distance_weights = distance_weights or {
            "num_nodes": 1.2,
            "num_edges": 1.2,
            "density": 0.8,
            "triangles": 1.0,
            "cycles": 1.0,
            "ports": 1.0,
            "avg_degree": 0.6,
        }
        self._build_index()

    def __len__(self) -> int:
        return len(self.codes)

    def assign(self, patch: Patch) -> int:
        if not self.codes:
            raise ValueError("Cannot assign a patch with an empty GPTok codebook.")
        scored = [(self.patch_distance(patch, c), c.code_id) for c in self.candidate_codes(patch)]
        return min(scored)[1]

    def assign_interface(self, patch: Patch) -> int:
        if not self.factorized_interfaces:
            return -1
        key = _interface_key(patch)
        schema = self._interface_by_key.get(key)
        if schema is not None:
            return schema.interface_id
        schema = InterfaceSchema(
            len(self.interfaces),
            tuple(p.prototype_schema() for p in patch.ports),
            tuple(a.prototype_schema() for a in patch.anchors),
            tuple(a.prototype_schema() for a in patch.edge_anchors),
            0,
        )
        self.interfaces.append(schema)
        self._interface_by_key[key] = schema
        return schema.interface_id

    def get_interface(self, interface_id: int) -> InterfaceSchema:
        idx = int(interface_id)
        if idx < 0 or idx >= len(self.interfaces):
            raise IndexError(f"Interface id {interface_id} is outside the interface table size {len(self.interfaces)}.")
        return self.interfaces[idx]

    def get(self, code_id: int) -> PrototypeCode:
        idx = int(code_id)
        if idx < 0 or idx >= len(self.codes):
            raise IndexError(f"Code id {code_id} is outside the codebook size {len(self.codes)}.")
        return self.codes[idx]

    def patch_distance(self, patch: Patch, code: PrototypeCode) -> float:
        d = 0.0
        for key, w in self.distance_weights.items():
            d += w * abs(float(patch.features.get(key, 0.0)) - float(code.feature_mean.get(key, 0.0)))
        d += 0.75 * (0 if _visual_key(patch) == _code_visual_key(code) else 1)
        if not self.factorized_interfaces:
            d += 4.0 * abs(len(patch.ports) - len(code.port_schema))
            d += 2.0 * abs(len(patch.anchors) - len(code.anchor_schema))
            d += 2.0 * abs(len(patch.edge_anchors) - len(code.edge_anchor_schema))
            d += 0.5 * _schema_distance(tuple(p.prototype_schema() for p in patch.ports), code.port_schema)
        return d

    def assign_all(self, patches: list[Patch]) -> dict[str, int]:
        assignments = {p.patch_id: self.assign(p) for p in patches}
        counts = Counter(assignments.values())
        for code in self.codes:
            code.usage = counts.get(code.code_id, 0)
        if self.factorized_interfaces:
            icounts = Counter(self.assign_interface(p) for p in patches)
            for schema in self.interfaces:
                schema.usage = icounts.get(schema.interface_id, 0)
        return assignments

    def candidate_codes(self, patch: Patch, limit: int = 256) -> list[PrototypeCode]:
        structural = self._by_structural_hash.get(patch.structural_hash)
        if structural:
            return structural
        n = int(patch.features.get("num_nodes", len(patch.nodes)))
        e = int(patch.features.get("num_edges", len(patch.edges)))
        p = len(patch.ports)
        candidates: list[PrototypeCode] = []
        seen: set[int] = set()

        def add(rows):
            for code in rows or []:
                if code.code_id not in seen:
                    seen.add(code.code_id)
                    candidates.append(code)

        add(self._by_bucket.get((n, e, p)))
        add(self._by_visual.get((n, e)))
        if len(candidates) < min(limit, len(self.codes)):
            for dn in (0, -1, 1, -2, 2):
                for de in (0, -1, 1, -2, 2, -4, 4):
                    for dp in (0, -1, 1):
                        add(self._by_bucket.get((n + dn, e + de, p + dp)))
                        if len(candidates) >= limit:
                            break
                    if len(candidates) >= limit:
                        break
                if len(candidates) >= limit:
                    break
        return candidates or self.codes

    def _build_index(self) -> None:
        self._by_structural_hash: dict[str, list[PrototypeCode]] = defaultdict(list)
        self._by_bucket: dict[tuple[int, int, int], list[PrototypeCode]] = defaultdict(list)
        self._by_visual: dict[tuple[int, int], list[PrototypeCode]] = defaultdict(list)
        self._interface_by_key: dict[tuple, InterfaceSchema] = {}
        for code in self.codes:
            self._by_structural_hash[code.structural_hash].append(code)
            n = int(code.prototype_num_nodes)
            e = int(len(code.prototype_edges))
            p = int(len(code.port_schema))
            self._by_bucket[(n, e, p)].append(code)
            self._by_visual[(n, e)].append(code)
        for schema in self.interfaces:
            self._interface_by_key[_schema_key(schema.port_schema, schema.anchor_schema, schema.edge_anchor_schema)] = schema

    def usage_stats(self) -> dict[str, float]:
        counts = [max(0, c.usage) for c in self.codes]
        total = sum(counts)
        used = sum(1 for x in counts if x > 0)
        probs = [x / total for x in counts if x > 0] if total else []
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        perplexity = math.exp(entropy) if probs else 0.0
        return {
            "codebook_size": float(len(self.codes)),
            "used_codes": float(used),
            "dead_code_ratio": float(1.0 - used / max(1, len(self.codes))),
            "usage_entropy": float(entropy),
            "codebook_perplexity": float(perplexity),
            "interface_vocab_size": float(len(self.interfaces)),
            "factorized_interfaces": float(self.factorized_interfaces),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "codes": [c.to_dict() for c in self.codes],
            "interfaces": [s.to_dict() for s in self.interfaces],
            "factorized_interfaces": self.factorized_interfaces,
            "distance_weights": self.distance_weights,
        }

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "PrototypeVQCodebook":
        return PrototypeVQCodebook(
            [PrototypeCode.from_dict(x) for x in row["codes"]],
            row.get("distance_weights"),
            [InterfaceSchema.from_dict(x) for x in row.get("interfaces", [])],
            bool(row.get("factorized_interfaces", False)),
        )


def learn_codebook(patches: list[Patch], config: dict) -> PrototypeVQCodebook:
    if not patches:
        return PrototypeVQCodebook([])
    cfg = config.get("vq", {})
    max_codes = int(cfg.get("num_codes", 64))
    min_usage = int(cfg.get("min_usage", 1))
    factorized = bool(cfg.get("factorize_interfaces", False))
    groups: dict[tuple | str, list[Patch]] = defaultdict(list)
    for patch in patches:
        groups[_visual_key(patch) if factorized else _coarse_key(patch)].append(patch)
    clusters = sorted(groups.values(), key=lambda g: (-len(g), _cluster_signature(g)))
    if len(clusters) > max_codes:
        head = [list(g) for g in clusters[:max_codes]]
        reps = [_representative(g) for g in head]
        for tail in clusters[max_codes:]:
            rep = _representative(tail)
            idx = min(range(len(head)), key=lambda i: _feature_distance(rep, reps[i]))
            head[idx].extend(tail)
        clusters = head
    clusters = [c for c in clusters if len(c) >= min_usage] or clusters
    codes = [_make_code(i, c, strip_interface=factorized) for i, c in enumerate(sorted(clusters, key=lambda x: (_cluster_signature(x), -len(x))))][:max_codes]
    interfaces = _learn_interfaces(patches) if factorized else []
    cb = PrototypeVQCodebook(codes, interfaces=interfaces, factorized_interfaces=factorized)
    cb.assign_all(patches)
    if cfg.get("dead_code_reinit", True):
        _dead_code_reinit(cb, patches)
        cb.assign_all(patches)
    if cfg.get("merge_near_duplicates", True):
        cb = _merge_near_duplicates(cb, patches, max_codes)
    return cb


def _make_code(code_id: int, patches: list[Patch], strip_interface: bool = False) -> PrototypeCode:
    medoid = _medoid(patches)
    feature_keys = sorted({k for p in patches for k in p.features})
    mean = {k: sum(p.features.get(k, 0.0) for p in patches) / len(patches) for k in feature_keys}
    return PrototypeCode(
        code_id=code_id,
        structural_hash=medoid.structural_hash,
        prototype_num_nodes=len(medoid.nodes),
        prototype_edges=tuple(tuple(sorted((medoid.nodes.index(u), medoid.nodes.index(v)))) for u, v in medoid.edges),
        port_schema=() if strip_interface else tuple(p.prototype_schema() for p in medoid.ports),
        anchor_schema=() if strip_interface else tuple(a.prototype_schema() for a in medoid.anchors),
        edge_anchor_schema=() if strip_interface else tuple(a.prototype_schema() for a in medoid.edge_anchors),
        feature_mean={k: float(v) for k, v in mean.items()},
        usage=len(patches),
        top_patch_ids=[p.patch_id for p in sorted(patches, key=lambda p: -p.score)[:8]],
    )


def _medoid(patches: list[Patch]) -> Patch:
    if len(patches) <= 16:
        return min(patches, key=lambda p: (sum(_feature_distance(p, q) for q in patches), -p.score, p.patch_id))
    keys = sorted({k for p in patches for k in p.features})
    mean = {k: sum(p.features.get(k, 0.0) for p in patches) / len(patches) for k in keys}
    return min(patches, key=lambda p: (sum(abs(p.features.get(k, 0.0) - mean[k]) for k in keys), -p.score, p.patch_id))


def _representative(patches: list[Patch]) -> Patch:
    return max(patches, key=lambda p: (p.score, len(p.edges), -len(p.ports), p.patch_id))


def _coarse_key(patch: Patch) -> str:
    f = patch.features
    return f"n{int(f.get('num_nodes', 0))}_e{int(f.get('num_edges', 0))}_tri{int(f.get('triangles', 0))}_cy{int(f.get('cycles', 0))}_p{int(f.get('ports', 0))}_{patch.structural_hash[:6]}"


def _visual_key(patch: Patch) -> tuple:
    local = {v: i for i, v in enumerate(patch.nodes)}
    edges = tuple(sorted(tuple(sorted((local[u], local[v]))) for u, v in patch.edges))
    return (len(patch.nodes), edges)


def _code_visual_key(code: PrototypeCode) -> tuple:
    return (int(code.prototype_num_nodes), tuple(sorted(tuple(sorted(e)) for e in code.prototype_edges)))


def _interface_key(patch: Patch) -> tuple:
    return _schema_key(
        tuple(p.prototype_schema() for p in patch.ports),
        tuple(a.prototype_schema() for a in patch.anchors),
        tuple(a.prototype_schema() for a in patch.edge_anchors),
    )


def _schema_key(port_schema: tuple, anchor_schema: tuple, edge_anchor_schema: tuple) -> tuple:
    return (_freeze(port_schema), _freeze(anchor_schema), _freeze(edge_anchor_schema))


def _learn_interfaces(patches: list[Patch]) -> list[InterfaceSchema]:
    groups: dict[tuple, tuple[tuple, tuple, tuple, int]] = {}
    counts: Counter[tuple] = Counter()
    for patch in patches:
        port_schema = tuple(p.prototype_schema() for p in patch.ports)
        anchor_schema = tuple(a.prototype_schema() for a in patch.anchors)
        edge_anchor_schema = tuple(a.prototype_schema() for a in patch.edge_anchors)
        key = _schema_key(port_schema, anchor_schema, edge_anchor_schema)
        groups[key] = (port_schema, anchor_schema, edge_anchor_schema, 0)
        counts[key] += 1
    interfaces = []
    for i, key in enumerate(sorted(groups, key=lambda k: (-counts[k], str(k)))):
        port_schema, anchor_schema, edge_anchor_schema, _ = groups[key]
        interfaces.append(InterfaceSchema(i, port_schema, anchor_schema, edge_anchor_schema, counts[key]))
    return interfaces


def _cluster_signature(cluster: list[Patch]) -> tuple:
    p = _medoid(cluster)
    return (len(p.nodes), len(p.edges), len(p.ports), p.structural_hash)


def _feature_distance(a: Patch, b: Patch) -> float:
    keys = {"num_nodes", "num_edges", "density", "triangles", "cycles", "ports", "avg_degree"}
    return sum(abs(a.features.get(k, 0.0) - b.features.get(k, 0.0)) for k in keys) + (0 if a.structural_hash == b.structural_hash else 1.0)


def _schema_distance(a: tuple, b: tuple) -> float:
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    return float(sum(abs(ca[k] - cb[k]) for k in keys))


def _freeze(value):
    if isinstance(value, list):
        return tuple(_freeze(x) for x in value)
    if isinstance(value, tuple):
        return tuple(_freeze(x) for x in value)
    return value


def _dead_code_reinit(cb: PrototypeVQCodebook, patches: list[Patch]) -> None:
    if not patches:
        return
    used_hashes = {c.structural_hash for c in cb.codes if c.usage > 0}
    hard = sorted(patches, key=lambda p: min(cb.patch_distance(p, c) for c in cb.candidate_codes(p)), reverse=True)
    cursor = 0
    for code in cb.codes:
        if code.usage > 0:
            continue
        while cursor < len(hard) and hard[cursor].structural_hash in used_hashes:
            cursor += 1
        if cursor >= len(hard):
            break
        fresh = _make_code(code.code_id, [hard[cursor]], strip_interface=cb.factorized_interfaces)
        code.structural_hash = fresh.structural_hash
        code.prototype_num_nodes = fresh.prototype_num_nodes
        code.prototype_edges = fresh.prototype_edges
        code.port_schema = fresh.port_schema
        code.anchor_schema = fresh.anchor_schema
        code.edge_anchor_schema = fresh.edge_anchor_schema
        code.feature_mean = fresh.feature_mean
        code.top_patch_ids = fresh.top_patch_ids
        used_hashes.add(fresh.structural_hash)
    cb._build_index()


def _merge_near_duplicates(cb: PrototypeVQCodebook, patches: list[Patch], max_codes: int) -> PrototypeVQCodebook:
    groups: dict[tuple, list[Patch]] = defaultdict(list)
    assigns = cb.assign_all(patches)
    by_id = {p.patch_id: p for p in patches}
    for pid, cid in assigns.items():
        c = cb.get(cid)
        groups[(c.structural_hash, c.prototype_num_nodes, len(c.prototype_edges), c.port_schema)].append(by_id[pid])
    if cb.factorized_interfaces:
        by_visual: dict[tuple, list[Patch]] = defaultdict(list)
        for patch in patches:
            by_visual[_visual_key(patch)].append(patch)
        codes = [_make_code(i, group, strip_interface=True) for i, group in enumerate(by_visual.values())]
        return PrototypeVQCodebook(codes[:max_codes], cb.distance_weights, cb.interfaces, True)
    codes = [_make_code(i, group) for i, group in enumerate(groups.values())]
    return PrototypeVQCodebook(codes[:max_codes], cb.distance_weights)

