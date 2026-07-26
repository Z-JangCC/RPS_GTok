"""Patch, port, and merge-anchor data structures for gptok2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PORT_ROLES = (
    "chain-end",
    "cycle-anchor",
    "branch-port",
    "bridge-port",
    "module-boundary",
    "generic-port",
)

PORT_MODES = ("ATTACH", "MERGE", "CLOSE")


@dataclass(frozen=True)
class Port:
    local_index: int
    anchor: int
    role: str
    mode: str
    capacity: int
    subtype: int

    def schema_key(self) -> tuple:
        return (self.role, self.mode, int(self.capacity), int(self.subtype))

    def prototype_schema(self) -> tuple:
        return (int(self.anchor), self.role, self.mode, int(self.capacity), int(self.subtype))

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_index": self.local_index,
            "anchor": self.anchor,
            "role": self.role,
            "mode": self.mode,
            "capacity": self.capacity,
            "subtype": self.subtype,
        }

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "Port":
        return Port(
            int(row["local_index"]),
            int(row["anchor"]),
            str(row["role"]),
            str(row["mode"]),
            int(row["capacity"]),
            int(row["subtype"]),
        )


@dataclass(frozen=True)
class Anchor:
    local_index: int
    kind: str
    role: str
    subtype: int
    nodes: tuple[int, ...]

    def schema_key(self) -> tuple:
        return (self.kind, self.role, int(self.subtype), len(self.nodes))

    def prototype_schema(self) -> tuple:
        return (self.kind, self.role, int(self.subtype), tuple(int(x) for x in self.nodes))

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_index": self.local_index,
            "kind": self.kind,
            "role": self.role,
            "subtype": self.subtype,
            "nodes": list(self.nodes),
        }

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "Anchor":
        return Anchor(
            int(row["local_index"]),
            str(row["kind"]),
            str(row["role"]),
            int(row["subtype"]),
            tuple(int(x) for x in row["nodes"]),
        )


@dataclass
class Patch:
    patch_id: str
    graph_id: str
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    ports: tuple[Port, ...]
    anchors: tuple[Anchor, ...]
    edge_anchors: tuple[Anchor, ...]
    score: float
    structural_hash: str
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self, include_original_nodes: bool = False) -> dict[str, Any]:
        # Program artifacts must call this with include_original_nodes=False.
        row = {
            "patch_id": self.patch_id,
            "graph_id": self.graph_id,
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "ports": [p.to_dict() for p in self.ports],
            "anchors": [a.to_dict() for a in self.anchors],
            "edge_anchors": [a.to_dict() for a in self.edge_anchors],
            "score": float(self.score),
            "structural_hash": self.structural_hash,
            "features": {k: float(v) for k, v in self.features.items()},
        }
        if include_original_nodes:
            row["nodes"] = list(self.nodes)
            row["edges"] = [list(e) for e in self.edges]
        return row

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "Patch":
        nodes = tuple(int(x) for x in row.get("nodes", range(int(row.get("num_nodes", 0)))))
        edges = tuple(tuple(sorted((int(u), int(v)))) for u, v in row.get("edges", []))
        return Patch(
            str(row["patch_id"]),
            str(row["graph_id"]),
            nodes,
            edges,
            tuple(Port.from_dict(x) for x in row.get("ports", [])),
            tuple(Anchor.from_dict(x) for x in row.get("anchors", [])),
            tuple(Anchor.from_dict(x) for x in row.get("edge_anchors", [])),
            float(row.get("score", 0.0)),
            str(row.get("structural_hash", "")),
            {str(k): float(v) for k, v in row.get("features", {}).items()},
        )

