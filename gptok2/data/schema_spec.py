"""Explicit reconstruction contracts for RPS-GTok.

The original release treated the graph schema implicitly.  This module keeps
the contract executable and serializable so reconstruction metrics can state
exactly which fields are retained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SchemaSpec:
    """Dataset-level retained-information contract."""

    name: str = "topology"
    directed: bool = False
    multiedge: bool = False
    self_loop: bool = False
    retain_graph_metadata: tuple[str, ...] = ()
    retain_node_type: bool = True
    retain_edge_type: bool = False
    retain_node_attr: bool = False
    retain_edge_attr: bool = False
    continuous_mode: str = "raw"
    quantization_decimals: int | None = None
    max_search_nodes: int = 100_000
    timeout_sec: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "SchemaSpec":
        row = dict((config or {}).get("schema", config or {}))
        retained = dict(row.get("retained", {}))
        quant = dict(row.get("quantization", {}))
        return cls(
            name=str(row.get("name", "topology")),
            directed=bool(row.get("directed", False)),
            multiedge=bool(row.get("multiedge", False)),
            self_loop=bool(row.get("self_loop", False)),
            retain_graph_metadata=tuple(str(x) for x in row.get("retain_graph_metadata", ())),
            retain_node_type=bool(row.get("retain_node_type", retained.get("node_type", True))),
            retain_edge_type=bool(row.get("retain_edge_type", retained.get("edge_type", False))),
            retain_node_attr=bool(row.get("retain_node_attr", retained.get("node_attr", False))),
            retain_edge_attr=bool(row.get("retain_edge_attr", retained.get("edge_attr", False))),
            continuous_mode=str(row.get("continuous_mode", quant.get("mode", "raw"))),
            quantization_decimals=(
                int(row["quantization_decimals"])
                if row.get("quantization_decimals") is not None
                else (
                    int(quant["decimals"])
                    if quant.get("decimals") is not None
                    else None
                )
            ),
            max_search_nodes=int(row.get("max_search_nodes", 100_000)),
            timeout_sec=float(row["timeout_sec"]) if row.get("timeout_sec") is not None else None,
            metadata=dict(row.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def quantize_value(self, value: Any) -> Any:
        if self.quantization_decimals is None or self.continuous_mode == "raw":
            return value
        try:
            return round(float(value), self.quantization_decimals)
        except (TypeError, ValueError):
            return str(value)

    @property
    def reconstruction_target(self) -> str:
        fields = ["topology"]
        if self.retain_node_type:
            fields.append("node_type")
        if self.retain_edge_type:
            fields.append("edge_type")
        if self.retain_node_attr:
            fields.append("node_attr_" + self.continuous_mode)
        if self.retain_edge_attr:
            fields.append("edge_attr_" + self.continuous_mode)
        if self.retain_graph_metadata:
            fields.append("graph_metadata")
        return "_".join(fields)
