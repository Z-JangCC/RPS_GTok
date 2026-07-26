"""Minimal executable graph program grammar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


OPS = (
    "BEGIN_GRAPH",
    "END_GRAPH",
    "BEGIN_BLOCK",
    "END_BLOCK",
    "EMIT",
    "INTERFACE",
    "ATTACH",
    "MERGE_NODE",
    "MERGE_EDGE",
    "CLOSE_CYCLE",
    "GLOBAL_LINK",
    "STOP",
)


@dataclass(frozen=True)
class Action:
    op: str
    args: tuple[Any, ...] = ()

    def __post_init__(self):
        if self.op not in OPS:
            raise ValueError(f"Unknown graph program op: {self.op}")

    def to_token(self) -> str:
        if not self.args:
            return self.op
        return self.op + "(" + ",".join(map(str, self.args)) + ")"

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "args": list(self.args)}

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "Action":
        return Action(str(row["op"]), tuple(row.get("args", [])))


def parse_token(token: str) -> Action:
    if "(" not in token:
        return Action(token)
    op, rest = token.split("(", 1)
    rest = rest.rstrip(")")
    args: list[Any] = []
    for arg in rest.split(",") if rest else []:
        if arg.lstrip("-").isdigit():
            args.append(int(arg))
        else:
            args.append(arg)
    return Action(op, tuple(args))


@dataclass
class GraphProgram:
    graph_id: str
    actions: list[Action]
    metadata: dict[str, Any] | None = None

    def to_tokens(self) -> list[str]:
        return [a.to_token() for a in self.actions]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "actions": [a.to_dict() for a in self.actions],
            "metadata": self.metadata or {},
        }

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "GraphProgram":
        return GraphProgram(str(row["graph_id"]), [Action.from_dict(x) for x in row.get("actions", [])], dict(row.get("metadata", {})))

