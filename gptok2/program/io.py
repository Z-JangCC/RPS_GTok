"""Graph program serialization with explicit leakage audit."""

from __future__ import annotations

from pathlib import Path

from gptok2.program.actions import GraphProgram
from gptok2.utils.io import read_jsonl, write_jsonl

FORBIDDEN_PROGRAM_KEYS = {
    "src_node",
    "dst_node",
    "covered_edges",
    "residual_edges",
    "original_edges",
    "edge_list",
    "node_ids",
    "nodes",
    "edges",
}


def save_programs(path, programs: list[GraphProgram]) -> None:
    rows = [p.to_dict() for p in programs]
    audit_rows_for_leakage(rows)
    write_jsonl(path, rows)


def load_programs(path) -> list[GraphProgram]:
    return [GraphProgram.from_dict(row) for row in read_jsonl(Path(path))]


def audit_rows_for_leakage(rows: list[dict]) -> None:
    bad: list[str] = []

    def walk(obj, trail=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k) in FORBIDDEN_PROGRAM_KEYS:
                    bad.append(trail + "." + str(k))
                walk(v, trail + "." + str(k))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, trail + f"[{i}]")

    for row in rows:
        walk(row)
    if bad:
        raise ValueError("Potential graph-structure leakage in program rows: " + ", ".join(bad[:20]))

