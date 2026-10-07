"""Pre-training guards for held-out and matched-consumer evidence."""
from __future__ import annotations

import math
from collections import defaultdict

from rps_gtok_consumption.data import TokenExample


def validate_heldout_splits(splits: dict[str, list[TokenExample]], task: str) -> dict:
    """Reject fallback validation, overlaps, non-finite targets and missing classes."""
    for name in ("train", "val", "test"):
        if not splits.get(name):
            raise ValueError(f"{name}: held-out split must not be empty")
        ids = [row.graph_id for row in splits[name]]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{name}: duplicate graph_id")
    id_sets = {name: {row.graph_id for row in splits[name]} for name in ("train", "val", "test")}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if id_sets[left] & id_sets[right]:
            raise ValueError(f"overlap between {left} and {right}")
    report = {"counts": {name: len(rows) for name, rows in splits.items()}}
    if task == "classification":
        classes = {}
        for name, rows in splits.items():
            values = [float(row.y) for row in rows]
            if any(not math.isfinite(y) or y < 0 or y != int(y) for y in values):
                raise ValueError(f"{name}: invalid class index")
            classes[name] = sorted({int(y) for y in values})
        if len(classes["train"]) < 2:
            raise ValueError("training data has fewer than two classes")
        if classes["train"] != list(range(max(classes["train"]) + 1)):
            raise ValueError("training class indices must be contiguous from zero")
        for name in ("val", "test"):
            if classes[name] != classes["train"]:
                raise ValueError(f"{name}: missing or unseen classification classes")
        report["classes"] = classes
    elif task == "regression":
        for name, rows in splits.items():
            if any(not math.isfinite(float(row.y)) for row in rows):
                raise ValueError(f"{name}: non-finite regression target")
    return report


def scenario_observability(examples: list[TokenExample], case_by_id: dict[str, str]) -> dict:
    """Detect physical scenarios with varying targets but identical model tokens."""
    groups = defaultdict(list)
    for example in examples:
        groups[case_by_id[example.graph_id]].append(example)
    blind = []
    details = []
    for case, rows in sorted(groups.items()):
        targets = {float(row.y) for row in rows}
        unique_tokens = len({tuple(row.tokens) for row in rows})
        varying = len(targets) > 1
        if varying and unique_tokens == 1:
            blind.append(case)
        details.append({"case": case, "samples": len(rows), "unique_targets": len(targets), "unique_tokens": unique_tokens})
    return {"passed": not blind, "unobservable_cases": blind, "case_details": details}


def require_equal_parameters(rows: list[dict]) -> None:
    counts = {int(row["parameter_count"]) for row in rows}
    if len(counts) != 1:
        raise ValueError(f"matched-consumer parameter counts differ: {sorted(counts)}")
