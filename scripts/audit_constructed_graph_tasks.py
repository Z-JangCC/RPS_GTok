"""Audit all constructed graph-level proxy labels for alignment and leakage."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from gptok2.data.io import load_records
from scripts.build_constructed_graph_tasks import DATASETS, descriptor


ROOT = Path("rebuttal")


def main() -> None:
    errors: list[str] = []
    audit_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        source = ROOT / "data_21" / f"{dataset}.jsonl"
        sidecar = ROOT / "constructed_tasks" / f"{dataset}.jsonl"
        protocol = ROOT / "constructed_tasks" / f"{dataset}.protocol.json"
        if not source.exists() or not sidecar.exists() or not protocol.exists():
            errors.append(f"{dataset}: missing source/sidecar/protocol")
            continue
        records = load_records(source)
        rows = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != len(records) or len({row.get("graph_id") for row in rows}) != len(rows):
            errors.append(f"{dataset}: label count or duplicate graph IDs")
            continue
        by_id = {row["graph_id"]: row for row in rows}
        for record in records:
            row = by_id.get(record.graph_id)
            if row is None:
                errors.append(f"{dataset}/{record.graph_id}: missing target")
                continue
            if record.y is not None:
                errors.append(f"{dataset}/{record.graph_id}: source y is not null")
            value, source_name = descriptor(record)
            if abs(float(row["target"]) - value) > 1e-8 or row.get("target_source") != source_name:
                errors.append(f"{dataset}/{record.graph_id}: target mismatch")
        splits = {split: [row for row in rows if row.get("split") == split] for split in ("train", "val", "test")}
        if any(not values for values in splits.values()):
            errors.append(f"{dataset}: empty graph-disjoint split")
        manifest = json.loads(protocol.read_text(encoding="utf-8"))
        if manifest.get("input_fields_used") != ["num_nodes", "edges", "directed"]:
            errors.append(f"{dataset}: unexpected label input fields")
        if set(manifest.get("input_fields_forbidden", [])) != {"y", "node_attr", "edge_attr", "metadata", "graph_id"}:
            errors.append(f"{dataset}: forbidden input field policy changed")
        audit_rows.append({"dataset": dataset, "records": len(records), "train": len(splits["train"]), "val": len(splits["val"]), "test": len(splits["test"]), "source_y_null": int(all(record.y is None for record in records)), "target_recomputed": 1, "input_leakage": 0})
    out = ROOT / "audits/constructed_task_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0])); writer.writeheader(); writer.writerows(audit_rows)
    if errors:
        newline = chr(10)
        raise SystemExit("constructed task audit failed:" + newline + "- " + (newline + "- ").join(errors))
    print(f"constructed task audit passed: {len(audit_rows)} datasets")


if __name__ == "__main__":
    main()
