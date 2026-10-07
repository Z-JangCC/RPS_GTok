"""Summarize auditable semantic-target availability after external recovery."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    audit = {r["dataset"]: r for r in csv.DictReader((ROOT / "rebuttal/audits/label_recovery_audit.csv").open())}
    node_audit = {r["dataset"]: r for r in json.loads((ROOT / "rebuttal/audits/node_ego_label_recovery.json").read_text())}
    rows = []
    for path in sorted((ROOT / "rebuttal/data_21").glob("*.jsonl")):
        dataset = path.stem
        source = audit.get(dataset, {})
        labeled = ROOT / "rebuttal/data_21_labeled" / path.name
        ys = []
        if labeled.exists():
            for line in labeled.read_text().splitlines():
                y = json.loads(line).get("y")
                if y is not None:
                    ys.append(y)
        status = source.get("status", "unscanned")
        if dataset in node_audit:
            status = node_audit[dataset]["status"]
        if dataset == "ieee_power_grid" and labeled.exists():
            status = "recovered_powerflow_target"
        multilabel = any(isinstance(y, list) for y in ys)
        scalar = [str(y) for y in ys if not isinstance(y, list)]
        class_count = len(set(scalar)) if scalar else 0
        if status == "recovered" and multilabel:
            eligibility = "multilabel_candidate_official_mask_required"
        elif status == "recovered" and class_count < 2:
            eligibility = "single_class_prefix_no_classification"
        elif status == "recovered":
            eligibility = "classification_candidate"
        elif status == "recovered_powerflow_target":
            eligibility = "regression_candidate"
        else:
            eligibility = "not_authorized"
        rows.append({"dataset": dataset, "records": sum(1 for _ in path.open()), "recovered_labels": len(ys), "source": source.get("source", ""), "status": status, "class_count": class_count, "multilabel": multilabel, "semantic_eligibility": eligibility, "target_policy": "official_graph_label" if dataset not in {"ogbg_molpcba", "ieee_power_grid"} else ("official_128_task_multilabel_with_observed_mask" if dataset == "ogbg_molpcba" else "mean_bus_vm_pu_from_pandapower_runpp")})
    out = ROOT / "rebuttal/tables/Table_R11b_label_recovery_expanded.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__": main()
