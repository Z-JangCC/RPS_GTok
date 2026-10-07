"""Aggregate locked no-truncation controls without mixing pilot summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def main() -> None:
    root = Path("rebuttal/experiments/E7_fair_grid")
    candidates = sorted(root.glob("*_no_truncation_v*/selected_results.csv"))
    selected = {}
    for path in candidates:
        name = path.parent.name
        try:
            version = int(name.rsplit("_v", 1)[1])
        except ValueError:
            version = 0
        dataset = name.split("_no_truncation", 1)[0]
        old = selected.get(dataset)
        protocol = path.parent / "protocol.json"
        adapter_mode = ""
        if protocol.exists():
            try:
                adapter_mode = json.loads(protocol.read_text(encoding="utf-8")).get("adapter_mode", "")
            except json.JSONDecodeError:
                pass
        mode_priority = {"shared_plain": 3, "shared_full_embed": 2, "native": 1, "": 0}.get(adapter_mode, 0)
        score = (mode_priority, version)
        if old is None or score > old[0]:
            selected[dataset] = (score, path)
    files = [item[1] for item in sorted(selected.values())]
    rows = []
    for path in files:
        payload = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
        for row in payload:
            row["run"] = path.parent.name
            rows.append(row)
    if not rows:
        raise SystemExit("no locked no-truncation runs")
    out = Path("rebuttal/tables/Table_R5_truncation_semantic_locked.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["view"], []).append(row)
    aggregate=[]
    for view, items in sorted(grouped.items()):
        def mean(key): return sum(float(x[key]) for x in items)/len(items)
        aggregate.append({"view":view,"seeds":len(items),"accuracy_mean":mean("test_accuracy"),"balanced_accuracy_mean":mean("test_balanced_accuracy"),"macro_f1_mean":mean("test_macro_f1"),"max_truncation":max(float(x["test_truncation_rate"]) for x in items),"max_discarded_fraction":max(float(x["test_discarded_fraction"]) for x in items),"prediction_collapse_runs":sum(int(x["prediction_unique"])<=1 for x in items)})
    agg_path=Path("rebuttal/tables/Table_R5_truncation_semantic_locked_aggregate.csv")
    with agg_path.open("w",encoding="utf-8",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(aggregate[0])); writer.writeheader(); writer.writerows(aggregate)
    print(json.dumps({"rows":len(rows),"aggregate":aggregate},indent=2))


if __name__ == "__main__": main()
