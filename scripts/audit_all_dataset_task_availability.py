"""Audit whether every materialized dataset has labels for supervised tasks."""

from __future__ import annotations

import csv
from pathlib import Path

from gptok2.data.io import load_records


def main() -> None:
    root = Path("rebuttal/data_21")
    rows=[]
    for path in sorted(root.glob("*.jsonl")):
        records=load_records(path)
        labels=[record.y for record in records if record.y is not None]
        rows.append({
            "dataset":path.stem,
            "graphs":len(records),
            "supervised_y_graphs":len(labels),
            "supervised_task_available":bool(labels),
            "structural_task_available":len(records)>=12,
            "domains":sorted({str(record.metadata.get("domain", "unknown")) for record in records}),
            "directed":any(bool(record.directed) for record in records),
        })
    out=Path("rebuttal/tables/Table_R11_all_dataset_task_availability.csv")
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__=="__main__": main()
