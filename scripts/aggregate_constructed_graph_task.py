"""Aggregate and audit the four parallel E16 constructed-task shards."""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path

import numpy as np


ROOT = Path("rebuttal")


def read(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="E16_constructed_topology_task_g")
    parser.add_argument("--out-name", default="Table_E16_constructed_topology_proxy_aggregate.csv")
    parser.add_argument("--expected-truncation", type=float, default=-1.0)
    args = parser.parse_args()
    shard_rows = []
    status_rows = []
    for path in sorted((ROOT / "experiments").glob(f"{args.prefix}*/results.csv")):
        shard_rows.extend(read(path))
    for path in sorted((ROOT / "experiments").glob(f"{args.prefix}*/dataset_status.csv")):
        status_rows.extend(read(path))
    if len({row["dataset"] for row in status_rows if row.get("status") == "ok"}) != 21:
        raise SystemExit(f"expected 21 completed E16 datasets, got {len(status_rows)} statuses")
    aggregate = []
    for dataset in sorted({row["dataset"] for row in shard_rows}):
        rows = [row for row in shard_rows if row["dataset"] == dataset]
        if len(rows) != 9 or len({(row["view"], row["seed"]) for row in rows}) != 9:
            raise SystemExit(f"{dataset}: incomplete 3-view/3-seed grid ({len(rows)})")
        if len({row["parameters"] for row in rows}) != 1:
            raise SystemExit(f"{dataset}: parameter mismatch")
        # Long raw edge streams may truncate at the fixed 512-token budget.
        # This is retained as an explicit representation diagnostic rather than
        # silently dropping the dataset or treating the metric as lossless.
        if args.expected_truncation >= 0 and any(float(row["test_truncation_rate"]) > args.expected_truncation for row in rows):
            raise SystemExit(f"{dataset}: truncation exceeds requested bound")
        if len({row["view_sha1"] for row in rows}) != 3:
            raise SystemExit(f"{dataset}: view sequence hash collision")
        for view in sorted({row["view"] for row in rows}):
            values = np.asarray([float(row["test_mae"]) for row in rows if row["view"] == view], dtype=float)
            rmses = np.asarray([float(row["test_rmse"]) for row in rows if row["view"] == view], dtype=float)
            r2 = np.asarray([float(row["test_r2"]) for row in rows if row["view"] == view], dtype=float)
            if not all(np.isfinite(values)) or not all(np.isfinite(rmses)) or not all(np.isfinite(r2)):
                raise SystemExit(f"{dataset}/{view}: non-finite metric")
            trunc = np.asarray([float(row["test_truncation_rate"]) for row in rows if row["view"] == view], dtype=float)
            discarded = np.asarray([float(row["test_discarded_fraction"]) for row in rows if row["view"] == view], dtype=float)
            aggregate.append({"dataset": dataset, "view": view, "seeds": len(values), "parameters": rows[0]["parameters"], "mean_mae": float(values.mean()), "std_mae": float(values.std(ddof=1)), "mean_rmse": float(rmses.mean()), "std_rmse": float(rmses.std(ddof=1)), "mean_r2": float(r2.mean()), "std_r2": float(r2.std(ddof=1)), "test_truncation_rate": float(trunc.mean()), "test_discarded_fraction": float(discarded.mean()), "target_source": "constructed_topology_proxy"})
    out = ROOT / "tables" / args.out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0])); writer.writeheader(); writer.writerows(aggregate)
    seed_name = args.out_name.replace("_aggregate.csv", "_seed_level.csv")
    with (ROOT / "tables" / seed_name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in shard_rows for key in row})); writer.writeheader(); writer.writerows(shard_rows)
    protocol_name = args.out_name.replace("_aggregate.csv", "_protocol.json")
    (ROOT / "tables" / protocol_name).write_text(json.dumps({"datasets": sorted({row["dataset"] for row in shard_rows}), "views": sorted({row["view"] for row in shard_rows}), "seeds": sorted({int(row["seed"]) for row in shard_rows}), "source": "constructed_tasks/*.jsonl", "input_graph_records_y": "null", "test_used_for_selection": False, "aggregate": str(out), "max_truncation_bound": args.expected_truncation}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"datasets": len(set(row["dataset"] for row in aggregate)), "rows": len(aggregate), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
