"""Validate completed corrected semantic runs and emit protocol manifests."""

from __future__ import annotations

import json
import re
from pathlib import Path


def main() -> None:
    root = Path("rebuttal/experiments/E4_semantic_tasks")
    candidates = sorted(root.glob("*_corrected*/summary.json"))
    selected = {}
    def rank(path: Path):
        match = re.search(r"_corrected_v(\d+)$", path.parent.name)
        return int(match.group(1)) if match else 0
    for summary in candidates:
        dataset = re.sub(r"_corrected(?:_v\d+)?$", "", summary.parent.name)
        if dataset not in selected or rank(summary) >= rank(selected[dataset]):
            selected[dataset] = summary
    for summary in sorted(selected.values()):
        rows = json.loads(summary.read_text(encoding="utf-8"))
        if not rows:
            continue
        run = summary.parent.name
        if not re.search(r"_corrected(?:_v\d+)?$", run):
            continue
        expected = len({int(row["seed"]) for row in rows}) * len({row["view"] for row in rows})
        if len(rows) != expected:
            raise RuntimeError(f"{run}: expected {expected} rows, found {len(rows)}")
        hashes = {}
        collapsed = []
        for row in rows:
            audit = row.get("view_audit", {}).get("test", {})
            if not audit.get("sha1"):
                raise RuntimeError(f"{run}: missing view audit for {row.get('view')}")
            hashes[row["view"]] = audit["sha1"]
            test = row.get("test", {})
            if test.get("prediction_unique") != 2:
                collapsed.append({"view": row.get("view"), "seed": row.get("seed"), "unique": test.get("prediction_unique")})
        if len(hashes) != len(set(hashes.values())):
            raise RuntimeError(f"{run}: view sequence hash collision")
        protocol = {
            "run": run,
            "dataset": re.sub(r"_corrected(?:_v\d+)?$", "", run),
            "views": sorted(hashes),
            "seeds": sorted({int(row["seed"]) for row in rows}),
            "expected_rows": expected,
            "completed_rows": len(rows),
            "class_weight": True,
            "view_test_sha1": hashes,
            "checks": {"complete": True, "distinct_view_sequences": True, "noncollapsed_predictions": not collapsed},
            "collapsed_predictions": collapsed,
        }
        (summary.parent / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
        print(f"audited {run}: {len(rows)} rows")


if __name__ == "__main__":
    main()
