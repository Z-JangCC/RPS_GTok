"""Aggregate validation-selected fair-grid results and evaluate the RPS gate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rps-view", default="rps_gtok_plus")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    with Path(args.selected).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["view"]].append(row)
    if args.rps_view not in grouped:
        raise SystemExit(f"missing RPS view {args.rps_view}")
    aggregate = []
    for view, items in sorted(grouped.items()):
        values = np.asarray([float(x["test_balanced_accuracy"]) for x in items])
        aggregate.append({
            "view": view,
            "seeds": len(values),
            "balanced_accuracy_mean": float(values.mean()),
            "balanced_accuracy_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "macro_f1_mean": float(np.mean([float(x["test_macro_f1"]) for x in items])),
            "collapsed_runs": sum(int(x["prediction_unique"]) <= 1 for x in items),
        })
    rps = {int(x["seed"]): float(x["test_balanced_accuracy"]) for x in grouped[args.rps_view]}
    rng = np.random.default_rng(2026)
    comparisons = []
    for view, items in sorted(grouped.items()):
        if view == args.rps_view:
            continue
        base = {int(x["seed"]): float(x["test_balanced_accuracy"]) for x in items}
        seeds = sorted(set(rps) & set(base))
        delta = np.asarray([rps[s] - base[s] for s in seeds])
        boot = np.asarray([
            rng.choice(delta, size=len(delta), replace=True).mean()
            for _ in range(args.bootstrap)
        ]) if len(delta) else np.asarray([np.nan])
        comparisons.append({
            "baseline": view,
            "paired_seeds": len(seeds),
            "mean_delta_rps_minus_baseline": float(delta.mean()) if len(delta) else None,
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
            "rps_strictly_better": bool(np.quantile(boot, 0.025) > 0),
        })
    gate = {
        "rps_view": args.rps_view,
        "primary_metric": "balanced_accuracy",
        "all_baseline_ci_lower_bounds_positive": bool(comparisons and all(x["rps_strictly_better"] for x in comparisons)),
        "no_collapsed_rps_runs": all(int(x["prediction_unique"]) > 1 for x in grouped[args.rps_view]),
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(aggregate[0])); writer.writeheader(); writer.writerows(aggregate)
    out.with_suffix(".gate.json").write_text(json.dumps({"gate": gate, "comparisons": comparisons}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"gate": gate, "comparisons": comparisons}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
