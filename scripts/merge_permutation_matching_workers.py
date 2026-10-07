"""Merge permutation-matching worker outputs with no imputation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read(path: Path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); parser.add_argument("--out", required=True); args = parser.parse_args()
    root=Path(args.root); out=Path(args.out); out.mkdir(parents=True,exist_ok=True); rows=[]
    for path in sorted(root.glob("**/results.csv")):
        rows.extend(read(path))
    rows.sort(key=lambda row:(row.get("dataset",""),row.get("view",""),row.get("seed","")))
    with (out/"results.csv").open("w",encoding="utf-8",newline="") as handle:
        fields=sorted({key for row in rows for key in row}); writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    grouped={}
    for row in rows: grouped.setdefault((row["dataset"],row["view"]),[]).append(row)
    aggregate=[]
    for (dataset,view),items in sorted(grouped.items()):
        def mean_sd(key):
            values=[float(row[key]) for row in items]; mean=sum(values)/len(values); sd=(sum((value-mean)**2 for value in values)/max(1,len(values)-1))**0.5; return mean,sd
        accuracy,accuracy_sd=mean_sd("test_accuracy"); balanced,balanced_sd=mean_sd("test_balanced_accuracy"); macro,macro_sd=mean_sd("test_macro_f1")
        aggregate.append({"dataset":dataset,"view":view,"seeds":len(items),"accuracy_mean":accuracy,"accuracy_sd":accuracy_sd,"balanced_accuracy_mean":balanced,"balanced_accuracy_sd":balanced_sd,"macro_f1_mean":macro,"macro_f1_sd":macro_sd,"collapse_runs":sum(int(row["prediction_unique"])<=1 for row in items)})
    with (out/"aggregate.csv").open("w",encoding="utf-8",newline="") as handle:
        fields=list(aggregate[0]) if aggregate else ["dataset"]; writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(aggregate)
    print(json.dumps({"rows":len(rows),"aggregate":len(aggregate)},indent=2))


if __name__=="__main__": main()
