"""Combine all-dataset structural downstream outputs into mean/std tables."""

from __future__ import annotations

import csv
import json
import argparse
from collections import defaultdict
from pathlib import Path


def read(path):
    if not path.exists(): return []
    with path.open(encoding="utf-8", newline="") as fh: return list(csv.DictReader(fh))


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--raw",default="rebuttal/experiments/E9_all_dataset_structural_edge_only/results.csv"); parser.add_argument("--rps",default="rebuttal/experiments/E9_all_dataset_structural_rps_merged_v4_clean/results.csv"); args=parser.parse_args()
    raw=read(Path(args.raw))
    rps=read(Path(args.rps))
    rows=raw+rps
    out=Path("rebuttal/tables/Table_R12_all_dataset_structural_downstream_seed_level.csv"); out.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        fields=sorted({key for row in rows for key in row})
        with out.open("w",encoding="utf-8",newline="") as fh:
            writer=csv.DictWriter(fh,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    grouped=defaultdict(list)
    for row in rows: grouped[(row["dataset"],row["view"])].append(row)
    aggregate=[]
    for (dataset,view),items in sorted(grouped.items()):
        def ms(key):
            xs=[float(x[key]) for x in items if x.get(key) not in (None,"")]
            if not xs:return "",""
            mean=sum(xs)/len(xs); sd=(sum((x-mean)**2 for x in xs)/max(1,len(xs)-1))**0.5
            return f"{mean:.6f}",f"{sd:.6f}"
        mae,mae_sd=ms("test_mae"); rmse,rmse_sd=ms("test_rmse"); r2,r2_sd=ms("test_r2")
        aggregate.append({"dataset":dataset,"view":view,"seeds":len(items),"mae_mean":mae,"mae_sd":mae_sd,"rmse_mean":rmse,"rmse_sd":rmse_sd,"r2_mean":r2,"r2_sd":r2_sd,"max_truncation":max(float(x.get("test_truncation_rate",0)) for x in items)})
    agg=Path("rebuttal/tables/Table_R12_all_dataset_structural_downstream_aggregate.csv")
    with agg.open("w",encoding="utf-8",newline="") as fh:
        writer=csv.DictWriter(fh,fieldnames=list(aggregate[0]) if aggregate else ["dataset"]); writer.writeheader(); writer.writerows(aggregate)
    print(json.dumps({"seed_rows":len(rows),"aggregate_rows":len(aggregate)},indent=2))


if __name__=="__main__": main()
