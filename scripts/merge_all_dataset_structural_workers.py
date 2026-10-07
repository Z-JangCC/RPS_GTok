"""Merge independent all-dataset structural workers without hiding failures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def rows(path: Path):
    if not path.exists(): return []
    with path.open(encoding="utf-8", newline="") as fh: return list(csv.DictReader(fh))


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--worker-root",required=True); parser.add_argument("--out",required=True); parser.add_argument("--workers",nargs="*",default=None); args=parser.parse_args()
    root=Path(args.worker_root); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    result_rows=[]; status_rows=[]; protocols=[]
    worker_paths = [root / name for name in args.workers] if args.workers else sorted(root.glob("*_v*"))
    for worker in worker_paths:
        result_rows.extend(rows(worker/"results.csv")); status_rows.extend(rows(worker/"dataset_status.csv"))
        if (worker/"protocol.json").exists(): protocols.append(json.loads((worker/"protocol.json").read_text(encoding="utf-8")))
    result_rows.sort(key=lambda r:(r.get("dataset",""),r.get("view",""),r.get("seed","")))
    status_rows.sort(key=lambda r:r.get("dataset",""))
    for name,data in (("results.csv",result_rows),("dataset_status.csv",status_rows)):
        if not data: continue
        with (out/name).open("w",encoding="utf-8",newline="") as fh:
            fields=sorted({key for row in data for key in row}); writer=csv.DictWriter(fh,fieldnames=fields); writer.writeheader(); writer.writerows(data)
    (out/"protocol.json").write_text(json.dumps({"workers":len(protocols),"protocols":protocols,"merge_policy":"include all completed rows; retain failed/too_small statuses; no imputation"},indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({"result_rows":len(result_rows),"status_rows":len(status_rows),"workers":len(protocols)},indent=2))


if __name__=="__main__": main()
