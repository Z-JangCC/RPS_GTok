"""Select a representation family using validation metrics only.

This makes the proposed ``RPS family`` comparison explicit: RPS may choose
among identifier-only, Full-Embed and anchor variants, while raw and canonical
families receive the same validation-only selection opportunity.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def family(view: str) -> str:
    if view.startswith("rps_gtok_"):
        return "rps"
    if view.startswith("canonical_"):
        return "canonical"
    return "raw"


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--candidates", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with Path(args.candidates).open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    groups=defaultdict(list)
    for row in rows:
        groups[(family(row["view"]), row["seed"])].append(row)
    selected=[]
    for (method_family, seed), items in sorted(groups.items()):
        best=max(items,key=lambda r:(float(r["val_balanced_accuracy"]),float(r["val_macro_f1"]),-int(r["parameters"])))
        selected.append({**best,"family":method_family,"selection":"validation_balanced_accuracy_then_macro_f1_then_parameters"})
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8",newline="") as fh:
        fields=sorted({key for row in selected for key in row}); writer=csv.DictWriter(fh,fieldnames=fields); writer.writeheader(); writer.writerows(selected)
    summary={}
    for method_family in sorted({r["family"] for r in selected}):
        vals=[float(r["test_balanced_accuracy"]) for r in selected if r["family"]==method_family]
        summary[method_family]={"seeds":len(vals),"mean_test_balanced_accuracy":sum(vals)/max(1,len(vals)),"selected_views":sorted({r["view"] for r in selected if r["family"]==method_family})}
    out.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=="__main__": main()
