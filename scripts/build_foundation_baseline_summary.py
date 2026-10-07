"""Summarize matched GIN graph-model reference results."""
from __future__ import annotations
import csv,json,glob
from pathlib import Path
def main():
    rows=[]
    for path in sorted(glob.glob("rebuttal/experiments/E8_foundation_baselines/*gin_v2.csv")):
        dataset=Path(path).stem.replace("_gin_v2","")
        with open(path,encoding="utf-8",newline="") as handle:
            for row in csv.DictReader(handle):
                row["dataset"]=dataset; rows.append(row)
    out=Path("rebuttal/tables/Table_E8_gin_foundation_baseline.csv");out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8",newline="") as handle:
        fields=sorted({key for row in rows for key in row}|{"dataset"});writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in rows: writer.writerow(row)
    print(json.dumps({"rows":len(rows)},indent=2))
if __name__=="__main__":main()
