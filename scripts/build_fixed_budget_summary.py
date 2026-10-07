"""Summarize fixed-token-budget topology retention runs."""
from __future__ import annotations
import csv,glob,json
from pathlib import Path
def main():
 rows=[]
 for path in sorted(glob.glob("rebuttal/experiments/E12_fixed_budget/*/results.csv")):
  budget=path.split("MUTAG_budget")[-1].split("/")[0]
  with open(path,encoding="utf-8",newline="") as handle:
   for row in csv.DictReader(handle): row["budget"]=int(budget); rows.append(row)
 out=Path("rebuttal/tables/Table_R13_fixed_budget_topology.csv");out.parent.mkdir(parents=True,exist_ok=True)
 with out.open("w",encoding="utf-8",newline="") as handle:
  fields=sorted({key for row in rows for key in row});writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
 grouped={}
 for row in rows: grouped.setdefault((row["budget"],row["view"]),[]).append(row)
 agg=[]
 for (budget,view),items in sorted(grouped.items()):
  def mean(key):return sum(float(r[key]) for r in items)/len(items)
  agg.append({"budget":budget,"view":view,"seeds":len(items),"mae_mean":mean("test_mae"),"rmse_mean":mean("test_rmse"),"r2_mean":mean("test_r2"),"max_truncation":max(float(r["test_truncation_rate"]) for r in items)})
 with Path("rebuttal/tables/Table_R13_fixed_budget_topology_aggregate.csv").open("w",encoding="utf-8",newline="") as handle:
  writer=csv.DictWriter(handle,fieldnames=list(agg[0]));writer.writeheader();writer.writerows(agg)
 print(json.dumps({"rows":len(rows),"aggregate":len(agg)},indent=2))
if __name__=="__main__":main()
