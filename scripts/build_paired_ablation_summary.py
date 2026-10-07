"""Aggregate corrected paired reconstruction/downstream diagnostics."""
from __future__ import annotations
import csv,glob
from pathlib import Path
def main():
 rows=[]
 for path in glob.glob('rebuttal/experiments/E5_paired_semantic/*_corrected_v*/results.csv'):
  with open(path,encoding='utf-8',newline='') as handle:rows.extend(list(csv.DictReader(handle)))
 out=Path('rebuttal/tables/Table_E5_paired_reconstruction_downstream.csv');out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',encoding='utf-8',newline='') as handle:
  fields=sorted({k for r in rows for k in r});w=csv.DictWriter(handle,fieldnames=fields);w.writeheader();w.writerows(rows)
 grouped={}
 for r in rows:grouped.setdefault((r['dataset'],r['drop']),[]).append(r)
 agg=[]
 for (ds,drop),items in sorted(grouped.items()):
  def m(k):return sum(float(r[k]) for r in items)/len(items)
  agg.append({'dataset':ds,'drop':drop,'seeds':len(items),'strict_reconstruction_mean':m('strict_reconstruction_test'),'balanced_accuracy_mean':m('test_balanced_accuracy'),'macro_f1_mean':m('test_macro_f1'),'collapse_runs':sum(int(r['prediction_unique'])<=1 for r in items)})
 with Path('rebuttal/tables/Table_E5_paired_reconstruction_downstream_aggregate.csv').open('w',encoding='utf-8',newline='') as handle:
  w=csv.DictWriter(handle,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
 print('wrote',len(rows),'rows')
if __name__=='__main__':main()
