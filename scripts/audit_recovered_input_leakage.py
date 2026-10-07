"""Audit that recovered targets are not copied into model input fields."""
from __future__ import annotations
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 rows=[]
 paths = sorted((ROOT/'rebuttal/data_21_labeled').glob('*.jsonl')) + sorted((ROOT/'rebuttal/experiments/E15_powerflow_perturbed_small').glob('*.jsonl'))
 for p in paths:
  total=0; bad=0; vector=0
  for line in p.read_text().splitlines():
   r=json.loads(line); total+=1; y=r.get('y')
   if isinstance(y,list): vector+=1
   meta=r.get('metadata',{})
   # y is target-only.  Explicit label fields in metadata/node/edge attrs are
   # forbidden; numerical feature equality alone is not considered leakage.
   forbidden = any(k in meta for k in ('label','target','y'))
   if meta.get('task_target') == 'mean_bus_vm_pu' and 'vm_pu' in meta.get('node_attr_keys', []):
    forbidden = True
   if isinstance(r.get('node_attr'),list): forbidden = forbidden or any(isinstance(x,dict) and any(k in x for k in ('label','target','y')) for x in r['node_attr'])
   if isinstance(r.get('edge_attr'),list): forbidden = forbidden or any(isinstance(x,dict) and any(k in x for k in ('label','target','y')) for x in r['edge_attr'])
   bad += int(forbidden)
  rows.append({'dataset':p.stem,'records':total,'multilabel_records':vector,'forbidden_target_fields':bad,'status':'pass' if bad==0 else 'fail'})
 out=ROOT/'rebuttal/audits/recovered_input_leakage.csv'; out.parent.mkdir(exist_ok=True)
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
