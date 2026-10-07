"""Audit raw converted records for label fields that would leak node targets."""
from __future__ import annotations
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 rows=[]
 for p in sorted((ROOT/'rebuttal/data_21').glob('*.jsonl')):
  first=json.loads(p.read_text().splitlines()[0]); keys=first.get('metadata',{}).get('node_attr_keys',[])
  rows.append({'dataset':p.stem,'node_attr_keys':json.dumps(keys,sort_keys=True),'contains_label_field':int(any(str(k).lower() in {'label','target','y'} for k in keys)),'semantic_authorized':0 if any(str(k).lower() in {'label','target','y'} for k in keys) else 1,'policy':'structural_only' if any(str(k).lower() in {'label','target','y'} for k in keys) else 'task_source_required'})
 out=ROOT/'rebuttal/audits/materialized_label_field_audit.csv';out.parent.mkdir(exist_ok=True)
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
