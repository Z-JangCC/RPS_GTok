"""Audit authoritative shared-adapter downstream runs."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def audit(name, root, expected, task):
 p=root/'protocol.json'; proto=json.loads(p.read_text()); metrics=list(root.glob('*/*/metrics.json'))
 errors=[]
 if proto.get('adapter_mode')!='shared_plain' or not proto.get('shared_vocabulary'):
  errors.append('protocol is not shared_plain with union vocabulary')
 if len(metrics)!=expected: errors.append(f'expected {expected} metrics, found {len(metrics)}')
 params=set(); trunc=[]
 for mpath in metrics:
  m=json.loads(mpath.read_text()); params.add(int(m['parameter_count']));trunc.append(float(m['sequence_audit']['test']['truncation_rate']))
  if task=='classification' and int(m['test'].get('prediction_unique',0))<2: errors.append(f'collapse:{mpath}')
 if len(params)!=1: errors.append(f'parameter_counts={sorted(params)}')
 if max(trunc,default=0)>0: errors.append(f'truncation={max(trunc)}')
 return {'name':name,'metrics':len(metrics),'parameter_counts':sorted(params),'max_test_truncation':max(trunc,default=0),'errors':errors}
def main():
 e14_root=ROOT/'rebuttal/experiments/E14_ogbg_molhiv_balanced/downstream_post_schema_union'
 if not e14_root.exists(): e14_root=ROOT/'rebuttal/experiments/E14_ogbg_molhiv_balanced/downstream_shared_plain_union'
 e15_root=ROOT/'rebuttal/experiments/E15_powerflow_perturbed_small/downstream_post_schema_union'
 if not e15_root.exists(): e15_root=ROOT/'rebuttal/experiments/E15_powerflow_perturbed_small/downstream_shared_plain_union'
 rows=[audit('E14_post_schema_union',e14_root,9,'classification'),audit('E15_post_schema_union',e15_root,9,'regression')]
 out=ROOT/'rebuttal/audits/matched_consumer_runs.json';out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
 if any(r['errors'] for r in rows): raise SystemExit(1)
if __name__=='__main__':main()
