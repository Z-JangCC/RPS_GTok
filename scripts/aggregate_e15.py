from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 rows=[]
 for p in sorted((ROOT/'rebuttal/experiments/E15_powerflow_perturbed_small/downstream_post_schema_union').glob('*/*/metrics.json')):
  m=json.loads(p.read_text());rows.append({'view':p.parts[-3],'seed':p.parent.name.removeprefix('seed'),'mae':m['test']['mae'],'rmse':m['test']['rmse'],'r2':m['test']['r2'],'truncation_rate':m['sequence_audit']['test']['truncation_rate'],'train_truncation_rate':m['sequence_audit']['train']['truncation_rate'],'val_truncation_rate':m['sequence_audit']['val']['truncation_rate']})
 out=ROOT/'rebuttal/tables/Table_E15_powerflow_perturbed_post_schema_seed_level.csv';out.parent.mkdir(exist_ok=True)
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 agg=[]
 for v in sorted({r['view'] for r in rows}):
  rr=[r for r in rows if r['view']==v];mean=sum(r['mae'] for r in rr)/len(rr);agg.append({'view':v,'seeds':len(rr),'mean_mae':mean,'std_mae':(sum((r['mae']-mean)**2 for r in rr)/len(rr))**.5,'mean_rmse':sum(r['rmse'] for r in rr)/len(rr),'mean_r2':sum(r['r2'] for r in rr)/len(rr),'max_test_truncation':max(r['truncation_rate'] for r in rr),'max_train_truncation':max(r['train_truncation_rate'] for r in rr),'max_val_truncation':max(r['val_truncation_rate'] for r in rr)})
 out2=ROOT/'rebuttal/tables/Table_E15_powerflow_perturbed_post_schema_aggregate.csv'
 with out2.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
 print(json.dumps(agg,indent=2))
if __name__=='__main__':main()
