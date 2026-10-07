from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
def main():
 rows=[]
 for p in sorted((ROOT/'rebuttal/experiments/E14_ogbg_molhiv_balanced/downstream_post_schema_union').glob('*/*/metrics.json')):
  m=json.loads(p.read_text()); rows.append({'view':p.parts[-3],'seed':'','path':str(p),'balanced_accuracy':m['test']['balanced_accuracy'],'macro_f1':m['test']['macro_f1'],'prediction_unique':m['test']['prediction_unique'],'truncation_rate':m['sequence_audit']['test']['truncation_rate']})
  rows[-1]['seed']=p.parent.name.removeprefix('seed')
 out=ROOT/'rebuttal/tables/Table_E14_ogbg_molhiv_balanced_post_schema_seed_level.csv';out.parent.mkdir(exist_ok=True)
 with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 agg=[]
 for view in sorted({r['view'] for r in rows}):
  rr=[r for r in rows if r['view']==view];mean=sum(r['balanced_accuracy'] for r in rr)/len(rr);agg.append({'view':view,'seeds':len(rr),'mean_balanced_accuracy':mean,'std_balanced_accuracy':(sum((r['balanced_accuracy']-mean)**2 for r in rr)/len(rr))**.5,'mean_macro_f1':sum(r['macro_f1'] for r in rr)/len(rr),'collapse_rate':sum(r['prediction_unique']<=1 for r in rr)/len(rr),'max_truncation_rate':max(r['truncation_rate'] for r in rr)})
 out2=ROOT/'rebuttal/tables/Table_E14_ogbg_molhiv_balanced_post_schema_aggregate.csv'
 with out2.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
 print(json.dumps(agg,indent=2))
if __name__=='__main__':main()
