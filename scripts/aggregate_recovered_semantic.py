"""Aggregate E13 recovered semantic runs without hiding collapse diagnostics."""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    rows=[]
    for p in sorted((ROOT/'rebuttal/experiments/E13_recovered_semantic').glob('mutag_30ep/*/seed*/metrics.json')):
        m=json.loads(p.read_text()); view=p.parts[-3]; dataset=p.parts[-4]; seed=p.parts[-2].removeprefix('seed'); test=m['test'];
        rows.append({'dataset':dataset,'view':view,'seed':seed,'balanced_accuracy':test.get('balanced_accuracy'),'macro_f1':test.get('macro_f1'),'accuracy':test.get('accuracy'),'prediction_unique':test.get('prediction_unique'),'majority_accuracy':test.get('majority_accuracy'),'truncation_rate':m['sequence_audit']['test'].get('truncation_rate'),'epochs_ran':m['epochs_ran'],'elapsed_sec':m['elapsed_sec']})
    out=ROOT/'rebuttal/tables/Table_E13_recovered_semantic_seed_level.csv'; out.parent.mkdir(parents=True,exist_ok=True)
    if rows:
        with out.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
    agg=[]
    by={}
    for r in rows: by.setdefault((r['dataset'],r['view']),[]).append(r)
    for (d,v),rr in sorted(by.items()):
        def mean(k): return sum(float(x[k]) for x in rr)/len(rr)
        agg.append({'dataset':d,'view':v,'seeds':len(rr),'mean_balanced_accuracy':mean('balanced_accuracy'),'std_balanced_accuracy':(sum((float(x['balanced_accuracy'])-mean('balanced_accuracy'))**2 for x in rr)/len(rr))**0.5,'mean_macro_f1':mean('macro_f1'),'collapse_rate':sum(int(x['prediction_unique']<=1) for x in rr)/len(rr),'mean_truncation_rate':mean('truncation_rate')})
    out2=ROOT/'rebuttal/tables/Table_E13_recovered_semantic_aggregate.csv'
    if agg:
        with out2.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
    print(json.dumps(agg,indent=2))

if __name__=='__main__': main()
