"""Build a machine-readable manifest for all numbers cited in rebuttal prose."""
from __future__ import annotations
import csv,json
from pathlib import Path
from collections import defaultdict

def read(path):
    with Path(path).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def mean(rows,key):
    values=[float(r[key]) for r in rows if r.get(key) not in (None,'')]
    return sum(values)/len(values) if values else None

def main():
    out={}
    r4=read('rebuttal/tables/Table_R4_semantic_downstream_corrected_aggregate.csv')
    out['semantic_balanced_accuracy']={f"{r['dataset']}/{r['view']}":float(r['balanced_accuracy_mean']) for r in r4}
    r5=read('rebuttal/tables/Table_R5_truncation_semantic_locked_aggregate.csv')
    out['no_truncation']={r['view']:{'balanced_accuracy':float(r['balanced_accuracy_mean']),'max_truncation':float(r['max_truncation']),'collapse_runs':int(r['prediction_collapse_runs'])} for r in r5}
    r12=read('rebuttal/tables/Table_R12_all_dataset_structural_downstream_aggregate.csv')
    grouped=defaultdict(list)
    for r in r12:grouped[r['view']].append(r)
    by_dataset=defaultdict(dict)
    for r in r12: by_dataset[r['dataset']][r['view']]=float(r['mae_mean'])
    out['structural_mae_wins']={view:sum(vals.get(view, float('inf')) < min((value for other,value in vals.items() if other != view), default=float('inf')) for vals in by_dataset.values()) for view in sorted({r['view'] for r in r12})}
    e8=read('rebuttal/tables/Table_E8_gin_foundation_baseline.csv')
    out['gin_balanced_accuracy']={dataset:mean([r for r in e8 if r['dataset']==dataset],'balanced_accuracy') for dataset in sorted({r['dataset'] for r in e8})}
    e1=read('rebuttal/tables/Table_E1_cachefixed_permutation_audit.csv')
    out['cachefixed_permutation']={r['dataset']:{k:float(r[k]) for k in ('raw_equality','canonical_equality','rps_equality','exact_pair_fraction')} for r in e1}
    r13=read('rebuttal/tables/Table_R13_fixed_budget_topology_aggregate.csv')
    out['fixed_budget_mae']={f"{r['budget']}/{r['view']}":float(r['mae_mean']) for r in r13}
    path=Path('rebuttal/rebuttal_number_manifest.json');path.write_text(json.dumps(out,indent=2,sort_keys=True),encoding='utf-8');print(f'wrote {path}')

if __name__=='__main__':main()
