"""Cache-corrected representation-level permutation audit."""
from __future__ import annotations
import argparse,csv,json,random,time
from pathlib import Path
from gptok2.data.io import load_records
from gptok2.canonical import canonicalize
from gptok2.data.schema_spec import SchemaSpec
from gptok2_tokenizer import GPTok2Tokenizer
from rps_gtok_consumption.views import TokenViewBuilder
from scripts.run_topology_shift_downstream import permute_record

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-dir',default='rebuttal/data_21');p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--out',required=True);p.add_argument('--max-graphs',type=int,default=20);p.add_argument('--permutations',type=int,default=3);p.add_argument('--max-search',type=int,default=100);p.add_argument('--timeout',type=float,default=.5);a=p.parse_args();rows=[]
 for dataset in a.datasets:
  path=Path(a.data_dir)/f'{dataset}.jsonl'; records=load_records(path)[:a.max_graphs];
  if not records:continue
  spec=SchemaSpec(max_search_nodes=a.max_search,timeout_sec=a.timeout); tok=GPTok2Tokenizer({'canonicalization':{'max_search_nodes':a.max_search,'timeout_sec':a.timeout},'patch':{'sparse_edge_patches':True,'sparse_density_threshold':1.0,'sparse_clustering_threshold':1.0,'max_patches_per_graph':128,'max_cycle_signatures':32,'max_triangle_signatures':128},'motif_macro':{'max_structural_macros':32,'min_structural_count':1000,'max_parameterized_span_len':32,'max_merge_schemas':32,'min_merge_schema_count':1000,'max_code_schemas':32,'min_code_schema_count':1000}}).fit(records[:max(1,int(.7*len(records)))])
  b=TokenViewBuilder(tok,seed=2026).fit(records,['rps_gtok_plus']); stats={'edge_list':[],'canonical_edge_list':[],'rps_gtok_plus':[]}; exact=[]
  for i,record in enumerate(records):
   base=canonicalize(record,spec,exact=True,max_search_nodes=a.max_search,timeout_sec=a.timeout); exact_flags=[]
   for seed in range(a.permutations):
    perm=permute_record(record,seed+8000+i); other=canonicalize(perm,spec,exact=True,max_search_nodes=a.max_search,timeout_sec=a.timeout); exact_flags.append(bool(base.exact_completed and other.exact_completed))
    for view in stats: stats[view].append(float(b.build(record,view)==b.build(perm,view)))
   exact.extend(exact_flags)
  rows.append({'dataset':dataset,'graphs':len(records),'permutations':a.permutations,'raw_equality':sum(stats['edge_list'])/max(1,len(stats['edge_list'])),'canonical_equality':sum(stats['canonical_edge_list'])/max(1,len(stats['canonical_edge_list'])),'rps_equality':sum(stats['rps_gtok_plus'])/max(1,len(stats['rps_gtok_plus'])),'exact_pair_fraction':sum(exact)/max(1,len(exact)),'max_search':a.max_search,'timeout_sec':a.timeout})
 out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('w',encoding='utf-8',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'rows':len(rows),'out':str(out)},indent=2))
if __name__=='__main__':main()
