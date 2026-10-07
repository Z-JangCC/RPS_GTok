"""Build a source-indexed, class-balanced OGB-MOLHIV audit subset."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
from torch.serialization import add_safe_globals
from torch_geometric.data.data import DataEdgeAttr,DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from ogb.graphproppred import PygGraphPropPredDataset
from gptok2.data.schema import record_from_dict,record_to_dict

ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT.parent/'graph_tokenizer_v2_base/data/kdd2027/formal_full_real_materialized/_ogb_source_cache'

def main():
 p=argparse.ArgumentParser();p.add_argument('--per-class',type=int,default=120);p.add_argument('--out',type=Path,default=ROOT/'rebuttal/experiments/E14_ogbg_molhiv_balanced');a=p.parse_args()
 add_safe_globals([DataEdgeAttr,DataTensorAttr,GlobalStorage])
 d=PygGraphPropPredDataset(name='ogbg-molhiv',root=str(CACHE))
 selected={0:[],1:[]}
 for idx in range(len(d)):
  y=int(d[idx].y.reshape(-1)[0].item())
  if len(selected[y])<a.per_class: selected[y].append(idx)
  if all(len(v)>=a.per_class for v in selected.values()): break
 records=[]
 for y,indices in selected.items():
  for idx in indices:
   data=d[idx]
   x=getattr(data,'x',None); ea=getattr(data,'edge_attr',None)
   node_type=(x.argmax(dim=1) if x is not None and x.ndim==2 and x.shape[1]>1 else (x.reshape(-1).long() if x is not None else None))
   edge_type=(ea.argmax(dim=1) if ea is not None and ea.ndim==2 and ea.shape[1]>1 else (ea.reshape(-1).long() if ea is not None else None))
   row={'graph_id':f'ogbg-molhiv_{idx}','num_nodes':int(data.num_nodes),'edges':data.edge_index.t().tolist(),'node_type':node_type.tolist() if node_type is not None else None,'edge_type':edge_type.tolist() if edge_type is not None else None,'node_attr':x.tolist() if x is not None else None,'edge_attr':ea.tolist() if ea is not None else None,'y':y,'metadata':{'domain':'molecule'},'directed':False}
   rec=record_from_dict(row)
   rec.metadata.update(label_source='ogbg-molhiv_graph-label.csv.gz',source_graph_index=idx,label_recovery='official_source_index')
   records.append(rec)
 random.Random(20261005).shuffle(records)
 # deterministic stratified split by label
 groups={0:[],1:[]}
 for r in records:groups[int(r.y.item())].append(r)
 splits={'train':[],'val':[],'test':[]}
 for rows in groups.values():
  n=len(rows); nt=int(n*.7);nv=int(n*.15);splits['train']+=rows[:nt];splits['val']+=rows[nt:nt+nv];splits['test']+=rows[nt+nv:]
 for s,rows in splits.items():
  path=a.out/(s+'.jsonl');path.parent.mkdir(parents=True,exist_ok=True);path.write_text('\n'.join(json.dumps(record_to_dict(r),sort_keys=True) for r in rows)+'\n')
 meta={'dataset':'ogbg-molhiv','source_cache':str(CACHE),'selected_indices':selected,'split_counts':{k:len(v) for k,v in splits.items()},'split_classes':{k:sorted({int(r.y.item()) for r in v}) for k,v in splits.items()},'official_label_source':True}
 (a.out/'protocol.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
