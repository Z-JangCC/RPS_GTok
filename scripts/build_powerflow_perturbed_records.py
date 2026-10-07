"""Generate deterministic load-perturbation power-flow graph records."""
from __future__ import annotations
import copy,json,random
from pathlib import Path
import pandapower as pp
import pandapower.networks as pn

ROOT=Path(__file__).resolve().parents[1]
CASES=["case4gs","case5","case6ww","case9","case14","case24_ieee_rts","case30","case33bw","case39","case57","case89pegase","case118","case145","case_ieee30","case_illinois200","case300"]

def record_for(name,perturb,idx):
 net=getattr(pn,name)()
 if hasattr(net,'load') and len(net.load):
  net.load.loc[:, 'p_mw'] *= perturb
  net.load.loc[:, 'q_mvar'] *= perturb
 if hasattr(net,'sgen') and len(net.sgen): net.sgen.loc[:, 'p_mw'] *= perturb
 pp.runpp(net,init='flat',numba=False)
 nodes=[];edges=[]
 def add(n,k,a):nodes.append((n,k,a))
 for i,row in net.bus.iterrows(): add(f'bus_{i}','bus',[float(row.get('vn_kv',0)),float(row.get('in_service',1))])
 for table,kind,attrs in [('ext_grid','ext_grid',['vm_pu']),('gen','gen',['p_mw','vm_pu']),('sgen','sgen',['p_mw','q_mvar']),('load','load',['p_mw','q_mvar'])]:
  if hasattr(net,table):
   for i,row in getattr(net,table).iterrows():
    nid=f'{kind}_{i}';add(nid,kind,[float(row.get(k,0)) for k in attrs]);edges.append((nid,f'bus_{int(row.bus)}',kind+'_attach',[0.0]))
 if hasattr(net,'line'):
  for i,row in net.line.iterrows():edges.append((f'bus_{int(row.from_bus)}',f'bus_{int(row.to_bus)}','power_line',[float(row.get(k,0)) for k in ['length_km','r_ohm_per_km','x_ohm_per_km','c_nf_per_km','max_i_ka']]))
 if hasattr(net,'trafo'):
  for i,row in net.trafo.iterrows():edges.append((f'bus_{int(row.hv_bus)}',f'bus_{int(row.lv_bus)}','transformer',[float(row.get(k,0)) for k in ['sn_mva','vn_hv_kv','vn_lv_kv','vk_percent','vkr_percent']]))
 nt={k:i for i,k in enumerate(sorted({k for _,k,_ in nodes}))}; et={k:i for i,k in enumerate(sorted({k for _,_,k,_ in edges}))}; ids={n:i for i,(n,_,_) in enumerate(nodes)}
 return {'graph_id':f'pandapower_{name}_p{idx:02d}','num_nodes':len(nodes),'edges':[[ids[u],ids[v]] for u,v,_,_ in edges],'node_type':[nt[k] for _,k,_ in nodes],'edge_type':[et[k] for _,_,k,_ in edges],'node_attr':[a[:3]+[0.0]*(3-len(a[:3])) for _,_,a in nodes],'edge_attr':[a+[0.0]*(5-len(a)) for *_,a in edges],'y':float(net.res_bus.vm_pu.mean()),'metadata':{'domain':'power_grid','case_name':name,'load_perturbation':perturb,'label_source':'pandapower.runpp','task_target':'mean_bus_vm_pu','node_attr_keys':['feature_0','feature_1','feature_2'],'target_leakage':'vm_pu removed from node_attr'},'directed':False}

def main():
 root=ROOT/'rebuttal/experiments/E15_powerflow_perturbed';root.mkdir(parents=True,exist_ok=True);rng=random.Random(20261005); rows=[];failed=[]
 case_rows={}
 for name in CASES:
  case_rows[name]=[]
  for i in range(10):
   perturb=0.82+0.04*i
   try: case_rows[name].append(record_for(name,perturb,i))
   except Exception as e: failed.append({'case':name,'perturbation':perturb,'error':repr(e)})
 cases=list(case_rows);rng.shuffle(cases);nt=max(1,int(.7*len(cases)));nv=max(1,int(.15*len(cases)));case_split={'train':cases[:nt],'val':cases[nt:nt+nv],'test':cases[nt+nv:]}
 for split,cs in case_split.items():
  items=[r for c in cs for r in case_rows[c]];rng.shuffle(items);(root/(split+'.jsonl')).write_text('\n'.join(json.dumps(r,sort_keys=True) for r in items)+'\n')
 protocol={'dataset':'pandapower_load_perturbation','cases':cases,'case_split':case_split,'perturbations':[.82+.04*i for i in range(10)],'failed':failed,'split_counts':{k:sum(len(case_rows[c]) for c in v) for k,v in case_split.items()},'target':'mean_bus_vm_pu','input_target_leakage':'vm_pu excluded'}
 (root/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(protocol,indent=2))
if __name__=='__main__':main()
