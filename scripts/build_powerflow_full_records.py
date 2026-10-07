"""Build a source-locked full pandapower case regression corpus."""
from __future__ import annotations
import json,random
from pathlib import Path
import pandapower as pp
import pandapower.networks as pn
from gptok2.data.schema import record_from_dict,record_to_dict

ROOT=Path(__file__).resolve().parents[1]
CASES=["case4gs","case5","case6ww","case9","case11_iwamoto","case14","case24_ieee_rts","case30","case33bw","case39","case57","case89pegase","case118","case145","case_ieee30","case_illinois200","case300","case1354pegase","case1888rte"]

def build(name):
 net=getattr(pn,name)(); pp.runpp(net,init='flat',numba=False)
 nodes=[]; edges=[]
 def add_node(node_id,node_type,attrs): nodes.append((node_id,node_type,attrs))
 for idx,row in net.bus.iterrows(): add_node(f'bus_{idx}','bus',[float(getattr(row,'vn_kv',0.0)),float(getattr(row,'in_service',1))])
 for table,kind,attrs in [('ext_grid','ext_grid',['vm_pu']),('gen','gen',['p_mw','vm_pu']),('sgen','sgen',['p_mw','q_mvar']),('load','load',['p_mw','q_mvar'])]:
  if hasattr(net,table):
   for idx,row in getattr(net,table).iterrows():
    node_id=f'{kind}_{idx}'; add_node(node_id,kind,[float(row.get(k,0.0)) for k in attrs]); bus=int(row.bus); edges.append((node_id,f'bus_{bus}',kind+'_attach',[0.0]))
 if hasattr(net,'line'):
  for idx,row in net.line.iterrows(): edges.append((f'bus_{int(row.from_bus)}',f'bus_{int(row.to_bus)}','power_line',[float(row.get(k,0.0)) for k in ['length_km','r_ohm_per_km','x_ohm_per_km','c_nf_per_km','max_i_ka']]))
 if hasattr(net,'trafo'):
  for idx,row in net.trafo.iterrows(): edges.append((f'bus_{int(row.hv_bus)}',f'bus_{int(row.lv_bus)}','transformer',[float(row.get(k,0.0)) for k in ['sn_mva','vn_hv_kv','vn_lv_kv','vk_percent','vkr_percent']]))
 node_types={k:i for i,k in enumerate(sorted({x[1] for x in nodes}))}; edge_types={k:i for i,k in enumerate(sorted({x[2] for x in edges}))}; ids={x[0]:i for i,x in enumerate(nodes)}
 row={'graph_id':f'pandapower_{name}','num_nodes':len(nodes),'edges':[[ids[u],ids[v]] for u,v,_,_ in edges],'node_type':[node_types[k] for _,k,_ in nodes],'edge_type':[edge_types[k] for _,_,k,_ in edges],'node_attr':[a+[0.0]*(2-len(a)) for _,_,a in nodes],'edge_attr':[a+[0.0]*(5-len(a)) for *_,a in edges],'y':float(net.res_bus.vm_pu.mean()),'metadata':{'domain':'power_grid','label_source':'pandapower.runpp','task_target':'mean_bus_vm_pu','node_attr_keys':['feature_0','feature_1'],'edge_attr_keys':['physical_parameters'],'case_name':name},'directed':False}
 return row

def main():
 out=ROOT/'rebuttal/experiments/E15_powerflow_full'; rows=[]; failed=[]
 for name in CASES:
  try: rows.append(build(name)); print(name,'ok',len(rows[-1]['edges']),rows[-1]['y'],flush=True)
  except Exception as e: failed.append({'case':name,'error':repr(e)});print(name,'FAIL',repr(e),flush=True)
 random.Random(20261005).shuffle(rows); n=len(rows);nt=max(1,int(.7*n));nv=max(1,int(.15*n));splits={'train':rows[:nt],'val':rows[nt:nt+nv],'test':rows[nt+nv:]}
 out.mkdir(parents=True,exist_ok=True)
 for s,items in splits.items():(out/(s+'.jsonl')).write_text('\n'.join(json.dumps(x,sort_keys=True) for x in items)+'\n')
 (out/'protocol.json').write_text(json.dumps({'dataset':'pandapower','cases_requested':CASES,'cases_failed':failed,'split_counts':{k:len(v) for k,v in splits.items()},'target':'mean_bus_vm_pu','target_leakage':'vm_pu result is not in node_attr'},indent=2))
if __name__=='__main__':main()
