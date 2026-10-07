"""Topology-sensitive downstream protocol with an explicit permutation shift.

Train graphs keep their original node IDs; test graphs are independently
relabelled before serialization. The target is selected once from a fixed
ordered family of topology statistics using training variance only. Thus raw
edge-list, canonical serialization and RPS-GTok++ use the same Transformer and
the same target, while the shift directly tests whether permutation stability
survives downstream consumption.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from gptok2.canonical import canonicalize
from gptok2.data.io import load_records
from gptok2.data.schema import GraphRecord
from gptok2.data.schema_spec import SchemaSpec
from gptok2_tokenizer import GPTok2Tokenizer
from rps_gtok_consumption.data import TokenExample
from rps_gtok_consumption.training import TrainConfig, train_model
from rps_gtok_consumption.views import TokenViewBuilder


DATASETS = [
    "cifar10_sp", "citeseer", "collab", "cora", "enzymes", "fb15k_237",
    "imdb_binary", "mnist_sp", "mutag", "ogbg_molhiv", "ogbg_molpcba",
    "ogbn_arxiv", "proteins", "pubmed", "synthetic_stress", "wn18rr",
]


def split_records(records, seed=2026):
    rows=list(records); random.Random(seed).shuffle(rows); n=len(rows); nt=max(1,int(round(.70*n))); nv=max(1,int(round(.15*n)))
    return {"train":rows[:nt],"val":rows[nt:nt+nv],"test":rows[nt+nv:]}


def topology_stats(record: GraphRecord) -> dict[str, float]:
    graph=nx.Graph()
    graph.add_nodes_from(range(int(record.num_nodes)))
    if record.edge_index.numel(): graph.add_edges_from(record.edge_index.t().tolist())
    n=graph.number_of_nodes(); m=graph.number_of_edges()
    triangles=sum(nx.triangles(graph).values())/3 if n else 0.0
    density=nx.density(graph) if n > 1 else 0.0
    components=nx.number_connected_components(graph) if n else 0
    degrees=[degree for _,degree in graph.degree()]
    degree_entropy=0.0
    if degrees:
        counts=np.bincount(degrees); probs=counts[counts>0]/sum(counts); degree_entropy=float(-(probs*np.log(probs)).sum())
    return {"log_edges":math.log1p(m),"density":float(density),"log_triangles":math.log1p(triangles),"log_components":math.log1p(components),"degree_entropy":degree_entropy}


def choose_target(train_records):
    ordered=["log_edges","density","log_triangles","log_components","degree_entropy"]
    values={key:np.asarray([topology_stats(record)[key] for record in train_records],dtype=float) for key in ordered}
    variances={key:float(np.var(value)) for key,value in values.items()}
    target=max(ordered,key=lambda key:(variances[key],-ordered.index(key)))
    return target,variances


def permute_record(record: GraphRecord, seed: int) -> GraphRecord:
    nodes=list(range(int(record.num_nodes))); random.Random(seed).shuffle(nodes); mapping={old:new for new,old in enumerate(nodes)}
    edges=[(mapping[int(u)],mapping[int(v)]) for u,v in record.edge_index.t().tolist()] if record.edge_index.numel() else []
    edge_index=torch.tensor(edges,dtype=torch.long).t().contiguous() if edges else torch.empty(2,0,dtype=torch.long)
    def reorder(value):
        return value[nodes].clone() if value is not None and value.ndim>0 and value.shape[0]==record.num_nodes else value.clone() if value is not None else None
    return GraphRecord(record.graph_id,record.num_nodes,edge_index,reorder(record.node_type),record.edge_type.clone() if record.edge_type is not None else None,reorder(record.node_attr),record.edge_attr.clone() if record.edge_attr is not None else None,record.y.clone() if torch.is_tensor(record.y) else record.y,dict(record.metadata),record.directed)


def make_examples(records, builder, dataset, split, view, target, permutation_seed=None):
    out=[]
    for index, record in enumerate(records):
        source=permute_record(record, permutation_seed + index) if permutation_seed is not None else record
        tokens=builder.build(source,view)
        out.append(TokenExample(graph_id=record.graph_id,dataset=dataset,split=split,view=view,tokens=tokens,y=float(topology_stats(record)[target]),task_type="regression",num_nodes=int(record.num_nodes),num_edges=int(record.edge_index.shape[1]) if record.edge_index.numel() else 0))
    return out


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--data-dir",default="rebuttal/data_21"); parser.add_argument("--out",required=True); parser.add_argument("--dataset",required=True); parser.add_argument("--views",nargs="+",default=["edge_list","canonical_edge_list","rps_gtok_plus"]); parser.add_argument("--max-graphs",type=int,default=60); parser.add_argument("--seeds",nargs="+",type=int,default=[2026,2027,2028]); parser.add_argument("--epochs",type=int,default=12); parser.add_argument("--batch-size",type=int,default=32); parser.add_argument("--max-len",type=int,default=512); parser.add_argument("--device",default="auto"); parser.add_argument("--canonical-max-search",type=int,default=10); parser.add_argument("--canonical-timeout-sec",type=float,default=.5)
    args=parser.parse_args(); path=Path(args.data_dir)/f"{args.dataset}.jsonl"; records=load_records(path)[:args.max_graphs]; out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if len(records)<12: raise SystemExit(f"{args.dataset}: too few graphs ({len(records)})")
    splits=split_records(records); target,variances=choose_target(splits["train"])
    spec=SchemaSpec(max_search_nodes=args.canonical_max_search); canonical_splits={}
    for split,items in splits.items(): canonical_splits[split]=[canonicalize(record,spec,exact=True,max_search_nodes=args.canonical_max_search,timeout_sec=args.canonical_timeout_sec).record for record in items]
    tokenizer=GPTok2Tokenizer({"canonicalization":{"enabled":False},"patch":{"sparse_edge_patches":True,"sparse_density_threshold":1.0,"sparse_clustering_threshold":1.0,"max_patches_per_graph":128,"max_cycle_signatures":32,"max_triangle_signatures":128},"motif_macro":{"max_structural_macros":32,"min_structural_count":1000,"max_parameterized_span_len":32,"max_merge_schemas":32,"min_merge_schema_count":1000,"max_code_schemas":32,"min_code_schema_count":1000}}).fit(canonical_splits["train"])
    builder=TokenViewBuilder(tokenizer,seed=2026).fit(canonical_splits["train"],list(args.views))
    materialized={}
    for view in args.views:
        materialized[view]={"train":make_examples(splits["train"],builder,args.dataset,"train",view,target),"val":make_examples(splits["val"],builder,args.dataset,"val",view,target),"test_permuted":make_examples(splits["test"],builder,args.dataset,"test",view,target,permutation_seed=987654)}
    union=[ex.tokens for view in args.views for ex in materialized[view]["train"]]
    rows=[]
    for view in args.views:
        for seed in args.seeds:
            split={"train":materialized[view]["train"],"val":materialized[view]["val"],"test":materialized[view]["test_permuted"]}
            metrics=train_model(split,TrainConfig(max_len=args.max_len,batch_size=args.batch_size,epochs=args.epochs,patience=max(4,args.epochs//3),task_type="regression",model={"adapter":"plain","dim":32,"layers":1,"heads":4,"dropout":.1},device=args.device,seed=seed,vocab_sequences=union),out_dir=out/view/f"seed{seed}")
            test=metrics["test"]; rows.append({"dataset":args.dataset,"view":view,"seed":seed,"target":target,"target_train_variance":variances[target],"test_permuted":True,"test_mae":test.get("mae"),"test_rmse":test.get("rmse"),"test_r2":test.get("r2"),"test_truncation_rate":metrics["sequence_audit"]["test"].get("truncation_rate"),"test_discarded_fraction":metrics["sequence_audit"]["test"].get("discarded_token_fraction"),"prediction_unique":None,"parameters":metrics["parameter_count"]})
    with (out/"results.csv").open("w",encoding="utf-8",newline="") as fh: writer=csv.DictWriter(fh,fieldnames=sorted({key for row in rows for key in row})); writer.writeheader(); writer.writerows(rows)
    (out/"protocol.json").write_text(json.dumps({"task":"permutation_shift_topology_regression","dataset":args.dataset,"target":target,"target_variances_train":variances,"views":args.views,"seeds":args.seeds,"test_permutation_seed":987654,"canonical_max_search":args.canonical_max_search,"canonical_timeout_sec":args.canonical_timeout_sec},indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({"dataset":args.dataset,"target":target,"rows":len(rows),"out":str(out)},indent=2))


if __name__=="__main__": main()
