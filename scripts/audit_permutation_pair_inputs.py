"""Audit pair-task inputs before interpreting learned accuracy."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from gptok2.canonical import canonicalize
from gptok2.data.io import load_records
from gptok2.data.schema_spec import SchemaSpec
from gptok2_tokenizer import GPTok2Tokenizer
from rps_gtok_consumption.views import TokenViewBuilder
from scripts.run_topology_shift_downstream import permute_record, split_records


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--data-dir",default="rebuttal/data_21"); parser.add_argument("--dataset",required=True); parser.add_argument("--out",required=True); parser.add_argument("--max-graphs",type=int,default=60); parser.add_argument("--canonical-max-search",type=int,default=100000); parser.add_argument("--canonical-timeout-sec",type=float,default=5.0); args=parser.parse_args()
    records=load_records(Path(args.data_dir)/f"{args.dataset}.jsonl")[:args.max_graphs]; splits=split_records(records); spec=SchemaSpec(max_search_nodes=args.canonical_max_search,timeout_sec=args.canonical_timeout_sec)
    exact_counts={}; exact_splits={}
    for split,items in splits.items():
        keep=[]
        for index,record in enumerate(items):
            left=canonicalize(record,spec,exact=True,max_search_nodes=args.canonical_max_search,timeout_sec=args.canonical_timeout_sec)
            right=canonicalize(permute_record(record,987654+index),spec,exact=True,max_search_nodes=args.canonical_max_search,timeout_sec=args.canonical_timeout_sec)
            if left.exact_completed and right.exact_completed: keep.append(record)
        exact_splits[split]=keep; exact_counts[split]=len(keep)
    tokenizer=GPTok2Tokenizer({"canonicalization":{"max_search_nodes":args.canonical_max_search,"timeout_sec":args.canonical_timeout_sec}}).fit(exact_splits["train"]); views=["edge_list","canonical_edge_list","rps_gtok_plus"]; builder=TokenViewBuilder(tokenizer).fit(exact_splits["train"],views)
    rows=[]
    for view in views:
        for split,items in exact_splits.items():
            equal=[]
            for index,record in enumerate(items): equal.append(builder.build(record,view)==builder.build(permute_record(record,987654+index),view))
            rows.append({"dataset":args.dataset,"view":view,"split":split,"graphs":len(items),"positive_token_equality_rate":sum(equal)/max(1,len(equal)),"canonical_exact_subset":True})
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (out.with_suffix(".protocol.json")).write_text(str({"exact_counts":exact_counts,"max_search_nodes":args.canonical_max_search,"timeout_sec":args.canonical_timeout_sec}),encoding="utf-8")
    print(rows)


if __name__=="__main__": main()
