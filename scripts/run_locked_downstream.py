"""Run a locked, no-truncation control after validation model selection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from gptok2.data.io import load_records
from gptok2_tokenizer import GPTok2Tokenizer
from rps_gtok_consumption.experiment import examples_for_view
from rps_gtok_consumption.training import TrainConfig, train_model
from rps_gtok_consumption.views import TokenViewBuilder


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--artifact",required=True); parser.add_argument("--train",required=True); parser.add_argument("--val",required=True); parser.add_argument("--test",required=True); parser.add_argument("--dataset",required=True); parser.add_argument("--views",nargs="+",required=True); parser.add_argument("--out",required=True); parser.add_argument("--seeds",nargs="+",type=int,default=[2026,2027,2028]); parser.add_argument("--max-len",type=int,default=2048); parser.add_argument("--epochs",type=int,default=8); parser.add_argument("--batch-size",type=int,default=32); parser.add_argument("--dim",type=int,default=32); parser.add_argument("--layers",type=int,default=1); parser.add_argument("--heads",type=int,default=4); parser.add_argument("--device",default="auto"); parser.add_argument("--class-weight",action="store_true"); parser.add_argument("--canonical-max-search",type=int,default=10); parser.add_argument("--adapter-mode",choices=["shared_plain","shared_full_embed","native"],default="shared_plain"); parser.add_argument("--max-train",type=int,default=0); parser.add_argument("--max-val",type=int,default=0); parser.add_argument("--max-test",type=int,default=0)
    args=parser.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    tokenizer=GPTok2Tokenizer.load(args.artifact); tokenizer.config.setdefault("canonicalization",{})["max_search_nodes"]=int(args.canonical_max_search)
    records={k:load_records(v) for k,v in {"train":args.train,"val":args.val,"test":args.test}.items()}
    for split, limit in (("train", args.max_train), ("val", args.max_val), ("test", args.max_test)):
        if int(limit) > 0:
            records[split] = records[split][: int(limit)]
    builder=TokenViewBuilder(tokenizer,seed=2026).fit(records["train"],list(args.views))
    materialized={view:{split:examples_for_view(rows,builder,args.dataset,split,view,"classification") for split,rows in records.items()} for view in args.views}
    union=[ex.tokens for view in args.views for ex in materialized[view]["train"]]
    rows=[]
    for view in args.views:
        for seed in args.seeds:
            is_rps=view.startswith("rps_gtok_")
            use_full_embed = args.adapter_mode == "shared_full_embed" or (args.adapter_mode == "native" and is_rps)
            cfg=TrainConfig(max_len=args.max_len,batch_size=args.batch_size,epochs=args.epochs,patience=max(3,args.epochs//3),task_type="classification",model={"adapter":"full_embed" if use_full_embed else "plain","dim":args.dim,"layers":args.layers,"heads":args.heads,"dropout":0.1},device=args.device,seed=seed,class_weight=args.class_weight,vocab_sequences=union)
            metrics=train_model(materialized[view],cfg,out_dir=out/view/f"seed{seed}")
            rows.append({"dataset":args.dataset,"view":view,"seed":seed,"parameters":metrics["parameter_count"],"test_accuracy":metrics["test"].get("accuracy"),"test_balanced_accuracy":metrics["test"].get("balanced_accuracy"),"test_macro_f1":metrics["test"].get("macro_f1"),"prediction_unique":metrics["test"].get("prediction_unique"),"test_truncation_rate":metrics["sequence_audit"]["test"].get("truncation_rate"),"test_discarded_fraction":metrics["sequence_audit"]["test"].get("discarded_token_fraction")})
    fields=sorted({key for row in rows for key in row})
    with (out/"selected_results.csv").open("w",encoding="utf-8",newline="") as fh: writer=csv.DictWriter(fh,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (out/"protocol.json").write_text(json.dumps({"dataset":args.dataset,"views":list(args.views),"seeds":list(args.seeds),"max_len":args.max_len,"no_truncation_control":True,"canonical_max_search":args.canonical_max_search,"adapter_mode":args.adapter_mode,"shared_vocabulary":True,"limits":{"train":args.max_train,"val":args.max_val,"test":args.max_test}},indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({"rows":len(rows),"out":str(out)},indent=2))


if __name__=="__main__": main()
