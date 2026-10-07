"""Run one common structural downstream task over every materialized dataset.

The 21-dataset token corpus has no supervised ``y`` labels.  This runner does
not invent domain labels; it evaluates a declared graph-structural task:
predicting log1p(number of nodes) from the token view.  The split, vocabulary,
architecture grid, seeds and test lock are identical for all views. Datasets
with too few graphs are reported as unevaluable rather than assigned fake
metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from gptok2.data.io import load_records
from gptok2.canonical import canonicalize
from gptok2.data.schema_spec import SchemaSpec
from gptok2_tokenizer import GPTok2Tokenizer
from rps_gtok_consumption.data import TokenExample
from rps_gtok_consumption.training import TrainConfig, train_model
from rps_gtok_consumption.views import TokenViewBuilder


DATASETS = [
    "ast_cfg_cpg", "cifar10_sp", "citeseer", "collab", "cora", "enzymes",
    "fb15k_237", "ieee_power_grid", "imdb_binary", "mnist_sp", "mutag",
    "ogbg_code2", "ogbg_molhiv", "ogbg_molpcba", "ogbg_ppa", "ogbn_arxiv",
    "proteins", "pubmed", "road_networks", "synthetic_stress", "wn18rr",
]


def split_records(records, seed: int = 2026):
    rows = list(records); random.Random(seed).shuffle(rows)
    n = len(rows); n_train = max(1, int(round(0.70 * n))); n_val = max(1, int(round(0.15 * n)))
    return {"train": rows[:n_train], "val": rows[n_train:n_train + n_val], "test": rows[n_train + n_val:]}


def examples(records, builder, dataset: str, split: str, view: str):
    out = []
    for record in records:
        tokens = builder.build(record, view)
        out.append(TokenExample(
            graph_id=record.graph_id,
            dataset=dataset,
            split=split,
            view=view,
            tokens=tokens,
            y=float(math.log1p(max(0, int(record.num_nodes)))),
            task_type="regression",
            num_nodes=int(record.num_nodes),
            num_edges=int(record.edge_index.shape[1]) if record.edge_index.numel() else 0,
        ))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="rebuttal/data_21")
    parser.add_argument("--out", default="rebuttal/experiments/E9_all_dataset_structural")
    parser.add_argument("--views", nargs="+", default=["edge_list", "canonical_edge_list", "rps_gtok_plus"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--max-graphs", type=int, default=60)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--canonical-max-search", type=int, default=10)
    parser.add_argument("--canonical-timeout-sec", type=float, default=0.5)
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--datasets", nargs="*", default=None, help="explicit dataset subset for parallel workers")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(args.out); root.mkdir(parents=True, exist_ok=True)
    rows=[]; statuses=[]
    if args.resume:
        result_path = root / "results.csv"
        status_path = root / "dataset_status.csv"
        if result_path.exists():
            with result_path.open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
        if status_path.exists():
            with status_path.open(encoding="utf-8", newline="") as fh:
                statuses = list(csv.DictReader(fh))
    completed = {str(row.get("dataset")) for row in statuses if row.get("status") in {"ok", "too_small", "missing", "skipped", "view_collision"}}
    datasets = list(args.datasets) if args.datasets else list(DATASETS)
    for dataset in datasets:
        if args.resume and dataset in completed:
            continue
        if dataset in set(args.skip):
            statuses.append({"dataset":dataset,"status":"skipped","reason":"explicit runtime-budget skip; run separately as stress case"})
            continue
        path=Path(args.data_dir)/f"{dataset}.jsonl"
        print(json.dumps({"stage":"dataset_start","dataset":dataset,"out":str(root)}, sort_keys=True), flush=True)
        if not path.exists():
            statuses.append({"dataset":dataset,"status":"missing"}); continue
        all_records=load_records(path)[:args.max_graphs]
        if len(all_records) < 12:
            statuses.append({"dataset":dataset,"status":"too_small","graphs":len(all_records),"reason":"insufficient graphs for 70/15/15 train/val/test split"}); continue
        splits=split_records(all_records)
        needs_tokenizer = any(view.startswith("rps_gtok_") or view.endswith("_bpe") for view in args.views)
        canonical_seconds = 0.0
        if needs_tokenizer:
            # Canonicalize each graph once.  GPTok2Tokenizer.fit/encode would
            # otherwise canonicalize the same record repeatedly, which made
            # the 21-dataset RPS run look non-terminating without changing the
            # emitted method semantics.
            spec = SchemaSpec(max_search_nodes=int(args.canonical_max_search))
            canonical_splits = {}
            import time as _time
            for split_name, split_rows in splits.items():
                canonical_splits[split_name] = []
                for record in split_rows:
                    started = _time.perf_counter()
                    result = canonicalize(record, spec, exact=True, max_search_nodes=int(args.canonical_max_search), timeout_sec=float(args.canonical_timeout_sec))
                    canonical_seconds += _time.perf_counter() - started
                    canonical_splits[split_name].append(result.record)
            splits_for_views = canonical_splits
            tokenizer = GPTok2Tokenizer({
                "canonicalization":{"enabled":False},
                "patch":{"sparse_edge_patches":True,"sparse_density_threshold":1.0,"sparse_clustering_threshold":1.0,"max_patches_per_graph":128,"max_cycle_signatures":32,"max_triangle_signatures":128},
                "motif_macro":{"max_structural_macros":32,"min_structural_count":1000,"max_parameterized_span_len":32,"max_merge_schemas":32,"min_merge_schema_count":1000,"max_code_schemas":32,"min_code_schema_count":1000},
                "compact_entropy":{"max_macros":32,"min_macro_count":1000,"max_bpe_merges":32,"min_bpe_count":1000},
            }).fit(canonical_splits["train"])
        else:
            # Raw edge-list only audits do not need a fitted tokenizer; this
            # keeps the all-dataset structural baseline fast and independent
            # of RPS motif mining cost.
            tokenizer = GPTok2Tokenizer({"canonicalization":{"max_search_nodes":args.canonical_max_search}})
            splits_for_views = splits
        tokenizer.config.setdefault("canonicalization",{})["max_search_nodes"]=int(args.canonical_max_search)
        builder=TokenViewBuilder(tokenizer,seed=2026).fit(splits_for_views["train"],list(args.views))
        materialized={view:{split:examples(records,builder,dataset,split,view) for split,records in splits_for_views.items()} for view in args.views}
        union=[ex.tokens for view in args.views for ex in materialized[view]["train"]]
        hashes={view:hashlib.sha1(json.dumps([x.tokens for x in materialized[view]["test"]],separators=(",",":")).encode()).hexdigest() for view in args.views}
        if len(set(hashes.values())) != len(hashes):
            statuses.append({"dataset":dataset,"status":"view_collision"}); continue
        try:
            for view in args.views:
                for seed in args.seeds:
                    cfg=TrainConfig(
                        max_len=args.max_len,batch_size=args.batch_size,epochs=args.epochs,
                        patience=max(4,args.epochs//3),task_type="regression",
                        model={"adapter":"plain","dim":32,"layers":1,"heads":4,"dropout":0.1},
                        device=args.device,seed=int(seed),vocab_sequences=union,
                    )
                    metrics=train_model(materialized[view],cfg,out_dir=root/dataset/view/f"seed{seed}")
                    test=metrics["test"]
                    rows.append({"dataset":dataset,"view":view,"seed":seed,"graphs_train":len(splits["train"]),"graphs_test":len(splits["test"]),"parameters":metrics["parameter_count"],"test_mae":test.get("mae"),"test_rmse":test.get("rmse"),"test_r2":test.get("r2"),"test_truncation_rate":metrics["sequence_audit"]["test"].get("truncation_rate"),"test_discarded_fraction":metrics["sequence_audit"]["test"].get("discarded_token_fraction"),"view_sha1":hashes[view],"epochs_ran":metrics["epochs_ran"]})
        except Exception as exc:
            statuses.append({"dataset":dataset,"status":"failed","graphs":len(all_records),"reason":repr(exc)})
            _write_progress(root, rows, statuses)
            continue
        statuses.append({"dataset":dataset,"status":"ok","graphs":len(all_records),"train":len(splits["train"]),"val":len(splits["val"]),"test":len(splits["test"]),"canonicalization_sec":canonical_seconds,"canonical_timeout_sec":args.canonical_timeout_sec,"canonical_max_search":args.canonical_max_search})
        _write_progress(root, rows, statuses)
        print(json.dumps({"stage":"dataset_complete","dataset":dataset,"status":"ok","rows":len(rows)}, sort_keys=True), flush=True)
    protocol={"task":"structural_regression_log1p_num_nodes","datasets":datasets,"views":args.views,"seeds":args.seeds,"epochs":args.epochs,"batch_size":args.batch_size,"max_len":args.max_len,"max_graphs":args.max_graphs,"canonical_max_search":args.canonical_max_search,"canonical_timeout_sec":args.canonical_timeout_sec,"tokenizer_budget":{"max_patches_per_graph":128,"max_cycle_signatures":32,"max_triangle_signatures":128,"max_structural_macros":32},"test_used_for_selection":False,"label_source":"GraphRecord.num_nodes; no domain labels fabricated"}
    (root/"protocol.json").write_text(json.dumps(protocol,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({"rows":len(rows),"statuses":statuses,"out":str(root)},indent=2))


def _write_progress(root: Path, rows: list[dict], statuses: list[dict]) -> None:
    if rows:
        with (root/"results.csv").open("w",encoding="utf-8",newline="") as fh:
            fields=sorted({key for row in rows for key in row}); writer=csv.DictWriter(fh,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    if statuses:
        with (root/"dataset_status.csv").open("w",encoding="utf-8",newline="") as fh:
            fields=sorted({key for row in statuses for key in row}); writer=csv.DictWriter(fh,fieldnames=fields); writer.writeheader(); writer.writerows(statuses)


if __name__ == "__main__": main()
