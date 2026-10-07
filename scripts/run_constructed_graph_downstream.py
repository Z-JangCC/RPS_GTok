"""Run the leakage-audited topology-only proxy task across all 21 datasets.

The runner consumes only the original ``GraphRecord`` as model input and joins
the separately generated target sidecar by ``graph_id``.  Raw edge list,
canonical edge list, and RPS-GTok++ are tokenized and consumed by one shared
plain Transformer with a frozen union vocabulary, equal parameter counts,
three fixed seeds, and no test-based model selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

from gptok2.canonical import canonicalize
from gptok2.data.io import load_records
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


def make_examples(records, label_map, builder, dataset, split, view):
    out = []
    for record in records:
        if record.graph_id not in label_map:
            raise ValueError(f"missing constructed target for {dataset}/{record.graph_id}")
        out.append(TokenExample(
            graph_id=record.graph_id,
            dataset=dataset,
            split=split,
            view=view,
            tokens=builder.build(record, view),
            y=float(label_map[record.graph_id]["target"]),
            task_type="regression",
            num_nodes=int(record.num_nodes),
            num_edges=int(record.edge_index.shape[1]) if record.edge_index.numel() else 0,
        ))
    return out


def load_targets(path: Path) -> dict[str, dict]:
    return {row["graph_id"]: row for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("rebuttal/data_21"))
    parser.add_argument("--task-dir", type=Path, default=Path("rebuttal/constructed_tasks"))
    parser.add_argument("--out", type=Path, default=Path("rebuttal/experiments/E16_constructed_topology_task"))
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--views", nargs="+", default=["edge_list", "canonical_edge_list", "rps_gtok_plus"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--canonical-max-search", type=int, default=10)
    parser.add_argument("--canonical-timeout-sec", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    datasets = list(args.datasets) if args.datasets else DATASETS
    args.out.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict] = []
    status_rows: list[dict] = []
    result_path, status_path = args.out / "results.csv", args.out / "dataset_status.csv"
    if args.resume:
        if result_path.exists():
            result_rows = list(csv.DictReader(result_path.open(encoding="utf-8", newline="")))
        if status_path.exists():
            status_rows = list(csv.DictReader(status_path.open(encoding="utf-8", newline="")))
    done = {row.get("dataset") for row in status_rows if row.get("status") == "ok"}
    for dataset in datasets:
        if args.resume and dataset in done:
            continue
        source = args.data_dir / f"{dataset}.jsonl"
        target_path = args.task_dir / f"{dataset}.jsonl"
        if not source.exists() or not target_path.exists():
            status_rows.append({"dataset": dataset, "status": "missing"})
            continue
        started = time.perf_counter()
        try:
            records = load_records(source)
            targets = load_targets(target_path)
            if {record.graph_id for record in records} != set(targets):
                raise ValueError("record/target graph-id sets differ")
            # The source files retain y=null.  Reject a materialized target in
            # the input record before any tokenization.
            if any(record.y is not None for record in records):
                raise ValueError("constructed task input contains a graph label")
            splits = {split: [r for r in records if targets[r.graph_id]["split"] == split] for split in ("train", "val", "test")}
            if any(not splits[split] for split in splits):
                raise ValueError(f"empty split: { {key: len(value) for key, value in splits.items()} }")
            # The tokenizer is fit only on the training graphs.  Canonical
            # controls use one explicit, bounded schema-aware pass.
            need_rps = any(view.startswith("rps_gtok_") for view in args.views)
            canonical_splits = splits
            canonical_seconds = 0.0
            if need_rps or any(view.startswith("canonical_") for view in args.views):
                spec = SchemaSpec(max_search_nodes=int(args.canonical_max_search))
                canonical_splits = {}
                for split, rows in splits.items():
                    canonical_splits[split] = []
                    for record in rows:
                        begin = time.perf_counter()
                        canonical_splits[split].append(canonicalize(record, spec, exact=True, max_search_nodes=int(args.canonical_max_search), timeout_sec=float(args.canonical_timeout_sec)).record)
                        canonical_seconds += time.perf_counter() - begin
            tokenizer = GPTok2Tokenizer({
                "canonicalization": {"enabled": False, "max_search_nodes": int(args.canonical_max_search)},
                "patch": {"sparse_edge_patches": True, "sparse_density_threshold": 1.0, "sparse_clustering_threshold": 1.0, "max_patches_per_graph": 128, "max_cycle_signatures": 32, "max_triangle_signatures": 128},
                "motif_macro": {"max_structural_macros": 32, "min_structural_count": 1000, "max_parameterized_span_len": 32, "max_merge_schemas": 32, "min_merge_schema_count": 1000, "max_code_schemas": 32, "min_code_schema_count": 1000},
                "compact_entropy": {"max_macros": 32, "min_macro_count": 1000, "max_bpe_merges": 32, "min_bpe_count": 1000},
            }).fit(canonical_splits["train"])
            builder = TokenViewBuilder(tokenizer, seed=2026).fit(canonical_splits["train"], list(args.views))
            materialized = {
                view: {split: make_examples(canonical_splits[split] if view.startswith(("canonical_", "rps_gtok_")) else splits[split], targets, builder, dataset, split, view) for split in splits}
                for view in args.views
            }
            union = [example.tokens for view in args.views for example in materialized[view]["train"]]
            hashes = {view: hashlib.sha1(json.dumps([example.tokens for example in materialized[view]["test"]], separators=(",", ":")).encode()).hexdigest() for view in args.views}
            for view in args.views:
                for seed in args.seeds:
                    metrics = train_model(
                        materialized[view],
                        TrainConfig(
                            max_len=int(args.max_len), batch_size=16, epochs=int(args.epochs), patience=max(3, int(args.epochs) // 2),
                            task_type="regression", model={"adapter": "plain", "dim": 32, "layers": 1, "heads": 4, "dropout": 0.1},
                            device=args.device, seed=int(seed), vocab_sequences=union,
                        ),
                        out_dir=args.out / dataset / view / f"seed{seed}",
                    )
                    test = metrics["test"]
                    result_rows.append({
                        "dataset": dataset, "view": view, "seed": seed,
                        "graphs_train": len(splits["train"]), "graphs_val": len(splits["val"]), "graphs_test": len(splits["test"]),
                        "parameters": metrics["parameter_count"], "test_mae": test.get("mae"), "test_rmse": test.get("rmse"), "test_r2": test.get("r2"),
                        "test_truncation_rate": metrics["sequence_audit"]["test"].get("truncation_rate"), "test_discarded_fraction": metrics["sequence_audit"]["test"].get("discarded_token_fraction"),
                        "prediction_unique": None, "view_sha1": hashes[view], "target_source": "constructed_topology_proxy", "epochs_ran": metrics["epochs_ran"],
                    })
            status_rows.append({"dataset": dataset, "status": "ok", "records": len(records), "train": len(splits["train"]), "val": len(splits["val"]), "test": len(splits["test"]), "canonicalization_sec": canonical_seconds, "elapsed_sec": time.perf_counter() - started, "shared_parameters": len({row["parameters"] for row in result_rows if row["dataset"] == dataset}) == 1, "view_collisions": len(set(hashes.values())) != len(hashes)})
        except Exception as exc:
            status_rows.append({"dataset": dataset, "status": "failed", "reason": repr(exc), "elapsed_sec": time.perf_counter() - started})
        write_csv(result_path, result_rows); write_csv(status_path, status_rows)
        print(json.dumps(status_rows[-1], sort_keys=True), flush=True)
    protocol = {"task": "constructed_topology_proxy_regression", "datasets": datasets, "views": args.views, "seeds": args.seeds, "epochs": args.epochs, "max_len": args.max_len, "adapter": "shared_plain", "vocabulary": "frozen_union_train_only", "label_sidecar": str(args.task_dir), "input_y_must_be_null": True, "test_used_for_selection": False, "canonical_max_search": args.canonical_max_search, "canonical_timeout_sec": args.canonical_timeout_sec}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"rows": len(result_rows), "statuses": status_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

