"""Command line interface for RPS-GTok downstream Transformer consumers."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from gptok2.data.io import load_records
from gptok2.data.synthetic import generate_synthetic_graphs, split_records
from gptok2_tokenizer import GPTok2Tokenizer
from rps_gtok_consumption.data import examples_from_records, load_examples, save_examples, split_examples
from rps_gtok_consumption.experiment import load_config as load_experiment_config
from rps_gtok_consumption.experiment import run_experiment_config
from rps_gtok_consumption.training import TrainConfig, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate Transformer consumers over RPS-GTok sequences.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare", help="Prepare tokenized examples from GraphRecord JSONL.")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--artifact", default="")
    prepare.add_argument("--mode", default="motif_hybrid")
    prepare.add_argument("--dataset", default="graph_records")
    prepare.add_argument("--split", default="all")
    prepare.add_argument("--task-type", choices=["classification", "regression", "multilabel"], default="classification")

    train = sub.add_parser("train", help="Train a consumer from prepared tokenized JSONL files.")
    train.add_argument("--train", required=True)
    train.add_argument("--val", default="")
    train.add_argument("--test", default="")
    train.add_argument("--config", default="")
    train.add_argument("--out", default="runs/rps_gtok_consumer")

    smoke = sub.add_parser("smoke", help="Run an end-to-end synthetic consumer smoke experiment.")
    smoke.add_argument("--out", default="runs/rps_gtok_consumer_smoke")
    smoke.add_argument("--epochs", type=int, default=2)

    run_config = sub.add_parser("run-config", help="Run a config-driven multi-view downstream experiment.")
    run_config.add_argument("--config", required=True)
    run_config.add_argument("--out", default="runs/rps_gtok_consumption_experiment")

    args = parser.parse_args()
    if args.cmd == "prepare":
        run_prepare(args)
    elif args.cmd == "train":
        run_train(args)
    elif args.cmd == "smoke":
        run_smoke(args)
    elif args.cmd == "run-config":
        result = run_experiment_config(load_experiment_config(args.config), args.out)
        print(json.dumps(result, sort_keys=True))


def run_prepare(args: argparse.Namespace) -> None:
    records = load_records(args.input)
    tokenizer = GPTok2Tokenizer.load(args.artifact) if args.artifact else GPTok2Tokenizer().fit(records)
    examples = examples_from_records(records, tokenizer, mode=args.mode, split=args.split, dataset=args.dataset, task_type=args.task_type)
    save_examples(args.output, examples)
    print(json.dumps({"output": args.output, "examples": len(examples), "mode": args.mode}, sort_keys=True))


def run_train(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    train_rows = load_examples(args.train)
    splits = {"train": train_rows}
    splits["val"] = load_examples(args.val) if args.val else train_rows
    splits["test"] = load_examples(args.test) if args.test else splits["val"]
    task_type = str(config.get("task_type", train_rows[0].task_type))
    cfg = TrainConfig(**{**asdict(TrainConfig()), **config, "task_type": task_type})
    result = train_model(splits, cfg, out_dir=args.out)
    print(json.dumps(result, sort_keys=True))


def run_smoke(args: argparse.Namespace) -> None:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "run": {"seed": 2026},
        "data": {
            "synthetic": {
                "num_graphs": 48,
                "num_nodes_min": 8,
                "num_nodes_max": 20,
                "families": ["cycle", "star", "tree", "motif_mix"],
            },
            "split": {"train_ratio": 0.7, "val_ratio": 0.15},
        },
    }
    records = generate_synthetic_graphs(config, seed=2026)
    split_records_map = split_records(records, config)
    tokenizer = GPTok2Tokenizer().fit(split_records_map["train"])
    splits = {
        split: examples_from_records(rows, tokenizer, mode="motif_hybrid", split=split, dataset="synthetic_smoke")
        for split, rows in split_records_map.items()
    }
    for split, rows in splits.items():
        save_examples(root / f"{split}.jsonl", rows)
    cfg = TrainConfig(
        max_len=128,
        batch_size=8,
        epochs=int(args.epochs),
        patience=max(1, int(args.epochs)),
        task_type="classification",
        model={"adapter": "full_embed", "dim": 48, "layers": 1, "heads": 4, "dropout": 0.1},
        device="cpu",
    )
    result = train_model(splits, cfg, out_dir=root)
    print(json.dumps(result, sort_keys=True))


def load_config(path: str) -> dict:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


if __name__ == "__main__":
    main()
