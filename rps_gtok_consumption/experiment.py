"""Config-driven downstream experiments for RPS-GTok token views."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from gptok2.data.io import load_records
from gptok2.data.synthetic import generate_synthetic_graphs, split_records
from gptok2_tokenizer import GPTok2Tokenizer
from rps_gtok_consumption.data import TokenExample, infer_token_node_refs, save_examples
from rps_gtok_consumption.training import TrainConfig, train_model
from rps_gtok_consumption.views import TokenViewBuilder


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def run_experiment_config(config: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    datasets = load_dataset_splits(config)
    views = [str(v) for v in config.get("data", {}).get("views", ["rps_gtok_full"])]
    training = dict(config.get("training", {}))
    seeds = [int(x) for x in training.get("seeds", [config.get("run", {}).get("seed", 2026)])]
    max_lens = [int(x) for x in training.get("max_lens", [training.get("max_len", 256)])]
    grid = list(training.get("grid", [training.get("model", {"adapter": "full_embed"})]))
    rows: list[dict[str, Any]] = []
    for dataset, splits in datasets.items():
        tokenizer = GPTok2Tokenizer().fit(splits["train"])
        builder = TokenViewBuilder(
            tokenizer,
            seed=int(config.get("run", {}).get("seed", 2026)),
            bpe_merges=int(config.get("data", {}).get("bpe", {}).get("merges", 500)),
            bpe_min_freq=int(config.get("data", {}).get("bpe", {}).get("min_freq", 2)),
        ).fit(splits["train"], views)
        for view in views:
            token_splits = {
                split: examples_for_view(records, builder, dataset, split, view, task_type=str(training.get("task_type", "classification")))
                for split, records in splits.items()
            }
            view_dir = root / "prepared" / dataset / view
            for split, examples in token_splits.items():
                save_examples(view_dir / f"{split}.jsonl", examples)
            for seed in seeds:
                for max_len in max_lens:
                    for idx, hp in enumerate(grid):
                        hp = dict(hp)
                        model_cfg = model_config_from_hparams(hp)
                        task_type = str(hp.get("task_type", training.get("task_type", "classification")))
                        train_cfg = TrainConfig(
                            max_len=max_len,
                            vocab_size=int(training.get("vocab_size", 100000)),
                            batch_size=int(hp.get("batch_size", training.get("batch_size", 32))),
                            epochs=int(hp.get("epochs", training.get("epochs", 40))),
                            lr=float(hp.get("lr", training.get("lr", 0.0003))),
                            weight_decay=float(hp.get("weight_decay", training.get("weight_decay", 0.01))),
                            patience=int(hp.get("patience", training.get("patience", 8))),
                            seed=seed,
                            task_type=task_type,
                            model=model_cfg,
                            device=str(config.get("run", {}).get("device", training.get("device", "auto"))),
                        )
                        run_dir = root / "models" / dataset / view / f"seed{seed}_len{max_len}_candidate{idx}"
                        metrics = train_model(token_splits, train_cfg, out_dir=run_dir)
                        row = flatten_metrics(dataset, view, seed, max_len, idx, hp, metrics)
                        rows.append(row)
                        write_csv(root / "candidate_results.csv", rows)
    summary = {"runs": len(rows), "datasets": sorted(datasets), "views": views, "out_dir": str(root)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def load_dataset_splits(config: dict[str, Any]) -> dict[str, dict[str, list]]:
    data = dict(config.get("data", {}))
    if data.get("format", "synthetic") == "synthetic":
        records = generate_synthetic_graphs(config, seed=int(config.get("run", {}).get("seed", 2026)))
        return {str(data.get("name", "synthetic")): split_records(records, config)}
    if data.get("format") == "graph_records_jsonl":
        out = {}
        for name, paths in dict(data.get("datasets", {})).items():
            out[str(name)] = {split: load_records(path) for split, path in dict(paths).items()}
        return out
    raise ValueError("data.format must be synthetic or graph_records_jsonl")


def examples_for_view(records: list, builder: TokenViewBuilder, dataset: str, split: str, view: str, task_type: str) -> list[TokenExample]:
    examples = []
    for record in records:
        tokens = builder.build(record, view)
        y = _record_target(record, task_type)
        node_refs = [infer_token_node_refs(tok) for tok in tokens]
        examples.append(
            TokenExample(
                graph_id=record.graph_id,
                dataset=dataset,
                split=split,
                view=view,
                tokens=tokens,
                y=y,
                task_type=task_type,
                num_nodes=int(record.num_nodes),
                num_edges=len(record.edge_index.t()) if record.edge_index.numel() else 0,
                node_refs=node_refs,
                edge_pairs=[[str(u), str(v)] for u, v in sorted(record.edge_index.t().tolist())] if record.edge_index.numel() else [],
            )
        )
    return examples


def model_config_from_hparams(hp: dict[str, Any]) -> dict[str, Any]:
    adapter = str(hp.get("adapter", hp.get("model", "full_embed"))).lower()
    if adapter == "rps_aware":
        adapter = "full_embed"
    return {
        "adapter": adapter,
        "dim": int(hp.get("dim", 128)),
        "layers": int(hp.get("layers", 4)),
        "heads": int(hp.get("heads", 4)),
        "dropout": float(hp.get("dropout", 0.1)),
        "use_feature_gate": bool(hp.get("use_feature_gate", adapter == "full_embed")),
        "attentive_pooling": bool(hp.get("use_attentive_pooling", hp.get("attentive_pooling", False))),
        "attn_pool_heads": int(hp.get("attn_pool_heads", 1)),
    }


def flatten_metrics(dataset: str, view: str, seed: int, max_len: int, candidate_idx: int, hp: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    row = {
        "dataset": dataset,
        "view": view,
        "seed": seed,
        "max_len": max_len,
        "candidate_idx": candidate_idx,
        "candidate_name": hp.get("name", f"candidate{candidate_idx}"),
        "adapter": model_config_from_hparams(hp)["adapter"],
    }
    for prefix in ["val", "test"]:
        for key, value in dict(metrics.get(prefix, {})).items():
            row[f"{prefix}_{key}"] = value
    for key, value in metrics.items():
        if key not in {"val", "test"}:
            row[key] = value
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _record_target(record, task_type: str) -> int | float | list[float]:
    if record.y is None:
        raise ValueError(f"record {record.graph_id} has no target")
    if task_type == "multilabel":
        return [float(x) for x in record.y.detach().cpu().flatten().tolist()]
    if task_type == "regression":
        return float(record.y.detach().cpu().flatten()[0].item())
    return int(record.y.detach().cpu().flatten()[0].item())
