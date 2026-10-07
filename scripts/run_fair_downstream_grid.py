"""Fair validation-selected downstream grid for all token views.

Every view receives the same candidate architectures, optimizer choices,
epochs, seeds and split.  Candidate selection uses validation balanced
accuracy only; the test split is read once for the selected candidate and is
never used to choose a method.  This prevents the common "RPS wins" failure
mode caused by giving one representation a larger or better-tuned model.
"""

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


DEFAULT_GRID = [
    {"name": "small", "dim": 32, "layers": 1, "heads": 4, "lr": 0.0003, "dropout": 0.1},
    {"name": "small_fast", "dim": 32, "layers": 1, "heads": 4, "lr": 0.001, "dropout": 0.1},
    {"name": "medium", "dim": 48, "layers": 1, "heads": 4, "lr": 0.0003, "dropout": 0.1},
    {"name": "medium_deep", "dim": 48, "layers": 2, "heads": 4, "lr": 0.0003, "dropout": 0.1},
]
COMPACT_GRID = [DEFAULT_GRID[0], DEFAULT_GRID[2]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--views", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grid-profile", choices=["full", "compact"], default="full")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--adapter-mode", choices=["shared_plain", "shared_full_embed", "native"], default="shared_plain",
        help="shared_plain/shared_full_embed are parameter-matched primary controls; native is auxiliary RPS Full-Embed",
    )
    parser.add_argument("--canonical-max-search", type=int, default=100)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = GPTok2Tokenizer.load(args.artifact)
    tokenizer.config.setdefault("canonicalization", {})["max_search_nodes"] = int(args.canonical_max_search)
    records = {
        "train": load_records(args.train),
        "val": load_records(args.val),
        "test": load_records(args.test),
    }
    builder = TokenViewBuilder(tokenizer, seed=2026).fit(records["train"], list(args.views))
    materialized = {
        view: {
            split: examples_for_view(rows, builder, args.dataset, split, view, "classification")
            for split, rows in records.items()
        }
        for view in args.views
    }
    print(json.dumps({"stage": "views_materialized", "dataset": args.dataset, "views": list(args.views), "graphs_train": len(records["train"])}, sort_keys=True), flush=True)
    view_hashes = {
        view: hashlib.sha1(
            json.dumps([ex.tokens for ex in materialized[view]["test"]], separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        for view in args.views
    }
    if len(view_hashes) != len(set(view_hashes.values())):
        raise RuntimeError("token-view collision: refusing to run a false comparison")
    union_sequences = [
        ex.tokens
        for view in args.views
        for ex in materialized[view]["train"]
    ]

    candidate_rows: list[dict] = []
    selected_rows: list[dict] = []
    shared_parameter_counts: dict[tuple[str, int], int] = {}
    for view in args.views:
        is_rps = view.startswith("rps_gtok_")
        use_full_embed = args.adapter_mode == "shared_full_embed" or (args.adapter_mode == "native" and is_rps)
        for seed in args.seeds:
            print(json.dumps({"stage": "seed_start", "dataset": args.dataset, "view": view, "seed": int(seed)}, sort_keys=True), flush=True)
            best = None
            grid = DEFAULT_GRID if args.grid_profile == "full" else COMPACT_GRID
            for index, hp in enumerate(grid):
                cfg = TrainConfig(
                    max_len=args.max_len,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    patience=max(3, args.epochs // 3),
                    lr=float(hp["lr"]),
                    task_type="classification",
                    model={
                        "adapter": "full_embed" if use_full_embed else "plain",
                        "dim": int(hp["dim"]),
                        "layers": int(hp["layers"]),
                        "heads": int(hp["heads"]),
                        "dropout": float(hp["dropout"]),
                    },
                    device=args.device,
                    seed=int(seed),
                    class_weight=bool(args.class_weight),
                    vocab_sequences=union_sequences,
                )
                metrics = train_model(
                    materialized[view], cfg,
                    out_dir=out / "candidates" / view / f"seed{seed}" / f"candidate{index}",
                )
                row = {
                    "dataset": args.dataset,
                    "view": view,
                    "seed": seed,
                    "candidate": index,
                    "candidate_name": hp["name"],
                    "parameters": metrics["parameter_count"],
                    "val_balanced_accuracy": metrics["val"].get("balanced_accuracy", 0.0),
                    "val_macro_f1": metrics["val"].get("macro_f1", 0.0),
                    "test_accuracy": metrics["test"].get("accuracy"),
                    "test_balanced_accuracy": metrics["test"].get("balanced_accuracy"),
                    "test_macro_f1": metrics["test"].get("macro_f1"),
                    "prediction_unique": metrics["test"].get("prediction_unique"),
                    "test_truncation_rate": metrics["sequence_audit"]["test"].get("truncation_rate"),
                    "test_discarded_fraction": metrics["sequence_audit"]["test"].get("discarded_token_fraction"),
                }
                candidate_rows.append(row)
                if args.adapter_mode in {"shared_plain", "shared_full_embed"}:
                    key = (str(hp["name"]), int(seed))
                    prior = shared_parameter_counts.setdefault(key, int(row["parameters"]))
                    if prior != int(row["parameters"]):
                        raise RuntimeError(
                            f"parameter mismatch under shared_plain for {key}: {prior} vs {row['parameters']}"
                        )
                # Validation-only selection.  Parameter count is a deterministic
                # tie-breaker; test metrics are intentionally not consulted.
                score = (float(row["val_balanced_accuracy"]), float(row["val_macro_f1"]), -int(row["parameters"]))
                if best is None or score > best[0]:
                    best = (score, row)
            assert best is not None
            selected_rows.append({**best[1], "selection": "validation_balanced_accuracy_then_macro_f1"})

    write_csv(out / "candidate_results.csv", candidate_rows)
    write_csv(out / "selected_results.csv", selected_rows)
    (out / "protocol.json").write_text(
        json.dumps({
            "dataset": args.dataset,
            "views": list(args.views),
            "seeds": list(args.seeds),
            "grid": DEFAULT_GRID if args.grid_profile == "full" else COMPACT_GRID,
            "grid_profile": args.grid_profile,
            "batch_size": args.batch_size,
            "selection": "validation balanced accuracy, then validation macro-F1, then fewer parameters",
            "test_used_for_selection": False,
            "class_weight": bool(args.class_weight),
            "adapter_mode": args.adapter_mode,
            "shared_vocabulary": True,
            "view_test_sha1": view_hashes,
            "canonical_max_search": int(args.canonical_max_search),
        }, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"candidates": len(candidate_rows), "selected": len(selected_rows), "out": str(out)}, indent=2))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
