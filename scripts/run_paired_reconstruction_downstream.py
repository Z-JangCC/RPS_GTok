"""Pair reconstruction fidelity with semantic downstream under controlled drops."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from gptok2.data.io import load_records
from gptok2.metrics.evaluate import structure_fidelity
from gptok2_tokenizer import GPTok2Tokenizer
from rps_gtok_consumption.data import TokenExample
from rps_gtok_consumption.training import TrainConfig, train_model
from rps_gtok_consumption.views import TokenViewBuilder
from scripts.run_topology_shift_downstream import split_records


def corrupt(tokens: list[str], graph_id: str, drop: float) -> list[str]:
    if drop <= 0:
        return list(tokens)
    removable = list(range(len(tokens)))
    rng = random.Random(f"paired-drop:{graph_id}:{drop}")
    rng.shuffle(removable)
    remove = set(removable[: max(0, int(round(len(tokens) * drop)))])
    return [token for index, token in enumerate(tokens) if index not in remove]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--drops", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--view", default="rps_gtok_full")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tokenizer = GPTok2Tokenizer.load(args.artifact)
    records = {"train": load_records(args.train), "val": load_records(args.val), "test": load_records(args.test)}
    for split, limit in (("train", args.max_train), ("val", args.max_val), ("test", args.max_test)):
        if int(limit) > 0:
            records[split] = records[split][: int(limit)]
    builder = TokenViewBuilder(tokenizer, seed=2026)
    builder.fit(records["train"], [args.view])
    examples_by_drop = {}
    fidelity_rows = []
    for drop in args.drops:
        split_examples = {}
        for split, rows in records.items():
            examples = []
            for record in rows:
                base = builder.build(record, args.view)
                program_tokens = list(base)
                tokens = corrupt(base, record.graph_id, float(drop))
                strict = 0.0; valid = 0.0
                try:
                    encoded = tokenizer.encode(record, mode="motif_hybrid")
                    corrupted_program = corrupt(program_tokens, record.graph_id, float(drop))
                    decoded = tokenizer.decode(corrupted_program, mode="motif_hybrid")
                    metric = structure_fidelity(record, decoded)
                    strict = float(metric.get("strict_topology_exact", metric.get("exact_reconstruction", 0.0)))
                    valid = 1.0
                except Exception:
                    pass
                fidelity_rows.append({"dataset": args.dataset, "split": split, "graph_id": record.graph_id, "drop": drop, "tokens": len(tokens), "strict_reconstruction": strict, "valid_program": valid})
                examples.append(TokenExample(graph_id=record.graph_id, dataset=args.dataset, split=split, view=f"{args.view}_drop{drop}", tokens=tokens, y=int(record.y.item()), task_type="classification", num_nodes=int(record.num_nodes), num_edges=int(record.edge_index.shape[1]) if record.edge_index.numel() else 0))
            split_examples[split] = examples
        examples_by_drop[drop] = split_examples
    union = [example.tokens for split_examples in examples_by_drop.values() for example in split_examples["train"]]
    result_rows = []
    for drop, splits in examples_by_drop.items():
        for seed in args.seeds:
            metrics = train_model(splits, TrainConfig(max_len=args.max_len, batch_size=args.batch_size, epochs=args.epochs, patience=max(4, args.epochs // 3), task_type="classification", model={"adapter": "plain", "dim": 32, "layers": 1, "heads": 4, "dropout": 0.1}, device=args.device, seed=seed, class_weight=True, vocab_sequences=union), out_dir=out / f"drop{drop}" / f"seed{seed}")
            test = metrics["test"]
            fidelity = [row for row in fidelity_rows if row["split"] == "test" and row["drop"] == drop]
            result_rows.append({"dataset": args.dataset, "drop": drop, "seed": seed, "test_accuracy": test.get("accuracy"), "test_balanced_accuracy": test.get("balanced_accuracy"), "test_macro_f1": test.get("macro_f1"), "prediction_unique": test.get("prediction_unique"), "test_truncation_rate": metrics["sequence_audit"]["test"].get("truncation_rate"), "strict_reconstruction_test": sum(row["strict_reconstruction"] for row in fidelity) / max(1, len(fidelity)), "valid_program_test": sum(row["valid_program"] for row in fidelity) / max(1, len(fidelity)), "parameters": metrics["parameter_count"]})
    def write(path, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            fields = sorted({key for row in rows for key in row}); writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    write(out / "results.csv", result_rows); write(out / "fidelity_per_graph.csv", fidelity_rows)
    (out / "protocol.json").write_text(json.dumps({"dataset": args.dataset, "task": "paired_reconstruction_downstream", "view": args.view, "drops": args.drops, "seeds": args.seeds, "test_used_for_selection": False, "fidelity_metric": "strict_topology_exact", "same_sequence_for_reconstruction_and_downstream": True, "limits": {"train": args.max_train, "val": args.max_val, "test": args.max_test}}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"result_rows": len(result_rows), "fidelity_rows": len(fidelity_rows), "out": str(out)}, indent=2))


if __name__ == "__main__": main()
