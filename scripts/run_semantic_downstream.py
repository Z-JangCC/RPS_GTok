"""Run matched semantic downstream views with explicit sequence audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from gptok2.data.io import load_records
from gptok2_tokenizer import GPTok2Tokenizer
from rps_gtok_consumption.experiment import examples_for_view
from rps_gtok_consumption.training import TrainConfig, train_model
from rps_gtok_consumption.views import TokenViewBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic graph downstream views.")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--views", nargs="+", default=["rps_gtok_identifier", "rps_gtok_full"])
    parser.add_argument("--out", default="rebuttal/experiments/E4_semantic_tasks")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--device", default="auto", help="torch device; use cuda when available")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--adapter-mode", choices=["shared_plain", "shared_full_embed", "native"], default="native")
    parser.add_argument("--task-type", choices=["classification", "regression", "multilabel"], default="classification")
    parser.add_argument("--canonical-max-search", type=int, default=100)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = GPTok2Tokenizer.load(args.artifact)
    tokenizer.config["patch"].update(
        {
            "sparse_edge_patches": True,
            "sparse_density_threshold": 1.0,
            "sparse_clustering_threshold": 1.0,
            "max_patches_per_graph": 12000,
        }
    )
    tokenizer.config["patch"]["max_patches_per_graph"] = 128
    tokenizer.config["program"]["noncanonical_order"] = True
    tokenizer.config["program"]["max_global_links"] = 64
    tokenizer.config["canonicalization"]["max_search_nodes"] = min(
        int(args.canonical_max_search), int(tokenizer.config["canonicalization"].get("max_search_nodes", args.canonical_max_search))
    )
    records = {
        "train": load_records(args.train),
        "val": load_records(args.val),
        "test": load_records(args.test),
    }
    for split, limit in [("train", args.max_train), ("val", args.max_val), ("test", args.max_test)]:
        if int(limit) > 0:
            records[split] = records[split][: int(limit)]
    builder = TokenViewBuilder(tokenizer, seed=2026).fit(records["train"], list(args.views))
    rows = []
    matched_parameter_counts = []
    # Materialize every view exactly once.  The prior implementation rebuilt
    # canonical views inside the seed loop, which made a 3-seed run needlessly
    # repeat expensive canonicalization and could leave incomplete summaries.
    materialized = {}
    view_audits = {}
    for view in args.views:
        materialized[view] = {
            split: examples_for_view(rows_, builder, args.dataset, split, view, args.task_type)
            for split, rows_ in records.items()
        }
        view_audits[view] = {
            split: token_view_audit(examples)
            for split, examples in materialized[view].items()
        }
    audit_hashes = {view: view_audits[view]["test"]["sha1"] for view in args.views}
    if len(audit_hashes) != len(set(audit_hashes.values())):
        collisions = [view for view, digest in audit_hashes.items() if list(audit_hashes.values()).count(digest) > 1]
        raise RuntimeError(f"view audit collision on test sequences; refusing invalid comparison: {collisions}")
    # A shared adapter is only fair if all views use the same embedding table.
    # Build one frozen union vocabulary from training sequences and pass it to
    # every consumer; native mode remains available as an explicitly enriched
    # auxiliary comparison.
    union_vocab_sequences = None
    if args.adapter_mode in {"shared_plain", "shared_full_embed"}:
        union_vocab_sequences = [example.tokens for view in args.views for example in materialized[view]["train"]]
    for seed in args.seeds:
      for view in args.views:
        splits = materialized[view]
        # Every RPS representation receives the same Full-Embed adapter.  The
        # old equality check only recognized ``rps_gtok_full`` and accidentally
        # trained new RPS variants with the plain baseline adapter.
        is_rps_view = view.startswith("rps_gtok_")
        if args.adapter_mode == "shared_plain":
            adapter = "plain"
        elif args.adapter_mode == "shared_full_embed":
            adapter = "full_embed"
        else:
            adapter = "full_embed" if is_rps_view else "plain"
        base_model = {"adapter": adapter, "dim": args.dim, "layers": args.layers, "heads": args.heads, "dropout": 0.1}
        result = train_model(
            splits,
            TrainConfig(
                max_len=args.max_len,
                batch_size=8,
                epochs=args.epochs,
                patience=max(1, args.epochs - 1),
                task_type=args.task_type,
                model=base_model,
                device=args.device,
                seed=int(seed),
                class_weight=bool(args.class_weight),
                vocab_sequences=union_vocab_sequences,
            ),
            out_dir=out / view / f"seed{int(seed)}",
        )
        rows.append({"seed": int(seed), "view": view, "view_audit": view_audits[view], **result})
        matched_parameter_counts.append(int(result["parameter_count"]))
    expected = len(args.seeds) * len(args.views)
    if len(rows) != expected:
        raise RuntimeError(f"incomplete semantic run: {len(rows)} rows, expected {expected}")
    if args.adapter_mode in {"shared_plain", "shared_full_embed"} and len(set(matched_parameter_counts)) != 1:
        raise RuntimeError(f"shared adapter parameter mismatch: {sorted(set(matched_parameter_counts))}")
    (out / "summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    (out / "protocol.json").write_text(
        json.dumps({
            "dataset": args.dataset,
            "views": list(args.views),
            "seeds": [int(x) for x in args.seeds],
            "epochs": int(args.epochs),
            "class_weight": bool(args.class_weight),
            "device": args.device,
            "adapter_mode": args.adapter_mode,
            "shared_vocabulary": args.adapter_mode in {"shared_plain", "shared_full_embed"},
            "task_type": args.task_type,
            "canonical_max_search": int(args.canonical_max_search),
            "max_len": int(args.max_len),
            "limits": {"train": int(args.max_train), "val": int(args.max_val), "test": int(args.max_test)},
            "view_test_sha1": audit_hashes,
            "completed_rows": len(rows),
            "expected_rows": len(args.seeds) * len(args.views),
        }, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(rows, indent=2, sort_keys=True))


def token_view_audit(examples) -> dict:
    """Audit whether a view actually differs before model fitting."""
    sequences = [list(ex.tokens) for ex in examples]
    digest = hashlib.sha1(
        json.dumps(sequences, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    unique_sequences = len({tuple(seq) for seq in sequences})
    lengths = [len(seq) for seq in sequences]
    return {
        "graphs": len(sequences),
        "unique_sequences": unique_sequences,
        "mean_tokens": float(sum(lengths) / max(1, len(lengths))),
        "sha1": digest,
    }


if __name__ == "__main__":
    main()
