"""Audit active vocabulary saturation on synthetic and real sample records."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from gptok2.data.synthetic import generate_synthetic_graphs
from gptok2_tokenizer import GPTok2Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run vocabulary-size sensitivity sweep.")
    parser.add_argument("--out", default="rebuttal/experiments/vocabulary_sensitivity")
    parser.add_argument("--sizes", nargs="+", type=int, default=[32, 64, 128, 256, 1024])
    parser.add_argument("--graphs", type=int, default=48)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "run": {"seed": 2026},
        "data": {
            "synthetic": {
                "num_graphs": int(args.graphs),
                "num_nodes_min": 8,
                "num_nodes_max": 18,
                "families": ["cycle", "star", "tree", "motif_mix"],
            }
        },
    }
    records = generate_synthetic_graphs(config, seed=2026)
    rows = []
    for size in args.sizes:
        tokenizer = GPTok2Tokenizer(
            {
                "vq": {"num_codes": int(size)},
                "compact_entropy": {"max_macros": min(64, size), "max_bpe_merges": min(64, size)},
                "motif_macro": {"max_structural_macros": min(64, size), "max_merge_schemas": min(64, size), "max_code_schemas": min(64, size)},
            }
        ).fit(records)
        values = [tokenizer.encode(record, mode="motif_hybrid") for record in records]
        rows.append(
            {
                "requested_vocab_size": size,
                "active_codebook_size": len(tokenizer.codebook.codes) if tokenizer.codebook is not None else 0,
                "interface_vocab_size": len(tokenizer.codebook.interfaces) if tokenizer.codebook is not None else 0,
                "mean_tokens": sum(value.token_count for value in values) / len(values),
                "mean_original_tokens": sum(value.original_token_count for value in values) / len(values),
                "mean_lossless_expand_match": sum(value.lossless_expand_match for value in values) / len(values),
                "mean_complete_bytes": sum(value.complete_token_bytes for value in values) / len(values),
            }
        )
    write_csv(out / "summary.csv", rows)
    print(f"wrote {len(rows)} rows to {out}")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
