"""Run deterministic primitive-coverage corruption ablations on synthetic graphs."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import networkx as nx

from gptok2.data.synthetic import generate_synthetic_graphs
from gptok2.metrics.evaluate import structure_fidelity
from gptok2_tokenizer import GPTok2Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run coverage ablation.")
    parser.add_argument("--out", default="rebuttal/experiments/E5_reconstruction_ablation")
    parser.add_argument("--graphs", type=int, default=24)
    parser.add_argument("--max-len", type=int, default=256)
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
    tokenizer = GPTok2Tokenizer().fit(records)
    rows = []
    for corruption in [0.0, 0.05, 0.10, 0.20]:
        for record in records:
            encoded = tokenizer.encode(record, mode="motif_hybrid")
            tokens = list(encoded.tokens)
            if corruption > 0:
                keep = max(1, int(round(len(tokens) * (1.0 - corruption))))
                rng = random.Random(f"coverage:{record.graph_id}:{corruption}")
                removable = [idx for idx, token in enumerate(tokens) if token not in {"BEGIN_GRAPH", "END_GRAPH", "STOP"}]
                rng.shuffle(removable)
                remove = set(removable[: max(0, len(tokens) - keep)])
                tokens = [token for idx, token in enumerate(tokens) if idx not in remove]
            try:
                decoded = tokenizer.decode(tokens, mode="motif_hybrid")
                metric = structure_fidelity(record, decoded)
                strict = float(metric.get("strict_topology_exact", metric.get("exact_reconstruction", 0.0)))
                valid = 1.0
            except Exception:
                strict = 0.0
                valid = 0.0
            rows.append(
                {
                    "graph_id": record.graph_id,
                    "variant": f"drop_{int(corruption * 100)}",
                    "nominal_coverage": 1.0 - corruption,
                    "token_count": len(tokens),
                    "valid_program": valid,
                    "strict_reconstruction": strict,
                }
            )
    write_csv(out / "per_graph.csv", rows)
    summary = []
    for variant in sorted({row["variant"] for row in rows}):
        group = [row for row in rows if row["variant"] == variant]
        summary.append(
            {
                "variant": variant,
                "graphs": len(group),
                "nominal_coverage": sum(row["nominal_coverage"] for row in group) / len(group),
                "mean_tokens": sum(row["token_count"] for row in group) / len(group),
                "valid_program_rate": sum(row["valid_program"] for row in group) / len(group),
                "strict_reconstruction_rate": sum(row["strict_reconstruction"] for row in group) / len(group),
            }
        )
    write_csv(out / "summary.csv", summary)
    print(f"wrote {len(rows)} rows to {out}")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
