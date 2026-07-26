"""Command line interface for the standalone GPTok2 tokenizer component."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import yaml

from gptok2.data.io import load_records
from gptok2.data.schema import GraphRecord, record_to_dict
from gptok2.data.synthetic import generate_synthetic_graphs
from gptok2.utils.io import write_jsonl
from gptok2_tokenizer import GPTok2Tokenizer

MODES = [
    "original",
    "compact",
    "entropy",
    "motif_macro",
    "motif_entropy",
    "motif_hybrid",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="GPTok2 tokenizer component CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    smoke = sub.add_parser("smoke", help="Run a tiny end-to-end tokenizer smoke test.")
    smoke.add_argument("--out", default="runs/rps_gtok_smoke")

    fit = sub.add_parser("fit", help="Fit a GPTok2 tokenizer artifact from GraphRecord JSONL.")
    fit.add_argument("--input", required=True)
    fit.add_argument("--artifact", required=True)
    fit.add_argument("--config", default="")

    encode = sub.add_parser("encode", help="Encode GraphRecord JSONL with a fitted artifact.")
    encode.add_argument("--artifact", required=True)
    encode.add_argument("--input", required=True)
    encode.add_argument("--output", required=True)
    encode.add_argument("--mode", choices=MODES + ["all"], default="all")

    evaluate = sub.add_parser("evaluate", help="Evaluate reconstruction on GraphRecord JSONL.")
    evaluate.add_argument("--artifact", required=True)
    evaluate.add_argument("--input", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--mode", choices=MODES, default="compact")

    args = parser.parse_args()
    if args.cmd == "smoke":
        _smoke(Path(args.out))
    elif args.cmd == "fit":
        records = load_records(args.input)
        config = _load_yaml(args.config) if args.config else {}
        tokenizer = GPTok2Tokenizer(config).fit(records)
        tokenizer.save(args.artifact)
        print(f"saved artifact: {args.artifact}")
    elif args.cmd == "encode":
        tokenizer = GPTok2Tokenizer.load(args.artifact)
        modes = MODES if args.mode == "all" else [args.mode]
        rows = []
        for record in load_records(args.input):
            for mode in modes:
                rows.append(tokenizer.encode(record, mode=mode).to_dict())
        write_jsonl(args.output, rows)
        print(f"saved encoded programs: {args.output}")
    elif args.cmd == "evaluate":
        tokenizer = GPTok2Tokenizer.load(args.artifact)
        rows = [tokenizer.evaluate_reconstruction(record, mode=args.mode) for record in load_records(args.input)]
        write_jsonl(args.output, rows)
        print(f"saved evaluation rows: {args.output}")


def _smoke(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "run": {"seed": 7},
        "data": {"synthetic": {"num_graphs": 24, "num_nodes_min": 8, "num_nodes_max": 16, "families": ["cycle", "star", "motif_mix"]}},
    }
    records = generate_synthetic_graphs(config, seed=7, ood=False)
    train, test = records, records
    tokenizer = GPTok2Tokenizer().fit(train)
    tokenizer.save(out_dir / "tokenizer.json")
    write_jsonl(out_dir / "test_graphs.jsonl", [record_to_dict(r) for r in test])
    rows = []
    for record in test:
        for mode in MODES:
            rows.append(tokenizer.evaluate_reconstruction(record, mode=mode))
    write_jsonl(out_dir / "eval.jsonl", rows)
    exact = sum(float(r.get("exact_reconstruction", 0.0)) for r in rows) / max(1, len(rows))
    print(json.dumps({"out": str(out_dir), "rows": len(rows), "avg_exact_reconstruction": exact}, indent=2))


def _load_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":
    main()
