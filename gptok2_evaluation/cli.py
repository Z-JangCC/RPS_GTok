"""CLI for GPTok2 component evaluation suites."""

from __future__ import annotations

import argparse
import sys

from gptok2_evaluation import cora_full_graph_eval, representative_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="GPTok2 tokenizer component evaluation CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rep = sub.add_parser("representative", help="Run representative dataset evaluation.")
    rep.add_argument("--out", default="runs/rps_gtok_representative_eval")
    rep.add_argument("--datasets", nargs="+", default=representative_eval.DATASETS)
    rep.add_argument("--seed", type=int, default=2026)
    rep.add_argument("--max-train", type=int, default=300)
    rep.add_argument("--max-val", type=int, default=80)
    rep.add_argument("--max-test", type=int, default=120)
    rep.add_argument("--max-nodes", type=int, default=256)
    rep.add_argument("--cora-ego-samples", type=int, default=500)
    rep.add_argument("--cora-ego-radius", type=int, default=1)
    rep.add_argument("--overwrite", action="store_true")

    cora = sub.add_parser("cora-full", help="Run GPTok2 on the whole Cora citation graph.")
    cora.add_argument("--out", default="runs/rps_gtok_cora_full_graph_eval")
    cora.add_argument("--root", default="runs/rps_gtok_representative_eval/CORA/pyg_cora")
    cora.add_argument("--variant", choices=["default", "high_fidelity", "both"], default="both")
    cora.add_argument("--save-artifact", action="store_true")

    args = parser.parse_args()
    if args.cmd == "representative":
        _dispatch(representative_eval.main, _argv_for_representative(args))
    elif args.cmd == "cora-full":
        _dispatch(cora_full_graph_eval.main, _argv_for_cora(args))


def _dispatch(fn, argv: list[str]) -> None:
    old = sys.argv[:]
    try:
        sys.argv = [old[0], *argv]
        fn()
    finally:
        sys.argv = old


def _argv_for_representative(args: argparse.Namespace) -> list[str]:
    argv = [
        "--out",
        args.out,
        "--seed",
        str(args.seed),
        "--max-train",
        str(args.max_train),
        "--max-val",
        str(args.max_val),
        "--max-test",
        str(args.max_test),
        "--max-nodes",
        str(args.max_nodes),
        "--cora-ego-samples",
        str(args.cora_ego_samples),
        "--cora-ego-radius",
        str(args.cora_ego_radius),
        "--datasets",
        *args.datasets,
    ]
    if args.overwrite:
        argv.append("--overwrite")
    return argv


def _argv_for_cora(args: argparse.Namespace) -> list[str]:
    argv = ["--out", args.out, "--root", args.root, "--variant", args.variant]
    if args.save_artifact:
        argv.append("--save-artifact")
    return argv


if __name__ == "__main__":
    main()
