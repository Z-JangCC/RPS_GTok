from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print(f"[run] {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    run("-m", "gptok2_tokenizer.cli", "smoke", "--out", "runs/rps_gtok_smoke")
    run(
        "-m",
        "gptok2_evaluation.cli",
        "representative",
        "--out",
        "runs/rps_gtok_representative_eval",
        "--datasets",
        "MUTAG",
        "PROTEINS",
        "IMDB-BINARY",
        "QM9",
        "ZINC",
        "OGBG-MOLHIV",
        "PEPTIDES-FUNC",
        "--overwrite",
    )
    run("-m", "gptok2_evaluation.cli", "cora-full", "--out", "runs/rps_gtok_cora_full_graph_eval", "--variant", "both", "--save-artifact")
    run("scripts/verify_release.py")


if __name__ == "__main__":
    main()
