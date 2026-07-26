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
    run("-m", "pytest", "tests", "-q")


if __name__ == "__main__":
    main()
