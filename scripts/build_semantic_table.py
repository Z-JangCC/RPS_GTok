"""Compatibility wrapper for the corrected semantic table builder.

The previous implementation aggregated pilot summaries and silently mixed
single-seed, truncated, and corrected runs.  It is intentionally no longer
allowed to regenerate ``Table_R4_semantic_downstream_all`` from those files.
"""

from __future__ import annotations

from pathlib import Path
import shutil

from build_semantic_corrected_table import main as build_corrected


def main() -> None:
    build_corrected()
    out = Path("rebuttal/tables")
    shutil.copyfile(out / "Table_R4_semantic_downstream_corrected.csv", out / "Table_R4_semantic_downstream_all.csv")
    print("Table_R4_semantic_downstream_all.csv now points only to corrected runs")


if __name__ == "__main__":
    main()
