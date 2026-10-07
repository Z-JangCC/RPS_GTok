"""Build the semantic truncation audit table from training summaries."""

from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    source = Path("rebuttal/tables/Table_R4_semantic_downstream_all.csv")
    rows = list(csv.DictReader(source.open("r", encoding="utf-8")))
    out = Path("rebuttal/tables")
    keys = ["dataset", "view", "graphs_test", "test_truncation_rate", "test_discarded_fraction"]
    with (out / "Table_R5_truncation_semantic.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: row[key] for key in keys} for row in rows)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Semantic-task sequence length and truncation audit at the configured context budget.}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & View & Test graphs & Truncation rate & Discarded fraction \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} & {row['view']} & {row['graphs_test']} & "
            f"{float(row['test_truncation_rate']):.3f} & {float(row['test_discarded_fraction']):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (out / "Table_R5_truncation_semantic.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
