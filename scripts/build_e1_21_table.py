"""Build the 21-dataset canonical-control summary table."""

from __future__ import annotations

import csv
from pathlib import Path


def main() -> None:
    source = Path("rebuttal/experiments/E1_21_canonical_controls/summary.csv")
    rows = list(csv.DictReader(source.open("r", encoding="utf-8")))
    out = Path("rebuttal/tables")
    keys = ["view", "datasets", "mean_stability", "mean_tokens"]
    grouped = []
    for view in sorted({row["view"] for row in rows}):
        group = [row for row in rows if row["view"] == view]
        grouped.append(
            {
                "view": view,
                "datasets": len(group),
                "mean_stability": sum(float(x["mean_permutation_stability"]) for x in group) / len(group),
                "mean_tokens": sum(float(x["mean_tokens"]) for x in group) / len(group),
            }
        )
    with (out / "Table_R1b_canonical_controls_21.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(grouped)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{21-dataset canonicalization control audit on sampled materialized records.}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"View & Datasets & Mean permutation stability & Mean tokens \\",
        r"\midrule",
    ]
    for row in grouped:
        lines.append(
            f"{row['view']} & {row['datasets']} & {row['mean_stability']:.3f} & {row['mean_tokens']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (out / "Table_R1b_canonical_controls_21.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
