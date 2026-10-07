"""Build the schema-contract LaTeX table from the executable manifest."""

from __future__ import annotations

import csv
from pathlib import Path

import yaml


def main() -> None:
    manifest = yaml.safe_load(Path("configs/schema_manifest.yaml").read_text(encoding="utf-8"))
    rows = []
    for name, spec in sorted(manifest.get("datasets", {}).items()):
        rows.append(
            {
                "dataset": name,
                "directed": spec.get("directed", False),
                "multiedge": spec.get("multiedge", False),
                "self_loop": spec.get("self_loop", False),
                "node_type": spec.get("retain_node_type", False),
                "edge_type": spec.get("retain_edge_type", False),
                "node_attr": spec.get("retain_node_attr", False),
                "edge_attr": spec.get("retain_edge_attr", False),
                "target": spec.get("reconstruction_target", "schema-derived"),
            }
        )
    out = Path("rebuttal/tables")
    out.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    with (out / "Table_R8_schema_contract.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Executable dataset-level reconstruction contracts.}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"Dataset & Dir. & Multi. & Loop & Node type & Edge type & Node attr. & Edge attr. & Target \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']} & {int(row['directed'])} & {int(row['multiedge'])} & {int(row['self_loop'])} & "
            f"{int(row['node_type'])} & {int(row['edge_type'])} & {int(row['node_attr'])} & {int(row['edge_attr'])} & "
            f"{row['target'].replace('_', r'\\_')} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (out / "Table_R8_schema_contract.tex").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
