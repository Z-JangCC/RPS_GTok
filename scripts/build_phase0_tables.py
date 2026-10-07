"""Build reproducible Phase-0 CSV and LaTeX tables from generated artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase-0 rebuttal tables.")
    parser.add_argument("--root", default="rebuttal")
    args = parser.parse_args()
    root = Path(args.root)
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    e1_path = root / "experiments/E1_canonicalized_baselines_fixed2/summary.csv"
    if not e1_path.exists():
        e1_path = root / "experiments/E1_canonicalized_baselines/summary.csv"
    e1 = read_csv(e1_path)
    r1 = [
        {
            "method": row["method"],
            "stability": f"{float(row['mean_stability_rate']):.3f}",
            "id_tokens": f"{float(row['mean_base_length']):.3f}",
            "id_bytes": f"{float(row['mean_identifier_bytes']):.3f}",
            "payload_bytes": f"{float(row['mean_payload_bytes']):.3f}",
            "complete_bytes": f"{float(row['mean_complete_token_bytes']):.3f}",
        }
        for row in e1
    ]
    write_csv(tables / "Table_R1_canonicalized_baselines.csv", r1)
    write_tex(
        tables / "Table_R1_canonicalized_baselines.tex",
        "Table R1: Canonicalization-isolation pilot.",
        ["Method", "Stability", "ID tokens", "ID bytes", "Payload bytes", "Complete bytes"],
        [[r["method"], r["stability"], r["id_tokens"], r["id_bytes"], r["payload_bytes"], r["complete_bytes"]] for r in r1],
    )

    e2_path = root / "experiments/E2_runtime_scaling_pilot/scaling.csv"
    if e2_path.exists():
        e2 = read_csv(e2_path)
        r2 = [
            {
                "family": row["family"],
                "nodes": row["nodes"],
                "edges": row["edges"],
                "canonical_sec": row["canonicalization_sec"],
                "fraction": row["canonical_fraction"],
                "branches": row["search_branches"],
                "exact": row["exact_completed"],
                "fallback": row["fallback_used"],
            }
            for row in e2
        ]
        write_csv(tables / "Table_R2_runtime_scaling_pilot.csv", r2)
        write_tex(
            tables / "Table_R2_runtime_scaling_pilot.tex",
            "Table R2: Canonicalization runtime-scaling pilot.",
            ["Family", "Nodes", "Edges", "Canonical sec", "Canonical fraction", "Branches", "Exact", "Fallback"],
            [[r["family"], r["nodes"], r["edges"], r["canonical_sec"], r["fraction"], r["branches"], r["exact"], r["fallback"]] for r in r2],
        )

    (tables / "Table_R3_full_footprint_pilot.csv").write_text(
        (tables / "Table_R1_canonicalized_baselines.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    semantic_path = root / "experiments/E4_semantic_tasks/PROTEINS_pilot/summary.json"
    if semantic_path.exists():
        semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
        r4 = [
            {
                "view": row["view"],
                "parameters": str(row["parameter_count"]),
                "test_accuracy": f"{float(row['test']['accuracy']):.3f}",
                "test_macro_f1": f"{float(row['test']['macro_f1']):.3f}",
                "truncation_rate": f"{float(row['sequence_audit']['test']['truncation_rate']):.3f}",
            }
            for row in semantic
        ]
        write_csv(tables / "Table_R4_semantic_downstream_pilot.csv", r4)
        write_tex(
            tables / "Table_R4_semantic_downstream_pilot.tex",
            "Table R4: PROTEINS semantic downstream diagnostic pilot.",
            ["View", "Parameters", "Accuracy", "Macro-F1", "Truncation"],
            [[r["view"], r["parameters"], r["test_accuracy"], r["test_macro_f1"], r["truncation_rate"]] for r in r4],
        )

    token_path = root / "experiments/token_level_21_final_bounded/summary.csv"
    if not token_path.exists():
        token_path = root / "experiments/token_level_21_fast60/summary.csv"
    artifact_rows = {}
    artifact_path = root / "experiments/E3_artifact_footprint_21/summary.csv"
    if artifact_path.exists():
        artifact_rows = {row["dataset"]: row for row in read_csv(artifact_path)}
    if token_path.exists():
        token_rows = read_csv(token_path)
        excluded = [row for row in token_rows if int(float(row.get("graphs", 0))) <= 0]
        token_rows = [row for row in token_rows if int(float(row.get("graphs", 0))) > 0]
        if excluded:
            write_csv(
                tables / "Table_R9_excluded_datasets.csv",
                [{"dataset": row.get("dataset"), "reason": "no held-out graphs; legacy run reported zeros"} for row in excluded],
            )
        r9 = [
            {
                "dataset": row["dataset"],
                "graphs": row["graphs"],
                "id_tok_edge": f"{float(row['mean_id_tokens_per_edge']):.3f}",
                "payload_bytes_edge": f"{float(row['mean_payload_bytes']) / max(1.0, float(row['mean_num_edges'])):.3f}",
                "complete_bytes_edge": f"{float(row['mean_complete_token_bytes']) / max(1.0, float(row['mean_num_edges'])):.3f}",
                "artifact_bytes_edge": "0.000",
                "lossless": f"{float(row['mean_lossless_expand_match']):.3f}",
                "perm_stability": f"{float(row['mean_permutation_stability']):.3f}",
                "canonical_exact": f"{float(row['mean_canonical_exact_completion']):.3f}",
            }
            for row in token_rows
        ]
        for row in r9:
            artifact = artifact_rows.get(row["dataset"], {})
            row["artifact_bytes_edge"] = (
                f"{float(artifact.get('amortized_artifact_bytes_per_edge', 0.0)):.3f}"
            )
        write_csv(tables / "Table_R9_token_level_21_fast60.csv", r9)
        write_tex(
            tables / "Table_R9_token_level_21_fast60.tex",
            "Table R9: 21-dataset token-level fast protocol.",
            ["Dataset", "Graphs", "ID Tok./edge", "Payload bytes/edge", "Complete bytes/edge", "Artifact bytes/edge", "Expansion", "Perm.", "Canonical exact"],
            [[r["dataset"], r["graphs"], r["id_tok_edge"], r["payload_bytes_edge"], r["complete_bytes_edge"], r["artifact_bytes_edge"], r["lossless"], r["perm_stability"], r["canonical_exact"]] for r in r9],
        )

    coverage_path = root / "experiments/E5_reconstruction_ablation/summary.csv"
    if coverage_path.exists():
        coverage = read_csv(coverage_path)
        write_csv(tables / "Table_R7_reconstruction_ablation.csv", coverage)
        write_tex(
            tables / "Table_R7_reconstruction_ablation.tex",
            "Table R7: Controlled token deletion and reconstruction ablation.",
            ["Variant", "Nominal coverage", "Mean tokens", "Valid program", "Strict reconstruction"],
            [
                [
                    row["variant"],
                    f"{float(row['nominal_coverage']):.3f}",
                    f"{float(row['mean_tokens']):.3f}",
                    f"{float(row['valid_program_rate']):.3f}",
                    f"{float(row['strict_reconstruction_rate']):.3f}",
                ]
                for row in coverage
            ],
        )

    vocab_path = root / "experiments/vocabulary_sensitivity/summary.csv"
    if vocab_path.exists():
        vocab = read_csv(vocab_path)
        write_csv(tables / "Table_R6_vocabulary_sensitivity.csv", vocab)
        write_tex(
            tables / "Table_R6_vocabulary_sensitivity.tex",
            "Table R6: Active vocabulary and footprint sensitivity.",
            ["Requested", "Active codes", "Interfaces", "Mean tokens", "Complete bytes"],
            [
                [
                    row["requested_vocab_size"],
                    row["active_codebook_size"],
                    row["interface_vocab_size"],
                    f"{float(row['mean_tokens']):.3f}",
                    f"{float(row['mean_complete_bytes']):.3f}",
                ]
                for row in vocab
            ],
        )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path: Path, caption: str, headers: list[str], rows: list[list[str]]) -> None:
    columns = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(str(cell).replace("_", r"\_") for cell in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
