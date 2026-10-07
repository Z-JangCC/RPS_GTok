"""Build the auditable semantic-downstream table from corrected multi-seed runs.

Only directories whose name ends in ``_corrected`` or ``_corrected_v*`` are
accepted.  This intentionally excludes the earlier pilot/degenerate summaries
so that an obsolete table cannot silently be used in the rebuttal.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


def main() -> None:
    root = Path("rebuttal/experiments/E4_semantic_tasks")
    out = Path("rebuttal/tables")
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    candidates = sorted(root.glob("*_corrected*/summary.json"))
    # A dataset may have an interrupted/old corrected run beside a later one.
    # Select the highest explicit version (or the plain corrected run when no
    # versioned replacement exists) instead of silently mixing them.
    selected: dict[str, Path] = {}
    def rank(path: Path) -> tuple[int, str]:
        name = path.parent.name
        match = re.search(r"_corrected_v(\d+)$", name)
        return (int(match.group(1)) if match else 0, name)
    for path in candidates:
        dataset_key = re.sub(r"_corrected(?:_v\d+)?$", "", path.parent.name)
        if dataset_key not in selected or rank(path) >= rank(selected[dataset_key]):
            selected[dataset_key] = path
    for path in sorted(selected.values()):
        run_name = path.parent.name
        dataset = re.sub(r"_corrected(?:_v\d+)?$", "", run_name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload:
            continue
        for item in payload:
            test = item.get("test", {})
            audit = item.get("sequence_audit", {}).get("test", {})
            view_audit = item.get("view_audit", {}).get("test", {})
            rows.append({
                "dataset": dataset,
                "run": run_name,
                "seed": item.get("seed"),
                "view": item.get("view"),
                "graphs_train": item.get("sequence_audit", {}).get("train", {}).get("graphs"),
                "graphs_test": audit.get("graphs"),
                "parameters": item.get("parameter_count"),
                "test_accuracy": test.get("accuracy"),
                "test_balanced_accuracy": test.get("balanced_accuracy"),
                "test_macro_f1": test.get("macro_f1"),
                "test_majority_accuracy": test.get("majority_accuracy"),
                "prediction_unique": test.get("prediction_unique"),
                "prediction_histogram": json.dumps(test.get("prediction_histogram", {}), sort_keys=True),
                "confusion_matrix": json.dumps(test.get("confusion_matrix", [])),
                "test_truncation_rate": audit.get("truncation_rate"),
                "test_discarded_fraction": audit.get("discarded_token_fraction"),
                "view_unique_sequences": view_audit.get("unique_sequences"),
                "view_sha1": view_audit.get("sha1"),
            })
    if not rows:
        raise SystemExit("No corrected semantic summaries found")
    fields = list(rows[0])
    with (out / "Table_R4_semantic_downstream_corrected.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    trunc_fields = ["dataset", "run", "seed", "view", "graphs_train", "graphs_test",
                    "test_truncation_rate", "test_discarded_fraction", "view_unique_sequences"]
    with (out / "Table_R5_truncation_semantic_corrected.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=trunc_fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in trunc_fields} for row in rows])
    trunc_tex = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Per-seed truncation audit for corrected semantic runs.}",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Dataset & View & Seed & Trunc. rate & Discarded fraction \\", r"\midrule",
    ]
    for row in rows:
        trunc_tex.append(
            f"{row['dataset']} & {row['view']} & {row['seed']} & "
            f"{float(row['test_truncation_rate']):.3f} & {float(row['test_discarded_fraction']):.3f} \\\\" 
        )
    trunc_tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (out / "Table_R5_truncation_semantic_corrected.tex").write_text("\n".join(trunc_tex), encoding="utf-8")
    (out / "Table_R5_truncation_semantic.tex").write_text("\n".join(trunc_tex), encoding="utf-8")

    # Aggregate by dataset/view across seeds, retaining uncertainty rather than
    # presenting one arbitrarily selected seed.
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["dataset"], row["view"]), []).append(row)
    agg = []
    for (dataset, view), items in sorted(grouped.items()):
        def vals(key):
            return [float(x[key]) for x in items if x[key] is not None]
        def mean_sd(key):
            xs = vals(key)
            if not xs:
                return "", ""
            mean = sum(xs) / len(xs)
            sd = (sum((x - mean) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5
            return f"{mean:.4f}", f"{sd:.4f}"
        acc, acc_sd = mean_sd("test_accuracy")
        bal, bal_sd = mean_sd("test_balanced_accuracy")
        f1, f1_sd = mean_sd("test_macro_f1")
        agg.append({"dataset": dataset, "view": view, "n_seeds": len(items),
                    "accuracy_mean": acc, "accuracy_sd": acc_sd,
                    "balanced_accuracy_mean": bal, "balanced_accuracy_sd": bal_sd,
                    "macro_f1_mean": f1, "macro_f1_sd": f1_sd})
    afields = list(agg[0])
    with (out / "Table_R4_semantic_downstream_corrected_aggregate.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=afields)
        writer.writeheader()
        writer.writerows(agg)
    tex = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Corrected semantic downstream results (mean $\pm$ standard deviation over three seeds).}",
        r"\begin{tabular}{llrrr}", r"\toprule",
        "Dataset & View & Accuracy & Balanced Acc. & Macro-F1 \\\\", r"\midrule",
    ]
    for item in agg:
        tex.append(
            f"{item['dataset']} & {item['view']} & {item['accuracy_mean']} $\\pm${item['accuracy_sd']} & "
            f"{item['balanced_accuracy_mean']} $\\pm${item['balanced_accuracy_sd']} & "
            f"{item['macro_f1_mean']} $\\pm${item['macro_f1_sd']} \\\\" 
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (out / "Table_R4_semantic_downstream_corrected.tex").write_text("\n".join(tex), encoding="utf-8")
    (out / "Table_R4_semantic_downstream_all.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"wrote {len(rows)} seed/view rows and {len(agg)} aggregate rows")


if __name__ == "__main__":
    main()
