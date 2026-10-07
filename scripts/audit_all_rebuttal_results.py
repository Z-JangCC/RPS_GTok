"""Fail-fast audit for all evidence tables used in the rebuttal.

This is intentionally stricter than a smoke test: it catches incomplete
seed/view grids, hidden majority-class predictions, zero-sized held-out sets,
duplicate dataset rows, and impossible footprint arithmetic before a table can
be cited in a response letter.
"""

from __future__ import annotations

import csv
import json
import numpy as np
from pathlib import Path


ROOT = Path("rebuttal")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def audit_semantic(errors: list[str]) -> None:
    root = ROOT / "experiments/E4_semantic_tasks"
    selected: dict[str, Path] = {}
    for path in root.glob("*_corrected*/summary.json"):
        name = path.parent.name
        dataset = name.split("_corrected", 1)[0]
        version = 0
        if "_v" in name:
            try:
                version = int(name.rsplit("_v", 1)[1])
            except ValueError:
                pass
        old = selected.get(dataset)
        old_version = 0
        if old is not None and "_v" in old.parent.name:
            try:
                old_version = int(old.parent.name.rsplit("_v", 1)[1])
            except ValueError:
                old_version = 0
        if old is None or version > old_version:
            selected[dataset] = path
    for dataset, path in sorted(selected.items()):
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not rows:
            fail(errors, f"{dataset}: empty semantic summary")
            continue
        expected = len({int(r["seed"]) for r in rows}) * len({r["view"] for r in rows})
        if len(rows) != expected:
            fail(errors, f"{dataset}: incomplete seed/view grid {len(rows)} != {expected}")
        hashes = {r["view"]: r.get("view_audit", {}).get("test", {}).get("sha1") for r in rows}
        if any(not value for value in hashes.values()):
            fail(errors, f"{dataset}: missing view hash")
        if len(set(hashes.values())) != len(hashes):
            fail(errors, f"{dataset}: view sequence hash collision")
        collapsed = [
            (r.get("seed"), r.get("view"))
            for r in rows
            if r.get("test", {}).get("prediction_unique") == 1
        ]
        if collapsed:
            fail(errors, f"{dataset}: collapsed predictions {collapsed}")


def audit_tables(errors: list[str]) -> None:
    table = ROOT / "tables/Table_R4_semantic_downstream_all.csv"
    if table.exists():
        rows = read_csv(table)
        if len(rows) != 36:
            fail(errors, f"Table R4: expected 36 corrected seed/view rows, found {len(rows)}")
        if len({(r.get("dataset"), r.get("seed"), r.get("view")) for r in rows}) != len(rows):
            fail(errors, "Table R4: duplicate dataset/seed/view rows")
        for r in rows:
            if not r.get("test_balanced_accuracy") or not r.get("prediction_unique"):
                fail(errors, f"Table R4: missing diagnostics in {r.get('dataset')}/{r.get('view')}")
    token = ROOT / "tables/Table_R9_token_level_21_fast60.csv"
    if token.exists():
        rows = read_csv(token)
        if any(float(r.get("graphs", 0)) <= 0 for r in rows):
            fail(errors, "Table R9: non-positive graph count in cited aggregate")
        if len({r.get("dataset") for r in rows}) != len(rows):
            fail(errors, "Table R9: duplicate dataset rows")
        for r in rows:
            complete = float(r.get("complete_bytes_edge", 0))
            ident = float(r.get("id_tok_edge", 0))
            if complete <= 0 or ident < 0:
                fail(errors, f"Table R9: invalid footprint for {r.get('dataset')}")
    runtime = ROOT / "experiments/E2_runtime_dataset/summary.csv"
    if runtime.exists():
        rows = read_csv(runtime)
        required = {"dataset", "canonicalization_sec_total", "canonical_fraction", "exact_fraction", "fallback_fraction"}
        for row in rows:
            if not required.issubset(row):
                fail(errors, f"E2 runtime: missing fields for {row.get('dataset')}")
            if float(row.get("canonicalization_sec_total", 0)) < 0:
                fail(errors, f"E2 runtime: negative time for {row.get('dataset')}")
    for selected in (ROOT / "experiments/E7_fair_grid").glob("*/selected_results.csv"):
        if "IMDB_corrected_no_truncation" in selected.parent.name and "shared_plain" not in selected.parent.name:
            # Historical diagnostics are retained but are not authoritative.
            continue
        rows = read_csv(selected)
        if not rows:
            fail(errors, f"E7 fair grid: empty {selected}")
            continue
        rps = [r for r in rows if r.get("view", "").startswith("rps_gtok_")]
        if any(int(r.get("prediction_unique", 0)) <= 1 for r in rps):
            fail(errors, f"E7 fair grid: collapsed RPS prediction in {selected}")
        protocol_path = selected.parent / "protocol.json"
        shared_mode = False
        if protocol_path.exists():
            shared_mode = json.loads(protocol_path.read_text(encoding="utf-8")).get("adapter_mode") in {"shared_plain", "shared_full_embed"}
        if shared_mode:
            by_candidate = {}
            for row in rows:
                key = (row.get("candidate_name"), row.get("seed"))
                by_candidate.setdefault(key, set()).add(row.get("parameters"))
            if any(len(values) != 1 for values in by_candidate.values()):
                fail(errors, f"E7 fair grid: parameter mismatch in {selected}")
    locked = ROOT / "tables/Table_R5_truncation_semantic_locked.csv"
    if locked.exists():
        rows = read_csv(locked)
        if any(float(row.get("test_truncation_rate", 1)) != 0.0 for row in rows):
            fail(errors, "locked no-truncation table contains truncated rows")
        if any(int(row.get("prediction_unique", 0)) <= 1 for row in rows):
            fail(errors, "locked no-truncation table contains collapsed predictions")
    structural = ROOT / "tables/Table_R12_all_dataset_structural_downstream_seed_level.csv"
    if structural.exists():
        rows = read_csv(structural)
        keys = {(row.get("dataset"), row.get("view"), row.get("seed")) for row in rows}
        if len(keys) != len(rows):
            fail(errors, "Table R12: duplicate dataset/view/seed rows")
        for row in rows:
            for key in ("test_mae", "test_rmse", "test_r2"):
                try:
                    value = float(row.get(key, "nan"))
                    if not np.isfinite(value):
                        fail(errors, f"Table R12: non-finite {key} for {row.get('dataset')}/{row.get('view')}")
                except ValueError:
                    fail(errors, f"Table R12: invalid {key} for {row.get('dataset')}/{row.get('view')}")
    exact21 = ROOT / "tables/Table_E11_nauty_all21_full_v12.csv"
    if not exact21.exists():
        exact21 = ROOT / "tables/Table_E11_nauty_all21_full_v7.csv"
    if not exact21.exists():
        exact21 = ROOT / "tables/Table_E11_nauty_all21_full_v3.csv"
    if not exact21.exists():
        exact21 = ROOT / "tables/Table_E11_nauty_all21_bounded3.csv"
    if exact21.exists():
        rows = read_csv(exact21)
        if len(rows) != 21 or len({row.get("dataset") for row in rows}) != 21:
            fail(errors, f"E11 all21: expected 21 unique datasets, found {len(rows)}")
        required = {"dataset", "graphs", "ok", "equal", "timeout", "error", "directed", "certified_equal_rate", "certificate"}
        for row in rows:
            if not required.issubset(row):
                fail(errors, f"E11 all21: missing fields for {row.get('dataset')}")
                continue
            ok = int(row["ok"]); equal = int(row["equal"]); total = int(row["graphs"])
            if equal > ok or ok + int(row["timeout"]) + int(row["error"]) + int(row["directed"]) != total:
                fail(errors, f"E11 all21: inconsistent counts for {row.get('dataset')}")
            rate = float(row["certified_equal_rate"])
            if not np.isfinite(rate) or rate < 0 or rate > 1:
                fail(errors, f"E11 all21: invalid equality rate for {row.get('dataset')}")
            if "directed_input" in row and int(row["directed_input"]) < 0:
                fail(errors, f"E11 all21: invalid directed input count for {row.get('dataset')}")
    for name in ("Table_E14_ogbg_molhiv_balanced_post_schema_aggregate.csv", "Table_E15_powerflow_perturbed_post_schema_aggregate.csv"):
        path = ROOT / "tables" / name
        if path.exists():
            rows = read_csv(path)
            if len(rows) != 3 or len({row.get("view") for row in rows}) != 3:
                fail(errors, f"{name}: expected three unique view rows")
            for row in rows:
                numeric = row.get("mean_balanced_accuracy") or row.get("mean_mae")
                try:
                    if not np.isfinite(float(numeric)):
                        fail(errors, f"{name}: non-finite aggregate for {row.get('view')}")
                except (TypeError, ValueError):
                    fail(errors, f"{name}: invalid aggregate for {row.get('view')}")
    raw_labels = ROOT / "audits/materialized_label_field_audit.csv"
    if raw_labels.exists():
        rows = read_csv(raw_labels)
        if len(rows) != 21 or len({row.get("dataset") for row in rows}) != 21:
            fail(errors, "materialized label-field audit: expected 21 unique datasets")
        for row in rows:
            if row.get("contains_label_field") == "1" and row.get("semantic_authorized") != "0":
                fail(errors, f"materialized label-field audit: semantic authorization leak for {row.get('dataset')}")
    constructed_audit = ROOT / "audits/constructed_task_audit.csv"
    if constructed_audit.exists():
        rows = read_csv(constructed_audit)
        if len(rows) != 21 or len({row.get("dataset") for row in rows}) != 21:
            fail(errors, "constructed task audit: expected 21 datasets")
        for row in rows:
            if row.get("input_leakage") != "0" or row.get("source_y_null") != "1" or row.get("target_recomputed") != "1":
                fail(errors, f"constructed task audit: leakage/alignment failure for {row.get('dataset')}")
    constructed_table = ROOT / "tables/Table_E16_constructed_topology_proxy_notrunc_aggregate.csv"
    if constructed_table.exists():
        rows = read_csv(constructed_table)
        if len(rows) != 63 or len({(row.get("dataset"), row.get("view")) for row in rows}) != 63:
            fail(errors, "E16 constructed no-truncation: expected 21x3 aggregate rows")
        for row in rows:
            if int(row.get("seeds", 0)) != 3 or float(row.get("test_truncation_rate", 1)) != 0.0:
                fail(errors, f"E16 constructed no-truncation: invalid protocol for {row.get('dataset')}/{row.get('view')}")
            for key in ("mean_mae", "mean_rmse", "mean_r2"):
                try:
                    if not np.isfinite(float(row[key])):
                        fail(errors, f"E16 constructed: non-finite {key} for {row.get('dataset')}/{row.get('view')}")
                except (KeyError, ValueError):
                    fail(errors, f"E16 constructed: invalid {key} for {row.get('dataset')}/{row.get('view')}")


def main() -> None:
    errors: list[str] = []
    audit_semantic(errors)
    audit_tables(errors)
    if errors:
        newline = chr(10)
        raise SystemExit("rebuttal evidence audit failed:" + newline + "- " + (newline + "- ").join(errors))
    print("rebuttal evidence audit passed")


if __name__ == "__main__":
    main()
