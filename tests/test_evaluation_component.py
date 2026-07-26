from __future__ import annotations

from pathlib import Path

from gptok2_evaluation import cli
from gptok2_evaluation.representative_eval import DATASETS, normalize_dataset_name


def test_evaluation_cli_exports_main() -> None:
    assert callable(cli.main)
    assert "CORA" not in DATASETS
    assert normalize_dataset_name("ogb_molhiv") == "OGBG-MOLHIV"


def test_representative_report_snapshot_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    report_dir = root / "reports" / "representative"
    assert (report_dir / "summary_all.csv").exists()
    assert (report_dir / "cora_full_graph_summary.csv").exists()
