from __future__ import annotations

import csv
import importlib
import re
import subprocess
import sys
from pathlib import Path


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRACKING_DIR = "wa" + "ndb"
GENERATED_ROOTS = {
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "logs",
    "runs",
    "tmp",
    TRACKING_DIR,
}
EXPECTED_MODES = {"original", "compact", "entropy", "motif_macro", "motif_entropy", "motif_hybrid"}
EXPECTED_DATASETS = {"CORA", "IMDB-BINARY", "MUTAG", "OGBG-MOLHIV", "PEPTIDES-FUNC", "PROTEINS", "QM9", "ZINC"}


def assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def tracked_release_files(root: Path) -> set[str] | None:
    if not (root / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except Exception:
        return None
    return {item.decode("utf-8") for item in result.stdout.split(b"\0") if item}


def is_release_path(rel: Path, tracked: set[str] | None) -> bool:
    if rel.parts and rel.parts[0] == ".git":
        return False
    if rel.parts and rel.parts[0] in GENERATED_ROOTS and tracked is None:
        return False
    if tracked is None:
        return True
    rel_text = rel.as_posix()
    return rel_text in tracked or any(item.startswith(rel_text + "/") for item in tracked)


def audit_tree_hygiene(root: Path) -> None:
    tracked = tracked_release_files(root)
    blocked_names = {
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "__pycache__",
        "logs",
        "runs",
        "tmp",
        TRACKING_DIR,
    }
    blocked_suffixes = {".log", ".pyc", ".pyo"}
    blocked: list[str] = []

    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if not is_release_path(rel, tracked):
            continue
        if path.name in blocked_names:
            blocked.append(str(rel))
        if path.is_file() and path.suffix.lower() in blocked_suffixes:
            blocked.append(str(rel))
        if path.is_symlink():
            target = path.readlink()
            if target.is_absolute():
                blocked.append(f"{rel} -> {target}")

    if blocked:
        sample = "\n".join(f"- {item}" for item in blocked[:20])
        raise RuntimeError(f"release tree hygiene audit failed:\n{sample}")


def audit_release_surface(root: Path) -> None:
    """Fail on process-version artifacts that should not be in the reviewer release."""
    tracked = tracked_release_files(root)
    this_file = Path(__file__).resolve()
    text_suffixes = {"", ".cfg", ".csv", ".gitignore", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
    blocked_paths: list[str] = []
    blocked_text: list[tuple[Path, int, str]] = []
    versioned_motif_package = "gptok2_motif_macro_" + r"v\d"
    versioned_report_dir = "representative_" + r"v\d"
    legacy_iteration_label = "v" + "8b"
    zh_suffix = "_" + "zh"
    chinese_readme = "README" + zh_suffix

    path_patterns = [
        ("process-version package", re.compile(versioned_motif_package, re.IGNORECASE)),
        ("process-version report directory", re.compile(versioned_report_dir, re.IGNORECASE)),
        ("legacy benchmark package", re.compile(r"gptok2_benchmark|benchmark", re.IGNORECASE)),
        (
            "Chinese release document",
            re.compile(rf"(?:^|/)(?:{re.escape(chinese_readme)}|.*{re.escape(zh_suffix)})\.md$", re.IGNORECASE),
        ),
        ("report-generation helper", re.compile(r"(?:generate|make|build).*report|report.*(?:generate|make|build)", re.IGNORECASE)),
        ("run log artifact", re.compile(r"(?:^|/)(?:run|experiment|debug).*\\.(?:log|txt|md)$", re.IGNORECASE)),
    ]
    text_patterns = [
        ("process version marker", re.compile(rf"\b{legacy_iteration_label}\b", re.IGNORECASE)),
        ("process package marker", re.compile(versioned_motif_package, re.IGNORECASE)),
        ("process report marker", re.compile(versioned_report_dir, re.IGNORECASE)),
        ("Chinese character", re.compile(r"[\u4e00-\u9fff]")),
    ]

    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if not is_release_path(rel, tracked):
            continue
        rel_text = rel.as_posix()
        for label, pattern in path_patterns:
            if pattern.search(rel_text):
                blocked_paths.append(f"{rel_text}: {label}")
        if not path.is_file() or path.resolve() == this_file or path.suffix.lower() not in text_suffixes:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(lines, start=1):
            for label, pattern in text_patterns:
                if pattern.search(line):
                    blocked_text.append((rel, lineno, label))

    if blocked_paths or blocked_text:
        samples = [f"- {item}" for item in blocked_paths[:10]]
        samples.extend(f"- {path}:{lineno}: {label}" for path, lineno, label in blocked_text[:20])
        raise RuntimeError("release surface audit failed:\n" + "\n".join(samples))


def audit_anonymity(root: Path) -> None:
    """Fail on identifying content that should not be in the public release."""
    tracked = tracked_release_files(root)
    this_file = Path(__file__).resolve()
    text_suffixes = {"", ".cfg", ".csv", ".gitignore", ".ini", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
    blocked: list[tuple[Path, int, str]] = []

    local_user = Path.home().name
    home_prefix = "/" + "home" + "/"
    user_prefix = "/" + "Users" + "/"
    forbidden_patterns = [
        ("current absolute release path", re.compile(re.escape(str(root)))),
        ("unix home path", re.compile(re.escape(home_prefix) + r"[A-Za-z0-9._-]+")),
        ("macOS user path", re.compile(re.escape(user_prefix) + r"[A-Za-z0-9._-]+")),
        ("local infrastructure path", re.compile(r"/(?:root|scratch|gpfs|lustre|nfs|mnt|tmp)/[A-Za-z0-9._/-]+")),
        ("Windows absolute path", re.compile(r"[A-Za-z]:\\[A-Za-z0-9._\\/-]+")),
        ("email address", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
        ("experiment tracking identifier", re.compile(r"\bW(?:ANDB|andb)\b")),
        ("high-risk secret token", re.compile(r"\b(?:ghp_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16})\b")),
        ("private key header", re.compile(r"BEGIN (?:RSA |OPENSSH |)PRIVATE KEY")),
    ]
    if local_user:
        forbidden_patterns.append(("current local username", re.compile(rf"\b{re.escape(local_user)}\b")))

    for path in root.rglob("*"):
        rel = path.relative_to(root)
        if not is_release_path(rel, tracked):
            continue
        if not path.is_file() or path.resolve() == this_file:
            continue
        if path.suffix.lower() not in text_suffixes:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(lines, start=1):
            for label, pattern in forbidden_patterns:
                if pattern.search(line):
                    blocked.append((rel, lineno, label))

    if blocked:
        sample = "\n".join(f"- {path}:{lineno}: {label}" for path, lineno, label in blocked[:20])
        raise RuntimeError(f"anonymity audit failed:\n{sample}")


def audit_artifact_consistency(root: Path) -> None:
    from gptok2_evaluation.cora_full_graph_eval import MODES as CORA_MODES
    from gptok2_evaluation.representative_eval import MODES as REPRESENTATIVE_MODES
    from gptok2_tokenizer.cli import MODES as TOKENIZER_MODES

    mode_sources = {
        "gptok2_tokenizer.cli": set(TOKENIZER_MODES),
        "gptok2_evaluation.representative_eval": set(REPRESENTATIVE_MODES),
        "gptok2_evaluation.cora_full_graph_eval": set(CORA_MODES),
    }
    mismatches = {name: sorted(modes) for name, modes in mode_sources.items() if modes != EXPECTED_MODES}
    if mismatches:
        raise RuntimeError(f"mode definitions do not match final release modes: {mismatches}")

    summary = read_csv(root / "reports/representative/summary_all.csv")
    test_rows = read_csv(root / "reports/representative/test_comparison.csv")
    cora_rows = read_csv(root / "reports/representative/cora_full_graph_summary.csv")
    status_rows = read_csv(root / "reports/representative/dataset_status.csv")

    report_modes = {row["mode"] for row in summary if row.get("mode") and row.get("mode") != "dataset"}
    test_modes = {row["mode"] for row in test_rows if row.get("mode")}
    cora_modes = {row["mode"] for row in cora_rows if row.get("mode")}
    report_datasets = {row["dataset"] for row in summary if row.get("dataset")}
    status_datasets = {row["dataset"] for row in status_rows if row.get("dataset")}
    if report_modes != EXPECTED_MODES or test_modes != EXPECTED_MODES or cora_modes != EXPECTED_MODES:
        raise RuntimeError(
            "report modes do not match final release modes: "
            f"summary={sorted(report_modes)}, test={sorted(test_modes)}, cora={sorted(cora_modes)}"
        )
    if report_datasets != EXPECTED_DATASETS or status_datasets != EXPECTED_DATASETS:
        raise RuntimeError(
            "report datasets do not match final release datasets: "
            f"summary={sorted(report_datasets)}, status={sorted(status_datasets)}"
        )


def main() -> None:
    audit_tree_hygiene(ROOT)
    audit_release_surface(ROOT)
    importlib.import_module("gptok2_tokenizer")
    importlib.import_module("gptok2_evaluation")
    importlib.import_module("rps_gtok_consumption")

    for rel in [
        "reports/representative/summary_all.csv",
        "reports/representative/test_comparison.csv",
        "reports/representative/dataset_status.csv",
        "reports/representative/cora_full_graph_summary.csv",
    ]:
        assert_exists(ROOT / rel)

    summary = read_csv(ROOT / "reports/representative/summary_all.csv")
    if not summary:
        raise RuntimeError("summary_all.csv is empty")
    audit_artifact_consistency(ROOT)
    audit_tree_hygiene(ROOT)
    audit_release_surface(ROOT)
    audit_anonymity(ROOT)
    print(f"loaded {len(summary)} rows from representative summary")
    print("package imports, release artifacts, consumer code, and anonymity audit look consistent")


if __name__ == "__main__":
    main()
