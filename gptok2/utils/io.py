"""I/O helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import yaml


def load_config(path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(config: dict) -> Path:
    return ensure_dir(Path(config.get("run", {}).get("output_dir", "runs/gptok")) / config.get("run", {}).get("name", "gptok_run"))


def write_jsonl(path, rows):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_json(path, obj):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_pickle(path, obj):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with Path(path).open("rb") as f:
        return pickle.load(f)

