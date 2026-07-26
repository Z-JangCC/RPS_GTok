"""GraphRecord serialization."""

from __future__ import annotations

from pathlib import Path

from gptok2.data.schema import record_from_dict, record_to_dict
from gptok2.utils.io import read_jsonl, write_jsonl


def save_records(path, records):
    write_jsonl(path, [record_to_dict(r) for r in records])


def load_records(path):
    return [record_from_dict(r) for r in read_jsonl(Path(path))]

