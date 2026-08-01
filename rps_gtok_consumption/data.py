"""Token datasets and RPS-GTok token-feature extraction for consumer models."""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from gptok2.data.schema import GraphRecord, edge_set
from gptok2_tokenizer import GPTok2Tokenizer


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"

TOKEN_FEATURE_KEYS = [
    "kind",
    "segment",
    "token_id_bin",
    "span_bin",
    "arity_bin",
    "ref_count_bin",
    "ref_role",
]

KIND_TO_ID = {
    "pad": 0,
    "special": 1,
    "rps_shape": 2,
    "rps_interface": 3,
    "rps_macro": 4,
    "rps_patch": 5,
    "rps_primitive": 6,
    "struct": 7,
    "edge": 8,
    "node": 9,
    "attr": 10,
    "other": 11,
}
SEGMENT_TO_ID = {"pad": 0, "program": 1, "struct": 2, "edge": 3, "other": 4}
REF_ROLE_TO_ID = {
    "pad": 0,
    "none": 1,
    "numeric": 2,
    "edge": 3,
    "shape": 4,
    "interface": 5,
    "patch": 6,
    "other": 7,
}


@dataclass
class TokenExample:
    graph_id: str
    dataset: str
    split: str
    view: str
    tokens: list[str]
    y: int | float | list[float]
    task_type: str = "classification"
    num_nodes: int = 0
    num_edges: int = 0
    node_refs: list[list[str]] = field(default_factory=list)
    edge_pairs: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "TokenExample":
        return cls(
            graph_id=str(row["graph_id"]),
            dataset=str(row.get("dataset", "")),
            split=str(row.get("split", "")),
            view=str(row.get("view", "")),
            tokens=[str(tok) for tok in row["tokens"]],
            y=row["y"],
            task_type=str(row.get("task_type", "classification")),
            num_nodes=int(row.get("num_nodes", 0)),
            num_edges=int(row.get("num_edges", 0)),
            node_refs=[[str(x) for x in xs] for xs in row.get("node_refs", [])],
            edge_pairs=[[str(x) for x in xs] for xs in row.get("edge_pairs", [])],
        )


class SequenceVocab:
    """Vocabulary for graph-token sequences.

    The first four IDs are fixed so model checkpoints and prepared datasets keep
    stable special-token semantics.
    """

    def __init__(self, sequences: Iterable[list[str]] = (), max_size: int = 100000):
        counts = Counter(tok for seq in sequences for tok in seq)
        base = [PAD, BOS, EOS, UNK]
        vocab = [tok for tok, _ in counts.most_common(max(0, int(max_size) - len(base))) if tok not in base]
        self.itos = base + vocab
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str], max_len: int) -> torch.LongTensor:
        ids = [self.stoi[BOS], *[self.stoi.get(tok, self.stoi[UNK]) for tok in tokens], self.stoi[EOS]]
        ids = ids[: int(max_len)]
        if len(ids) < int(max_len):
            ids.extend([self.stoi[PAD]] * (int(max_len) - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def to_dict(self) -> dict[str, Any]:
        return {"itos": list(self.itos)}

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SequenceVocab":
        obj = cls([])
        obj.itos = [str(tok) for tok in row["itos"]]
        obj.stoi = {tok: idx for idx, tok in enumerate(obj.itos)}
        return obj


class TokenizedGraphDataset(Dataset):
    def __init__(
        self,
        examples: list[TokenExample],
        vocab: SequenceVocab,
        max_len: int,
        task_type: str = "classification",
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ):
        self.examples = list(examples)
        self.vocab = vocab
        self.max_len = int(max_len)
        self.task_type = str(task_type)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std) if float(target_std) > 0 else 1.0

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        ids = self.vocab.encode(ex.tokens, self.max_len)
        item: dict[str, torch.Tensor] = {
            "ids": ids,
            "length": torch.tensor(min(len(ex.tokens) + 2, self.max_len), dtype=torch.long),
            "structural_bias": structural_bias_tensor(ex.node_refs, ex.edge_pairs, self.max_len),
        }
        item.update(token_feature_vectors(ex.tokens, self.max_len))
        if self.task_type == "regression":
            y = float(ex.y)  # type: ignore[arg-type]
            item["target"] = torch.tensor((y - self.target_mean) / self.target_std, dtype=torch.float32)
            item["target_raw"] = torch.tensor(y, dtype=torch.float32)
        elif self.task_type == "multilabel":
            values = torch.tensor([float(v) for v in ex.y], dtype=torch.float32)  # type: ignore[arg-type]
            item["target"] = values
            item["target_raw"] = values.clone()
        else:
            item["target"] = torch.tensor(int(ex.y), dtype=torch.long)  # type: ignore[arg-type]
            item["target_raw"] = torch.tensor(float(ex.y), dtype=torch.float32)  # type: ignore[arg-type]
        return item


def collate_tokenized_graphs(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = set().union(*(item.keys() for item in batch))
    out: dict[str, torch.Tensor] = {}
    for key in sorted(keys):
        values = [item[key] for item in batch if key in item]
        if key == "structural_bias":
            width = max(v.shape[-1] for v in values)
            tensor = torch.zeros((len(batch), width, width), dtype=torch.float32)
            for idx, value in enumerate(values):
                tensor[idx, : value.shape[0], : value.shape[1]] = value
            out[key] = tensor
        else:
            out[key] = torch.stack(values)
    return out


def examples_from_records(
    records: Iterable[GraphRecord],
    tokenizer: GPTok2Tokenizer,
    mode: str = "motif_hybrid",
    split: str = "train",
    dataset: str = "graph_records",
    task_type: str = "classification",
) -> list[TokenExample]:
    examples: list[TokenExample] = []
    for record in records:
        encoded = tokenizer.encode(record, mode=mode)
        y = _record_target(record, task_type)
        node_refs = [infer_token_node_refs(tok) for tok in encoded.tokens]
        examples.append(
            TokenExample(
                graph_id=record.graph_id,
                dataset=dataset,
                split=split,
                view=mode,
                tokens=encoded.tokens,
                y=y,
                task_type=task_type,
                num_nodes=int(record.num_nodes),
                num_edges=len(edge_set(record)),
                node_refs=node_refs,
                edge_pairs=[[str(u), str(v)] for u, v in sorted(edge_set(record))],
            )
        )
    return examples


def split_examples(examples: list[TokenExample], seed: int = 2026, train_ratio: float = 0.7, val_ratio: float = 0.1) -> dict[str, list[TokenExample]]:
    rows = list(examples)
    random.Random(int(seed)).shuffle(rows)
    n_train = int(round(float(train_ratio) * len(rows)))
    n_val = int(round(float(val_ratio) * len(rows)))
    splits = {
        "train": rows[:n_train],
        "val": rows[n_train : n_train + n_val],
        "test": rows[n_train + n_val :],
    }
    for split, items in splits.items():
        for item in items:
            item.split = split
    return splits


def save_examples(path: str | Path, examples: Iterable[TokenExample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), sort_keys=True) + "\n")


def load_examples(path: str | Path) -> list[TokenExample]:
    rows: list[TokenExample] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(TokenExample.from_dict(json.loads(line)))
    return rows


def token_feature_vectors(tokens: list[str], max_len: int) -> dict[str, torch.LongTensor]:
    features = {key: torch.zeros(int(max_len), dtype=torch.long) for key in TOKEN_FEATURE_KEYS}
    full = [BOS, *tokens, EOS][: int(max_len)]
    for idx, tok in enumerate(full):
        meta = token_metadata(tok)
        features["kind"][idx] = KIND_TO_ID.get(meta["kind"], KIND_TO_ID["other"])
        features["segment"][idx] = SEGMENT_TO_ID.get(meta["segment"], SEGMENT_TO_ID["other"])
        features["token_id_bin"][idx] = bin_int(meta["token_id"])
        features["span_bin"][idx] = bin_int(meta["span"])
        features["arity_bin"][idx] = bin_int(meta["arity"])
        features["ref_count_bin"][idx] = bin_int(meta["ref_count"])
        features["ref_role"][idx] = REF_ROLE_TO_ID.get(meta["ref_role"], REF_ROLE_TO_ID["other"])
    return features


def token_metadata(token: str) -> dict[str, Any]:
    if token in {PAD, BOS, EOS, UNK}:
        return {"kind": "special", "segment": "other", "token_id": 0, "span": 0, "arity": 0, "ref_count": 0, "ref_role": "none"}
    upper = token.upper()
    nums = [int(x) for x in re.findall(r"-?\d+", token)]
    kind = "other"
    segment = "program"
    ref_role = "none"
    if upper.startswith(("MOTIF", "MACRO", "BPE")) or "MOTIF" in upper or "MACRO" in upper:
        kind = "rps_macro"
        ref_role = "shape"
    elif upper.startswith(("CODE", "SHAPE", "INTERFACE")) or "CODE" in upper:
        kind = "rps_shape"
        ref_role = "interface" if "INTERFACE" in upper else "shape"
    elif upper.startswith(("PATCH", "BEGIN", "END", "MERGE", "EMIT")):
        kind = "rps_patch"
        ref_role = "patch"
    elif upper.startswith(("ADD_NODE", "ADD_EDGE", "NODE", "EDGE")):
        kind = "rps_primitive"
        ref_role = "edge" if "EDGE" in upper else "numeric"
    elif upper.startswith(("STRUCT", "DEG", "TRI", "WL", "LEN_BIN")):
        kind = "struct"
        segment = "struct"
    elif upper.startswith("E(") or upper.startswith("EDGE"):
        kind = "edge"
        segment = "edge"
        ref_role = "edge"
    arity = token.count("|") + token.count(",")
    return {
        "kind": kind,
        "segment": segment,
        "token_id": abs(nums[0]) if nums else 0,
        "span": max(nums) - min(nums) if len(nums) >= 2 else 0,
        "arity": arity,
        "ref_count": len(set(nums)),
        "ref_role": ref_role if nums else "none",
    }


def infer_token_node_refs(token: str) -> list[str]:
    return [str(int(x)) for x in re.findall(r"-?\d+", token)[:16]]


def structural_bias_tensor(node_refs: list[list[str]], edge_pairs: list[list[str]], max_len: int) -> torch.Tensor:
    refs = [[]] + [list(map(str, xs)) for xs in node_refs] + [[]]
    refs = refs[: int(max_len)]
    width = int(max_len)
    bias = torch.zeros((width, width), dtype=torch.float32)
    edge_set_local = {(str(u), str(v)) for u, v in edge_pairs}
    edge_set_local.update((v, u) for u, v in list(edge_set_local))
    ref_sets = [set(xs) for xs in refs]
    for i in range(len(refs)):
        bias[i, i] = 1.0
        if i + 1 < len(refs):
            bias[i, i + 1] = 1.0
            bias[i + 1, i] = 1.0
        for j in range(i + 1, len(refs)):
            if ref_sets[i] & ref_sets[j] or any((u, v) in edge_set_local for u in ref_sets[i] for v in ref_sets[j]):
                bias[i, j] = 1.0
                bias[j, i] = 1.0
    return bias


def bin_int(value: int | float) -> int:
    value = int(abs(value))
    if value <= 0:
        return 0
    return min(10, int(math.floor(math.log2(value))) + 1)


def _record_target(record: GraphRecord, task_type: str) -> int | float | list[float]:
    if record.y is None:
        raise ValueError(f"record {record.graph_id} has no target")
    if task_type == "multilabel":
        return [float(x) for x in record.y.detach().cpu().flatten().tolist()]
    if task_type == "regression":
        return float(record.y.detach().cpu().flatten()[0].item())
    return int(record.y.detach().cpu().flatten()[0].item())
