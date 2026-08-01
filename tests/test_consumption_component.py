from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gptok2.data.synthetic import generate_synthetic_graphs, split_records  # noqa: E402
from gptok2_tokenizer import GPTok2Tokenizer  # noqa: E402
from rps_gtok_consumption.data import SequenceVocab, TokenizedGraphDataset, collate_tokenized_graphs, examples_from_records  # noqa: E402
from rps_gtok_consumption.model import FullEmbedTokenAdapter, PlainTokenAdapter, build_model  # noqa: E402
from rps_gtok_consumption.training import TrainConfig, train_model  # noqa: E402


def test_full_embed_and_plain_share_backbone_shape() -> None:
    model_cfg = {"dim": 32, "layers": 1, "heads": 4, "dropout": 0.0}
    plain = build_model(32, 3, 64, {**model_cfg, "adapter": "plain"})
    full = build_model(32, 3, 64, {**model_cfg, "adapter": "full_embed"})
    assert isinstance(plain.adapter, PlainTokenAdapter)
    assert isinstance(full.adapter, FullEmbedTokenAdapter)
    assert type(plain.backbone) is type(full.backbone)
    assert sum(p.numel() for p in plain.backbone.parameters()) == sum(p.numel() for p in full.backbone.parameters())


def test_consumer_forward_on_tokenized_graphs() -> None:
    cfg = {
        "run": {"seed": 11},
        "data": {"synthetic": {"num_graphs": 12, "num_nodes_min": 6, "num_nodes_max": 12, "families": ["cycle", "star"]}},
    }
    records = generate_synthetic_graphs(cfg, seed=11)
    tokenizer = GPTok2Tokenizer().fit(records)
    examples = examples_from_records(records[:4], tokenizer, mode="motif_hybrid", split="train", dataset="smoke")
    vocab = SequenceVocab([ex.tokens for ex in examples])
    ds = TokenizedGraphDataset(examples, vocab, max_len=96)
    batch = collate_tokenized_graphs([ds[0], ds[1]])
    model = build_model(len(vocab), 8, 96, {"adapter": "full_embed", "dim": 32, "layers": 1, "heads": 4, "dropout": 0.0})
    logits = model(batch["ids"], **{k: v for k, v in batch.items() if k != "ids"})
    assert logits.shape == (2, 8)
    assert torch.isfinite(logits).all()


def test_training_loop_runs_one_epoch(tmp_path) -> None:
    cfg = {
        "run": {"seed": 13},
        "data": {
            "synthetic": {"num_graphs": 24, "num_nodes_min": 6, "num_nodes_max": 14, "families": ["cycle", "star", "tree"]},
            "split": {"train_ratio": 0.7, "val_ratio": 0.15},
        },
    }
    records = split_records(generate_synthetic_graphs(cfg, seed=13), cfg)
    tokenizer = GPTok2Tokenizer().fit(records["train"])
    splits = {split: examples_from_records(rows, tokenizer, mode="motif_hybrid", split=split, dataset="smoke") for split, rows in records.items()}
    result = train_model(
        splits,
        TrainConfig(
            max_len=96,
            batch_size=4,
            epochs=1,
            patience=1,
            task_type="classification",
            model={"adapter": "full_embed", "dim": 32, "layers": 1, "heads": 4, "dropout": 0.0},
            device="cpu",
        ),
        out_dir=tmp_path,
    )
    assert result["epochs_ran"] == 1
    assert result["test"]["split"] == "test"
    assert (tmp_path / "metrics.json").exists()
