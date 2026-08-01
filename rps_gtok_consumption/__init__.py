"""Downstream Transformer consumers for RPS-GTok token sequences."""

from rps_gtok_consumption.data import (
    SequenceVocab,
    TokenExample,
    TokenizedGraphDataset,
    collate_tokenized_graphs,
    examples_from_records,
    load_examples,
    save_examples,
    split_examples,
)
from rps_gtok_consumption.model import (
    FullEmbedTokenAdapter,
    PlainTokenAdapter,
    TransformerBackbone,
    TransformerGraphClassifier,
    build_model,
    parameter_counts,
)
from rps_gtok_consumption.experiment import run_experiment_config
from rps_gtok_consumption.training import (
    EvaluationResult,
    TrainConfig,
    evaluate,
    train_model,
)
from rps_gtok_consumption.views import TokenBPE, TokenViewBuilder

__all__ = [
    "EvaluationResult",
    "FullEmbedTokenAdapter",
    "PlainTokenAdapter",
    "SequenceVocab",
    "TokenExample",
    "TokenBPE",
    "TokenViewBuilder",
    "TokenizedGraphDataset",
    "TrainConfig",
    "TransformerBackbone",
    "TransformerGraphClassifier",
    "build_model",
    "collate_tokenized_graphs",
    "evaluate",
    "examples_from_records",
    "load_examples",
    "parameter_counts",
    "run_experiment_config",
    "save_examples",
    "split_examples",
    "train_model",
]
