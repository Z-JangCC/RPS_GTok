"""Training and evaluation loop for RPS-GTok Transformer consumers."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from rps_gtok_consumption.data import SequenceVocab, TokenExample, TokenizedGraphDataset, TOKEN_FEATURE_KEYS, collate_tokenized_graphs
from rps_gtok_consumption.model import build_model, parameter_counts


@dataclass
class TrainConfig:
    max_len: int = 256
    vocab_size: int = 100000
    batch_size: int = 32
    epochs: int = 40
    lr: float = 0.0003
    weight_decay: float = 0.01
    patience: int = 8
    seed: int = 2026
    task_type: str = "classification"
    model: dict[str, Any] | None = None
    device: str = "auto"


@dataclass
class EvaluationResult:
    split: str
    loss: float
    accuracy: float | None = None
    macro_f1: float | None = None
    roc_auc: float | None = None
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def train_model(
    splits: dict[str, list[TokenExample]],
    config: TrainConfig,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    task_type = str(config.task_type)
    train_rows = splits["train"]
    val_rows = splits.get("val") or splits["train"]
    test_rows = splits.get("test") or val_rows
    vocab = SequenceVocab([ex.tokens for ex in train_rows], max_size=config.vocab_size)
    num_outputs = infer_num_outputs(train_rows, task_type)
    target_mean, target_std = regression_stats(train_rows) if task_type == "regression" else (0.0, 1.0)
    train_ds = TokenizedGraphDataset(train_rows, vocab, config.max_len, task_type, target_mean, target_std)
    val_ds = TokenizedGraphDataset(val_rows, vocab, config.max_len, task_type, target_mean, target_std)
    test_ds = TokenizedGraphDataset(test_rows, vocab, config.max_len, task_type, target_mean, target_std)
    loader_args = {"batch_size": int(config.batch_size), "collate_fn": collate_tokenized_graphs}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    model = build_model(len(vocab), num_outputs, config.max_len, config.model or {}).to(device)
    criterion = loss_fn(task_type)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.lr), weight_decay=float(config.weight_decay))
    best_state = None
    best_score = -float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    start = time.time()
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = forward_batch(model, batch, device)
            target = batch["target"].to(device)
            loss = compute_loss(logits, target, criterion, task_type)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * int(target.shape[0])
            total_seen += int(target.shape[0])
        val = evaluate(model, val_loader, task_type, device, split="val", target_mean=target_mean, target_std=target_std)
        score = selection_score(val, task_type)
        row = {"epoch": epoch, "train_loss": total_loss / max(1, total_seen), **val.to_dict()}
        history.append(row)
        if score > best_score:
            best_score = score
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(config.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_result = evaluate(model, val_loader, task_type, device, split="val", target_mean=target_mean, target_std=target_std)
    test_result = evaluate(model, test_loader, task_type, device, split="test", target_mean=target_mean, target_std=target_std)
    result = {
        "task_type": task_type,
        "epochs_ran": len(history),
        "elapsed_sec": time.time() - start,
        "vocab_size": len(vocab),
        "num_outputs": num_outputs,
        "val": val_result.to_dict(),
        "test": test_result.to_dict(),
        **parameter_counts(model),
    }
    if out_dir is not None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        (path / "history.json").write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
        (path / "vocab.json").write_text(json.dumps(vocab.to_dict(), indent=2), encoding="utf-8")
        torch.save({"model": model.state_dict(), "config": asdict(config), "metrics": result}, path / "model.pt")
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    task_type: str,
    device: torch.device,
    split: str,
    target_mean: float = 0.0,
    target_std: float = 1.0,
) -> EvaluationResult:
    model.eval()
    criterion = loss_fn(task_type)
    losses: list[float] = []
    y_true: list[Any] = []
    y_score: list[Any] = []
    for batch in loader:
        logits = forward_batch(model, batch, device)
        target = batch["target"].to(device)
        loss = compute_loss(logits, target, criterion, task_type)
        losses.append(float(loss.detach().cpu()))
        if task_type == "regression":
            pred = logits.squeeze(-1).detach().cpu() * float(target_std) + float(target_mean)
            truth = batch["target_raw"].detach().cpu()
            y_score.extend(pred.tolist())
            y_true.extend(truth.tolist())
        elif task_type == "multilabel":
            y_score.extend(torch.sigmoid(logits).detach().cpu().tolist())
            y_true.extend(target.detach().cpu().tolist())
        else:
            y_score.extend(torch.softmax(logits, dim=-1).detach().cpu().tolist())
            y_true.extend(target.detach().cpu().tolist())
    mean_loss = float(np.mean(losses)) if losses else 0.0
    if task_type == "regression":
        pred = np.asarray(y_score, dtype=float)
        truth = np.asarray(y_true, dtype=float)
        mse = mean_squared_error(truth, pred) if len(truth) else 0.0
        return EvaluationResult(split, mean_loss, mae=mean_absolute_error(truth, pred), rmse=float(np.sqrt(mse)), r2=r2_score(truth, pred) if len(truth) > 1 else 0.0)
    if task_type == "multilabel":
        truth = np.asarray(y_true, dtype=float)
        score = np.asarray(y_score, dtype=float)
        roc = safe_multilabel_roc_auc(truth, score)
        return EvaluationResult(split, mean_loss, roc_auc=roc)
    pred_labels = np.asarray(y_score, dtype=float).argmax(axis=1) if y_score else np.asarray([], dtype=int)
    truth = np.asarray(y_true, dtype=int)
    return EvaluationResult(
        split,
        mean_loss,
        accuracy=accuracy_score(truth, pred_labels) if len(truth) else 0.0,
        macro_f1=f1_score(truth, pred_labels, average="macro", zero_division=0) if len(truth) else 0.0,
    )


def forward_batch(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    features = {key: batch[key].to(device) for key in TOKEN_FEATURE_KEYS if key in batch}
    return model(batch["ids"].to(device), **features)


def compute_loss(logits: torch.Tensor, target: torch.Tensor, criterion: nn.Module, task_type: str) -> torch.Tensor:
    if task_type == "regression":
        return criterion(logits.squeeze(-1), target.float())
    if task_type == "multilabel":
        return criterion(logits, target.float())
    return criterion(logits, target.long())


def loss_fn(task_type: str) -> nn.Module:
    if task_type == "regression":
        return nn.MSELoss()
    if task_type == "multilabel":
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def infer_num_outputs(rows: list[TokenExample], task_type: str) -> int:
    if task_type == "regression":
        return 1
    if task_type == "multilabel":
        return len(rows[0].y)  # type: ignore[arg-type]
    return max(int(ex.y) for ex in rows) + 1  # type: ignore[arg-type]


def regression_stats(rows: list[TokenExample]) -> tuple[float, float]:
    values = np.asarray([float(ex.y) for ex in rows], dtype=float)
    return float(values.mean()), float(values.std() if values.std() > 0 else 1.0)


def selection_score(result: EvaluationResult, task_type: str) -> float:
    if task_type == "regression":
        return -float(result.mae if result.mae is not None else result.loss)
    if task_type == "multilabel":
        return float(result.roc_auc if result.roc_auc is not None else -result.loss)
    return float(result.macro_f1 if result.macro_f1 is not None else -result.loss)


def safe_multilabel_roc_auc(truth: np.ndarray, score: np.ndarray) -> float:
    values = []
    for idx in range(truth.shape[1] if truth.ndim == 2 else 0):
        if len(np.unique(truth[:, idx])) < 2:
            continue
        values.append(roc_auc_score(truth[:, idx], score[:, idx]))
    return float(np.mean(values)) if values else 0.0


def resolve_device(value: str) -> torch.device:
    if str(value) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(str(value))


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
