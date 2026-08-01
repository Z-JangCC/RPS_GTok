"""Full-Embed token adapters and Transformer graph consumers."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rps_gtok_consumption.data import KIND_TO_ID, REF_ROLE_TO_ID, SEGMENT_TO_ID


class PlainTokenAdapter(nn.Module):
    """Token-ID and position embedding adapter used by sequence baselines."""

    def __init__(self, vocab_size: int, max_len: int, dim: int, dropout: float = 0.1):
        super().__init__()
        self.token = nn.Embedding(int(vocab_size), int(dim), padding_idx=0)
        self.position = nn.Embedding(int(max_len), int(dim))
        self.norm = nn.LayerNorm(int(dim))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, ids: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = ids.shape
        pos = torch.arange(seqlen, device=ids.device).unsqueeze(0).expand(bsz, seqlen)
        return self.dropout(self.norm(self.token(ids) + self.position(pos)))


class FullEmbedTokenAdapter(PlainTokenAdapter):
    """RPS-GTok Full-Embed adapter before the shared Transformer backbone."""

    def __init__(self, vocab_size: int, max_len: int, dim: int, dropout: float = 0.1, use_feature_gate: bool = True):
        super().__init__(vocab_size, max_len, dim, dropout)
        self.kind = nn.Embedding(len(KIND_TO_ID), int(dim), padding_idx=0)
        self.segment = nn.Embedding(len(SEGMENT_TO_ID), int(dim), padding_idx=0)
        self.token_id_bin = nn.Embedding(11, int(dim), padding_idx=0)
        self.span_bin = nn.Embedding(11, int(dim), padding_idx=0)
        self.arity_bin = nn.Embedding(11, int(dim), padding_idx=0)
        self.ref_count_bin = nn.Embedding(11, int(dim), padding_idx=0)
        self.ref_role = nn.Embedding(len(REF_ROLE_TO_ID), int(dim), padding_idx=0)
        self.use_feature_gate = bool(use_feature_gate)
        self.feature_gate = nn.Sequential(nn.Linear(int(dim), int(dim)), nn.Sigmoid())

    def forward(self, ids: torch.Tensor, **features: torch.Tensor) -> torch.Tensor:
        base = super().forward(ids)
        ctx = torch.zeros_like(base)
        for key in ["kind", "segment", "token_id_bin", "span_bin", "arity_bin", "ref_count_bin", "ref_role"]:
            if key in features:
                ctx = ctx + getattr(self, key)(features[key])
        if self.use_feature_gate:
            base = base * (1.0 + self.feature_gate(ctx))
        return self.dropout(self.norm(base + ctx))


class TransformerBackbone(nn.Module):
    """Shared Transformer encoder used for baseline and Full-Embed variants."""

    def __init__(self, dim: int, layers: int, heads: int, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            int(dim),
            int(heads),
            dim_feedforward=4 * int(dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, int(layers), enable_nested_tensor=False)
        self.norm = nn.LayerNorm(int(dim))

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(x, src_key_padding_mask=pad))


class AttentivePooling(nn.Module):
    def __init__(self, dim: int, heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.heads = max(1, int(heads))
        self.query = nn.Parameter(torch.empty(self.heads, int(dim)))
        self.key = nn.Linear(int(dim), int(dim))
        self.value = nn.Linear(int(dim), int(dim))
        self.dropout = nn.Dropout(float(dropout))
        nn.init.normal_(self.query, mean=0.0, std=float(dim) ** -0.5)

    def forward(self, h: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("bld,hd->blh", torch.tanh(self.key(h)), self.query) / (h.shape[-1] ** 0.5)
        scores = scores.masked_fill(pad.unsqueeze(-1), float("-inf"))
        weights = torch.softmax(scores, dim=1)
        values = self.value(h)
        return torch.einsum("blh,bld->bhd", self.dropout(weights), values).reshape(h.shape[0], -1)


class TransformerGraphClassifier(nn.Module):
    """Graph-level prediction head over tokenized graph programs."""

    def __init__(
        self,
        adapter: nn.Module,
        backbone: TransformerBackbone,
        num_outputs: int,
        dim: int,
        dropout: float = 0.1,
        attentive_pooling: bool = False,
        attn_pool_heads: int = 1,
    ):
        super().__init__()
        self.adapter = adapter
        self.backbone = backbone
        self.attentive_pooling = bool(attentive_pooling)
        pool_dim = int(dim) * 3
        self.attn_pool = nn.Identity()
        if self.attentive_pooling:
            self.attn_pool = AttentivePooling(int(dim), int(attn_pool_heads), float(dropout))
            pool_dim += int(dim) * max(1, int(attn_pool_heads))
        self.head = nn.Sequential(
            nn.Linear(pool_dim, int(dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(dim), int(num_outputs)),
        )

    def forward(self, ids: torch.Tensor, **features: torch.Tensor) -> torch.Tensor:
        pad = ids.eq(0)
        h = self.backbone(self.adapter(ids, **features), pad)
        valid = (~pad).float().unsqueeze(-1)
        mean_pool = (h * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        max_pool = h.masked_fill(pad.unsqueeze(-1), -1e4).max(1).values
        pooled = [h[:, 0], mean_pool, max_pool]
        if self.attentive_pooling:
            pooled.append(self.attn_pool(h, pad))
        return self.head(torch.cat(pooled, dim=-1))


def build_model(vocab_size: int, num_outputs: int, max_len: int, config: dict[str, Any]) -> TransformerGraphClassifier:
    dim = int(config.get("dim", 128))
    dropout = float(config.get("dropout", 0.1))
    adapter_type = str(config.get("adapter", config.get("model", "full_embed"))).lower()
    adapter_cls = PlainTokenAdapter if adapter_type in {"plain", "baseline", "token"} else FullEmbedTokenAdapter
    adapter = adapter_cls(
        vocab_size=int(vocab_size),
        max_len=int(max_len),
        dim=dim,
        dropout=dropout,
        **({"use_feature_gate": bool(config.get("use_feature_gate", True))} if adapter_cls is FullEmbedTokenAdapter else {}),
    )
    backbone = TransformerBackbone(
        dim=dim,
        layers=int(config.get("layers", 4)),
        heads=int(config.get("heads", 4)),
        dropout=dropout,
    )
    return TransformerGraphClassifier(
        adapter=adapter,
        backbone=backbone,
        num_outputs=int(num_outputs),
        dim=dim,
        dropout=dropout,
        attentive_pooling=bool(config.get("attentive_pooling", False)),
        attn_pool_heads=int(config.get("attn_pool_heads", 1)),
    )


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("adapter."))
    backbone = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone."))
    head = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and n.startswith("head."))
    return {
        "parameter_count": int(total),
        "adapter_parameter_count": int(adapter),
        "transformer_parameter_count": int(backbone),
        "head_parameter_count": int(head),
        "other_parameter_count": int(total - adapter - backbone - head),
    }
