"""Identifier/payload separation and deterministic footprint accounting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GraphTokenInstance:
    identifier: str
    payload: tuple[str, ...]
    raw_token: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "payload": list(self.payload),
            "raw_token": self.raw_token,
        }


def split_token(token: str) -> GraphTokenInstance:
    text = str(token)
    if "(" not in text or not text.endswith(")"):
        return GraphTokenInstance(text, (), text)
    op, raw = text.split("(", 1)
    payload = raw[:-1]
    fields = tuple(x for x in payload.split(",") if x != "")
    identifier = _identifier_for(op, fields)
    return GraphTokenInstance(identifier, fields, text)


def _identifier_for(op: str, fields: tuple[str, ...]) -> str:
    if op in {"EMIT", "EMIT_CODE", "INTERFACE", "INTERFACE_CODE"} and fields:
        return f"{op}:{fields[0]}"
    return op


def token_instances(tokens: list[str]) -> list[GraphTokenInstance]:
    return [split_token(token) for token in tokens]


def footprint_stats(tokens: list[str], encoding: str = "utf-8") -> dict[str, float]:
    instances = token_instances(tokens)
    id_bytes = sum(len(item.identifier.encode(encoding)) for item in instances)
    payload_bytes = sum(
        len(
            json.dumps(
                list(item.payload),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode(encoding)
        )
        for item in instances
    )
    complete_bytes = sum(
        len(
            json.dumps(
                item.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode(encoding)
        )
        for item in instances
    )
    return {
        "identifier_token_count": float(len(instances)),
        "identifier_bytes": float(id_bytes),
        "payload_bytes": float(payload_bytes),
        "complete_token_bytes": float(complete_bytes),
        "payload_fraction": payload_bytes / max(1.0, complete_bytes),
    }


def serialize_complete_stream(
    tokens: list[str],
    graph_payload: dict[str, Any] | None = None,
    encoding: str = "utf-8",
) -> bytes:
    """Serialize identifiers, token-local payloads and graph payload deterministically."""
    body = {
        "tokens": [item.to_dict() for item in token_instances(tokens)],
        "graph_payload": graph_payload or {},
    }
    return json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode(encoding)


def deserialize_complete_stream(blob: bytes, encoding: str = "utf-8") -> dict[str, Any]:
    return json.loads(blob.decode(encoding))


def complete_stream_tokens(
    tokens: list[str],
    graph_payload: dict[str, Any] | None = None,
) -> list[str]:
    """Return a deterministic self-contained token stream.

    The ordinary consumer stream exposes only identifiers.  This projection
    makes the missing generation contract explicit: every identifier is
    followed by its ordered payload fields, and the graph-level schema payload
    is emitted as a canonical JSON token.  It is intended for the constrained
    generation/footprint control, not as a claim that an unconstrained AR model
    can invent valid ports without a grammar.
    """
    out = ["<GRAPH_BOS>"]
    for index, item in enumerate(token_instances(tokens)):
        out.append(f"<TOKEN_{index}>")
        out.append(f"ID={item.identifier}")
        # Keeping the raw surface form makes this a reversible control stream
        # even for identifiers that intentionally collapse an operation and a
        # code (e.g. ``EMIT_CODE:17``).
        out.append("RAW=" + json.dumps(item.raw_token, ensure_ascii=False, separators=(",", ":")))
        out.extend(f"ARG_{arg_index}={value}" for arg_index, value in enumerate(item.payload))
        out.append("<TOKEN_END>")
    payload = json.dumps(
        graph_payload or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    out.extend(["<GRAPH_PAYLOAD_BEGIN>", f"GRAPH_PAYLOAD={payload}", "<GRAPH_PAYLOAD_END>", "<GRAPH_EOS>"])
    return out


def validate_complete_stream_tokens(stream: list[str]) -> dict[str, Any]:
    """Validate the finite-state structure of a self-contained stream."""
    if not stream or stream[0] != "<GRAPH_BOS>" or stream[-1] != "<GRAPH_EOS>":
        return {"valid": False, "reason": "missing_graph_boundaries"}
    payload_begin = [idx for idx, token in enumerate(stream) if token == "<GRAPH_PAYLOAD_BEGIN>"]
    payload_end = [idx for idx, token in enumerate(stream) if token == "<GRAPH_PAYLOAD_END>"]
    if len(payload_begin) != 1 or len(payload_end) != 1 or payload_begin[0] >= payload_end[0]:
        return {"valid": False, "reason": "invalid_graph_payload_boundaries"}
    for idx, token in enumerate(stream[1:payload_begin[0]], start=1):
        if token.startswith("<TOKEN_") and token != "<TOKEN_END>":
            if idx + 1 >= len(stream) or not stream[idx + 1].startswith("ID="):
                return {"valid": False, "reason": "token_without_identifier", "index": idx}
    payload_tokens = [token for token in stream[payload_begin[0] + 1 : payload_end[0]] if token.startswith("GRAPH_PAYLOAD=")]
    if len(payload_tokens) != 1:
        return {"valid": False, "reason": "missing_graph_payload"}
    try:
        json.loads(payload_tokens[0].split("=", 1)[1])
    except (ValueError, json.JSONDecodeError):
        return {"valid": False, "reason": "invalid_graph_payload_json"}
    return {"valid": True, "token_count": len(stream), "payload_included": True}


def complete_stream_to_tokens(stream: list[str]) -> tuple[list[str], dict[str, Any]]:
    """Recover raw token instances and graph payload from a valid stream."""
    audit = validate_complete_stream_tokens(stream)
    if not audit.get("valid"):
        raise ValueError(f"invalid complete stream: {audit}")
    raw_tokens: list[str] = []
    for token in stream:
        if token.startswith("RAW="):
            raw_tokens.append(json.loads(token.split("=", 1)[1]))
    begin = stream.index("<GRAPH_PAYLOAD_BEGIN>")
    end = stream.index("<GRAPH_PAYLOAD_END>")
    payload_token = next(token for token in stream[begin + 1 : end] if token.startswith("GRAPH_PAYLOAD="))
    payload = json.loads(payload_token.split("=", 1)[1])
    return raw_tokens, payload
