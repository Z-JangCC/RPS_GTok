"""Audit identifier-only versus self-contained RPS token streams.

This directly addresses the reviewer objection that identifiers alone do not
carry the expansion payload.  It does not claim unconstrained generation is
valid; it measures the explicit grammar/payload control stream and verifies
that it round-trips to the exact emitted token instances.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from gptok2.data.io import load_records
from gptok2_tokenizer import GPTok2Tokenizer
from gptok2_tokenizer.payload import complete_stream_to_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", default="motif_hybrid")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    tokenizer = GPTok2Tokenizer.load(args.artifact)
    records = load_records(args.data)
    if args.limit > 0:
        records = records[: args.limit]
    rows = []
    for record in records:
        encoded = tokenizer.encode(record, mode=args.mode)
        stream = encoded.complete_stream_tokens
        recovered_tokens, recovered_payload = complete_stream_to_tokens(stream)
        rows.append({
            "graph_id": record.graph_id,
            "identifier_tokens": len(encoded.identifier_tokens),
            "complete_stream_tokens": len(stream),
            "identifier_bytes": encoded.identifier_bytes,
            "payload_bytes": encoded.payload_bytes,
            "complete_bytes": encoded.complete_token_bytes,
            "raw_token_roundtrip": float(recovered_tokens == encoded.tokens),
            "payload_roundtrip": float(recovered_payload == encoded.payload),
            "grammar_valid": float(encoded.validate_complete_stream().get("valid", False)),
        })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "graphs": len(rows),
        "identifier_tokens_mean": sum(r["identifier_tokens"] for r in rows) / max(1, len(rows)),
        "complete_stream_tokens_mean": sum(r["complete_stream_tokens"] for r in rows) / max(1, len(rows)),
        "raw_token_roundtrip": sum(r["raw_token_roundtrip"] for r in rows) / max(1, len(rows)),
        "payload_roundtrip": sum(r["payload_roundtrip"] for r in rows) / max(1, len(rows)),
        "grammar_valid": sum(r["grammar_valid"] for r in rows) / max(1, len(rows)),
        "mode": args.mode,
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
