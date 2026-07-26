"""Structure-grounded reversible motif macros for GPTok2 programs.

This module is deliberately independent from the public tokenizer API package.
It adds a higher-level reversible compression layer on top of GPTok2 program
tokens. The codec learns macro rules only from emitted prototype-code shapes,
so the rules are graph-structure aware while still expanding exactly back to
the original GPTok2 action tokens.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, unquote

from gptok2.vq.codebook import PrototypeVQCodebook
from gptok2_compact.codec import CompactCodec, CompactProfile
from gptok2_compact.codec import apply_bpe_rules as _compact_apply_bpe_rules
from gptok2_compact.codec import apply_macro_rules as _compact_apply_macro_rules
from gptok2_compact.codec import compact_symbol_bits as _compact_symbol_bits


STRUCTURAL_RULE_PREFIX = {
    "edge_chain": "EMIT_CHAIN_BLOCK",
    "chain": "EMIT_CHAIN_BLOCK",
    "cycle": "EMIT_CYCLE_BLOCK",
    "star": "EMIT_STAR_BLOCK",
    "clique": "EMIT_CLIQUE_BLOCK",
    "treelet": "EMIT_TREELET_BLOCK",
    "motif": "EMIT_MOTIF_BLOCK",
}

PARAMETERIZED_MACRO_OPS = {
    "EMIT_CHAIN_BLOCK",
    "EMIT_CYCLE_BLOCK",
    "EMIT_STAR_BLOCK",
    "EMIT_CLIQUE_BLOCK",
    "EMIT_TREELET_BLOCK",
    "EMIT_MOTIF_BLOCK",
}


@dataclass
class MotifMacroProfile:
    """Serializable profile for a reversible motif macro codec."""

    compact_profile: CompactProfile
    structural_rules: dict[str, list[str]]
    rule_categories: dict[str, str]
    code_schema_rules: dict[str, list[str]]
    merge_schema_rules: dict[str, list[str]]
    shape_categories: dict[str, str]
    train_original_tokens: int
    train_motif_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "compact_profile": self.compact_profile.to_dict(),
            "structural_rules": self.structural_rules,
            "rule_categories": self.rule_categories,
            "code_schema_rules": self.code_schema_rules,
            "merge_schema_rules": self.merge_schema_rules,
            "shape_categories": self.shape_categories,
            "train_original_tokens": self.train_original_tokens,
            "train_motif_tokens": self.train_motif_tokens,
        }

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "MotifMacroProfile":
        return MotifMacroProfile(
            compact_profile=CompactProfile.from_dict(row.get("compact_profile", {})),
            structural_rules={str(k): list(v) for k, v in row.get("structural_rules", {}).items()},
            rule_categories={str(k): str(v) for k, v in row.get("rule_categories", {}).items()},
            code_schema_rules={str(k): list(v) for k, v in row.get("code_schema_rules", {}).items()},
            merge_schema_rules={str(k): list(v) for k, v in row.get("merge_schema_rules", {}).items()},
            shape_categories={str(k): str(v) for k, v in row.get("shape_categories", {}).items()},
            train_original_tokens=int(row.get("train_original_tokens", 0)),
            train_motif_tokens=int(row.get("train_motif_tokens", 0)),
        )


class MotifMacroCodec:
    """Reversible graph-structure macro codec.

    The pipeline is:

    original GPTok2 tokens -> compact base tokens -> learned motif macros
    -> compact BPE tokens.

    Decoding performs the exact inverse and must recover the original token
    list byte-for-byte.
    """

    def __init__(
        self,
        codebook: PrototypeVQCodebook,
        max_structural_macros: int = 256,
        min_structural_count: int = 2,
        max_structural_len: int = 24,
        max_compact_macros: int = 256,
        min_compact_macro_count: int = 3,
        max_compact_macro_len: int = 12,
        max_bpe_merges: int = 256,
        min_bpe_count: int = 3,
        use_parameterized_macros: bool = True,
        min_parameterized_emit_count: int = 3,
        max_parameterized_span_len: int = 96,
        max_merge_schemas: int = 256,
        min_merge_schema_count: int = 2,
        max_code_schemas: int = 256,
        min_code_schema_count: int = 2,
    ):
        self.codebook = codebook
        self.max_structural_macros = int(max_structural_macros)
        self.min_structural_count = int(min_structural_count)
        self.max_structural_len = int(max_structural_len)
        self.use_parameterized_macros = bool(use_parameterized_macros)
        self.min_parameterized_emit_count = int(min_parameterized_emit_count)
        self.max_parameterized_span_len = int(max_parameterized_span_len)
        self.max_merge_schemas = int(max_merge_schemas)
        self.min_merge_schema_count = int(min_merge_schema_count)
        self.max_code_schemas = int(max_code_schemas)
        self.min_code_schema_count = int(min_code_schema_count)
        self.base_codec = CompactCodec(
            max_macros=max_compact_macros,
            min_macro_count=min_compact_macro_count,
            max_macro_len=max_compact_macro_len,
            max_bpe_merges=max_bpe_merges,
            min_bpe_count=min_bpe_count,
        )
        self.profile = MotifMacroProfile(
            compact_profile=self.base_codec.profile,
            structural_rules={},
            rule_categories={},
            code_schema_rules={},
            merge_schema_rules={},
            shape_categories=code_shape_categories(codebook),
            train_original_tokens=0,
            train_motif_tokens=0,
        )

    def fit(self, train_sequences: list[list[str]]) -> "MotifMacroCodec":
        self.base_codec.fit(train_sequences)
        shape_categories = code_shape_categories(self.codebook)
        self.profile = MotifMacroProfile(
            compact_profile=self.base_codec.profile,
            structural_rules={},
            rule_categories={},
            code_schema_rules={},
            merge_schema_rules={},
            shape_categories=shape_categories,
            train_original_tokens=0,
            train_motif_tokens=0,
        )
        raw_base_sequences = [_compact_without_learned_rules(seq) for seq in train_sequences]
        merge_schema_rules = learn_merge_schema_rules(
            raw_base_sequences,
            min_emit_count=self.min_parameterized_emit_count,
            max_span_len=self.max_parameterized_span_len,
            max_rules=self.max_merge_schemas,
            min_count=self.min_merge_schema_count,
        )
        code_schema_rules = learn_code_schema_rules(
            raw_base_sequences,
            min_emit_count=self.min_parameterized_emit_count,
            max_span_len=self.max_parameterized_span_len,
            max_rules=self.max_code_schemas,
            min_count=self.min_code_schema_count,
        )
        self.profile.merge_schema_rules = merge_schema_rules
        self.profile.code_schema_rules = code_schema_rules
        base_sequences = [self._apply_parameterized(seq) for seq in raw_base_sequences]
        structural_rules, rule_categories, _ = learn_structural_rules(
            base_sequences,
            shape_categories,
            max_rules=self.max_structural_macros,
            min_count=self.min_structural_count,
            max_len=self.max_structural_len,
        )
        # Reuse the BPE stage from the already fitted compact codec after
        # applying the structural rules. This keeps motif-macro comparable to
        # compact while adding structure-grounded larger symbols first.
        self.profile = MotifMacroProfile(
            compact_profile=self.base_codec.profile,
            structural_rules=structural_rules,
            rule_categories=rule_categories,
            code_schema_rules=code_schema_rules,
            merge_schema_rules=merge_schema_rules,
            shape_categories=shape_categories,
            train_original_tokens=sum(len(s) for s in train_sequences),
            train_motif_tokens=0,
        )
        self.profile.train_motif_tokens = sum(len(self.encode(seq)) for seq in train_sequences)
        return self

    def encode(self, tokens: list[str]) -> list[str]:
        compact = self.base_codec.encode(tokens)
        seq = _compact_without_learned_rules(tokens)
        seq = self._apply_parameterized(seq)
        seq = self._apply_structural(seq, self.profile.structural_rules)
        seq = self._apply_bpe(seq)
        return seq if len(seq) < len(compact) else compact

    def encode_hybrid(
        self,
        tokens: list[str],
        compact_entropy: Any | None = None,
        motif_entropy: "MotifEntropyModel | None" = None,
    ) -> list[str]:
        """Encode with local span-level compact-vs-macro selection.

        Unlike the older graph-level hybrid score, this method decides at each
        parameterized emit/glue span whether the macro payload is cheaper than
        the equivalent compact span. The output is still a single reversible
        stream: motif macro tokens and compact macro/BPE tokens may coexist.
        """

        compact = self.base_codec.encode(tokens)
        seq = _compact_without_learned_rules(tokens)
        seq = self._apply_parameterized_hybrid(seq, compact_entropy, motif_entropy)
        seq = self._apply_structural(seq, self.profile.structural_rules)
        seq = self._apply_bpe(seq)
        if compact_entropy is not None and motif_entropy is not None:
            hybrid_bits = _sum_token_bits(motif_entropy, seq)
            compact_bits = _sum_token_bits(compact_entropy, compact)
            return seq if hybrid_bits <= compact_bits else compact
        return seq if len(seq) < len(compact) else compact

    def decode(self, motif_tokens: list[str]) -> list[str]:
        motif_decoded = self._decode_motif_stream(motif_tokens)
        if _looks_like_original_program(motif_decoded):
            return motif_decoded
        return self.base_codec.decode(motif_tokens)

    def macro_usage(self, sequences: list[list[str]]) -> dict[str, float]:
        counts: Counter[str] = Counter()
        total = 0
        for seq in sequences:
            encoded = self.encode(seq)
            total += len(encoded)
            for tok in encoded:
                op, _ = _split_token(tok)
                if op.startswith("EMIT_") and op.endswith("_BLOCK"):
                    counts[op] += 1
                elif op.startswith("STRUCT_BPE_"):
                    counts["STRUCT_BPE"] += 1
                elif op.startswith("ACTION_BPE_"):
                    counts["ACTION_BPE"] += 1
        return {
            "total_tokens": float(total),
            **{f"usage_{k.lower()}": float(v) for k, v in sorted(counts.items())},
        }

    def _apply_structural(self, tokens: list[str], rules: dict[str, list[str]]) -> list[str]:
        seq = list(tokens)
        for token, parts in rules.items():
            seq = _replace_ngram(seq, tuple(parts), token)
        return seq

    def _apply_bpe(self, tokens: list[str]) -> list[str]:
        seq = list(tokens)
        for token, pair in self.profile.compact_profile.bpe_rules.items():
            seq = _replace_pair(seq, tuple(pair), token)
        return seq

    def _decode_motif_stream(self, motif_tokens: list[str]) -> list[str]:
        seq: list[str] = []
        for tok in motif_tokens:
            seq.extend(_expand_bpe(tok, self.profile.compact_profile.bpe_rules))
        expanded_compact: list[str] = []
        for tok in seq:
            expanded_compact.extend(_expand_compact_macro(tok, self.profile.compact_profile.macro_rules))
        expanded_structural: list[str] = []
        for tok in expanded_compact:
            expanded_structural.extend(_expand_structural(tok, self.profile.structural_rules))
        expanded_parameterized: list[str] = []
        for tok in expanded_structural:
            expanded_parameterized.extend(_expand_parameterized_macro(tok, self.profile.merge_schema_rules, self.profile.code_schema_rules))
        return _expand_base_tokens(expanded_parameterized)

    def _apply_parameterized(self, tokens: list[str]) -> list[str]:
        if not self.use_parameterized_macros:
            return list(tokens)
        return apply_parameterized_macros(
            tokens,
            shape_categories=self.profile.shape_categories,
            merge_schema_rules=self.profile.merge_schema_rules,
            code_schema_rules=self.profile.code_schema_rules,
            min_emit_count=self.min_parameterized_emit_count,
            max_span_len=self.max_parameterized_span_len,
        )

    def _apply_parameterized_hybrid(
        self,
        tokens: list[str],
        compact_entropy: Any | None,
        motif_entropy: "MotifEntropyModel | None",
    ) -> list[str]:
        if not self.use_parameterized_macros:
            return list(tokens)
        return apply_parameterized_macros_hybrid(
            tokens,
            shape_categories=self.profile.shape_categories,
            merge_schema_rules=self.profile.merge_schema_rules,
            code_schema_rules=self.profile.code_schema_rules,
            min_emit_count=self.min_parameterized_emit_count,
            max_span_len=self.max_parameterized_span_len,
            compact_profile=self.profile.compact_profile,
            compact_entropy=compact_entropy,
            motif_entropy=motif_entropy,
        )


class MotifEntropyModel:
    """Entropy estimator for motif-macro sequences."""

    def __init__(self):
        self.token_counts: Counter[str] = Counter()
        self.op_counts: Counter[str] = Counter()
        self.category_counts: Counter[str] = Counter()
        self.payload_counts: Counter[str] = Counter()
        self.length_counts: Counter[str] = Counter()
        self.code_counts: Counter[str] = Counter()
        self.code_schema_counts: Counter[str] = Counter()
        self.schema_counts: Counter[str] = Counter()
        self.glue_op_counts: Counter[str] = Counter()
        self.merge_family_counts: Counter[str] = Counter()
        self.merge_pair_count_counts: Counter[str] = Counter()
        self.merge_local_start_counts: Counter[str] = Counter()
        self.merge_local_step_counts: Counter[str] = Counter()
        self.merge_ref_start_delta_counts: Counter[str] = Counter()
        self.merge_ref_step_counts: Counter[str] = Counter()
        self.local_counts: Counter[str] = Counter()
        self.ref_delta_counts: Counter[str] = Counter()
        self.close_counts: Counter[str] = Counter()
        self.merge_schema_rules: dict[str, list[str]] = {}
        self.code_schema_rules: dict[str, list[str]] = {}
        self.num_sequences = 0
        self.rulebook_bits = 0.0

    def fit(self, motif_sequences: list[list[str]], profile: MotifMacroProfile | None = None) -> "MotifEntropyModel":
        self.num_sequences = len(motif_sequences)
        self.merge_schema_rules = dict(profile.merge_schema_rules) if profile is not None else {}
        self.code_schema_rules = dict(profile.code_schema_rules) if profile is not None else {}
        for seq in motif_sequences:
            for tok in seq:
                self._observe(tok, profile)
        if profile is not None:
            self.rulebook_bits = _profile_rulebook_bits(profile)
        return self

    def bits(self, motif_tokens: list[str], include_rulebook: bool = False) -> float:
        total = 0.0
        for tok in motif_tokens:
            total += self._token_bits(tok)
        if include_rulebook and self.num_sequences:
            total += self.rulebook_bits / max(1, self.num_sequences)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_counts": dict(self.token_counts),
            "op_counts": dict(self.op_counts),
            "category_counts": dict(self.category_counts),
            "payload_counts": dict(self.payload_counts),
            "length_counts": dict(self.length_counts),
            "code_counts": dict(self.code_counts),
            "code_schema_counts": dict(self.code_schema_counts),
            "schema_counts": dict(self.schema_counts),
            "glue_op_counts": dict(self.glue_op_counts),
            "merge_family_counts": dict(self.merge_family_counts),
            "merge_pair_count_counts": dict(self.merge_pair_count_counts),
            "merge_local_start_counts": dict(self.merge_local_start_counts),
            "merge_local_step_counts": dict(self.merge_local_step_counts),
            "merge_ref_start_delta_counts": dict(self.merge_ref_start_delta_counts),
            "merge_ref_step_counts": dict(self.merge_ref_step_counts),
            "local_counts": dict(self.local_counts),
            "ref_delta_counts": dict(self.ref_delta_counts),
            "close_counts": dict(self.close_counts),
            "num_sequences": self.num_sequences,
            "rulebook_bits": self.rulebook_bits,
        }

    def _observe(self, tok: str, profile: MotifMacroProfile | None) -> None:
        op, payload = _split_token(tok)
        self.token_counts[tok] += 1
        self.op_counts[op] += 1
        category = profile.rule_categories.get(tok, op) if profile is not None else op
        self.category_counts[category] += 1
        parsed = _parse_parameterized_payload_raw(
            payload,
            profile.merge_schema_rules if profile is not None else None,
            profile.code_schema_rules if profile is not None else None,
        )
        if op in PARAMETERIZED_MACRO_OPS and parsed is not None:
            codes, glues, close, is_code_run, merge_schema_id, code_schema_id, glue_family = parsed
            self.length_counts[str(len(codes))] += 1
            if code_schema_id:
                self.code_schema_counts[code_schema_id] += 1
            else:
                observed_codes = [codes[0]] if is_code_run and codes else codes
                for code in observed_codes:
                    self.code_counts[code] += 1
            if glue_family:
                self._observe_glue_family(glue_family)
            elif merge_schema_id:
                self.schema_counts[merge_schema_id] += 1
            else:
                self._observe_glue_sequence(glues)
            if close:
                close_op, close_payload = _split_token(close)
                self.close_counts[close_op] += 1
                self._observe_pairs(close_payload)
            return
        if payload:
            self.payload_counts[payload] += 1

    def _token_bits(self, tok: str) -> float:
        op, payload = _split_token(tok)
        whole = _neglog(self.token_counts, tok)
        component = _neglog(self.op_counts, op)
        parsed = _parse_parameterized_payload_raw(
            payload,
            self.merge_schema_rules if hasattr(self, "merge_schema_rules") else None,
            self.code_schema_rules if hasattr(self, "code_schema_rules") else None,
        )
        if op in PARAMETERIZED_MACRO_OPS and parsed is not None:
            codes, glues, close, is_code_run, merge_schema_id, code_schema_id, glue_family = parsed
            component += _neglog(self.length_counts, str(len(codes)))
            if code_schema_id:
                component += _neglog(self.code_schema_counts, code_schema_id)
            else:
                charged_codes = [codes[0]] if is_code_run and codes else codes
                component += sum(_neglog(self.code_counts, code) for code in charged_codes)
            if glue_family:
                component += self._glue_family_bits(glue_family)
            elif merge_schema_id:
                component += _neglog(self.schema_counts, merge_schema_id)
            else:
                component += self._glue_sequence_bits(glues)
            if close:
                close_op, close_payload = _split_token(close)
                component += _neglog(self.close_counts, close_op)
                component += self._pairs_bits(close_payload)
            schema_encoded_glue = bool(glue_family) or bool(merge_schema_id) or not glues
            schema_encoded_code = bool(code_schema_id) or is_code_run or len(codes) <= 1
            if schema_encoded_glue and schema_encoded_code:
                return min(whole, component)
            return component
        if payload:
            component += _neglog(self.payload_counts, payload)
        return min(whole, component)

    def _observe_pairs(self, payload: str) -> None:
        prev_ref = 0
        for item in [p for p in payload.split(";") if p]:
            parts = item.split(":")
            if len(parts) >= 2:
                local, ref = parts[0], parts[1]
                self.local_counts[local] += 1
                try:
                    ref_i = int(ref)
                    self.ref_delta_counts[str(ref_i - prev_ref)] += 1
                    prev_ref = ref_i
                except ValueError:
                    self.ref_delta_counts[ref] += 1

    def _observe_glue_sequence(self, glues: list[str]) -> None:
        prev_ref = 0
        for glue in glues:
            glue_op, glue_payload = _split_token(glue)
            self.glue_op_counts[glue_op] += 1
            family = _merge_arithmetic_family(glue_op, glue_payload)
            if family is not None:
                count, local_start, local_step, ref_start, ref_step = family
                self.merge_family_counts["arith"] += 1
                self.merge_pair_count_counts[str(count)] += 1
                self.merge_local_start_counts[str(local_start)] += 1
                self.merge_local_step_counts[str(local_step)] += 1
                self.merge_ref_start_delta_counts[str(ref_start - prev_ref)] += 1
                self.merge_ref_step_counts[str(ref_step)] += 1
                prev_ref = ref_start + ref_step * max(0, count - 1)
                continue
            for item in [p for p in glue_payload.split(";") if p]:
                parts = item.split(":")
                if len(parts) >= 2:
                    local, ref = parts[0], parts[1]
                    self.local_counts[local] += 1
                    try:
                        ref_i = int(ref)
                        self.ref_delta_counts[str(ref_i - prev_ref)] += 1
                        prev_ref = ref_i
                    except ValueError:
                        self.ref_delta_counts[ref] += 1

    def _observe_glue_family(self, fields: dict[str, str]) -> None:
        family = fields.get("family", "")
        self.merge_family_counts[family] += 1
        if family == "P":
            self._observe_packed_glue_payload(fields.get("payload", ""))
            return
        for key in ["count", "local_start", "local_step", "ref_step"]:
            val = fields.get(key)
            if val is None:
                continue
            if key == "count":
                self.merge_pair_count_counts[val] += 1
            elif key == "local_start":
                self.merge_local_start_counts[val] += 1
            elif key == "local_step":
                self.merge_local_step_counts[val] += 1
            elif key == "ref_step":
                self.merge_ref_step_counts[val] += 1
        for key in ["ref_start", "delta", "pair_sum", "ref"]:
            val = fields.get(key)
            if val is not None:
                self.merge_ref_start_delta_counts[val] += 1
        if family == "R" and fields.get("token"):
            self.payload_counts[fields["token"]] += 1

    def _pairs_bits(self, payload: str) -> float:
        bits = 0.0
        prev_ref = 0
        for item in [p for p in payload.split(";") if p]:
            parts = item.split(":")
            if len(parts) >= 2:
                local, ref = parts[0], parts[1]
                bits += _neglog(self.local_counts, local)
                try:
                    ref_i = int(ref)
                    bits += _neglog(self.ref_delta_counts, str(ref_i - prev_ref))
                    prev_ref = ref_i
                except ValueError:
                    bits += _neglog(self.ref_delta_counts, ref)
        return bits

    def _glue_sequence_bits(self, glues: list[str]) -> float:
        bits = 0.0
        prev_ref = 0
        for glue in glues:
            glue_op, glue_payload = _split_token(glue)
            bits += _neglog(self.glue_op_counts, glue_op)
            family = _merge_arithmetic_family(glue_op, glue_payload)
            if family is not None:
                count, local_start, local_step, ref_start, ref_step = family
                bits += _neglog(self.merge_family_counts, "arith")
                bits += _neglog(self.merge_pair_count_counts, str(count))
                bits += _neglog(self.merge_local_start_counts, str(local_start))
                bits += _neglog(self.merge_local_step_counts, str(local_step))
                bits += _neglog(self.merge_ref_start_delta_counts, str(ref_start - prev_ref))
                bits += _neglog(self.merge_ref_step_counts, str(ref_step))
                prev_ref = ref_start + ref_step * max(0, count - 1)
                continue
            for item in [p for p in glue_payload.split(";") if p]:
                parts = item.split(":")
                if len(parts) >= 2:
                    local, ref = parts[0], parts[1]
                    bits += _neglog(self.local_counts, local)
                    try:
                        ref_i = int(ref)
                        bits += _neglog(self.ref_delta_counts, str(ref_i - prev_ref))
                        prev_ref = ref_i
                    except ValueError:
                        bits += _neglog(self.ref_delta_counts, ref)
        return bits

    def _glue_family_bits(self, fields: dict[str, str]) -> float:
        bits = _neglog(self.merge_family_counts, fields.get("family", ""))
        if fields.get("family") == "P":
            return bits + self._packed_glue_payload_bits(fields.get("payload", ""))
        mapping = [
            ("count", self.merge_pair_count_counts),
            ("local_start", self.merge_local_start_counts),
            ("local_step", self.merge_local_step_counts),
            ("ref_start", self.merge_ref_start_delta_counts),
            ("ref_step", self.merge_ref_step_counts),
            ("delta", self.merge_ref_start_delta_counts),
            ("pair_sum", self.merge_ref_start_delta_counts),
            ("ref", self.merge_ref_start_delta_counts),
        ]
        for key, counts in mapping:
            val = fields.get(key)
            if val is not None:
                bits += _neglog(counts, val)
        if fields.get("family") == "R" and fields.get("token"):
            bits += _neglog(self.payload_counts, fields["token"])
        return bits

    def _observe_packed_glue_payload(self, payload: str) -> None:
        for raw in [p for p in payload.split(".") if p]:
            item = unquote(raw)
            fields = _single_glue_item_family_fields(item)
            if fields is None:
                self.payload_counts[item] += 1
            else:
                self._observe_single_glue_item_family(fields)

    def _packed_glue_payload_bits(self, payload: str) -> float:
        bits = 0.0
        for raw in [p for p in payload.split(".") if p]:
            item = unquote(raw)
            fields = _single_glue_item_family_fields(item)
            if fields is None:
                bits += _neglog(self.payload_counts, item)
            else:
                bits += self._single_glue_item_family_bits(fields)
        return bits

    def _observe_single_glue_item_family(self, fields: dict[str, str]) -> None:
        self.merge_family_counts["M:" + fields.get("family", "")] += 1
        for key in ["count", "local_start", "local_step", "ref_step"]:
            val = fields.get(key)
            if val is None:
                continue
            if key == "count":
                self.merge_pair_count_counts[val] += 1
            elif key == "local_start":
                self.merge_local_start_counts[val] += 1
            elif key == "local_step":
                self.merge_local_step_counts[val] += 1
            elif key == "ref_step":
                self.merge_ref_step_counts[val] += 1
        for key in ["ref_start", "delta", "pair_sum", "ref", "local_deltas", "ref_deltas"]:
            val = fields.get(key)
            if val is not None:
                self.payload_counts[key + "=" + val] += 1

    def _single_glue_item_family_bits(self, fields: dict[str, str]) -> float:
        bits = _neglog(self.merge_family_counts, "M:" + fields.get("family", ""))
        for key, counts in [
            ("count", self.merge_pair_count_counts),
            ("local_start", self.merge_local_start_counts),
            ("local_step", self.merge_local_step_counts),
            ("ref_step", self.merge_ref_step_counts),
        ]:
            val = fields.get(key)
            if val is not None:
                bits += _neglog(counts, val)
        for key in ["ref_start", "delta", "pair_sum", "ref", "local_deltas", "ref_deltas"]:
            val = fields.get(key)
            if val is not None:
                bits += _neglog(self.payload_counts, key + "=" + val)
        return bits


def learn_structural_rules(
    sequences: list[list[str]],
    shape_categories: dict[str, str],
    max_rules: int,
    min_count: int,
    max_len: int,
) -> tuple[dict[str, list[str]], dict[str, str], list[list[str]]]:
    work = [list(seq) for seq in sequences]
    rules: dict[str, list[str]] = {}
    categories: dict[str, str] = {}
    counters_by_category: dict[str, int] = Counter()
    for _ in range(max_rules):
        counts: Counter[tuple[str, ...]] = Counter()
        category_votes: dict[tuple[str, ...], str] = {}
        for seq in work:
            candidates = _structural_spans(seq, shape_categories, max_len)
            for start, end, category in candidates:
                gram = tuple(seq[start:end])
                if len(gram) >= 2:
                    counts[gram] += 1
                    category_votes.setdefault(gram, category)
        if not counts:
            break
        gram, count = max(counts.items(), key=lambda kv: (_savings(kv[0], kv[1]), kv[1], len(kv[0])))
        if count < min_count or _savings(gram, count) <= 0:
            break
        category = category_votes.get(gram, "motif")
        prefix = STRUCTURAL_RULE_PREFIX.get(category, "EMIT_MOTIF_BLOCK")
        idx = counters_by_category[prefix]
        counters_by_category[prefix] += 1
        token = f"{prefix}({idx:04d})"
        rules[token] = list(gram)
        categories[token] = category
        work = [_replace_ngram(seq, gram, token) for seq in work]
    return rules, categories, work


def code_shape_categories(codebook: PrototypeVQCodebook) -> dict[str, str]:
    categories: dict[str, str] = {}
    for code in codebook.codes:
        categories[str(code.code_id)] = _shape_category(int(code.prototype_num_nodes), tuple(code.prototype_edges))
    return categories


def motif_symbol_bits(tokens: list[str], vocab_size: int | None = None) -> float:
    return _compact_symbol_bits(tokens, vocab_size)


def _compact_without_learned_rules(tokens: list[str]) -> list[str]:
    """Base reversible compaction without corpus-learned macro/BPE rules."""

    out: list[str] = []
    i = 0
    group_names = {
        "ATTACH": "ATTACH_PATTERN",
        "MERGE_NODE": "MERGE_NODE_PATTERN",
        "MERGE_EDGE": "MERGE_EDGE_PATTERN",
        "CLOSE_CYCLE": "CLOSE_CYCLE_PATTERN",
        "GLOBAL_LINK": "GLOBAL_LINK_GROUP",
    }
    while i < len(tokens):
        tok = tokens[i]
        op, payload = _split_token(tok)
        if op in group_names:
            pairs = []
            while i < len(tokens):
                op2, payload2 = _split_token(tokens[i])
                if op2 != op:
                    break
                pairs.append(payload2.replace(",", ":"))
                i += 1
            out.append(f"{group_names[op]}(" + ";".join(pairs) + ")")
            continue
        if op == "EMIT":
            out.append(f"EMIT_CODE({payload})")
        elif op == "INTERFACE":
            out.append(f"INTERFACE_CODE({payload})")
        else:
            out.append(tok)
        i += 1
    return out


def _expand_base_tokens(tokens: list[str]) -> list[str]:
    reverse = {
        "ATTACH_PATTERN": "ATTACH",
        "MERGE_NODE_PATTERN": "MERGE_NODE",
        "MERGE_EDGE_PATTERN": "MERGE_EDGE",
        "CLOSE_CYCLE_PATTERN": "CLOSE_CYCLE",
        "GLOBAL_LINK_GROUP": "GLOBAL_LINK",
    }
    out: list[str] = []
    for tok in tokens:
        op, payload = _split_token(tok)
        if op in reverse:
            original_op = reverse[op]
            for pair in [p for p in payload.split(";") if p]:
                out.append(f"{original_op}(" + pair.replace(":", ",") + ")")
        elif op == "EMIT_CODE":
            out.append(f"EMIT({payload})")
        elif op == "INTERFACE_CODE":
            out.append(f"INTERFACE({payload})")
        elif op == "REPEAT_EMIT_MERGE":
            code, pattern, count = payload.split("|")
            for _ in range(int(count)):
                out.append(f"EMIT({code})")
                for pair in [p for p in pattern.split(";") if p]:
                    out.append("MERGE_NODE(" + pair.replace(":", ",") + ")")
        else:
            out.append(tok)
    return out


def _looks_like_original_program(tokens: list[str]) -> bool:
    valid_ops = {
        "BEGIN_GRAPH",
        "END_GRAPH",
        "BEGIN_BLOCK",
        "END_BLOCK",
        "EMIT",
        "INTERFACE",
        "ATTACH",
        "MERGE_NODE",
        "MERGE_EDGE",
        "CLOSE_CYCLE",
        "GLOBAL_LINK",
        "STOP",
    }
    if not tokens:
        return False
    for tok in tokens:
        op, _ = _split_token(tok)
        if op not in valid_ops:
            return False
    return True


def apply_parameterized_macros(
    tokens: list[str],
    shape_categories: dict[str, str],
    merge_schema_rules: dict[str, list[str]],
    code_schema_rules: dict[str, list[str]],
    min_emit_count: int,
    max_span_len: int,
) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        macro, end = _parameterized_span(tokens, i, shape_categories, merge_schema_rules, code_schema_rules, min_emit_count, max_span_len)
        if macro is not None and end > i:
            out.append(macro)
            i = end
        else:
            out.append(tokens[i])
            i += 1
    return out


def apply_parameterized_macros_hybrid(
    tokens: list[str],
    shape_categories: dict[str, str],
    merge_schema_rules: dict[str, list[str]],
    code_schema_rules: dict[str, list[str]],
    min_emit_count: int,
    max_span_len: int,
    compact_profile: CompactProfile,
    compact_entropy: Any | None,
    motif_entropy: "MotifEntropyModel | None",
) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        macro, end = _parameterized_span(tokens, i, shape_categories, merge_schema_rules, code_schema_rules, min_emit_count, max_span_len)
        if macro is None or end <= i:
            out.append(tokens[i])
            i += 1
            continue
        base_span = tokens[i:end]
        compact_span = _encode_base_span_with_compact_rules(base_span, compact_profile)
        if compact_entropy is not None and motif_entropy is not None:
            macro_bits = _sum_token_bits(motif_entropy, [macro])
            compact_bits = _sum_token_bits(compact_entropy, compact_span)
            if compact_bits < macro_bits:
                out.extend(compact_span)
            else:
                out.append(macro)
        elif len(compact_span) < 1:
            out.extend(compact_span)
        else:
            out.append(macro)
        i = end
    return out


def _parameterized_span(
    tokens: list[str],
    start: int,
    shape_categories: dict[str, str],
    merge_schema_rules: dict[str, list[str]],
    code_schema_rules: dict[str, list[str]],
    min_emit_count: int,
    max_span_len: int,
) -> tuple[str | None, int]:
    op, payload = _split_token(tokens[start])
    if op != "EMIT_CODE":
        return None, start
    codes = [payload]
    glues: list[str] = []
    j = start + 1
    glue_ops = {"MERGE_NODE_PATTERN", "MERGE_EDGE_PATTERN", "ATTACH_PATTERN"}
    while j + 1 < len(tokens) and j - start < max_span_len:
        glue_op, _ = _split_token(tokens[j])
        next_op, next_payload = _split_token(tokens[j + 1])
        if glue_op not in glue_ops or next_op != "EMIT_CODE":
            break
        glues.append(tokens[j])
        codes.append(next_payload)
        j += 2
    close = ""
    if j < len(tokens):
        close_op, _ = _split_token(tokens[j])
        if close_op == "CLOSE_CYCLE_PATTERN" and len(codes) >= min_emit_count:
            close = tokens[j]
            j += 1
    if len(codes) < min_emit_count:
        return None, start
    # Require a real token-count win. One macro replaces len(codes)+len(glues)+close.
    original_len = len(codes) + len(glues) + (1 if close else 0)
    if original_len <= 2:
        return None, start
    category = _parameterized_category(codes, glues, close, shape_categories)
    op_name = STRUCTURAL_RULE_PREFIX.get(category, "EMIT_MOTIF_BLOCK")
    token = _make_parameterized_token(op_name, codes, glues, close, merge_schema_rules, code_schema_rules)
    return token, j


def _encode_base_span_with_compact_rules(tokens: list[str], profile: CompactProfile) -> list[str]:
    seq = _compact_apply_macro_rules(list(tokens), profile.macro_rules)
    return _compact_apply_bpe_rules(seq, profile.bpe_rules)


def _sum_token_bits(model: Any, tokens: list[str]) -> float:
    token_bits = getattr(model, "_token_bits", None)
    if callable(token_bits):
        return float(sum(token_bits(tok) for tok in tokens))
    bits = getattr(model, "bits", None)
    if callable(bits):
        return float(bits(tokens, include_rulebook=False))
    vocab = max(2, len(set(tokens)))
    return float(len(tokens) * math.log2(vocab))


def _parameterized_category(codes: list[str], glues: list[str], close: str, shape_categories: dict[str, str]) -> str:
    if close:
        return "cycle"
    categories = [shape_categories.get(str(c), "motif") for c in codes]
    if categories and all(c == "clique" for c in categories):
        return "clique"
    if _star_like(glues):
        return "star"
    if categories and all(c in {"edge", "chain", "treelet"} for c in categories):
        return "chain"
    if categories and all(c == "cycle" for c in categories):
        return "cycle"
    return "motif"


def _star_like(glues: list[str]) -> bool:
    refs: list[str] = []
    locals_: list[str] = []
    for glue in glues:
        op, payload = _split_token(glue)
        if op != "MERGE_NODE_PATTERN":
            return False
        for item in [p for p in payload.split(";") if p]:
            parts = item.split(":")
            if len(parts) >= 2:
                locals_.append(parts[0])
                refs.append(parts[1])
    return len(glues) >= 3 and refs and len(set(refs)) == 1 and len(set(locals_)) >= 2


def _make_parameterized_token(
    op_name: str,
    codes: list[str],
    glues: list[str],
    close: str,
    merge_schema_rules: dict[str, list[str]],
    code_schema_rules: dict[str, list[str]],
) -> str:
    if codes and len(set(codes)) == 1:
        code_payload = "R:" + quote(codes[0], safe="")
    elif codes:
        reverse = {tuple(v): k for k, v in code_schema_rules.items()}
        schema_id = reverse.get(tuple(codes))
        code_payload = "C:" + quote(schema_id, safe="") if schema_id is not None else _pack_items(codes)
    else:
        code_payload = _pack_items(codes)
    glue_payload = _pack_glues(glues, merge_schema_rules)
    payload = "|".join(
        [
            "v2",
            str(len(codes)),
            code_payload,
            glue_payload,
            quote(close, safe=""),
        ]
    )
    return f"{op_name}({payload})"


def _expand_parameterized_macro(
    tok: str,
    merge_schema_rules: dict[str, list[str]] | None = None,
    code_schema_rules: dict[str, list[str]] | None = None,
) -> list[str]:
    op, payload = _split_token(tok)
    parsed = _parse_parameterized_payload(payload, merge_schema_rules, code_schema_rules)
    if op not in PARAMETERIZED_MACRO_OPS or parsed is None:
        return [tok]
    codes, glues, close = parsed
    out = [f"EMIT_CODE({codes[0]})"] if codes else []
    for glue, code in zip(glues, codes[1:]):
        out.append(glue)
        out.append(f"EMIT_CODE({code})")
    if close:
        out.append(close)
    return out


def _parse_parameterized_payload(
    payload: str,
    merge_schema_rules: dict[str, list[str]] | None = None,
    code_schema_rules: dict[str, list[str]] | None = None,
) -> tuple[list[str], list[str], str] | None:
    parsed = _parse_parameterized_payload_raw(payload, merge_schema_rules, code_schema_rules)
    if parsed is None:
        return None
    codes, glues, close, _, _, _, _ = parsed
    return codes, glues, close


def _parse_parameterized_payload_raw(
    payload: str,
    merge_schema_rules: dict[str, list[str]] | None = None,
    code_schema_rules: dict[str, list[str]] | None = None,
) -> tuple[list[str], list[str], str, bool, str, str, dict[str, str] | None] | None:
    if not payload.startswith("v2|"):
        return None
    parts = payload.split("|", 4)
    if len(parts) != 5:
        return None
    _, count_s, codes_s, glues_s, close_s = parts
    try:
        count = int(count_s)
    except ValueError:
        return None
    is_code_run = codes_s.startswith("R:")
    code_schema_id = ""
    if is_code_run:
        codes = [unquote(codes_s[2:])] * count
    elif codes_s.startswith("C:"):
        code_schema_id = unquote(codes_s[2:])
        codes = list((code_schema_rules or {}).get(code_schema_id, []))
    else:
        codes = _unpack_items(codes_s)
    schema_id = ""
    glue_family: dict[str, str] | None = None
    if glues_s.startswith("S:"):
        schema_id = unquote(glues_s[2:])
        glues = list((merge_schema_rules or {}).get(schema_id, []))
    elif glues_s.startswith("F:"):
        glues = _unpack_glue_family(glues_s)
        if glues is None:
            return None
        glue_family = _glue_family_fields(glues_s)
        if glue_family is None:
            return None
    elif glues_s.startswith("P:"):
        glues = _unpack_compact_glue_items(glues_s[2:])
        if glues is None:
            return None
        glue_family = {"family": "P", "payload": glues_s[2:]}
    else:
        glues = _unpack_items(glues_s)
    close = unquote(close_s) if close_s else ""
    if len(codes) != count or len(glues) not in {max(0, count - 1), count}:
        return None
    return codes, glues, close, is_code_run, schema_id, code_schema_id, glue_family


def _pack_items(items: list[str]) -> str:
    return ".".join(quote(str(x), safe="") for x in items)


def _unpack_items(payload: str) -> list[str]:
    if not payload:
        return []
    return [unquote(x) for x in payload.split(".")]


def _pack_glues(glues: list[str], merge_schema_rules: dict[str, list[str]]) -> str:
    raw = _pack_items(glues)
    candidates = [raw]
    reverse = {tuple(v): k for k, v in merge_schema_rules.items()}
    schema_id = reverse.get(tuple(glues))
    if schema_id is not None:
        candidates.append("S:" + quote(schema_id, safe=""))
    family = _pack_glue_family(glues)
    if family:
        candidates.append(family)
    compact_items = _pack_compact_glue_items(glues)
    if compact_items:
        candidates.append(compact_items)
    return min(candidates, key=len)


def _pack_compact_glue_items(glues: list[str]) -> str:
    packed: list[str] = []
    changed = False
    for glue in glues:
        item = _pack_single_glue_item(glue)
        packed.append(item)
        changed = changed or item != glue
    if not changed:
        return ""
    return "P:" + ".".join(quote(x, safe="") for x in packed)


def _unpack_compact_glue_items(payload: str) -> list[str] | None:
    if not payload:
        return []
    out: list[str] = []
    for raw in payload.split("."):
        item = unquote(raw)
        glue = _unpack_single_glue_item(item)
        if glue is None:
            return None
        out.append(glue)
    return out


def _pack_single_glue_item(glue: str) -> str:
    op, payload = _split_token(glue)
    if op != "MERGE_NODE_PATTERN":
        return glue
    family = _pack_pair_payload_family(payload)
    return family if family and len(family) < len(glue) else glue


def _unpack_single_glue_item(item: str) -> str | None:
    if not item.startswith("M:"):
        return item
    payload = _unpack_pair_payload_family(item)
    if payload is None:
        return None
    return f"MERGE_NODE_PATTERN({payload})"


def _single_glue_item_family_fields(item: str) -> dict[str, str] | None:
    if not item.startswith("M:") or _unpack_pair_payload_family(item) is None:
        return None
    parts = item[2:].split(",", 5)
    family = parts[0]
    try:
        if family == "I" and len(parts) == 4:
            count, local_start, local_step = map(int, parts[1:])
            return {"family": "I", "count": str(count), "local_start": str(local_start), "local_step": str(local_step)}
        if family == "O" and len(parts) == 5:
            count, local_start, local_step, delta = map(int, parts[1:])
            return {
                "family": "O",
                "count": str(count),
                "local_start": str(local_start),
                "local_step": str(local_step),
                "delta": str(delta),
            }
        if family == "V" and len(parts) == 5:
            count, local_start, local_step, pair_sum = map(int, parts[1:])
            return {
                "family": "V",
                "count": str(count),
                "local_start": str(local_start),
                "local_step": str(local_step),
                "pair_sum": str(pair_sum),
            }
        if family == "W" and len(parts) == 5:
            count, local_start, local_step, ref = map(int, parts[1:])
            return {
                "family": "W",
                "count": str(count),
                "local_start": str(local_start),
                "local_step": str(local_step),
                "ref": str(ref),
            }
        if family == "A" and len(parts) == 6:
            count, local_start, local_step, ref_start, ref_step = map(int, parts[1:])
            return {
                "family": "A",
                "count": str(count),
                "local_start": str(local_start),
                "local_step": str(local_step),
                "ref_start": str(ref_start),
                "ref_step": str(ref_step),
            }
        if family == "D" and len(parts) == 6:
            count = int(parts[1])
            int(parts[2])
            int(parts[3])
            return {
                "family": "D",
                "count": str(count),
                "local_start": parts[2],
                "ref_start": parts[3],
                "local_deltas": parts[4],
                "ref_deltas": parts[5],
            }
    except ValueError:
        return None
    return None


def _pack_pair_payload_family(payload: str) -> str:
    pairs = _parse_int_pairs(payload)
    if len(pairs) < 2:
        return ""
    local_steps = [pairs[i + 1][0] - pairs[i][0] for i in range(len(pairs) - 1)]
    ref_steps = [pairs[i + 1][1] - pairs[i][1] for i in range(len(pairs) - 1)]
    count = len(pairs)
    local_start = pairs[0][0]
    ref_start = pairs[0][1]
    if len(set(local_steps)) == 1:
        local_step = local_steps[0]
        if all(ref == local for local, ref in pairs):
            return f"M:I,{count},{local_start},{local_step}"
        offsets = [ref - local for local, ref in pairs]
        if len(set(offsets)) == 1:
            return f"M:O,{count},{local_start},{local_step},{offsets[0]}"
        reverse_sums = [ref + local for local, ref in pairs]
        if len(set(reverse_sums)) == 1:
            return f"M:V,{count},{local_start},{local_step},{reverse_sums[0]}"
        if len({ref for _, ref in pairs}) == 1:
            return f"M:W,{count},{local_start},{local_step},{ref_start}"
        if len(set(ref_steps)) == 1:
            return f"M:A,{count},{local_start},{local_step},{ref_start},{ref_steps[0]}"
    # Residual delta coding is useful for recent-window refs that are not
    # perfectly arithmetic. It is still reversible and often shorter than the
    # raw local:ref list.
    local_deltas = [str(x) for x in _deltas([p[0] for p in pairs])]
    ref_deltas = [str(x) for x in _deltas([p[1] for p in pairs])]
    residual = f"M:D,{count},{pairs[0][0]},{pairs[0][1]},{';'.join(local_deltas)},{';'.join(ref_deltas)}"
    raw = f"MERGE_NODE_PATTERN({payload})"
    return residual if len(residual) < len(raw) else ""


def _unpack_pair_payload_family(item: str) -> str | None:
    parts = item[2:].split(",", 5)
    if not parts:
        return None
    family = parts[0]
    try:
        if family == "I" and len(parts) == 4:
            count, local_start, local_step = map(int, parts[1:])
            pairs = [(local_start + local_step * i, local_start + local_step * i) for i in range(count)]
        elif family == "O" and len(parts) == 5:
            count, local_start, local_step, delta = map(int, parts[1:])
            pairs = [(local_start + local_step * i, local_start + local_step * i + delta) for i in range(count)]
        elif family == "V" and len(parts) == 5:
            count, local_start, local_step, pair_sum = map(int, parts[1:])
            pairs = [(local_start + local_step * i, pair_sum - (local_start + local_step * i)) for i in range(count)]
        elif family == "W" and len(parts) == 5:
            count, local_start, local_step, ref = map(int, parts[1:])
            pairs = [(local_start + local_step * i, ref) for i in range(count)]
        elif family == "A" and len(parts) == 6:
            count, local_start, local_step, ref_start, ref_step = map(int, parts[1:])
            pairs = [(local_start + local_step * i, ref_start + ref_step * i) for i in range(count)]
        elif family == "D" and len(parts) == 6:
            count = int(parts[1])
            local_start = int(parts[2])
            ref_start = int(parts[3])
            local_deltas = [int(x) for x in parts[4].split(";") if x]
            ref_deltas = [int(x) for x in parts[5].split(";") if x]
            if len(local_deltas) != max(0, count - 1) or len(ref_deltas) != max(0, count - 1):
                return None
            locals_ = _undeltas(local_start, local_deltas)
            refs = _undeltas(ref_start, ref_deltas)
            pairs = list(zip(locals_, refs))
        else:
            return None
    except ValueError:
        return None
    return ";".join(f"{local}:{ref}" for local, ref in pairs)


def _deltas(values: list[int]) -> list[int]:
    return [values[i] - values[i - 1] for i in range(1, len(values))]


def _undeltas(start: int, deltas: list[int]) -> list[int]:
    out = [start]
    for delta in deltas:
        out.append(out[-1] + delta)
    return out


def _pack_glue_family(glues: list[str]) -> str:
    if len(glues) < 2:
        return ""
    if len(set(glues)) == 1:
        return "F:R," + str(len(glues)) + "," + quote(glues[0], safe="")
    single_pairs: list[tuple[int, int]] = []
    for glue in glues:
        op, payload = _split_token(glue)
        if op != "MERGE_NODE_PATTERN":
            return ""
        pairs = _parse_int_pairs(payload)
        if len(pairs) != 1:
            return ""
        single_pairs.append(pairs[0])
    if len(single_pairs) < 2:
        return ""
    local_steps = [single_pairs[i + 1][0] - single_pairs[i][0] for i in range(len(single_pairs) - 1)]
    ref_steps = [single_pairs[i + 1][1] - single_pairs[i][1] for i in range(len(single_pairs) - 1)]
    if len(set(local_steps)) != 1 or len(set(ref_steps)) != 1:
        return ""
    count = len(single_pairs)
    local_start = single_pairs[0][0]
    local_step = local_steps[0]
    ref_start = single_pairs[0][1]
    ref_step = ref_steps[0]
    if all(ref == local for local, ref in single_pairs):
        return f"F:I,{count},{local_start},{local_step}"
    offsets = [ref - local for local, ref in single_pairs]
    if len(set(offsets)) == 1:
        return f"F:O,{count},{local_start},{local_step},{offsets[0]}"
    reverse_sums = [ref + local for local, ref in single_pairs]
    if len(set(reverse_sums)) == 1:
        return f"F:V,{count},{local_start},{local_step},{reverse_sums[0]}"
    if len({ref for _, ref in single_pairs}) == 1:
        return f"F:W,{count},{local_start},{local_step},{ref_start}"
    return f"F:A,{count},{local_start},{local_step},{ref_start},{ref_step}"


def _unpack_glue_family(payload: str) -> list[str] | None:
    if not payload.startswith("F:"):
        return None
    parts = payload[2:].split(",")
    if not parts:
        return None
    family = parts[0]
    try:
        if family == "R" and len(parts) == 3:
            return [unquote(parts[2])] * int(parts[1])
        if family == "I" and len(parts) == 4:
            count, local_start, local_step = map(int, parts[1:])
            return [f"MERGE_NODE_PATTERN({local_start + local_step * i}:{local_start + local_step * i})" for i in range(count)]
        if family == "O" and len(parts) == 5:
            count, local_start, local_step, delta = map(int, parts[1:])
            return [
                f"MERGE_NODE_PATTERN({local_start + local_step * i}:{local_start + local_step * i + delta})"
                for i in range(count)
            ]
        if family == "V" and len(parts) == 5:
            count, local_start, local_step, pair_sum = map(int, parts[1:])
            return [
                f"MERGE_NODE_PATTERN({local_start + local_step * i}:{pair_sum - (local_start + local_step * i)})"
                for i in range(count)
            ]
        if family == "W" and len(parts) == 5:
            count, local_start, local_step, ref = map(int, parts[1:])
            return [f"MERGE_NODE_PATTERN({local_start + local_step * i}:{ref})" for i in range(count)]
        if family == "A" and len(parts) == 6:
            count, local_start, local_step, ref_start, ref_step = map(int, parts[1:])
            return [
                f"MERGE_NODE_PATTERN({local_start + local_step * i}:{ref_start + ref_step * i})"
                for i in range(count)
            ]
    except ValueError:
        return None
    return None


def _glue_family_fields(payload: str) -> dict[str, str] | None:
    if not payload.startswith("F:"):
        return None
    parts = payload[2:].split(",")
    if not parts:
        return None
    family = parts[0]
    try:
        if family == "R" and len(parts) == 3:
            int(parts[1])
            return {"family": "R", "count": parts[1], "token": unquote(parts[2])}
        if family == "I" and len(parts) == 4:
            count, local_start, local_step = map(int, parts[1:])
            return {"family": "I", "count": str(count), "local_start": str(local_start), "local_step": str(local_step)}
        if family == "O" and len(parts) == 5:
            count, local_start, local_step, delta = map(int, parts[1:])
            return {
                "family": "O",
                "count": str(count),
                "local_start": str(local_start),
                "local_step": str(local_step),
                "delta": str(delta),
            }
        if family == "V" and len(parts) == 5:
            count, local_start, local_step, pair_sum = map(int, parts[1:])
            return {
                "family": "V",
                "count": str(count),
                "local_start": str(local_start),
                "local_step": str(local_step),
                "pair_sum": str(pair_sum),
            }
        if family == "W" and len(parts) == 5:
            count, local_start, local_step, ref = map(int, parts[1:])
            return {
                "family": "W",
                "count": str(count),
                "local_start": str(local_start),
                "local_step": str(local_step),
                "ref": str(ref),
            }
        if family == "A" and len(parts) == 6:
            count, local_start, local_step, ref_start, ref_step = map(int, parts[1:])
            return {
                "family": "A",
                "count": str(count),
                "local_start": str(local_start),
                "local_step": str(local_step),
                "ref_start": str(ref_start),
                "ref_step": str(ref_step),
            }
    except ValueError:
        return None
    return None


def learn_merge_schema_rules(
    sequences: list[list[str]],
    min_emit_count: int,
    max_span_len: int,
    max_rules: int,
    min_count: int,
) -> dict[str, list[str]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for seq in sequences:
        i = 0
        while i < len(seq):
            parts = _parameterized_parts(seq, i, min_emit_count, max_span_len)
            if parts is None:
                i += 1
                continue
            _, glues, _, end = parts
            if len(glues) >= 2:
                counts[tuple(glues)] += 1
            i = max(i + 1, end)
    rules: dict[str, list[str]] = {}
    for idx, (schema, count) in enumerate(
        sorted(counts.items(), key=lambda kv: ((len(kv[0]) - 1) * (kv[1] - 1), kv[1], len(kv[0])), reverse=True)
    ):
        if idx >= max_rules or count < min_count:
            break
        if (len(schema) - 1) * (count - 1) <= 0:
            continue
        rules[f"{idx:04d}"] = list(schema)
    return rules


def learn_code_schema_rules(
    sequences: list[list[str]],
    min_emit_count: int,
    max_span_len: int,
    max_rules: int,
    min_count: int,
) -> dict[str, list[str]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for seq in sequences:
        i = 0
        while i < len(seq):
            parts = _parameterized_parts(seq, i, min_emit_count, max_span_len)
            if parts is None:
                i += 1
                continue
            codes, _, _, end = parts
            if len(codes) >= 2 and len(set(codes)) > 1:
                counts[tuple(codes)] += 1
            i = max(i + 1, end)
    rules: dict[str, list[str]] = {}
    for idx, (schema, count) in enumerate(
        sorted(counts.items(), key=lambda kv: ((len(kv[0]) - 1) * (kv[1] - 1), kv[1], len(kv[0])), reverse=True)
    ):
        if idx >= max_rules or count < min_count:
            break
        if (len(schema) - 1) * (count - 1) <= 0:
            continue
        rules[f"{idx:04d}"] = list(schema)
    return rules


def _parameterized_parts(seq: list[str], start: int, min_emit_count: int, max_span_len: int) -> tuple[list[str], list[str], str, int] | None:
    op, payload = _split_token(seq[start])
    if op != "EMIT_CODE":
        return None
    codes = [payload]
    glues: list[str] = []
    j = start + 1
    glue_ops = {"MERGE_NODE_PATTERN", "MERGE_EDGE_PATTERN", "ATTACH_PATTERN"}
    while j + 1 < len(seq) and j - start < max_span_len:
        glue_op, _ = _split_token(seq[j])
        next_op, next_payload = _split_token(seq[j + 1])
        if glue_op not in glue_ops or next_op != "EMIT_CODE":
            break
        glues.append(seq[j])
        codes.append(next_payload)
        j += 2
    close = ""
    if j < len(seq):
        close_op, _ = _split_token(seq[j])
        if close_op == "CLOSE_CYCLE_PATTERN" and len(codes) >= min_emit_count:
            close = seq[j]
            j += 1
    if len(codes) < min_emit_count:
        return None
    return codes, glues, close, j


def _merge_arithmetic_family(op: str, payload: str) -> tuple[int, int, int, int, int] | None:
    if op != "MERGE_NODE_PATTERN":
        return None
    pairs = _parse_int_pairs(payload)
    if len(pairs) < 2:
        return None
    local_steps = [pairs[i + 1][0] - pairs[i][0] for i in range(len(pairs) - 1)]
    ref_steps = [pairs[i + 1][1] - pairs[i][1] for i in range(len(pairs) - 1)]
    if len(set(local_steps)) != 1 or len(set(ref_steps)) != 1:
        return None
    local_step = local_steps[0]
    ref_step = ref_steps[0]
    if local_step not in {-1, 0, 1} or ref_step not in {-1, 0, 1}:
        return None
    return len(pairs), pairs[0][0], local_step, pairs[0][1], ref_step


def _parse_int_pairs(payload: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in [p for p in payload.split(";") if p]:
        parts = item.split(":")
        if len(parts) != 2:
            return []
        try:
            pairs.append((int(parts[0]), int(parts[1])))
        except ValueError:
            return []
    return pairs


def _structural_spans(seq: list[str], shape_categories: dict[str, str], max_len: int) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    i = 0
    while i < len(seq):
        tok = seq[i]
        op, payload = _split_token(tok)
        if op != "EMIT_CODE":
            i += 1
            continue
        cat = shape_categories.get(payload, "motif")
        j = i + 1
        emit_count = 1
        while j < len(seq) and j - i < max_len:
            opj, payloadj = _split_token(seq[j])
            if opj == "EMIT_CODE":
                cat_j = shape_categories.get(payloadj, "motif")
                if cat_j != cat and "motif" not in {cat, cat_j}:
                    break
                emit_count += 1
            elif opj not in {"MERGE_NODE_PATTERN", "MERGE_EDGE_PATTERN", "ATTACH_PATTERN", "CLOSE_CYCLE_PATTERN", "INTERFACE_CODE"}:
                break
            j += 1
            if emit_count >= 2 and j - i >= 2:
                spans.append((i, j, _span_category(cat, emit_count, seq[i:j])))
        i += 1
    return spans


def _span_category(base_category: str, emit_count: int, span: list[str]) -> str:
    if emit_count >= 3 and base_category in {"edge", "chain", "treelet"}:
        return "edge_chain"
    if base_category in STRUCTURAL_RULE_PREFIX:
        return base_category
    if any(t.startswith("GLOBAL_LINK_GROUP(") for t in span):
        return "motif"
    return "motif"


def _shape_category(num_nodes: int, edges: tuple[tuple[int, int], ...]) -> str:
    n = int(num_nodes)
    e = len(edges)
    if n <= 0:
        return "empty"
    if n == 1:
        return "atom"
    if n == 2 and e == 1:
        return "edge"
    degrees = [0 for _ in range(n)]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for u, v in edges:
        if 0 <= int(u) < n and 0 <= int(v) < n:
            degrees[int(u)] += 1
            degrees[int(v)] += 1
            union(int(u), int(v))
    connected = len({find(i) for i in range(n)}) == 1
    if connected and e == n * (n - 1) // 2:
        return "clique"
    if connected and n >= 3 and e == n and all(d == 2 for d in degrees):
        return "cycle"
    if connected and n >= 3 and e == n - 1 and max(degrees) == n - 1 and degrees.count(1) == n - 1:
        return "star"
    if connected and e == n - 1 and max(degrees) <= 2:
        return "chain"
    if connected and e == n - 1:
        return "treelet"
    return "motif"


def _savings(gram: tuple[str, ...], count: int) -> int:
    return (len(gram) - 1) * (count - 1)


def _replace_ngram(seq: list[str], gram: tuple[str, ...], token: str) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(gram)
    while i < len(seq):
        if i + n <= len(seq) and tuple(seq[i : i + n]) == gram:
            out.append(token)
            i += n
        else:
            out.append(seq[i])
            i += 1
    return out


def _replace_pair(seq: list[str], pair: tuple[str, str], token: str) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(seq):
        if i + 1 < len(seq) and seq[i] == pair[0] and seq[i + 1] == pair[1]:
            out.append(token)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _expand_bpe(tok: str, rules: dict[str, tuple[str, str]]) -> list[str]:
    if tok not in rules:
        return [tok]
    left, right = rules[tok]
    return _expand_bpe(left, rules) + _expand_bpe(right, rules)


def _expand_structural(tok: str, rules: dict[str, list[str]]) -> list[str]:
    if tok not in rules:
        return [tok]
    out: list[str] = []
    for part in rules[tok]:
        out.extend(_expand_structural(part, rules))
    return out


def _expand_compact_macro(tok: str, rules: dict[str, list[str]]) -> list[str]:
    if tok not in rules:
        return [tok]
    out: list[str] = []
    for part in rules[tok]:
        out.extend(_expand_compact_macro(part, rules))
    return out


def _split_token(tok: str) -> tuple[str, str]:
    if "(" not in tok:
        return tok, ""
    op, rest = tok.split("(", 1)
    return op, rest.rstrip(")")


def _neglog(counts: Counter[str], key: str) -> float:
    total = sum(counts.values())
    vocab = max(1, len(counts))
    count = counts.get(key, 0)
    return -math.log2((count + 1.0) / (total + vocab + 1.0))


def _profile_rulebook_bits(profile: MotifMacroProfile) -> float:
    raw = 0
    for schema_id, parts in profile.code_schema_rules.items():
        raw += len(schema_id) + sum(len(p) for p in parts)
    for schema_id, parts in profile.merge_schema_rules.items():
        raw += len(schema_id) + sum(len(p) for p in parts)
    for token, parts in profile.structural_rules.items():
        raw += len(token) + sum(len(p) for p in parts)
    raw += len(str(profile.rule_categories))
    raw += len(str(profile.shape_categories))
    compact = profile.compact_profile
    for token, parts in compact.macro_rules.items():
        raw += len(token) + sum(len(p) for p in parts)
    for token, pair in compact.bpe_rules.items():
        raw += len(token) + len(pair[0]) + len(pair[1])
    return float(raw * 8)
