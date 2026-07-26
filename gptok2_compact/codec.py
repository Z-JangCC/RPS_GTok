"""Reversible GPTok2 program compaction and entropy-cost estimation."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any


PAIR_OPS = {"ATTACH", "MERGE_NODE", "MERGE_EDGE", "CLOSE_CYCLE"}
GROUP_NAMES = {
    "ATTACH": "ATTACH_PATTERN",
    "MERGE_NODE": "MERGE_NODE_PATTERN",
    "MERGE_EDGE": "MERGE_EDGE_PATTERN",
    "CLOSE_CYCLE": "CLOSE_CYCLE_PATTERN",
    "GLOBAL_LINK": "GLOBAL_LINK_GROUP",
}
REVERSE_GROUP_NAMES = {v: k for k, v in GROUP_NAMES.items()}


@dataclass
class CompactProfile:
    macro_rules: dict[str, list[str]]
    bpe_rules: dict[str, tuple[str, str]]
    train_original_tokens: int
    train_compact_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "macro_rules": self.macro_rules,
            "bpe_rules": {k: list(v) for k, v in self.bpe_rules.items()},
            "train_original_tokens": self.train_original_tokens,
            "train_compact_tokens": self.train_compact_tokens,
        }

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "CompactProfile":
        return CompactProfile(
            macro_rules={str(k): list(v) for k, v in row.get("macro_rules", {}).items()},
            bpe_rules={str(k): tuple(v) for k, v in row.get("bpe_rules", {}).items()},
            train_original_tokens=int(row.get("train_original_tokens", 0)),
            train_compact_tokens=int(row.get("train_compact_tokens", 0)),
        )


class CompactCodec:
    """A reversible token-level codec.

    The codec never changes graph semantics. `encode()` returns compact tokens;
    `decode()` expands them back to the exact original GPTok2 action tokens.
    """

    def __init__(
        self,
        max_macros: int = 256,
        min_macro_count: int = 3,
        max_macro_len: int = 10,
        max_bpe_merges: int = 256,
        min_bpe_count: int = 3,
    ):
        self.max_macros = int(max_macros)
        self.min_macro_count = int(min_macro_count)
        self.max_macro_len = int(max_macro_len)
        self.max_bpe_merges = int(max_bpe_merges)
        self.min_bpe_count = int(min_bpe_count)
        self.profile = CompactProfile({}, {}, 0, 0)

    def fit(self, train_sequences: list[list[str]]) -> "CompactCodec":
        base = [base_compact_tokens(seq) for seq in train_sequences]
        macro_rules, after_macro = learn_macro_rules(
            base,
            max_macros=self.max_macros,
            min_count=self.min_macro_count,
            max_len=self.max_macro_len,
        )
        bpe_rules = learn_bpe_rules(after_macro, max_merges=self.max_bpe_merges, min_count=self.min_bpe_count)
        compact = [apply_bpe_rules(apply_macro_rules(seq, macro_rules), bpe_rules) for seq in base]
        self.profile = CompactProfile(
            macro_rules=macro_rules,
            bpe_rules=bpe_rules,
            train_original_tokens=sum(len(s) for s in train_sequences),
            train_compact_tokens=sum(len(s) for s in compact),
        )
        return self

    def encode(self, tokens: list[str]) -> list[str]:
        seq = base_compact_tokens(tokens)
        seq = apply_macro_rules(seq, self.profile.macro_rules)
        seq = apply_bpe_rules(seq, self.profile.bpe_rules)
        return seq

    def decode(self, compact_tokens: list[str]) -> list[str]:
        seq: list[str] = []
        for tok in compact_tokens:
            seq.extend(_expand_bpe(tok, self.profile.bpe_rules))
        expanded_macros: list[str] = []
        for tok in seq:
            expanded_macros.extend(_expand_macro(tok, self.profile.macro_rules))
        return expand_base_tokens(expanded_macros)


def base_compact_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        op, payload = _split_token(tok)
        if op in GROUP_NAMES:
            pairs = []
            while i < len(tokens):
                op2, payload2 = _split_token(tokens[i])
                if op2 != op:
                    break
                pairs.append(payload2.replace(",", ":"))
                i += 1
            out.append(f"{GROUP_NAMES[op]}(" + ";".join(pairs) + ")")
            continue
        if op == "EMIT":
            out.append(f"EMIT_CODE({payload})")
        elif op == "INTERFACE":
            out.append(f"INTERFACE_CODE({payload})")
        else:
            out.append(tok)
        i += 1
    return compact_repeated_emit_merge(out)


def expand_base_tokens(tokens: list[str]) -> list[str]:
    out: list[str] = []
    for tok in tokens:
        op, payload = _split_token(tok)
        if op in REVERSE_GROUP_NAMES:
            original_op = REVERSE_GROUP_NAMES[op]
            for pair in [p for p in payload.split(";") if p]:
                out.append(f"{original_op}(" + pair.replace(":", ",") + ")")
        elif op == "REPEAT_EMIT_MERGE":
            code, pattern, count = payload.split("|")
            for _ in range(int(count)):
                out.append(f"EMIT({code})")
                for pair in [p for p in pattern.split(";") if p]:
                    out.append("MERGE_NODE(" + pair.replace(":", ",") + ")")
        elif op == "EMIT_CODE":
            out.append(f"EMIT({payload})")
        elif op == "INTERFACE_CODE":
            out.append(f"INTERFACE({payload})")
        else:
            out.append(tok)
    return out


def compact_repeated_emit_merge(tokens: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and tokens[i].startswith("EMIT_CODE(") and tokens[i + 1].startswith("MERGE_NODE_PATTERN("):
            code = _payload(tokens[i])
            pattern = _payload(tokens[i + 1])
            count = 0
            j = i
            while j + 1 < len(tokens) and tokens[j] == tokens[i] and tokens[j + 1] == tokens[i + 1]:
                count += 1
                j += 2
            if count >= 2:
                out.append(f"REPEAT_EMIT_MERGE({code}|{pattern}|{count})")
                i = j
                continue
        out.append(tokens[i])
        i += 1
    return out


def learn_macro_rules(sequences: list[list[str]], max_macros: int, min_count: int, max_len: int) -> tuple[dict[str, list[str]], list[list[str]]]:
    work = [list(seq) for seq in sequences]
    rules: dict[str, list[str]] = {}
    for idx in range(max_macros):
        counts: Counter[tuple[str, ...]] = Counter()
        for seq in work:
            for n in range(3, max_len + 1):
                if len(seq) < n:
                    continue
                for i in range(len(seq) - n + 1):
                    gram = tuple(seq[i : i + n])
                    if _valid_macro_ngram(gram):
                        counts[gram] += 1
        if not counts:
            break
        gram, count = max(counts.items(), key=lambda kv: ((len(kv[0]) - 1) * (kv[1] - 1), kv[1], len(kv[0])))
        if count < min_count or (len(gram) - 1) * (count - 1) <= 0:
            break
        token = f"MOTIF_MACRO_{idx:04d}"
        rules[token] = list(gram)
        work = [_replace_ngram(seq, gram, token) for seq in work]
    return rules, work


def _valid_macro_ngram(gram: tuple[str, ...]) -> bool:
    if any(t in {"BEGIN_GRAPH", "END_GRAPH", "STOP"} for t in gram):
        return False
    return any(t.startswith(("EMIT_CODE(", "REPEAT_EMIT_MERGE(", "MERGE_NODE_PATTERN(", "ATTACH_PATTERN(", "GLOBAL_LINK_GROUP(")) for t in gram)


def apply_macro_rules(tokens: list[str], rules: dict[str, list[str]]) -> list[str]:
    seq = list(tokens)
    for token, parts in rules.items():
        seq = _replace_ngram(seq, tuple(parts), token)
    return seq


def learn_bpe_rules(sequences: list[list[str]], max_merges: int, min_count: int) -> dict[str, tuple[str, str]]:
    work = [list(seq) for seq in sequences]
    rules: dict[str, tuple[str, str]] = {}
    for idx in range(max_merges):
        counts: Counter[tuple[str, str]] = Counter()
        for seq in work:
            counts.update(zip(seq, seq[1:]))
        if not counts:
            break
        pair, count = counts.most_common(1)[0]
        if count < min_count:
            break
        token = f"ACTION_BPE_{idx:04d}"
        rules[token] = pair
        work = [_replace_pair(seq, pair, token) for seq in work]
    return rules


def apply_bpe_rules(tokens: list[str], rules: dict[str, tuple[str, str]]) -> list[str]:
    seq = list(tokens)
    for token, pair in rules.items():
        seq = _replace_pair(seq, pair, token)
    return seq


class EntropyModel:
    """Component entropy model for compact GPTok2 programs."""

    def __init__(self):
        self.token_counts: Counter[str] = Counter()
        self.op_counts: Counter[str] = Counter()
        self.symbol_counts: Counter[str] = Counter()
        self.code_counts: Counter[str] = Counter()
        self.local_counts: Counter[str] = Counter()
        self.ref_delta_counts: Counter[str] = Counter()
        self.count_counts: Counter[str] = Counter()
        self.num_sequences = 0
        self.rulebook_bits = 0.0

    def fit(self, compact_sequences: list[list[str]], profile: CompactProfile | None = None) -> "EntropyModel":
        self.num_sequences = len(compact_sequences)
        for seq in compact_sequences:
            for tok in seq:
                self._observe(tok)
        if profile is not None:
            self.rulebook_bits = _profile_rulebook_bits(profile)
        return self

    def bits(self, compact_tokens: list[str], include_rulebook: bool = False) -> float:
        total = 0.0
        for tok in compact_tokens:
            total += self._token_bits(tok)
        if include_rulebook and self.num_sequences:
            total += self.rulebook_bits / max(1, self.num_sequences)
        return total

    def to_dict(self) -> dict[str, Any]:
        return {
            "op_counts": dict(self.op_counts),
            "token_counts": dict(self.token_counts),
            "symbol_counts": dict(self.symbol_counts),
            "code_counts": dict(self.code_counts),
            "local_counts": dict(self.local_counts),
            "ref_delta_counts": dict(self.ref_delta_counts),
            "count_counts": dict(self.count_counts),
            "num_sequences": self.num_sequences,
            "rulebook_bits": self.rulebook_bits,
        }

    def _observe(self, tok: str) -> None:
        op, payload = _split_token(tok)
        self.token_counts[tok] += 1
        self.op_counts[op] += 1
        if op.startswith(("ACTION_BPE_", "MOTIF_MACRO_")):
            self.symbol_counts[tok] += 1
            return
        if op in {"EMIT_CODE", "INTERFACE_CODE"}:
            self.code_counts[payload] += 1
            return
        if op == "REPEAT_EMIT_MERGE":
            code, pattern, count = payload.split("|")
            self.code_counts[code] += 1
            self.count_counts[count] += 1
            self._observe_pairs(pattern)
            return
        if op in REVERSE_GROUP_NAMES:
            self._observe_pairs(payload)

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

    def _token_bits(self, tok: str) -> float:
        op, payload = _split_token(tok)
        whole_bits = _neglog(self.token_counts, tok)
        bits = _neglog(self.op_counts, op)
        if op.startswith(("ACTION_BPE_", "MOTIF_MACRO_")):
            return min(whole_bits, bits + _neglog(self.symbol_counts, tok))
        if op in {"EMIT_CODE", "INTERFACE_CODE"}:
            return min(whole_bits, bits + _neglog(self.code_counts, payload))
        if op == "REPEAT_EMIT_MERGE":
            code, pattern, count = payload.split("|")
            component_bits = bits + _neglog(self.code_counts, code) + _neglog(self.count_counts, count) + self._pairs_bits(pattern)
            return min(whole_bits, component_bits)
        if op in REVERSE_GROUP_NAMES:
            return min(whole_bits, bits + self._pairs_bits(payload))
        return min(whole_bits, bits)

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


def original_program_bits(tokens: list[str], codebook_size: int, ref_window: int) -> float:
    bits = 0.0
    n = max(2, len(tokens))
    for tok in tokens:
        op, _ = _split_token(tok)
        bits += 1.5
        if op in {"EMIT", "INTERFACE"}:
            bits += math.log2(max(2, codebook_size))
        elif op in PAIR_OPS:
            bits += math.log2(max(2, ref_window)) + 4.0
        elif op == "GLOBAL_LINK":
            bits += 2 * math.log2(n)
    return bits


def compact_symbol_bits(tokens: list[str], vocab_size: int | None = None) -> float:
    vocab = max(2, int(vocab_size) if vocab_size is not None else len(set(tokens)))
    return len(tokens) * math.log2(vocab)


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


def _expand_macro(tok: str, rules: dict[str, list[str]]) -> list[str]:
    if tok not in rules:
        return [tok]
    out: list[str] = []
    for part in rules[tok]:
        out.extend(_expand_macro(part, rules))
    return out


def _split_token(tok: str) -> tuple[str, str]:
    if "(" not in tok:
        return tok, ""
    op, rest = tok.split("(", 1)
    return op, rest.rstrip(")")


def _payload(tok: str) -> str:
    return _split_token(tok)[1]


def _neglog(counts: Counter[str], key: str) -> float:
    total = sum(counts.values())
    vocab = max(1, len(counts))
    count = counts.get(key, 0)
    return -math.log2((count + 1.0) / (total + vocab + 1.0))


def _profile_rulebook_bits(profile: CompactProfile) -> float:
    raw = 0
    for token, parts in profile.macro_rules.items():
        raw += len(token) + sum(len(p) for p in parts)
    for token, pair in profile.bpe_rules.items():
        raw += len(token) + len(pair[0]) + len(pair[1])
    return float(raw * 8)
