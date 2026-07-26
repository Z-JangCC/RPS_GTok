"""Standalone motif-macro compression layer for GPTok2 programs."""

from gptok2_motif_macro.codec import (
    MotifEntropyModel,
    MotifMacroCodec,
    MotifMacroProfile,
    code_shape_categories,
    motif_symbol_bits,
)

__all__ = [
    "MotifEntropyModel",
    "MotifMacroCodec",
    "MotifMacroProfile",
    "code_shape_categories",
    "motif_symbol_bits",
]
