"""Positional embedding implementations.

Supported PE methods:
- NoPE: No Positional Embeddings (with optional QK normalization)
- RoPE: Rotary Position Embeddings (HuggingFace native)
- YaRN: Yet another RoPE extensioN (HuggingFace rope_scaling)
- PoPE: Polar Positional Embeddings
"""

from .nope import (
    NoPELlamaAttention,
    QKNormNoPELlamaAttention,
    QNormNoPELlamaAttention,
    KNormNoPELlamaAttention,
    NOPE_ATTENTION_VARIANTS,
    convert_to_nope,
    nope,
    qk_norm_nope,
)
from .pope import (
    PolarEmbedding,
    PoPELlamaAttention,
    POPE_ATTENTION_VARIANTS,
    apply_polar_pos_emb,
    convert_to_pope,
    pope_attention,
)
from .rope import apply_rotary_pos_emb, rotate_half
from .yarn import get_yarn_config, get_yarn_scaling_dict

__all__ = [
    # NoPE
    "NoPELlamaAttention",
    "QKNormNoPELlamaAttention",
    "QNormNoPELlamaAttention",
    "KNormNoPELlamaAttention",
    "NOPE_ATTENTION_VARIANTS",
    "convert_to_nope",
    "nope",
    "qk_norm_nope",
    # PoPE
    "PolarEmbedding",
    "PoPELlamaAttention",
    "POPE_ATTENTION_VARIANTS",
    "apply_polar_pos_emb",
    "convert_to_pope",
    "pope_attention",
    # RoPE
    "apply_rotary_pos_emb",
    "rotate_half",
    # YaRN
    "get_yarn_config",
    "get_yarn_scaling_dict",
]
