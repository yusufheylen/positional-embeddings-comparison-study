"""RoPE (Rotary Position Embeddings) utilities.

RoPE is the default positional embedding in LLaMA/SmolLM models.
This module provides utilities for working with HuggingFace's native RoPE.

Reference: https://arxiv.org/abs/2104.09864
"""

import torch
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary position embeddings to query and key tensors.

    This is a wrapper around HuggingFace's implementation for consistency.

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim).
        k: Key tensor of shape (batch, num_heads, seq_len, head_dim).
        cos: Cosine component of rotary embeddings.
        sin: Sine component of rotary embeddings.
        position_ids: Optional position indices.
        unsqueeze_dim: Dimension to unsqueeze cos/sin tensors.

    Returns:
        Tuple of (rotated_q, rotated_k).
    """
    return hf_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input.

    Args:
        x: Input tensor of shape (..., head_dim).

    Returns:
        Tensor with second half negated and swapped with first half.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
