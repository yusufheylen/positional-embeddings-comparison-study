"""NoPE (No Positional Embeddings) attention implementation.

NoPE removes positional information from attention, relying on causal masking
and content-based attention only. Used as the target state in DroPE training.

Reference: Adapted from references/DroPE/custom_models/attention.py
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
)


class NoPEAttention(nn.Module):
    """Attention layer without positional embeddings.

    This removes the apply_rotary_pos_emb() call from standard LLaMA attention,
    passing Q/K directly to the attention computation.
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    @classmethod
    def from_source(cls, source_attention: LlamaAttention) -> "NoPEAttention":
        """Create NoPEAttention from an existing LlamaAttention layer.

        Args:
            source_attention: Source attention layer to copy weights from.

        Returns:
            NoPEAttention with copied weights.
        """
        nope_attn = cls(source_attention.config, source_attention.layer_idx)

        # Copy projection weights
        nope_attn.q_proj.weight.data.copy_(source_attention.q_proj.weight.data)
        nope_attn.k_proj.weight.data.copy_(source_attention.k_proj.weight.data)
        nope_attn.v_proj.weight.data.copy_(source_attention.v_proj.weight.data)
        nope_attn.o_proj.weight.data.copy_(source_attention.o_proj.weight.data)

        return nope_attn

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass without positional embeddings.

        Key difference from standard attention: No rotary position embedding applied.
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV for GQA
        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute attention (NO positional encoding applied)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / (self.head_dim**0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def convert_to_nope(model: nn.Module) -> nn.Module:
    """Convert a model's attention layers to NoPE (no positional embeddings).

    Args:
        model: Model with standard attention layers.

    Returns:
        Model with NoPE attention layers.
    """
    for layer in model.model.layers:
        if hasattr(layer, "self_attn") and isinstance(layer.self_attn, LlamaAttention):
            layer.self_attn = NoPEAttention.from_source(layer.self_attn)

    return model
