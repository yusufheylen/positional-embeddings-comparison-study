"""PoPE (Polar Positional Embeddings) implementation.

PoPE represents attention in polar coordinates, disentangling magnitude (what)
from phase (where) for improved length generalization.

Reference:
- Paper: https://arxiv.org/abs/2509.10534
- Implementation: references/x-transformers/x_transformers/x_transformers.py:782-812
- Formulas: references/PoPE/arXiv-2509.10534v2/sections/method.tex
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig


class PolarEmbedding(nn.Module):
    """Polar positional embedding module.

    Generates frequency-based positional information with learnable per-head bias.
    """

    def __init__(self, dim: int, num_heads: int, base: float = 10000.0, bias_init_zero: bool = True):
        """Initialize PolarEmbedding.

        Args:
            dim: Head dimension (full dim, not dim/2 like RoPE).
            num_heads: Number of attention heads.
            base: Base for frequency computation.
            bias_init_zero: If True, init bias to 0 for length generalization.
        """
        super().__init__()

        # Frequencies: theta_c = base^((c-1)/d) for c in [0, dim)
        # Note: PoPE uses d frequencies, not d/2 like RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Learnable per-head bias: delta_c in [-2pi, 0]
        self.learned_bias = nn.Parameter(torch.zeros(num_heads, 1, dim))
        if not bias_init_zero:
            self.learned_bias.data.uniform_(-2.0 * math.pi, 0.0)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute positional frequencies and bias.

        Args:
            seq_len: Sequence length.
            device: Target device.

        Returns:
            Tuple of (freqs, bias) where:
            - freqs: (seq_len, dim) position-dependent frequencies
            - bias: (num_heads, 1, dim) learnable bias clamped to [-2pi, 0]
        """
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, dim)
        bias = self.learned_bias.clamp(-2.0 * math.pi, 0.0)
        return freqs, bias


class PoPEAttention(nn.Module):
    """Attention layer with Polar Positional Embeddings.

    Key differences from RoPE:
    1. Uses d frequencies (not d/2 pairs)
    2. Disentangled magnitude (softplus) and phase (position)
    3. Learnable per-head phase bias
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

        # PoPE-specific: polar embedding per head
        self.polar_emb = PolarEmbedding(self.head_dim, self.num_heads, bias_init_zero=True)

    @classmethod
    def from_source(cls, source_attention: LlamaAttention) -> "PoPEAttention":
        """Create PoPEAttention from an existing LlamaAttention layer."""
        pope_attn = cls(source_attention.config, source_attention.layer_idx)

        pope_attn.q_proj.weight.data.copy_(source_attention.q_proj.weight.data)
        pope_attn.k_proj.weight.data.copy_(source_attention.k_proj.weight.data)
        pope_attn.v_proj.weight.data.copy_(source_attention.v_proj.weight.data)
        pope_attn.o_proj.weight.data.copy_(source_attention.o_proj.weight.data)

        return pope_attn

    def _apply_polar_pos_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        freqs: torch.Tensor,
        bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply polar positional embeddings to Q and K.

        Converts to Cartesian form for efficient computation:
        x_q = mu_q * cos(t * theta)
        y_q = mu_q * sin(t * theta)
        x_k = mu_k * cos(s * theta + delta)
        y_k = mu_k * sin(s * theta + delta)

        Args:
            q: Query tensor (batch, heads, seq, dim).
            k: Key tensor (batch, heads, seq, dim).
            freqs: Position frequencies (seq, dim).
            bias: Learnable bias (heads, 1, dim).

        Returns:
            Tuple of (q_polar, k_polar) in Cartesian form.
        """
        # Magnitude via softplus (ensures non-negative)
        mu_q = F.softplus(q)  # (batch, heads, seq, dim)
        mu_k = F.softplus(k)

        # Expand freqs for broadcasting: (1, 1, seq, dim)
        freqs = freqs.unsqueeze(0).unsqueeze(0)

        # Query: no bias
        q_real = mu_q * freqs.cos()
        q_imag = mu_q * freqs.sin()

        # Key: with learnable bias
        k_freqs = freqs + bias.unsqueeze(0)  # Add head-specific bias
        k_real = mu_k * k_freqs.cos()
        k_imag = mu_k * k_freqs.sin()

        # Stack real/imag as last dimension for attention computation
        q_polar = torch.stack([q_real, q_imag], dim=-1)  # (batch, heads, seq, dim, 2)
        k_polar = torch.stack([k_real, k_imag], dim=-1)

        return q_polar, k_polar

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
        """Forward pass with PoPE attention."""
        batch_size, seq_len, _ = hidden_states.size()

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Get polar frequencies and bias
        freqs, bias = self.polar_emb(seq_len, hidden_states.device)

        # Apply polar positional embeddings
        q_polar, k_polar = self._apply_polar_pos_emb(query_states, key_states, freqs, bias)

        # Compute attention scores: Re[q^H * k] = sum(q_real*k_real + q_imag*k_imag)
        # Shape: (batch, heads, seq_q, seq_k)
        attn_weights = (q_polar * k_polar.transpose(2, 3).unsqueeze(2)).sum(dim=(-1, -2))
        attn_weights = attn_weights / (self.head_dim**0.5)

        # Handle KV cache (simplified - full implementation needs polar cache)
        if past_key_value is not None:
            # Note: Full implementation should cache polar-transformed K/V
            pass

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV for GQA
        if self.num_key_value_groups > 1:
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

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


def convert_to_pope(model: nn.Module) -> nn.Module:
    """Convert a model's attention layers to PoPE.

    Args:
        model: Model with standard attention layers.

    Returns:
        Model with PoPE attention layers.
    """
    for layer in model.model.layers:
        if hasattr(layer, "self_attn") and isinstance(layer.self_attn, LlamaAttention):
            layer.self_attn = PoPEAttention.from_source(layer.self_attn)

    return model
