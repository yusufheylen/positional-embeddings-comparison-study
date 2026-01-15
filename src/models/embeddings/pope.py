"""PoPE (Polar Positional Embeddings) implementation.

PoPE represents attention in polar coordinates, disentangling magnitude (what)
from phase (where) for improved length generalization.

Key differences from RoPE:
- Uses d frequencies (not d/2 pairs like RoPE)
- Applies softplus to get non-negative magnitudes
- Output is Cartesian form: [t * cos(θ), t * sin(θ)] (doubles dimension)
- Learnable per-head phase bias for keys

Reference:
- Paper: https://arxiv.org/abs/2509.10534
- Implementation: references/x-transformers/x_transformers/x_transformers.py:782-824
"""

import inspect
import logging
import math
from typing import Any, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import LlamaAttention

logger = logging.getLogger(__name__)


class PolarEmbedding(nn.Module):
    """Polar positional embedding module.

    Generates frequency-based positional information with learnable per-head bias.
    Reference: x-transformers/x_transformers.py:782-812
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        base: float = 10000.0,
        bias_init_zero: bool = True,
    ):
        """Initialize PolarEmbedding.

        Args:
            dim: Head dimension (uses full dim, not dim/2 like RoPE).
            num_heads: Number of attention heads.
            base: Base for frequency computation.
            bias_init_zero: If True, initialize bias to 0 for length generalization.
        """
        super().__init__()

        # Frequencies: inv_freq[c] = 1 / (base^(c/dim)) for c in [0, dim)
        # Note: PoPE uses d frequencies, not d/2 like RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Learnable per-head bias: delta_c in [-2π, 0]
        # Initialize to 0 for better length generalization
        self.learned_bias = nn.Parameter(torch.zeros(num_heads, 1, dim))

        if not bias_init_zero:
            self.learned_bias.data.uniform_(-2.0 * math.pi, 0.0)

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute positional frequencies and bias.

        Args:
            positions: Position indices of shape (batch, seq_len) or (seq_len,).

        Returns:
            Tuple of (freqs, bias):
            - freqs: (batch, seq_len, dim) positional frequencies
            - bias: (num_heads, 1, dim) learnable bias clamped to [-2π, 0]
        """
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)

        # freqs[b, i, c] = positions[b, i] * inv_freq[c]
        freqs = torch.einsum("b i, j -> b i j", positions.float(), self.inv_freq)

        # Clamp bias to valid range
        bias = self.learned_bias.clamp(-2.0 * math.pi, 0.0)

        return freqs, bias


@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
def apply_polar_pos_emb(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply polar positional embeddings to a tensor.

    Converts to Cartesian form: [μ * cos(θ), μ * sin(θ)]
    where μ = softplus(t) ensures non-negative magnitude.

    Args:
        t: Input tensor of shape (..., dim).
        freqs: Frequency tensor of shape (..., dim).

    Returns:
        Tensor of shape (..., 2*dim) in Cartesian form.
    """
    seq_len = t.shape[-2]
    freqs = freqs[:, -seq_len:]  # Handle offset

    # Softplus for non-negative magnitude
    t = F.softplus(t)

    # Convert to Cartesian form (doubles dimension)
    out = torch.cat((t * freqs.cos(), t * freqs.sin()), dim=-1)

    return out


def pope_attention(BaseAttentionClass: Type[nn.Module]) -> Type[nn.Module]:
    """Factory function to create PoPE attention from a base attention class.

    Args:
        BaseAttentionClass: Base attention class (e.g., LlamaAttention).

    Returns:
        Attention class with PoPE positional embeddings.
    """

    class PoPEAttention(BaseAttentionClass):
        """Attention with Polar Positional Embeddings (PoPE)."""

        def __init__(self, *args, **kwargs):
            # Get config before super().__init__
            signature = inspect.signature(super().__init__)
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()
            config = bound_args.arguments["config"]

            super().__init__(*args, **kwargs)

            # Add PoPE embedding - bias is per KV head for GQA compatibility
            self.polar_emb = PolarEmbedding(
                dim=self.head_dim,
                num_heads=config.num_key_value_heads,  # Use KV heads for bias
                base=getattr(config, "rope_theta", 10000.0),
                bias_init_zero=True,
            )

            # Adjust scaling for doubled dimension
            # Standard: sqrt(head_dim), PoPE: sqrt(2 * head_dim)
            self.pope_scaling = 1.0 / math.sqrt(2 * self.head_dim)

        def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_value: Optional[Any] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """Forward pass with PoPE attention.

            Note: This implementation uses eager attention for simplicity.
            PoPE's dimension doubling is not directly compatible with Flash Attention.
            """
            bsz, seq_len, _ = hidden_states.size()
            device = hidden_states.device

            # Get head counts from config (transformers 5.x compatibility)
            num_heads = self.config.num_attention_heads
            num_kv_heads = self.config.num_key_value_heads
            num_kv_groups = num_heads // num_kv_heads

            # Project Q, K, V
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            # Reshape for multi-head attention
            query_states = query_states.view(bsz, seq_len, num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, seq_len, num_kv_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, seq_len, num_kv_heads, self.head_dim).transpose(1, 2)

            # Get position indices
            if cache_position is not None:
                positions = cache_position
            else:
                positions = torch.arange(seq_len, device=device)

            # Compute PoPE frequencies and bias
            freqs, bias = self.polar_emb(positions)

            # Expand freqs for batch and heads: (batch, 1, seq, dim) -> broadcast to (batch, heads, seq, dim)
            freqs = freqs.unsqueeze(1)

            # Apply PoPE to Q (no bias) and K (with bias)
            # After this, Q and K have shape (batch, heads, seq, 2*head_dim)
            query_states = apply_polar_pos_emb(query_states, freqs)
            # For K, add per-head bias: bias has shape (heads, 1, dim)
            key_freqs = freqs + bias.unsqueeze(0)  # (batch, heads, seq, dim)
            key_states = apply_polar_pos_emb(key_states, key_freqs)

            # Repeat KV for grouped query attention if needed
            if num_kv_groups > 1:
                key_states = key_states.repeat_interleave(num_kv_groups, dim=1)
                value_states = value_states.repeat_interleave(num_kv_groups, dim=1)

            # Compute attention scores with adjusted scaling
            # Q and K now have shape (batch, heads, seq, 2*head_dim)
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.pope_scaling

            # Apply attention mask
            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask

            # Softmax
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
            attn_weights = F.dropout(attn_weights, p=getattr(self, 'attention_dropout', 0.0) if self.training else 0.0)

            # Apply attention to values
            # Note: V has original dimension, output also has original dimension
            attn_output = torch.matmul(attn_weights, value_states)

            # Reshape and project output
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
            attn_output = self.o_proj(attn_output)

            return attn_output, attn_weights

        @classmethod
        def from_source(cls, source_module: nn.Module, config: Any = None) -> "PoPEAttention":
            """Create PoPEAttention from an existing attention module."""
            config = source_module.config
            layer_idx = source_module.layer_idx
            device = source_module.o_proj.weight.device

            new_module = cls(config=config, layer_idx=layer_idx).to(device)
            # Load weights (strict=False because we have extra parameters)
            new_module.load_state_dict(source_module.state_dict(), strict=False)

            return new_module

    PoPEAttention.__name__ = f"PoPE{BaseAttentionClass.__name__}"
    return PoPEAttention


# Pre-built PoPE attention class
PoPELlamaAttention = pope_attention(LlamaAttention)

# Attention variants map
POPE_ATTENTION_VARIANTS = {
    LlamaAttention: {
        "pope": PoPELlamaAttention,
    },
}


def convert_to_pope(model: nn.Module) -> nn.Module:
    """Convert a model's attention layers to PoPE.

    Args:
        model: Model with standard attention layers.

    Returns:
        Model with PoPE attention layers.
    """
    # Get the model core
    model_core = getattr(model, model.base_model_prefix, model)

    for i, layer in enumerate(model_core.layers):
        if hasattr(layer, "self_attn"):
            original_attn = layer.self_attn
            base_class = type(original_attn)

            if base_class in POPE_ATTENTION_VARIANTS:
                AttentionClass = POPE_ATTENTION_VARIANTS[base_class]["pope"]
                new_attn = AttentionClass.from_source(original_attn, model.config)
                layer.self_attn = new_attn
                logger.info(f"Layer {i}: Converted to {new_attn.__class__.__name__}")
            else:
                logger.warning(f"No PoPE variant for {base_class.__name__}")

    return model
