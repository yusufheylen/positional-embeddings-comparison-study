"""NoPE (No Positional Embeddings) attention implementation.

NoPE removes positional information from attention by nullifying RoPE -
setting cos=1 and sin=0 (identity rotation). This approach works with
any attention implementation (Flash Attention, SDPA, eager).

Reference: Adapted from references/DroPE/custom_models/attention.py
"""

import inspect
import logging
from typing import Any, Callable, Optional, Type

import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
    eager_attention_forward as llama_eager_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

logger = logging.getLogger(__name__)


def nope(BaseAttentionClass: Type[nn.Module]) -> Type[nn.Module]:
    """Factory function that creates a NoPE attention class from a base attention class.

    This works by intercepting the forward pass and nullifying RoPE by setting
    cos=1 and sin=0 in the position_embeddings argument.

    Args:
        BaseAttentionClass: The base attention class (e.g., LlamaAttention).

    Returns:
        A new attention class with RoPE disabled.
    """

    class NoPEAttention(BaseAttentionClass):
        """Attention module with RoPE disabled (No Positional Embeddings)."""

        def forward(self, *args, **kwargs):
            # Bind arguments to get access by name
            signature = inspect.signature(super().forward)
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Nullify RoPE by setting cos=1, sin=0 (identity rotation)
            if "position_embeddings" in bound_args.arguments:
                cos, sin = bound_args.arguments["position_embeddings"]
                bound_args.arguments["position_embeddings"] = (
                    torch.ones_like(cos),
                    torch.zeros_like(sin),
                )
                logger.debug("NoPE: Nullified position_embeddings")
            else:
                raise NotImplementedError(
                    "NoPE only supports models that pass position_embeddings to attention. "
                    "This model architecture may not be supported."
                )

            return super().forward(*bound_args.args, **bound_args.kwargs)

        @classmethod
        def from_source(cls, source_module: nn.Module, config: Any = None) -> "NoPEAttention":
            """Create NoPEAttention from an existing attention module."""
            config = source_module.config
            layer_idx = source_module.layer_idx
            device = source_module.o_proj.weight.device
            dtype = source_module.o_proj.weight.dtype

            new_module = cls(config=config, layer_idx=layer_idx).to(device=device, dtype=dtype)
            new_module.load_state_dict(source_module.state_dict())

            # Handle custom softmax scale if present
            if hasattr(config, "softmax_scale"):
                new_module.scaling = config.softmax_scale

            return new_module

    NoPEAttention.__name__ = f"NoPE{BaseAttentionClass.__name__}"
    return NoPEAttention


def qk_norm_nope(
    BaseAttentionClass: Type[nn.Module],
    forward_fn: Callable,
) -> Type[nn.Module]:
    """Factory for NoPE attention with QK normalization.

    Args:
        BaseAttentionClass: The base attention class.
        forward_fn: Custom forward function with normalization.

    Returns:
        Attention class with QK normalization and NoPE.
    """

    class QKNormNoPEAttention(BaseAttentionClass):
        """NoPE attention with query/key normalization."""

        def __init__(self, *args, **kwargs):
            # Transformers versions differ in how __init__ is exposed (*args/**kwargs vs named config).
            # Prefer explicit kwargs, then positional arg[0], and only then signature binding fallback.
            config = kwargs.get("config")
            if config is None and args:
                config = args[0]
            if config is None:
                signature = inspect.signature(super().__init__)
                bound_args = signature.bind(*args, **kwargs)
                bound_args.apply_defaults()
                config = bound_args.arguments.get("config")
            if config is None:
                raise ValueError("Could not determine config for QKNormNoPEAttention initialization")

            super().__init__(*args, **kwargs)

            # Add normalization layers
            self.q_norm = nn.RMSNorm(
                config.num_attention_heads * self.head_dim,
                config.rms_norm_eps,
            )
            self.k_norm = nn.RMSNorm(
                config.num_key_value_heads * self.head_dim,
                config.rms_norm_eps,
            )

            # Bind custom forward
            self.forward = forward_fn.__get__(self, self.__class__)

        @classmethod
        def from_source(cls, source_module: nn.Module, config: Any = None) -> "QKNormNoPEAttention":
            """Create from existing attention module."""
            config = source_module.config
            device = next(source_module.parameters()).device
            dtype = next(source_module.parameters()).dtype
            new_module = cls(config, source_module.layer_idx).to(device=device, dtype=dtype)
            new_module.load_state_dict(source_module.state_dict(), strict=False)
            return new_module

    QKNormNoPEAttention.__name__ = f"QKNormNoPE{BaseAttentionClass.__name__}"
    return QKNormNoPEAttention


def llama_qk_norm_nope_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Any] = None,
    cache_position: Optional[torch.LongTensor] = None,
    norm_type: str = "qk",
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for LLaMA with QK normalization and NoPE.

    Args:
        norm_type: "qk" for both, "q" for query only, "k" for key only.
    """
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Project with optional normalization
    query_states = self.q_norm(self.q_proj(hidden_states)) if "q" in norm_type else self.q_proj(hidden_states)
    key_states = self.k_norm(self.k_proj(hidden_states)) if "k" in norm_type else self.k_proj(hidden_states)

    query_states = query_states.view(hidden_shape).transpose(1, 2)
    key_states = key_states.view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    # Nullify positional embeddings (NoPE)
    original_cos, original_sin = position_embeddings
    cos = torch.ones_like(original_cos)
    sin = torch.zeros_like(original_sin)
    query_states, key_states = llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # Handle KV cache
    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # Select attention implementation
    attention_fn: Callable = llama_eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_fn(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def llama_q_norm_nope_forward(self, *args, **kwargs):
    """Forward with Q-only normalization."""
    return llama_qk_norm_nope_forward(self, *args, norm_type="q", **kwargs)


def llama_k_norm_nope_forward(self, *args, **kwargs):
    """Forward with K-only normalization."""
    return llama_qk_norm_nope_forward(self, *args, norm_type="k", **kwargs)


# Pre-built attention classes
NoPELlamaAttention = nope(LlamaAttention)
QKNormNoPELlamaAttention = qk_norm_nope(LlamaAttention, llama_qk_norm_nope_forward)
QNormNoPELlamaAttention = qk_norm_nope(LlamaAttention, llama_q_norm_nope_forward)
KNormNoPELlamaAttention = qk_norm_nope(LlamaAttention, llama_k_norm_nope_forward)

# Attention variants map
NOPE_ATTENTION_VARIANTS = {
    LlamaAttention: {
        "nope": NoPELlamaAttention,
        "qk_norm_nope": QKNormNoPELlamaAttention,
        "q_norm_nope": QNormNoPELlamaAttention,
        "k_norm_nope": KNormNoPELlamaAttention,
    },
}


def convert_to_nope(model: nn.Module, attention_type: str = "nope") -> nn.Module:
    """Convert a model's attention layers to NoPE.

    Args:
        model: Model with standard attention layers.
        attention_type: Type of NoPE attention ("nope", "qk_norm_nope", "q_norm_nope", "k_norm_nope").

    Returns:
        Model with NoPE attention layers.
    """
    # Get the model core (usually model.model for CausalLM)
    model_core = getattr(model, model.base_model_prefix, model)

    for i, layer in enumerate(model_core.layers):
        if hasattr(layer, "self_attn"):
            original_attn = layer.self_attn
            base_class = type(original_attn)

            # Find the appropriate NoPE variant
            if base_class in NOPE_ATTENTION_VARIANTS:
                variants = NOPE_ATTENTION_VARIANTS[base_class]
                if attention_type in variants:
                    AttentionClass = variants[attention_type]
                    new_attn = AttentionClass.from_source(original_attn, model.config)
                    layer.self_attn = new_attn
                    logger.info(f"Layer {i}: Converted to {new_attn.__class__.__name__}")
                else:
                    logger.warning(f"Unknown attention_type '{attention_type}' for {base_class.__name__}")
            else:
                logger.warning(f"No NoPE variant for {base_class.__name__}")

    return model
