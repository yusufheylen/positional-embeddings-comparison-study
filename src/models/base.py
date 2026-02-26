"""Model factory and registration for PE variants.

This module provides a unified interface for creating models with different
positional embedding strategies (NoPE, RoPE, PoPE, YaRN, DroPE).

Factory pattern adapted from: references/DroPE/custom_models/drope.py
"""

import logging
from typing import Optional, Type

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention, LlamaConfig

logger = logging.getLogger(__name__)


def get_best_attn_implementation() -> str:
    """Detect the best available attention implementation.

    Returns:
        "flash_attention_2" for NVIDIA GPUs with Flash Attention installed,
        "sdpa" for Apple Silicon or when Flash Attention unavailable,
        "eager" as fallback.
    """
    # Check for CUDA + Flash Attention
    if torch.cuda.is_available():
        try:
            import flash_attn

            return "flash_attention_2"
        except ImportError:
            pass

    # SDPA works on CUDA, MPS, and CPU (PyTorch 2.0+)
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return "sdpa"

    return "eager"


def get_model_config(
    model_name_or_path: str,
    pe_type: str = "rope",
    **kwargs,
) -> AutoConfig:
    """Get model configuration with PE-specific settings.

    Args:
        model_name_or_path: HuggingFace model identifier or path.
        pe_type: Positional embedding type ("nope", "rope", "pope", "yarn").
        **kwargs: Additional config overrides.

    Returns:
        Model configuration with PE settings applied.
    """
    config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)

    # Apply YaRN scaling if requested
    if pe_type == "yarn":
        config.rope_scaling = {
            "rope_type": "yarn",
            "factor": kwargs.get("yarn_factor", 2.0),
        }

    return config


def create_model(
    model_name_or_path: str,
    pe_type: str = "rope",
    attention_type: str = "nope",
    config: Optional[AutoConfig] = None,
    attn_implementation: str = "auto",
    from_scratch: bool = False,
    **kwargs,
) -> PreTrainedModel:
    """Create a model with the specified positional embedding strategy.

    Args:
        model_name_or_path: HuggingFace model identifier or path.
        pe_type: Positional embedding type ("nope", "rope", "pope", "yarn").
        attention_type: For NoPE, the variant ("nope", "qk_norm_nope", "q_norm_nope", "k_norm_nope").
        config: Optional pre-configured model config.
        attn_implementation: Attention implementation ("auto", "flash_attention_2", "sdpa", "eager").
        from_scratch: If True, initialize model with random weights instead of pretrained.
        **kwargs: Additional model loading arguments.

    Returns:
        Model instance with the specified PE strategy.
    """
    if config is None:
        config = get_model_config(model_name_or_path, pe_type, **kwargs)

    # Auto-detect best attention implementation
    if attn_implementation == "auto":
        # PoPE requires eager attention (dimension doubling not compatible with FA/SDPA)
        if pe_type == "pope":
            attn_implementation = "eager"
            logger.info("PoPE requires eager attention implementation")
        else:
            attn_implementation = get_best_attn_implementation()
        logger.info(f"Using attention implementation: {attn_implementation}")

    # Set attention implementation in config
    config._attn_implementation = attn_implementation

    if from_scratch:
        # Initialize model with random weights (from config only)
        logger.info(f"Initializing model from scratch with config from {model_name_or_path}")

        # Get the model class for this config
        model_class = AutoModelForCausalLM._model_mapping.get(type(config), None)
        if model_class is None:
            # Fallback: try to infer from config
            from transformers import LlamaForCausalLM
            model_class = LlamaForCausalLM
            logger.warning(f"Could not find model class for config type {type(config)}, using LlamaForCausalLM")

        # Initialize with random weights
        model = model_class(config)

        # Convert to specified dtype if provided
        dtype = kwargs.get("dtype")
        if dtype is not None:
            model = model.to(dtype)
            logger.info(f"Converted model to {dtype}")

        logger.info(f"Initialized {model_class.__name__} with {model.num_parameters():,} parameters (random weights)")
    else:
        # Load pretrained weights
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            attn_implementation=attn_implementation,
            **kwargs,
        )

    # Apply PE-specific modifications
    if pe_type == "nope":
        from .embeddings.nope import convert_to_nope

        model = convert_to_nope(model, attention_type=attention_type)
        logger.info(f"Converted model to NoPE (variant: {attention_type})")

    elif pe_type == "pope":
        from .embeddings.pope import convert_to_pope

        model = convert_to_pope(model)
        logger.info("Converted model to PoPE")

    # rope and yarn use native HF implementation (no conversion needed)
    elif pe_type in ("rope", "yarn"):
        logger.info(f"Using native HF {pe_type.upper()} implementation")

    return model


# ============================================================================
# DroPE-style Factory Pattern for Model Registration
# ============================================================================

# Map of supported model architectures
MODEL_ARCH_MAP = {
    LlamaForCausalLM: (LlamaAttention, LlamaConfig),
}


def _create_pe_config_class(
    BaseConfigClass: Type[PretrainedConfig],
    pe_type: str,
    **custom_fields,
) -> Type[PretrainedConfig]:
    """Create a custom config class for PE variants.

    Args:
        BaseConfigClass: Base config class (e.g., LlamaConfig).
        pe_type: PE type identifier.
        **custom_fields: Additional config fields with defaults.

    Returns:
        Custom config class.
    """

    class PEConfig(BaseConfigClass):
        model_type = f"{BaseConfigClass.model_type}_{pe_type}"

        def __init__(self, **kwargs):
            for field, default in custom_fields.items():
                setattr(self, field, kwargs.pop(field, default))
            super().__init__(**kwargs)

        @classmethod
        def from_base_config(cls, base_config: BaseConfigClass):
            return cls(**base_config.to_dict())

    PEConfig.__name__ = f"{pe_type.upper()}{BaseConfigClass.__name__}"
    AutoConfig.register(PEConfig.model_type, PEConfig)
    PEConfig.register_for_auto_class()

    return PEConfig


def _create_pe_model_class(
    BaseModelClass: Type[PreTrainedModel],
    pe_type: str,
    convert_fn,
    **custom_config_fields,
) -> Type[PreTrainedModel]:
    """Create a PE model class using factory pattern.

    Args:
        BaseModelClass: Base model class (e.g., LlamaForCausalLM).
        pe_type: PE type identifier.
        convert_fn: Function to convert attention layers.
        **custom_config_fields: Custom config fields.

    Returns:
        Custom model class with PE.
    """
    if BaseModelClass not in MODEL_ARCH_MAP:
        raise ValueError(f"Unsupported model: {BaseModelClass}")

    _, BaseConfigClass = MODEL_ARCH_MAP[BaseModelClass]

    PEConfigClass = _create_pe_config_class(
        BaseConfigClass,
        pe_type,
        **custom_config_fields,
    )

    class PEModel(BaseModelClass):
        config_class = PEConfigClass

        def __init__(self, config, *args, **kwargs):
            super().__init__(config, *args, **kwargs)
            convert_fn(self)

    architecture_name = f"{pe_type.upper()}{BaseModelClass.__name__}"
    PEModel.__name__ = architecture_name

    AutoModelForCausalLM.register(PEModel.config_class, PEModel)
    PEModel.register_for_auto_class("AutoModelForCausalLM")

    return PEModel


# Create registered model classes for easy loading
def _create_nope_model():
    from .embeddings.nope import convert_to_nope

    return _create_pe_model_class(
        LlamaForCausalLM,
        "nope",
        lambda m: convert_to_nope(m, "nope"),
        attention_type="nope",
    )


def _create_pope_model():
    from .embeddings.pope import convert_to_pope

    return _create_pe_model_class(
        LlamaForCausalLM,
        "pope",
        convert_to_pope,
    )


# Lazy initialization to avoid import issues
_NoPELlamaForCausalLM = None
_PoPELlamaForCausalLM = None


def get_nope_model_class():
    """Get the registered NoPE model class."""
    global _NoPELlamaForCausalLM
    if _NoPELlamaForCausalLM is None:
        _NoPELlamaForCausalLM = _create_nope_model()
    return _NoPELlamaForCausalLM


def get_pope_model_class():
    """Get the registered PoPE model class."""
    global _PoPELlamaForCausalLM
    if _PoPELlamaForCausalLM is None:
        _PoPELlamaForCausalLM = _create_pope_model()
    return _PoPELlamaForCausalLM
