"""Model factory and registration for PE variants.

This module provides a unified interface for creating models with different
positional embedding strategies (NoPE, RoPE, PoPE, YaRN, DroPE).
"""

from typing import Optional

from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel


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
            "type": "yarn",
            "factor": kwargs.get("yarn_factor", 2.0),
        }

    return config


def create_model(
    model_name_or_path: str,
    pe_type: str = "rope",
    config: Optional[AutoConfig] = None,
    **kwargs,
) -> PreTrainedModel:
    """Create a model with the specified positional embedding strategy.

    Args:
        model_name_or_path: HuggingFace model identifier or path.
        pe_type: Positional embedding type ("nope", "rope", "pope", "yarn").
        config: Optional pre-configured model config.
        **kwargs: Additional model loading arguments.

    Returns:
        Model instance with the specified PE strategy.
    """
    if config is None:
        config = get_model_config(model_name_or_path, pe_type, **kwargs)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        **kwargs,
    )

    # Apply PE-specific modifications
    if pe_type == "nope":
        from .embeddings.nope import convert_to_nope

        model = convert_to_nope(model)
    elif pe_type == "pope":
        from .embeddings.pope import convert_to_pope

        model = convert_to_pope(model)
    # rope and yarn use native HF implementation (no conversion needed)

    return model
