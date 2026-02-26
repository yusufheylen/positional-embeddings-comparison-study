"""YaRN (Yet another RoPE extensioN) scaling configuration.

YaRN enables RoPE-based models to handle longer sequences than they were
trained on by applying NTK-aware interpolation.

Reference: https://arxiv.org/abs/2309.00071
HuggingFace: rope_scaling config parameter
"""

from typing import Any, Dict, Optional

from transformers import AutoConfig


def get_yarn_config(
    model_name_or_path: str,
    factor: float = 2.0,
    original_max_position_embeddings: Optional[int] = None,
    **kwargs,
) -> AutoConfig:
    """Get model config with YaRN scaling enabled.

    Args:
        model_name_or_path: HuggingFace model identifier or path.
        factor: Scaling factor for context extension (e.g., 2.0 = 2x length).
        original_max_position_embeddings: Original context length (auto-detected if None).
        **kwargs: Additional config overrides.

    Returns:
        Model configuration with YaRN scaling.
    """
    config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)

    # YaRN scaling configuration
    config.rope_scaling = {
        "rope_type": "yarn",
        "factor": factor,
    }

    # Optionally set original context length
    if original_max_position_embeddings is not None:
        config.rope_scaling["original_max_position_embeddings"] = original_max_position_embeddings

    return config


def get_yarn_scaling_dict(
    factor: float = 2.0,
    attention_factor: Optional[float] = None,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> Dict[str, Any]:
    """Get YaRN scaling parameters dictionary.

    Args:
        factor: Scaling factor for context extension.
        attention_factor: Attention scaling (auto-computed if None).
        beta_fast: Fast interpolation boundary.
        beta_slow: Slow interpolation boundary.

    Returns:
        Dictionary of YaRN scaling parameters for rope_scaling config.
    """
    scaling = {
        "rope_type": "yarn",
        "factor": factor,
        "beta_fast": beta_fast,
        "beta_slow": beta_slow,
    }

    if attention_factor is not None:
        scaling["attention_factor"] = attention_factor

    return scaling
