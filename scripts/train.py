#!/usr/bin/env python3
"""Training script for PE comparison study.

Usage:
    # Baseline training from scratch
    python scripts/train.py --config configs/rope.yaml

    # Scaffold training (load PE checkpoint, convert to NoPE, train remaining steps)
    python scripts/train.py --config configs/scaffold_rope_10k.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import get_data_collator, load_dataset_for_training, tokenize_and_chunk_dataset
from src.models import create_model, get_model_config
from src.models.embeddings.nope import convert_to_nope
from src.training import DroPECallback, PETrainer, create_training_args


def load_config(config_path: str) -> dict:
    """Load and merge configuration files."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Handle defaults/inheritance
    if "defaults" in config:
        base_configs = config.pop("defaults")
        merged = {}
        for base in base_configs:
            base_path = Path(config_path).parent / f"{base}.yaml"
            if base_path.exists():
                with open(base_path) as f:
                    base_config = yaml.safe_load(f)
                    merged = deep_merge(merged, base_config)
        config = deep_merge(merged, config)

    return config


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_model_from_checkpoint_as_nope(
    checkpoint_path: str,
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "auto",
) -> AutoModelForCausalLM:
    """Load model weights from a PE checkpoint and convert to NoPE.

    This loads ONLY the model weights (not optimizer/scheduler state) from a
    checkpoint that was trained with positional embeddings (RoPE, YaRN, PoPE),
    then converts the model to use NoPE attention.

    For PoPE checkpoints: Extra PoPE-specific parameters (polar_emb) are ignored
    since they're not needed for NoPE attention.

    Args:
        checkpoint_path: Path to the checkpoint directory (e.g., run1_rope/checkpoint-15000)
        model_name_or_path: Original model identifier for config (e.g., HuggingFaceTB/SmolLM-360M)
        dtype: Model dtype (default: bfloat16)
        attn_implementation: Attention implementation to use

    Returns:
        Model with NoPE attention and weights from the PE checkpoint
    """
    from transformers import AutoConfig
    import logging

    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

    # Load config from original model (not checkpoint) to get clean architecture
    config = AutoConfig.from_pretrained(model_name_or_path)

    # Auto-detect attention implementation
    if attn_implementation == "auto":
        from src.models import get_best_attn_implementation
        attn_implementation = get_best_attn_implementation()

    config._attn_implementation = attn_implementation

    # Load model weights from checkpoint
    # ignore_mismatched_sizes handles PoPE's extra parameters gracefully
    print(f"Loading model weights from: {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        config=config,
        attn_implementation=attn_implementation,
        dtype=dtype,  # Use dtype (not torch_dtype) for transformers 5.x
        ignore_mismatched_sizes=True,  # Handle any size mismatches from PE-specific params
    )

    # Convert to NoPE
    print("Converting model to NoPE attention...")
    model = convert_to_nope(model, attention_type="nope")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Loaded model with {param_count:,} parameters, converted to NoPE")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train PE comparison models")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override seed if provided
    if args.seed is not None:
        config["seed"] = args.seed

    # Set seed
    set_seed(config["seed"])
    print(f"Using seed: {config['seed']}")

    # Setup W&B
    if config["training"].get("report_to") == "wandb":
        import wandb

        wandb.init(
            project=config["wandb"].get("project", "pe-comparison-study"),
            entity=config["wandb"].get("entity"),
            name=config["wandb"].get("name"),
            tags=config["wandb"].get("tags", []),
            config=config,
        )

    # Load tokenizer
    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"])

    # Set pad token if not set
    if tokenizer.pad_token is None:
        if "smollm" in model_cfg["name_or_path"].lower():
            tokenizer.pad_token = "<|reserved_special_token_5|>"
        else:
            tokenizer.pad_token = tokenizer.eos_token

    # Create model - either from scratch, pretrained, or from PE checkpoint (scaffold)
    scaffold_cfg = config.get("scaffold")
    pe_type = config.get("pe_type", "rope")
    from_scratch = model_cfg.get("from_scratch", False)

    if scaffold_cfg and scaffold_cfg.get("enabled", False):
        # Scaffold training: load from PE checkpoint, convert to NoPE
        checkpoint_path = scaffold_cfg["checkpoint_path"]
        source_pe = scaffold_cfg.get("source_pe", "unknown")
        print(f"Scaffold training: loading {source_pe} checkpoint from {checkpoint_path}")
        print(f"Will convert to NoPE and train for remaining steps")

        model = load_model_from_checkpoint_as_nope(
            checkpoint_path=checkpoint_path,
            model_name_or_path=model_cfg["name_or_path"],
            dtype=getattr(torch, model_cfg.get("dtype", "bfloat16")),
            attn_implementation=model_cfg.get("attn_implementation", "auto"),
        )
    else:
        # Standard training: from scratch or pretrained
        print(f"Creating model with PE type: {pe_type}, from_scratch: {from_scratch}")

        model = create_model(
            model_cfg["name_or_path"],
            pe_type=pe_type,
            dtype=getattr(torch, model_cfg.get("dtype", "bfloat16")),
            attn_implementation=model_cfg.get("attn_implementation", "auto"),
            from_scratch=from_scratch,
        )

    # Load dataset
    data_cfg = config["data"]
    dataset_config = data_cfg.get("dataset_config")
    print(f"Loading dataset: {data_cfg['dataset_name']}" + (f" ({dataset_config})" if dataset_config else ""))
    dataset = load_dataset_for_training(
        dataset_name=data_cfg["dataset_name"],
        dataset_config=dataset_config,
        streaming=data_cfg.get("streaming", True),
    )

    # Tokenize and chunk dataset
    print(f"Tokenizing dataset with max_length={data_cfg.get('max_length', 2048)}")
    dataset = tokenize_and_chunk_dataset(
        dataset,
        tokenizer,
        text_column=data_cfg.get("text_column", "text"),
        max_length=data_cfg.get("max_length", 2048),
        streaming=data_cfg.get("streaming", True),
    )

    # Create collator based on attention implementation
    # Detect actual attention implementation from model (important when "auto" is used)
    actual_attn_impl = getattr(model.config, "_attn_implementation", None)
    if actual_attn_impl is None:
        actual_attn_impl = model_cfg.get("attn_implementation", "sdpa")
    print(f"Using attention implementation: {actual_attn_impl}")

    collator = get_data_collator(
        tokenizer,
        attn_implementation=actual_attn_impl,
        mask_past_sequences=True,
    )

    # Create training arguments
    train_cfg = config["training"]
    training_args = create_training_args(
        output_dir=train_cfg["output_dir"],
        max_steps=train_cfg.get("max_steps", 100000),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 6e-4),
        min_learning_rate=train_cfg.get("min_learning_rate"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_steps=train_cfg.get("warmup_steps", 1000),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 5000),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=train_cfg.get("gradient_checkpointing_kwargs"),
        report_to=train_cfg.get("report_to", "wandb"),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        seed=config["seed"],
    )

    # Setup callbacks
    callbacks = []

    # Add DroPE callback if enabled
    drope_cfg = config.get("drope", {})
    if drope_cfg.get("enabled", False):
        print(f"DroPE enabled: will switch at {drope_cfg.get('switch_fraction', 0.7)*100:.0f}% of training")
        callbacks.append(
            DroPECallback(
                switch_step=drope_cfg.get("switch_step"),
                switch_fraction=drope_cfg.get("switch_fraction", 0.7),
                reset_optimizer=drope_cfg.get("reset_optimizer", False),
            )
        )

    # Create trainer
    trainer = PETrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=callbacks,
    )

    # Train
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume)

    # Save final model
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(train_cfg["output_dir"])

    # Save training config for evaluation auto-detection
    config_save_path = Path(train_cfg["output_dir"]) / "training_config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved training config to {config_save_path}")

    print("Training complete!")


if __name__ == "__main__":
    main()
