"""Custom trainer with enhanced logging and checkpoint resumption.

Features:
- Per-step loss logging
- Gradient norm tracking
- Learning rate logging
- Data skipping for checkpoint resumption

Reference: Adapted from references/DroPE/trainers/
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback

logger = logging.getLogger(__name__)


class PETrainer(Trainer):
    """Trainer with enhanced logging and checkpoint resumption support.

    Features:
    - Per-step loss logging
    - Gradient norm tracking
    - Learning rate logging
    - Data skipping for checkpoint resumption with streaming datasets
    """

    def __init__(
        self,
        *args,
        log_grad_norm: bool = True,
        skip_samples: int = 0,
        **kwargs,
    ):
        """Initialize PETrainer.

        Args:
            *args: Arguments passed to parent Trainer.
            log_grad_norm: Whether to log gradient norms.
            skip_samples: Number of samples to skip (for resumption).
            **kwargs: Keyword arguments passed to parent Trainer.
        """
        super().__init__(*args, **kwargs)
        self.log_grad_norm = log_grad_norm
        self.skip_samples = skip_samples
        self._current_loss = None
        self._current_grad_norm = None
        self._samples_seen = 0

    def get_train_dataloader(self):
        """Get training dataloader with optional sample skipping."""
        dataloader = super().get_train_dataloader()

        if self.skip_samples > 0:
            logger.info(f"Skipping {self.skip_samples} samples for resumption")
            # For streaming datasets, use skip()
            if hasattr(self.train_dataset, "skip"):
                self.train_dataset = self.train_dataset.skip(self.skip_samples)
                dataloader = super().get_train_dataloader()

        return dataloader

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss and store for logging."""
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        if isinstance(loss, tuple):
            self._current_loss = loss[0].item()
        else:
            self._current_loss = loss.item()

        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Training step with grad_norm capture after backward pass."""
        # Call parent training_step (does forward + backward)
        loss = super().training_step(model, inputs, num_items_in_batch)

        # Capture grad_norm after backward, before optimizer.step() zeros them
        if self.log_grad_norm:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            self._current_grad_norm = total_norm ** 0.5

        return loss

    def log(self, logs: Dict[str, Any], start_time: Optional[float] = None) -> None:
        """Enhanced logging with additional metrics."""
        # Add gradient norm captured during training_step (before zero_grad)
        if self.log_grad_norm and self._current_grad_norm is not None:
            logs["grad_norm"] = self._current_grad_norm

        # Add current learning rate
        if self.optimizer is not None:
            logs["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        super().log(logs, start_time)

    def _save_checkpoint(self, model, trial):
        """Save checkpoint with additional metadata for resumption."""
        checkpoint_folder = super()._save_checkpoint(model, trial)

        # Save additional resumption metadata
        if checkpoint_folder is not None:
            metadata = {
                "global_step": self.state.global_step,
                "samples_seen": self._samples_seen,
                "epoch": self.state.epoch,
            }
            metadata_path = os.path.join(checkpoint_folder, "pe_trainer_state.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        return checkpoint_folder

    @classmethod
    def get_resumption_state(cls, checkpoint_path: str) -> Dict[str, Any]:
        """Get resumption state from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory.

        Returns:
            Dictionary with resumption metadata.
        """
        metadata_path = os.path.join(checkpoint_path, "pe_trainer_state.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                return json.load(f)
        return {}


def create_training_args(
    output_dir: str,
    num_train_epochs: int = 1,
    max_steps: int = -1,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 6e-4,
    min_learning_rate: float = None,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    warmup_steps: int = 1000,
    lr_scheduler_type: str = "cosine",
    logging_steps: int = 10,
    save_steps: int = 1000,
    save_total_limit: int = 3,
    bf16: bool = True,
    tf32: bool = True,
    gradient_checkpointing: bool = True,
    gradient_checkpointing_kwargs: dict = None,
    report_to: str = "wandb",
    dataloader_num_workers: int = 4,
    seed: int = 42,
    **kwargs,
) -> TrainingArguments:
    """Create TrainingArguments with sensible defaults for PE experiments.

    Uses hyperparameters from PoPE paper (Table 2) as defaults.

    Args:
        output_dir: Directory for checkpoints and logs.
        num_train_epochs: Number of training epochs.
        max_steps: Max training steps (-1 for epoch-based).
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Peak learning rate (default: 6e-4 from PoPE paper).
        min_learning_rate: Minimum learning rate for cosine scheduler.
        weight_decay: Weight decay coefficient (default: 0.01).
        max_grad_norm: Max gradient norm for clipping (default: 1.0).
        warmup_steps: LR warmup steps (default: 1000).
        lr_scheduler_type: LR scheduler type.
        logging_steps: Steps between logging.
        save_steps: Steps between checkpoints.
        save_total_limit: Max checkpoints to keep.
        bf16: Use bfloat16 training.
        tf32: Use TF32 for matmuls.
        gradient_checkpointing: Use gradient checkpointing.
        gradient_checkpointing_kwargs: Kwargs for gradient checkpointing (e.g., {"use_reentrant": False}).
        report_to: Logging backend ("wandb", "tensorboard", etc.).
        dataloader_num_workers: Number of dataloader workers.
        seed: Random seed.
        **kwargs: Additional TrainingArguments.

    Returns:
        Configured TrainingArguments.
    """
    # Build lr_scheduler_kwargs for minimum LR support
    lr_scheduler_kwargs = kwargs.pop("lr_scheduler_kwargs", {})
    if min_learning_rate is not None and lr_scheduler_type == "cosine":
        lr_scheduler_kwargs["min_lr"] = min_learning_rate

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs if lr_scheduler_kwargs else None,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        bf16=bf16,
        tf32=tf32,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {"use_reentrant": False},
        report_to=report_to,
        dataloader_num_workers=dataloader_num_workers,
        seed=seed,
        remove_unused_columns=False,
        # AdamW beta2 = 0.95 from PoPE paper
        adam_beta2=0.95,
        **kwargs,
    )


class CheckpointResumptionCallback(TrainerCallback):
    """Callback to handle checkpoint resumption with data skipping."""

    def __init__(self, checkpoint_path: Optional[str] = None):
        """Initialize callback.

        Args:
            checkpoint_path: Path to checkpoint to resume from.
        """
        self.checkpoint_path = checkpoint_path
        self.resumption_state = {}

    def on_train_begin(self, args, state, control, **kwargs):
        """Load resumption state at training start."""
        if self.checkpoint_path:
            self.resumption_state = PETrainer.get_resumption_state(self.checkpoint_path)
            if self.resumption_state:
                logger.info(f"Resuming from step {self.resumption_state.get('global_step', 0)}")
                logger.info(f"Skipping {self.resumption_state.get('samples_seen', 0)} samples")
