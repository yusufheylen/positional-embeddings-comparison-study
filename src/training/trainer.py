"""Custom trainer with enhanced logging for PE comparison study.

Reference: Adapted from references/DroPE/trainers/logging_trainer.py
"""

from typing import Any, Dict, Optional

import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback


class PETrainer(Trainer):
    """Trainer with enhanced logging for positional embedding experiments.

    Features:
    - Per-step loss logging
    - Gradient norm tracking
    - Learning rate logging
    - Custom metric logging for PE analysis
    """

    def __init__(
        self,
        *args,
        log_grad_norm: bool = True,
        **kwargs,
    ):
        """Initialize PETrainer.

        Args:
            *args: Arguments passed to parent Trainer.
            log_grad_norm: Whether to log gradient norms.
            **kwargs: Keyword arguments passed to parent Trainer.
        """
        super().__init__(*args, **kwargs)
        self.log_grad_norm = log_grad_norm
        self._current_loss = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute loss and store for logging."""
        loss = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

        if isinstance(loss, tuple):
            self._current_loss = loss[0].item()
        else:
            self._current_loss = loss.item()

        return loss

    def log(self, logs: Dict[str, Any]) -> None:
        """Enhanced logging with additional metrics."""
        # Add gradient norm if requested
        if self.log_grad_norm and hasattr(self, "accelerator"):
            try:
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm**0.5
                logs["grad_norm"] = total_norm
            except Exception:
                pass

        # Add current learning rate
        if self.optimizer is not None:
            logs["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        super().log(logs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Training step with optional per-step logging."""
        loss = super().training_step(model, inputs, num_items_in_batch)

        # Log per-step loss at configurable intervals
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({"train/step_loss": loss.item()})

        return loss


def create_training_args(
    output_dir: str,
    num_train_epochs: int = 1,
    max_steps: int = -1,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 6e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    lr_scheduler_type: str = "cosine",
    logging_steps: int = 10,
    save_steps: int = 1000,
    save_total_limit: int = 3,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    report_to: str = "wandb",
    **kwargs,
) -> TrainingArguments:
    """Create TrainingArguments with sensible defaults for PE experiments.

    Args:
        output_dir: Directory for checkpoints and logs.
        num_train_epochs: Number of training epochs.
        max_steps: Max training steps (-1 for epoch-based).
        per_device_train_batch_size: Batch size per device.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Peak learning rate.
        weight_decay: Weight decay coefficient.
        warmup_steps: LR warmup steps.
        lr_scheduler_type: LR scheduler type.
        logging_steps: Steps between logging.
        save_steps: Steps between checkpoints.
        save_total_limit: Max checkpoints to keep.
        bf16: Use bfloat16 training.
        gradient_checkpointing: Use gradient checkpointing.
        report_to: Logging backend ("wandb", "tensorboard", etc.).
        **kwargs: Additional TrainingArguments.

    Returns:
        Configured TrainingArguments.
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        logging_steps=logging_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        report_to=report_to,
        remove_unused_columns=False,
        **kwargs,
    )
