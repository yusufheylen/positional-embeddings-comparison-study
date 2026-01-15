"""DroPE callback for mid-training PE ablation.

DroPE (Drop Positional Embeddings) trains with a PE (RoPE/PoPE) and then
removes it mid-training to achieve length generalization.

Reference: https://arxiv.org/abs/2512.12167
"""

from typing import Optional

import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from ..models.embeddings.nope import convert_to_nope


class DroPECallback(TrainerCallback):
    """Callback to switch from PE (RoPE/PoPE) to NoPE mid-training.

    This implements the DroPE strategy: train with positional embeddings
    for the first N steps, then ablate them for the remaining training.
    """

    def __init__(
        self,
        switch_step: Optional[int] = None,
        switch_fraction: float = 0.7,
        reset_optimizer: bool = False,
    ):
        """Initialize DroPE callback.

        Args:
            switch_step: Absolute step to switch at (overrides switch_fraction).
            switch_fraction: Fraction of training to use PE (default 70%).
            reset_optimizer: Whether to reset optimizer state at switch.
        """
        self.switch_step = switch_step
        self.switch_fraction = switch_fraction
        self.reset_optimizer = reset_optimizer
        self._switched = False

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Compute switch step if using fraction."""
        if self.switch_step is None and args.max_steps > 0:
            self.switch_step = int(args.max_steps * self.switch_fraction)
            print(f"DroPE: Will switch to NoPE at step {self.switch_step}")

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Check if it's time to switch PE strategy."""
        if self._switched:
            return

        if self.switch_step is not None and state.global_step >= self.switch_step:
            self._perform_switch(model, kwargs.get("optimizer"))

    def _perform_switch(self, model, optimizer=None):
        """Perform the PE to NoPE switch."""
        print(f"DroPE: Switching to NoPE attention")

        # Convert model to NoPE
        convert_to_nope(model)

        # Optionally reset optimizer state
        if self.reset_optimizer and optimizer is not None:
            print("DroPE: Resetting optimizer state")
            optimizer.state.clear()

        self._switched = True
        print("DroPE: Switch complete")

    @property
    def has_switched(self) -> bool:
        """Check if PE has been ablated."""
        return self._switched


class DroPEFromPoPECallback(DroPECallback):
    """DroPE callback specifically for PoPE to NoPE transition.

    Handles any PoPE-specific cleanup during the switch.
    """

    def _perform_switch(self, model, optimizer=None):
        """Perform PoPE to NoPE switch with cleanup."""
        print("DroPE: Switching from PoPE to NoPE attention")

        # First convert to NoPE
        convert_to_nope(model)

        # Clean up any PoPE-specific buffers/parameters
        for layer in model.model.layers:
            if hasattr(layer.self_attn, "polar_emb"):
                delattr(layer.self_attn, "polar_emb")

        if self.reset_optimizer and optimizer is not None:
            print("DroPE: Resetting optimizer state")
            optimizer.state.clear()

        self._switched = True
        print("DroPE: PoPE -> NoPE switch complete")
