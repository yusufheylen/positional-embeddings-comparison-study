"""Dataset loading and collation utilities.

Reference: Adapted from references/DroPE/custom_data/pretraining_data.py
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer


def load_dataset_for_training(
    dataset_name: str = "cerebras/SlimPajama-627B",
    split: str = "train",
    streaming: bool = True,
    num_samples: Optional[int] = None,
    **kwargs,
) -> Dataset:
    """Load a dataset for pretraining.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split to load.
        streaming: Whether to use streaming mode.
        num_samples: Optional limit on number of samples.
        **kwargs: Additional arguments to load_dataset.

    Returns:
        HuggingFace Dataset.
    """
    dataset = load_dataset(dataset_name, split=split, streaming=streaming, **kwargs)

    if num_samples is not None and not streaming:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    return dataset


def tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    text_column: str = "text",
    max_length: int = 2048,
    num_proc: int = 4,
) -> Dataset:
    """Tokenize a dataset.

    Args:
        dataset: Dataset to tokenize.
        tokenizer: Tokenizer to use.
        text_column: Column containing text.
        max_length: Maximum sequence length.
        num_proc: Number of processes for tokenization.

    Returns:
        Tokenized dataset.
    """

    def tokenize_fn(examples):
        return tokenizer(
            examples[text_column],
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )

    return tokenized


@dataclass
class BlockDiagonalCollator:
    """Data collator that creates block diagonal attention masks.

    This allows packing multiple sequences into a single batch while
    preventing cross-sequence attention, improving training efficiency.

    Reference: references/DroPE/custom_data/pretraining_data.py
    """

    tokenizer: PreTrainedTokenizer
    max_length: int = 2048
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate features into a batch with block diagonal attention.

        Args:
            features: List of tokenized examples.

        Returns:
            Batch dictionary with input_ids, attention_mask, labels.
        """
        # Extract input_ids
        input_ids_list = [f["input_ids"] for f in features]

        # Pack sequences
        packed_ids = []
        packed_labels = []
        attention_mask = []
        current_length = 0

        for ids in input_ids_list:
            seq_len = len(ids)

            if current_length + seq_len > self.max_length:
                # Start new sequence
                if packed_ids:
                    # Pad current batch
                    pad_len = self.max_length - current_length
                    packed_ids.extend([self.tokenizer.pad_token_id] * pad_len)
                    packed_labels.extend([-100] * pad_len)
                    attention_mask.extend([0] * pad_len)

                packed_ids = list(ids)
                packed_labels = list(ids)
                attention_mask = [1] * seq_len
                current_length = seq_len
            else:
                # Append to current sequence
                packed_ids.extend(ids)
                packed_labels.extend(ids)
                attention_mask.extend([1] * seq_len)
                current_length += seq_len

        # Pad final sequence
        if current_length < self.max_length:
            pad_len = self.max_length - current_length
            packed_ids.extend([self.tokenizer.pad_token_id] * pad_len)
            packed_labels.extend([-100] * pad_len)
            attention_mask.extend([0] * pad_len)

        # Convert to tensors
        batch = {
            "input_ids": torch.tensor([packed_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            "labels": torch.tensor([packed_labels], dtype=torch.long),
        }

        return batch


def create_block_diagonal_mask(
    sequence_lengths: List[int],
    total_length: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a block diagonal attention mask.

    Args:
        sequence_lengths: Length of each sequence in the packed batch.
        total_length: Total padded length.
        dtype: Data type for the mask.

    Returns:
        Block diagonal attention mask of shape (1, 1, total_length, total_length).
    """
    mask = torch.zeros(total_length, total_length, dtype=dtype)

    start = 0
    for length in sequence_lengths:
        end = start + length
        # Create causal mask for this block
        block_mask = torch.triu(torch.ones(length, length, dtype=dtype), diagonal=1)
        mask[start:end, start:end] = -block_mask * float("inf")
        start = end

    # Mask padding positions
    if start < total_length:
        mask[start:, :] = float("-inf")
        mask[:, start:] = float("-inf")

    return mask.unsqueeze(0).unsqueeze(0)
