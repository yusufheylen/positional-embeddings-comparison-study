"""Dataset loading and collation utilities.

Includes block diagonal attention collators for efficient sequence packing
during pretraining.

Reference: Adapted from references/DroPE/custom_data/pretraining_data.py
"""

import logging
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, IterableDataset, load_dataset
from transformers import PreTrainedTokenizer, default_data_collator

logger = logging.getLogger(__name__)


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


def tokenize_and_chunk_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    text_column: str = "text",
    max_length: int = 2048,
    add_eos: bool = True,
    streaming: bool = True,
    num_proc: int = 4,
) -> Dataset:
    """Tokenize dataset and chunk into fixed-length sequences.

    Args:
        dataset: Dataset to process.
        tokenizer: Tokenizer to use.
        text_column: Column containing text.
        max_length: Maximum sequence length.
        add_eos: Whether to add EOS between documents.
        streaming: Whether dataset is streaming.
        num_proc: Number of processes for non-streaming.

    Returns:
        Tokenized and chunked dataset.
    """
    eos_id = tokenizer.eos_token_id

    def tokenize_fn(examples):
        return tokenizer(
            examples[text_column],
            add_special_tokens=False,
            return_attention_mask=True,
        )

    def group_texts(examples):
        """Concatenate and chunk into fixed-length sequences."""
        ids_cat, attn_cat = [], []

        for ids, attn in zip(examples["input_ids"], examples["attention_mask"]):
            ids_cat.extend(ids)
            attn_cat.extend(attn)
            if add_eos and eos_id is not None:
                ids_cat.append(eos_id)
                attn_cat.append(1)

        # Chunk into max_length blocks
        total = (len(ids_cat) // max_length) * max_length
        input_blocks = [ids_cat[i : i + max_length] for i in range(0, total, max_length)]
        attn_blocks = [attn_cat[i : i + max_length] for i in range(0, total, max_length)]

        return {
            "input_ids": input_blocks,
            "attention_mask": attn_blocks,
            "labels": [b[:] for b in input_blocks],
        }

    # Get columns to remove
    if streaming:
        # For streaming, features can be None - check both hasattr and not None
        if hasattr(dataset, "features") and dataset.features is not None:
            cols = list(dataset.features.keys())
        elif hasattr(dataset, "column_names") and dataset.column_names is not None:
            cols = dataset.column_names
        else:
            cols = [text_column]
    else:
        cols = dataset.column_names

    # Tokenize
    if streaming:
        tok_ds = dataset.map(tokenize_fn, batched=True, remove_columns=cols)
    else:
        tok_ds = dataset.map(
            tokenize_fn,
            batched=True,
            num_proc=num_proc,
            remove_columns=cols,
        )

    # Group into chunks
    if streaming:
        chunked_ds = tok_ds.map(group_texts, batched=True)
    else:
        chunked_ds = tok_ds.map(group_texts, batched=True, num_proc=num_proc)

    return chunked_ds


class BlockDiagFromEOSCollator:
    """Collator that creates block diagonal attention masks from EOS tokens.

    This enables efficient sequence packing: multiple documents are concatenated
    into a single sequence, with attention masked to prevent cross-document attention.
    EOS tokens mark document boundaries.

    Reference: references/DroPE/custom_data/pretraining_data.py
    """

    def __init__(self, tokenizer: PreTrainedTokenizer):
        """Initialize collator.

        Args:
            tokenizer: Tokenizer to get EOS token ID.
        """
        self.eos_id = tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Create batch with block diagonal attention mask.

        Args:
            features: List of tokenized examples.

        Returns:
            Batch dict with input_ids, attention_mask (4D), and labels.
        """
        batch = default_data_collator(features)
        ids = batch["input_ids"]
        pad_mask = batch.get("attention_mask")

        if pad_mask is None:
            pad_mask = torch.ones_like(ids)

        b, seq_len = ids.shape
        device = ids.device

        # Find EOS positions to determine segment boundaries
        is_eos = ids.eq(self.eos_id)

        # Segment starts: position 0 or right after EOS
        seg_start = torch.zeros_like(ids)
        seg_start[:, 0] = 1
        seg_start[:, 1:] = is_eos[:, :-1].int()

        # Assign segment IDs
        seg_id = torch.cumsum(seg_start, dim=1) - 1

        # Build attention mask: attend only within same segment
        same_seg = seg_id.unsqueeze(2) == seg_id.unsqueeze(1)  # (b, seq, seq)

        # Causal mask
        positions = torch.arange(seq_len, device=device)
        causal = positions[None, :, None] >= positions[None, None, :]  # (1, seq, seq)

        # Combine: same segment AND causal AND not padding
        keep = same_seg & causal
        keep = keep & pad_mask[:, None, :].bool() & pad_mask[:, :, None].bool()

        # Convert to attention mask format (0 for attend, -inf for ignore)
        attn_mask = torch.zeros(b, 1, seq_len, seq_len, device=device, dtype=torch.float32)
        attn_mask.masked_fill_(~keep.unsqueeze(1), torch.finfo(attn_mask.dtype).min)
        batch["attention_mask"] = attn_mask

        # Prepare labels: mask padding and first token after EOS
        labels = batch.get("labels")
        if labels is None:
            labels = ids.clone()
        else:
            labels = labels.clone()

        # Mask padding
        labels[pad_mask.eq(0)] = -100

        # Mask first token of each segment (can't predict from previous doc)
        ignore_first = torch.zeros_like(is_eos, dtype=torch.bool)
        ignore_first[:, 1:] = is_eos[:, :-1]
        labels[ignore_first] = -100

        batch["labels"] = labels

        return batch


class BlockDiagFA2Collator:
    """Block diagonal collator optimized for Flash Attention 2.

    Creates the cu_seqlens format required by Flash Attention for variable
    length sequences with block diagonal masking.

    Reference: references/DroPE/custom_data/pretraining_data.py
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        ignore_idx: int = -100,
        return_seq_idx: bool = False,
    ):
        """Initialize collator.

        Args:
            tokenizer: Tokenizer to get EOS token ID.
            ignore_idx: Index for ignored labels.
            return_seq_idx: Whether to return sequence indices.
        """
        self.eos_id = tokenizer.eos_token_id
        self.ignore_idx = ignore_idx
        self.return_seq_idx = return_seq_idx

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Create batch with Flash Attention 2 format.

        Args:
            features: List of tokenized examples.

        Returns:
            Batch dict with input_ids, labels, position_ids, cu_seq_lens, max_length.
        """
        flat_ids, flat_labels, pos_ids = [], [], []
        seq_idx = [] if self.return_seq_idx else None
        cu = [0]  # Cumulative sequence lengths
        max_len = 0

        for si, ex in enumerate(features):
            ids = ex["input_ids"]
            lbls = ex.get("labels", ids)
            n = len(ids)

            if n == 0:
                continue

            # Split on EOS tokens
            start = 0
            split_on_eos = self.eos_id is not None

            for i, tok in enumerate(ids):
                if split_on_eos and tok == self.eos_id:
                    seg_ids = ids[start : i + 1]
                    seg_labels = lbls[start : i + 1]

                    if seg_ids:
                        # First token of segment gets ignored label
                        seg_labels = [self.ignore_idx] + list(seg_labels[1:])

                        flat_ids.extend(seg_ids)
                        flat_labels.extend(seg_labels)
                        pos_ids.extend(range(len(seg_ids)))

                        if self.return_seq_idx:
                            seq_idx.extend([si] * len(seg_ids))

                        cu.append(cu[-1] + len(seg_ids))
                        max_len = max(max_len, len(seg_ids))

                    start = i + 1

            # Handle remaining tokens after last EOS
            if start < n:
                seg_ids = ids[start:n]
                if seg_ids:
                    seg_labels = lbls[start:n]
                    seg_labels = [self.ignore_idx] + list(seg_labels[1:])

                    flat_ids.extend(seg_ids)
                    flat_labels.extend(seg_labels)
                    pos_ids.extend(range(len(seg_ids)))

                    if self.return_seq_idx:
                        seq_idx.extend([si] * len(seg_ids))

                    cu.append(cu[-1] + len(seg_ids))
                    max_len = max(max_len, len(seg_ids))

        assert cu[-1] == len(flat_ids) == len(flat_labels) == len(pos_ids)

        batch = {
            "input_ids": torch.tensor([flat_ids], dtype=torch.int64),
            "labels": torch.tensor([flat_labels], dtype=torch.int64),
            "position_ids": torch.tensor([pos_ids], dtype=torch.int64),
            "cu_seq_lens_q": torch.tensor(cu, dtype=torch.int32),
            "cu_seq_lens_k": torch.tensor(cu, dtype=torch.int32),
            "max_length_q": int(max_len),
            "max_length_k": int(max_len),
        }

        if self.return_seq_idx:
            batch["seq_idx"] = torch.tensor([seq_idx], dtype=torch.int32)

        return batch


def get_data_collator(
    tokenizer: PreTrainedTokenizer,
    attn_implementation: str = "sdpa",
    mask_past_sequences: bool = True,
) -> Any:
    """Get appropriate data collator based on attention implementation.

    Args:
        tokenizer: Tokenizer for EOS token.
        attn_implementation: Attention implementation being used.
        mask_past_sequences: Whether to use block diagonal masking.

    Returns:
        Data collator instance.
    """
    if not mask_past_sequences:
        return default_data_collator

    if "flash" in attn_implementation:
        return BlockDiagFA2Collator(tokenizer)
    else:
        return BlockDiagFromEOSCollator(tokenizer)


# Legacy alias for backwards compatibility
BlockDiagonalCollator = BlockDiagFromEOSCollator
