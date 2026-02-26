"""Perplexity evaluation at various context lengths.

Evaluates model perplexity on held-out data, particularly for context lengths
beyond training distribution to assess length generalization.
"""

from typing import Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


class PerplexityEvaluator:
    """Evaluate perplexity at multiple context lengths."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,
    ):
        """Initialize evaluator.

        Args:
            model: Model to evaluate.
            tokenizer: Tokenizer for the model.
            device: Device to run evaluation on.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(
        self,
        texts: List[str],
        context_lengths: List[int] = [2048, 4096, 8192, 16384],
        stride: Optional[int] = None,
        batch_size: int = 1,
    ) -> Dict[int, float]:
        """Evaluate perplexity at multiple context lengths.

        Args:
            texts: List of text samples to evaluate on.
            context_lengths: Context lengths to evaluate.
            stride: Sliding window stride (default: context_length // 2).
            batch_size: Evaluation batch size.

        Returns:
            Dictionary mapping context_length -> perplexity.
        """
        results = {}

        for ctx_len in context_lengths:
            print(f"Evaluating at context length {ctx_len}...")
            ppl = self._evaluate_at_length(texts, ctx_len, stride, batch_size)
            results[ctx_len] = ppl
            print(f"  Perplexity: {ppl:.2f}")

        return results

    @torch.no_grad()
    def _evaluate_at_length(
        self,
        texts: List[str],
        context_length: int,
        stride: Optional[int],
        batch_size: int,
    ) -> float:
        """Evaluate perplexity at a specific context length.

        Concatenates all texts into a single token sequence, then uses a
        sliding window approach. This is the standard method (per HF's
        perplexity tutorial) and avoids skipping short texts.
        """
        if stride is None:
            stride = context_length // 2

        # Concatenate all texts and tokenize as one sequence
        full_text = "\n\n".join(texts)
        encodings = self.tokenizer(
            full_text,
            return_tensors="pt",
            truncation=False,
            add_special_tokens=True,
        )
        input_ids = encodings.input_ids.to(self.device)
        seq_len = input_ids.size(1)

        if seq_len < context_length:
            print(f"  Warning: total tokens ({seq_len}) < context_length ({context_length}), skipping")
            return float("inf")

        total_loss = 0.0
        total_tokens = 0

        num_windows = max(1, (seq_len - context_length) // stride + 1)
        for i, begin in enumerate(tqdm(range(0, seq_len - context_length + 1, stride), desc=f"ctx={context_length}", total=num_windows)):
            end = begin + context_length
            window_ids = input_ids[:, begin:end]
            labels = window_ids.clone()

            # Mask tokens in the overlap region to avoid double-counting
            if begin > 0:
                overlap = context_length - stride
                labels[:, :overlap] = -100

            outputs = self.model(window_ids, labels=labels)
            loss = outputs.loss

            num_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

        if total_tokens == 0:
            return float("inf")

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return perplexity

    def evaluate_dataset(
        self,
        dataset,
        text_column: str = "text",
        context_lengths: List[int] = [2048, 4096, 8192, 16384],
        max_samples: int = 100,
        **kwargs,
    ) -> Dict[int, float]:
        """Evaluate perplexity on a HuggingFace dataset.

        Args:
            dataset: HuggingFace dataset.
            text_column: Column containing text data.
            context_lengths: Context lengths to evaluate.
            max_samples: Maximum number of samples to use.
            **kwargs: Additional arguments to evaluate().

        Returns:
            Dictionary mapping context_length -> perplexity.
        """
        texts = [sample[text_column] for sample in dataset.select(range(min(max_samples, len(dataset))))]
        return self.evaluate(texts, context_lengths=context_lengths, **kwargs)
