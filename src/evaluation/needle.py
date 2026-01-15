"""Needle-in-a-Haystack evaluation for long context retrieval.

Tests a model's ability to retrieve a specific fact ("needle") hidden
at various positions within a long context ("haystack").
"""

import random
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


class NeedleInHaystackEvaluator:
    """Evaluate needle retrieval at various positions and context lengths."""

    # Default needle template
    DEFAULT_NEEDLE = "The secret code is: {code}"
    DEFAULT_QUERY = "What is the secret code?"

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

    def generate_haystack(self, target_length: int, filler_text: str) -> str:
        """Generate haystack text of approximately target token length.

        Args:
            target_length: Target length in tokens.
            filler_text: Text to repeat as filler.

        Returns:
            Haystack text.
        """
        # Tokenize filler to know its length
        filler_tokens = self.tokenizer.encode(filler_text, add_special_tokens=False)
        filler_len = len(filler_tokens)

        # Repeat to reach target length
        repeats = (target_length // filler_len) + 1
        haystack = (filler_text + " ") * repeats

        # Truncate to exact length
        haystack_tokens = self.tokenizer.encode(haystack, add_special_tokens=False)[:target_length]
        haystack = self.tokenizer.decode(haystack_tokens)

        return haystack

    def insert_needle(
        self,
        haystack: str,
        needle: str,
        position_fraction: float,
    ) -> str:
        """Insert needle at a specific position in haystack.

        Args:
            haystack: Haystack text.
            needle: Needle text to insert.
            position_fraction: Position as fraction of haystack (0=start, 1=end).

        Returns:
            Haystack with needle inserted.
        """
        # Find insertion point (at sentence boundary if possible)
        words = haystack.split()
        insert_idx = int(len(words) * position_fraction)

        # Insert needle
        words.insert(insert_idx, needle)
        return " ".join(words)

    @torch.no_grad()
    def evaluate_single(
        self,
        haystack: str,
        needle: str,
        query: str,
        expected_answer: str,
        max_new_tokens: int = 50,
    ) -> Tuple[bool, str]:
        """Evaluate a single needle retrieval.

        Args:
            haystack: Full context with needle.
            needle: The needle that was inserted.
            query: Question to ask.
            expected_answer: Expected answer (the code).
            max_new_tokens: Max tokens to generate.

        Returns:
            Tuple of (success, generated_answer).
        """
        # Create prompt
        prompt = f"{haystack}\n\nQuestion: {query}\nAnswer:"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)

        # Check if answer is correct
        success = expected_answer.lower() in response.lower()

        return success, response.strip()

    def evaluate(
        self,
        context_lengths: List[int] = [2048, 4096, 8192],
        position_fractions: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
        num_trials: int = 5,
        filler_text: str = "The quick brown fox jumps over the lazy dog. " * 10,
        needle_template: str = DEFAULT_NEEDLE,
        query: str = DEFAULT_QUERY,
    ) -> Dict[Tuple[int, float], float]:
        """Run full needle-in-haystack evaluation.

        Args:
            context_lengths: Context lengths to test.
            position_fractions: Needle positions as fractions (0=start, 1=end).
            num_trials: Number of trials per (length, position) pair.
            filler_text: Text to use as haystack filler.
            needle_template: Template for needle with {code} placeholder.
            query: Question to ask about the needle.

        Returns:
            Dictionary mapping (context_length, position) -> accuracy.
        """
        results = {}

        for ctx_len in tqdm(context_lengths, desc="Context lengths"):
            for pos in tqdm(position_fractions, desc="Positions", leave=False):
                successes = 0

                for trial in range(num_trials):
                    # Generate random code
                    code = f"{random.randint(1000, 9999)}"
                    needle = needle_template.format(code=code)

                    # Create haystack with needle
                    haystack = self.generate_haystack(ctx_len - 100, filler_text)  # Reserve space
                    full_context = self.insert_needle(haystack, needle, pos)

                    # Evaluate
                    success, _ = self.evaluate_single(full_context, needle, query, code)
                    if success:
                        successes += 1

                accuracy = successes / num_trials
                results[(ctx_len, pos)] = accuracy
                print(f"  ctx={ctx_len}, pos={pos:.2f}: {accuracy:.1%}")

        return results

    def results_to_heatmap_data(
        self,
        results: Dict[Tuple[int, float], float],
    ) -> Tuple[List[int], List[float], List[List[float]]]:
        """Convert results to format suitable for heatmap plotting.

        Returns:
            Tuple of (context_lengths, positions, accuracy_matrix).
        """
        context_lengths = sorted(set(k[0] for k in results.keys()))
        positions = sorted(set(k[1] for k in results.keys()))

        matrix = []
        for ctx in context_lengths:
            row = [results.get((ctx, pos), 0.0) for pos in positions]
            matrix.append(row)

        return context_lengths, positions, matrix
