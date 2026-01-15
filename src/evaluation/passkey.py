"""Passkey retrieval evaluation for long context.

Tests a model's ability to retrieve a hidden passkey from within
distractor text at various context lengths.
"""

import random
import string
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer


class PasskeyRetrievalEvaluator:
    """Evaluate passkey retrieval at various context lengths."""

    # Prompt templates
    PASSKEY_TEMPLATE = "The passkey is {passkey}. Remember it."
    DISTRACTOR = "The grass is green. The sky is blue. The sun is yellow. "
    QUERY = "What is the passkey?"

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

    def generate_passkey(self, length: int = 5) -> str:
        """Generate a random passkey.

        Args:
            length: Length of passkey.

        Returns:
            Random alphanumeric passkey.
        """
        return "".join(random.choices(string.digits, k=length))

    def create_prompt(
        self,
        passkey: str,
        target_length: int,
        position_fraction: float = 0.5,
    ) -> str:
        """Create a prompt with passkey hidden in distractor text.

        Args:
            passkey: The passkey to hide.
            target_length: Target prompt length in tokens.
            position_fraction: Where to place passkey (0=start, 1=end).

        Returns:
            Full prompt with hidden passkey.
        """
        passkey_text = self.PASSKEY_TEMPLATE.format(passkey=passkey)

        # Tokenize components to estimate lengths
        passkey_tokens = len(self.tokenizer.encode(passkey_text, add_special_tokens=False))
        query_tokens = len(self.tokenizer.encode(self.QUERY, add_special_tokens=False))
        distractor_tokens = len(self.tokenizer.encode(self.DISTRACTOR, add_special_tokens=False))

        # Calculate distractor repetitions
        available_tokens = target_length - passkey_tokens - query_tokens - 50  # Buffer
        num_distractors = max(1, available_tokens // distractor_tokens)

        # Create distractor blocks
        before_ratio = position_fraction
        num_before = int(num_distractors * before_ratio)
        num_after = num_distractors - num_before

        before_text = self.DISTRACTOR * num_before
        after_text = self.DISTRACTOR * num_after

        # Assemble prompt
        prompt = f"{before_text}{passkey_text} {after_text}\n\n{self.QUERY} The passkey is"

        return prompt

    @torch.no_grad()
    def evaluate_single(
        self,
        prompt: str,
        passkey: str,
        max_new_tokens: int = 20,
    ) -> Tuple[bool, str]:
        """Evaluate a single passkey retrieval.

        Args:
            prompt: Full prompt with hidden passkey.
            passkey: The correct passkey.
            max_new_tokens: Max tokens to generate.

        Returns:
            Tuple of (success, generated_response).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)

        # Check if passkey is in response
        success = passkey in response

        return success, response.strip()

    def evaluate(
        self,
        context_lengths: List[int] = [2048, 4096, 8192, 16384],
        position_fractions: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
        num_trials: int = 10,
        passkey_length: int = 5,
    ) -> Dict[Tuple[int, float], float]:
        """Run full passkey retrieval evaluation.

        Args:
            context_lengths: Context lengths to test.
            position_fractions: Passkey positions (0=start, 1=end).
            num_trials: Number of trials per configuration.
            passkey_length: Length of generated passkeys.

        Returns:
            Dictionary mapping (context_length, position) -> accuracy.
        """
        results = {}

        for ctx_len in tqdm(context_lengths, desc="Context lengths"):
            for pos in tqdm(position_fractions, desc="Positions", leave=False):
                successes = 0

                for _ in range(num_trials):
                    passkey = self.generate_passkey(passkey_length)
                    prompt = self.create_prompt(passkey, ctx_len, pos)

                    success, _ = self.evaluate_single(prompt, passkey)
                    if success:
                        successes += 1

                accuracy = successes / num_trials
                results[(ctx_len, pos)] = accuracy
                print(f"  ctx={ctx_len}, pos={pos:.2f}: {accuracy:.1%}")

        return results

    def evaluate_lengths_only(
        self,
        context_lengths: List[int] = [2048, 4096, 8192, 16384],
        num_trials: int = 20,
        passkey_length: int = 5,
    ) -> Dict[int, float]:
        """Evaluate at different context lengths with passkey at middle.

        Args:
            context_lengths: Context lengths to test.
            num_trials: Number of trials per length.
            passkey_length: Length of generated passkeys.

        Returns:
            Dictionary mapping context_length -> accuracy.
        """
        results = {}

        for ctx_len in tqdm(context_lengths, desc="Evaluating"):
            successes = 0

            for _ in range(num_trials):
                passkey = self.generate_passkey(passkey_length)
                prompt = self.create_prompt(passkey, ctx_len, position_fraction=0.5)

                success, _ = self.evaluate_single(prompt, passkey)
                if success:
                    successes += 1

            accuracy = successes / num_trials
            results[ctx_len] = accuracy
            print(f"  ctx={ctx_len}: {accuracy:.1%}")

        return results
