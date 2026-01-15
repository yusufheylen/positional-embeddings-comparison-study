#!/usr/bin/env python3
"""Evaluation script for PE comparison study.

Usage:
    python scripts/evaluate.py --checkpoint outputs/rope --eval perplexity
    python scripts/evaluate.py --checkpoint outputs/rope --eval all
    python scripts/evaluate.py --checkpoint outputs/rope --eval all --wandb
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import NeedleInHaystackEvaluator, PasskeyRetrievalEvaluator, PerplexityEvaluator
from src.models import create_model


def load_pe_type_from_checkpoint(checkpoint_path: str) -> Optional[str]:
    """Try to detect PE type from checkpoint config."""
    config_path = Path(checkpoint_path) / "training_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config.get("pe_type")

    # Try to detect from model config
    model_config_path = Path(checkpoint_path) / "config.json"
    if model_config_path.exists():
        with open(model_config_path) as f:
            config = json.load(f)
            # Check for our custom PE markers
            if config.get("_pe_type"):
                return config["_pe_type"]

    return None


def load_model_for_eval(
    checkpoint_path: str,
    pe_type: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load model with correct PE configuration.

    Args:
        checkpoint_path: Path to model checkpoint.
        pe_type: PE type override. If None, attempts auto-detection.
        device: Device to load model on.
        dtype: Model dtype.

    Returns:
        Tuple of (model, tokenizer).
    """
    # Try to detect PE type
    if pe_type is None:
        pe_type = load_pe_type_from_checkpoint(checkpoint_path)

    if pe_type is None:
        print("Warning: Could not detect PE type, defaulting to 'rope'")
        pe_type = "rope"

    print(f"Loading model with PE type: {pe_type}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Load model with correct PE configuration
    model = create_model(
        checkpoint_path,
        pe_type=pe_type,
        dtype=dtype,
        attn_implementation="eager",  # Use eager for eval compatibility
    )

    model.to(device)
    model.eval()

    return model, tokenizer


def run_perplexity_eval(
    model,
    tokenizer,
    context_lengths: List[int],
    device: str,
    max_samples: int = 50,
) -> Dict[int, float]:
    """Run perplexity evaluation."""
    from datasets import load_dataset

    print("\n=== Perplexity Evaluation ===")
    evaluator = PerplexityEvaluator(model, tokenizer, device=device)

    # Load eval data
    print("Loading WikiText-2 evaluation data...")
    eval_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in eval_data["text"] if len(t) > 500][:max_samples]

    results = evaluator.evaluate(texts, context_lengths=context_lengths)
    return results


def run_needle_eval(
    model,
    tokenizer,
    context_lengths: List[int],
    device: str,
    num_trials: int = 5,
) -> Dict[str, float]:
    """Run needle-in-haystack evaluation."""
    print("\n=== Needle-in-Haystack Evaluation ===")
    evaluator = NeedleInHaystackEvaluator(model, tokenizer, device=device)

    results = evaluator.evaluate(
        context_lengths=context_lengths,
        position_fractions=[0.0, 0.25, 0.5, 0.75, 1.0],
        num_trials=num_trials,
    )

    # Convert tuple keys to strings for JSON
    return {f"{k[0]}_{k[1]}": v for k, v in results.items()}


def run_passkey_eval(
    model,
    tokenizer,
    context_lengths: List[int],
    device: str,
    num_trials: int = 10,
) -> Dict[int, float]:
    """Run passkey retrieval evaluation."""
    print("\n=== Passkey Retrieval Evaluation ===")
    evaluator = PasskeyRetrievalEvaluator(model, tokenizer, device=device)

    results = evaluator.evaluate_lengths_only(
        context_lengths=context_lengths,
        num_trials=num_trials,
    )
    return results


def log_to_wandb(results: dict, checkpoint_name: str, pe_type: str):
    """Log evaluation results to W&B."""
    import wandb

    # Create tables for each evaluation type
    if "perplexity" in results:
        ppl_table = wandb.Table(columns=["context_length", "perplexity"])
        for ctx_len, ppl in results["perplexity"].items():
            ppl_table.add_data(ctx_len, ppl)
        wandb.log({"perplexity_table": ppl_table})

        # Also log as scalars for easier comparison
        for ctx_len, ppl in results["perplexity"].items():
            wandb.log({f"perplexity/ctx_{ctx_len}": ppl})

    if "needle" in results:
        needle_table = wandb.Table(columns=["context_length", "position", "accuracy"])
        for key, acc in results["needle"].items():
            ctx_len, pos = key.split("_")
            needle_table.add_data(int(ctx_len), float(pos), acc)
        wandb.log({"needle_table": needle_table})

    if "passkey" in results:
        passkey_table = wandb.Table(columns=["context_length", "accuracy"])
        for ctx_len, acc in results["passkey"].items():
            passkey_table.add_data(ctx_len, acc)
        wandb.log({"passkey_table": passkey_table})

        for ctx_len, acc in results["passkey"].items():
            wandb.log({f"passkey/ctx_{ctx_len}": acc})


def main():
    parser = argparse.ArgumentParser(description="Evaluate PE comparison models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--pe-type",
        type=str,
        choices=["rope", "nope", "pope", "yarn"],
        default=None,
        help="PE type (auto-detected if not specified)",
    )
    parser.add_argument(
        "--eval",
        type=str,
        choices=["perplexity", "needle", "passkey", "all"],
        default="all",
        help="Evaluation type",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192, 16384],
        help="Context lengths to evaluate",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--wandb", action="store_true", help="Log results to W&B")
    parser.add_argument("--wandb-project", type=str, default="pe-comparison-eval", help="W&B project name")
    parser.add_argument("--num-trials", type=int, default=5, help="Number of trials for needle/passkey")
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples for perplexity eval")
    args = parser.parse_args()

    # Auto-detect device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    print(f"Using device: {args.device}")

    # Load model
    model, tokenizer = load_model_for_eval(
        args.checkpoint,
        pe_type=args.pe_type,
        device=args.device,
    )

    # Determine PE type for results
    pe_type = args.pe_type or load_pe_type_from_checkpoint(args.checkpoint) or "unknown"
    checkpoint_name = Path(args.checkpoint).name

    # Initialize W&B if requested
    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=f"eval-{checkpoint_name}",
            config={
                "checkpoint": args.checkpoint,
                "pe_type": pe_type,
                "context_lengths": args.context_lengths,
                "eval_types": args.eval,
            },
        )

    results = {
        "checkpoint": args.checkpoint,
        "pe_type": pe_type,
        "context_lengths": args.context_lengths,
    }

    # Run evaluations
    if args.eval in ["perplexity", "all"]:
        results["perplexity"] = run_perplexity_eval(
            model,
            tokenizer,
            args.context_lengths,
            args.device,
            max_samples=args.max_samples,
        )
        print(f"\nPerplexity results: {results['perplexity']}")

    if args.eval in ["needle", "all"]:
        results["needle"] = run_needle_eval(
            model,
            tokenizer,
            args.context_lengths,
            args.device,
            num_trials=args.num_trials,
        )
        print(f"\nNeedle results: {results['needle']}")

    if args.eval in ["passkey", "all"]:
        results["passkey"] = run_passkey_eval(
            model,
            tokenizer,
            args.context_lengths,
            args.device,
            num_trials=args.num_trials * 2,  # More trials for passkey
        )
        print(f"\nPasskey results: {results['passkey']}")

    # Log to W&B
    if args.wandb:
        log_to_wandb(results, checkpoint_name, pe_type)
        wandb.finish()

    # Save results
    output_path = args.output or f"{args.checkpoint}/eval_results.json"
    os.makedirs(Path(output_path).parent, exist_ok=True)
    print(f"\nSaving results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print("Evaluation complete!")
    print("=" * 50)

    return results


if __name__ == "__main__":
    main()
