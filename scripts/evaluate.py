#!/usr/bin/env python3
"""Evaluation script for PE comparison study.

Usage:
    python scripts/evaluate.py --checkpoint outputs/rope --eval perplexity
    python scripts/evaluate.py --checkpoint outputs/rope --eval all
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import NeedleInHaystackEvaluator, PasskeyRetrievalEvaluator, PerplexityEvaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate PE comparison models")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
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
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    # Load model and tokenizer
    print(f"Loading model from {args.checkpoint}...")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        dtype=torch.bfloat16,
        device_map=args.device,
    )

    results = {"checkpoint": args.checkpoint, "context_lengths": args.context_lengths}

    # Perplexity evaluation
    if args.eval in ["perplexity", "all"]:
        print("\n=== Perplexity Evaluation ===")
        evaluator = PerplexityEvaluator(model, tokenizer, device=args.device)

        # Load eval data (using a small subset)
        from datasets import load_dataset

        eval_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [t for t in eval_data["text"] if len(t) > 100][:50]

        ppl_results = evaluator.evaluate(texts, context_lengths=args.context_lengths)
        results["perplexity"] = ppl_results
        print(f"Perplexity results: {ppl_results}")

    # Needle-in-haystack evaluation
    if args.eval in ["needle", "all"]:
        print("\n=== Needle-in-Haystack Evaluation ===")
        evaluator = NeedleInHaystackEvaluator(model, tokenizer, device=args.device)

        needle_results = evaluator.evaluate(
            context_lengths=args.context_lengths,
            position_fractions=[0.0, 0.25, 0.5, 0.75, 1.0],
            num_trials=5,
        )
        # Convert tuple keys to strings for JSON
        results["needle"] = {f"{k[0]}_{k[1]}": v for k, v in needle_results.items()}
        print(f"Needle results: {needle_results}")

    # Passkey retrieval evaluation
    if args.eval in ["passkey", "all"]:
        print("\n=== Passkey Retrieval Evaluation ===")
        evaluator = PasskeyRetrievalEvaluator(model, tokenizer, device=args.device)

        passkey_results = evaluator.evaluate_lengths_only(
            context_lengths=args.context_lengths,
            num_trials=10,
        )
        results["passkey"] = passkey_results
        print(f"Passkey results: {passkey_results}")

    # Save results
    output_path = args.output or f"{args.checkpoint}/eval_results.json"
    print(f"\nSaving results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation complete!")
    return results


if __name__ == "__main__":
    main()
