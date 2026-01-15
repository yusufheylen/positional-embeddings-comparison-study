#!/usr/bin/env python3
"""Aggregate evaluation results from all checkpoints.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --output results/summary.csv
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def load_results(outputs_dir: str) -> dict:
    """Load all eval_results.json files from outputs directory."""
    results = {}
    outputs_path = Path(outputs_dir)

    for checkpoint_dir in outputs_path.iterdir():
        if not checkpoint_dir.is_dir():
            continue

        results_file = checkpoint_dir / "eval_results.json"
        if results_file.exists():
            with open(results_file) as f:
                results[checkpoint_dir.name] = json.load(f)
            print(f"Loaded: {checkpoint_dir.name}")

    return results


def create_perplexity_table(results: dict) -> pd.DataFrame:
    """Create perplexity comparison table."""
    rows = []
    for name, data in results.items():
        if "perplexity" not in data:
            continue

        row = {"method": name, "pe_type": data.get("pe_type", "unknown")}
        for ctx_len, ppl in data["perplexity"].items():
            row[f"ppl_{ctx_len}"] = ppl
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("method")
    return df


def create_needle_table(results: dict) -> pd.DataFrame:
    """Create needle-in-haystack comparison table."""
    rows = []
    for name, data in results.items():
        if "needle" not in data:
            continue

        for key, acc in data["needle"].items():
            ctx_len, pos = key.split("_")
            rows.append(
                {
                    "method": name,
                    "pe_type": data.get("pe_type", "unknown"),
                    "context_length": int(ctx_len),
                    "position": float(pos),
                    "accuracy": acc,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def create_passkey_table(results: dict) -> pd.DataFrame:
    """Create passkey retrieval comparison table."""
    rows = []
    for name, data in results.items():
        if "passkey" not in data:
            continue

        row = {"method": name, "pe_type": data.get("pe_type", "unknown")}
        for ctx_len, acc in data["passkey"].items():
            row[f"passkey_{ctx_len}"] = acc
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("method")
    return df


def print_summary(results: dict):
    """Print summary of all results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)

    # Perplexity
    ppl_df = create_perplexity_table(results)
    if not ppl_df.empty:
        print("\n--- Perplexity (lower is better) ---")
        print(ppl_df.to_string(index=False))

    # Passkey
    passkey_df = create_passkey_table(results)
    if not passkey_df.empty:
        print("\n--- Passkey Retrieval Accuracy (higher is better) ---")
        print(passkey_df.to_string(index=False))

    # Needle summary (just average per method)
    needle_df = create_needle_table(results)
    if not needle_df.empty:
        print("\n--- Needle-in-Haystack Average Accuracy (higher is better) ---")
        avg_needle = needle_df.groupby(["method", "pe_type"])["accuracy"].mean().reset_index()
        avg_needle.columns = ["method", "pe_type", "avg_accuracy"]
        print(avg_needle.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for CSV files",
    )
    args = parser.parse_args()

    # Load all results
    results = load_results(args.outputs_dir)

    if not results:
        print(f"No eval_results.json files found in {args.outputs_dir}")
        return

    # Print summary
    print_summary(results)

    # Save CSV files if output specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        ppl_df = create_perplexity_table(results)
        if not ppl_df.empty:
            ppl_df.to_csv(output_dir / "perplexity.csv", index=False)
            print(f"\nSaved: {output_dir / 'perplexity.csv'}")

        needle_df = create_needle_table(results)
        if not needle_df.empty:
            needle_df.to_csv(output_dir / "needle.csv", index=False)
            print(f"Saved: {output_dir / 'needle.csv'}")

        passkey_df = create_passkey_table(results)
        if not passkey_df.empty:
            passkey_df.to_csv(output_dir / "passkey.csv", index=False)
            print(f"Saved: {output_dir / 'passkey.csv'}")

        # Also save raw JSON
        with open(output_dir / "all_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {output_dir / 'all_results.json'}")


if __name__ == "__main__":
    main()
