#!/bin/bash
# Scaffold experiments runner
# Usage: ./scripts/run_scaffold_experiments.sh

set -e

cd "$(dirname "$0")/.."

echo "=== PE Study: Scaffold Experiments ==="
echo "Working directory: $(pwd)"
echo ""

# Check checkpoints exist
CHECKPOINTS=(
    "../initial-run-outputs/outputs/run1_rope/checkpoint-10000"
    "../initial-run-outputs/outputs/run1_rope/checkpoint-15000"
    "../initial-run-outputs/outputs/run2_yarn/checkpoint-10000"
    "../initial-run-outputs/outputs/run2_yarn/checkpoint-15000"
    "../initial-run-outputs/outputs/run3_pope/checkpoint-10000"
    "../initial-run-outputs/outputs/run3_pope/checkpoint-15000"
)

echo "Checking checkpoints..."
for ckpt in "${CHECKPOINTS[@]}"; do
    if [ ! -d "$ckpt" ]; then
        echo "ERROR: Missing checkpoint: $ckpt"
        exit 1
    fi
done
echo "All checkpoints found."
echo ""

# Run experiments
CONFIGS=(
    "configs/scaffold_rope_10k.yaml"
    "configs/scaffold_rope_15k.yaml"
    "configs/scaffold_yarn_10k.yaml"
    "configs/scaffold_yarn_15k.yaml"
    "configs/scaffold_pope_10k.yaml"
    "configs/scaffold_pope_15k.yaml"
)

for config in "${CONFIGS[@]}"; do
    name=$(basename "$config" .yaml)
    echo "=========================================="
    echo "Running: $name"
    echo "=========================================="
    python scripts/train.py --config "$config"
    echo ""
done

echo "=== All scaffold experiments complete ==="
