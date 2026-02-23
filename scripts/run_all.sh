#!/bin/bash
# Run experiments for PE comparison study (from-scratch training)
#
# Usage:
#   ./scripts/run_all.sh rope                    # Run single experiment
#   ./scripts/run_all.sh rope --accel zero1      # With accelerate config
#   ./scripts/run_all.sh baselines               # Run all 4 baseline experiments
#   ./scripts/run_all.sh baselines --seeds       # Run all baselines with multiple seeds

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIGS_DIR="$PROJECT_DIR/configs"

# Default settings
SEEDS=(42 43 44)
RUN_SEEDS=false
ACCEL_CONFIG="single_gpu"

# Parse arguments
EXPERIMENT="$1"
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)
            RUN_SEEDS=true
            shift
            ;;
        --accel)
            ACCEL_CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to run a single experiment
run_experiment() {
    local config=$1
    local seed=$2

    echo "========================================"
    echo "Running: $config (seed=$seed, accel=$ACCEL_CONFIG)"
    echo "========================================"

    "$SCRIPT_DIR/launch.sh" "$CONFIGS_DIR/$config.yaml" "$ACCEL_CONFIG" --seed "$seed"
}

# Function to run experiment with optional multiple seeds
run_with_seeds() {
    local config=$1

    if [ "$RUN_SEEDS" = true ]; then
        for seed in "${SEEDS[@]}"; do
            run_experiment "$config" "$seed"
        done
    else
        run_experiment "$config" 42
    fi
}

# Main logic - Updated for from-scratch training configs
case "$EXPERIMENT" in
    nope|run0)
        run_with_seeds "nope_scratch"
        ;;
    rope|run1)
        run_with_seeds "rope_scratch"
        ;;
    yarn|run2)
        run_with_seeds "yarn_scratch"
        ;;
    pope|run3)
        run_with_seeds "pope_scratch"
        ;;
    baselines)
        echo "Running baseline experiments (runs 0-3)..."
        run_with_seeds "nope_scratch"
        run_with_seeds "rope_scratch"
        run_with_seeds "yarn_scratch"
        run_with_seeds "pope_scratch"
        ;;
    all)
        # alias for baselines
        echo "Running baseline experiments (runs 0-3)..."
        run_with_seeds "nope_scratch"
        run_with_seeds "rope_scratch"
        run_with_seeds "yarn_scratch"
        run_with_seeds "pope_scratch"
        ;;
    *)
        echo "Usage: $0 {nope|rope|yarn|pope|baselines|all} [options]"
        echo ""
        echo "Experiments (from-scratch training, 16k steps):"
        echo "  nope (run0)       - No Positional Embeddings"
        echo "  rope (run1)       - Rotary Position Embeddings"
        echo "  yarn (run2)       - RoPE with YaRN scaling"
        echo "  pope (run3)       - Polar Position Embeddings"
        echo ""
        echo "Groups:"
        echo "  baselines (all)   - Run all 4 baseline experiments (runs 0-3)"
        echo ""
        echo "Options:"
        echo "  --seeds              - Run with multiple seeds (42, 43, 44)"
        echo "  --accel <config>     - Accelerate config (single_gpu, zero1, zero3, zero3_offload)"
        echo ""
        echo "Examples:"
        echo "  $0 rope                          # Single GPU training"
        echo "  $0 rope --accel zero1            # Multi-GPU with ZeRO-1"
        echo "  $0 baselines --seeds --accel zero3  # All baselines, multi-seed, ZeRO-3"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Experiments complete!"
echo "========================================"
