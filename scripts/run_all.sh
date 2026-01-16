#!/bin/bash
# Run experiments for PE comparison study (from-scratch training)
#
# Usage:
#   ./scripts/run_all.sh rope                    # Run single experiment
#   ./scripts/run_all.sh rope --accel zero1      # With accelerate config
#   ./scripts/run_all.sh all                     # Run all 7 experiments
#   ./scripts/run_all.sh all --seeds             # Run all with multiple seeds

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
    drope_rope|run4a)
        run_with_seeds "drope_rope"
        ;;
    drope_yarn|run4b)
        run_with_seeds "drope_yarn"
        ;;
    drope_pope|run4c)
        run_with_seeds "drope_pope"
        ;;
    baselines)
        echo "Running baseline experiments (runs 0-3)..."
        run_with_seeds "nope_scratch"
        run_with_seeds "rope_scratch"
        run_with_seeds "yarn_scratch"
        run_with_seeds "pope_scratch"
        ;;
    drope)
        echo "Running DroPE experiments (runs 4a-4c)..."
        run_with_seeds "drope_rope"
        run_with_seeds "drope_yarn"
        run_with_seeds "drope_pope"
        ;;
    all)
        echo "Running all 7 experiments..."
        echo "Run 0: NoPE"
        run_with_seeds "nope_scratch"
        echo "Run 1: RoPE"
        run_with_seeds "rope_scratch"
        echo "Run 2: YaRN"
        run_with_seeds "yarn_scratch"
        echo "Run 3: PoPE"
        run_with_seeds "pope_scratch"
        echo "Run 4a: DroPE (RoPE → NoPE)"
        run_with_seeds "drope_rope"
        echo "Run 4b: DroPE (YaRN → NoPE)"
        run_with_seeds "drope_yarn"
        echo "Run 4c: DroPE (PoPE → NoPE)"
        run_with_seeds "drope_pope"
        ;;
    *)
        echo "Usage: $0 {nope|rope|yarn|pope|drope_rope|drope_yarn|drope_pope|baselines|drope|all} [options]"
        echo ""
        echo "Experiments (from-scratch training, 16k steps):"
        echo "  nope (run0)       - No Positional Embeddings"
        echo "  rope (run1)       - Rotary Position Embeddings"
        echo "  yarn (run2)       - RoPE with YaRN scaling"
        echo "  pope (run3)       - Polar Position Embeddings"
        echo "  drope_rope (run4a) - DroPE (RoPE → NoPE at 87.5%)"
        echo "  drope_yarn (run4b) - DroPE (YaRN → NoPE at 87.5%)"
        echo "  drope_pope (run4c) - DroPE (PoPE → NoPE at 87.5%)"
        echo ""
        echo "Groups:"
        echo "  baselines         - Run baseline experiments (runs 0-3)"
        echo "  drope             - Run DroPE experiments (runs 4a-4c)"
        echo "  all               - Run all 7 experiments"
        echo ""
        echo "Options:"
        echo "  --seeds              - Run with multiple seeds (42, 43, 44)"
        echo "  --accel <config>     - Accelerate config (single_gpu, zero1, zero3, zero3_offload)"
        echo ""
        echo "Examples:"
        echo "  $0 rope                          # Single GPU training"
        echo "  $0 rope --accel zero1            # Multi-GPU with ZeRO-1"
        echo "  $0 all --seeds --accel zero3     # All experiments, multi-seed, ZeRO-3"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "Experiments complete!"
echo "========================================"
