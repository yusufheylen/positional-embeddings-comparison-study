#!/bin/bash
# Run experiments for PE comparison study
#
# Usage:
#   ./scripts/run_all.sh rope                    # Run single experiment
#   ./scripts/run_all.sh rope --accel zero1      # With accelerate config
#   ./scripts/run_all.sh all                     # Run all experiments
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

# Main logic
case "$EXPERIMENT" in
    nope)
        run_with_seeds "nope"
        ;;
    rope)
        run_with_seeds "rope"
        ;;
    rope_yarn|yarn)
        run_with_seeds "rope_yarn"
        ;;
    pope)
        run_with_seeds "pope"
        ;;
    drope_from_rope|drope_rope)
        run_with_seeds "drope_from_rope"
        ;;
    drope_from_pope|drope_pope)
        run_with_seeds "drope_from_pope"
        ;;
    baselines)
        echo "Running baseline experiments..."
        run_with_seeds "nope"
        run_with_seeds "rope"
        ;;
    all)
        echo "Running all experiments..."
        run_with_seeds "nope"
        run_with_seeds "rope"
        run_with_seeds "rope_yarn"
        run_with_seeds "pope"
        run_with_seeds "drope_from_rope"
        run_with_seeds "drope_from_pope"
        ;;
    *)
        echo "Usage: $0 {nope|rope|rope_yarn|pope|drope_from_rope|drope_from_pope|baselines|all} [options]"
        echo ""
        echo "Experiments:"
        echo "  nope            - No Positional Embeddings"
        echo "  rope            - Rotary Position Embeddings"
        echo "  rope_yarn       - RoPE with YaRN scaling"
        echo "  pope            - Polar Position Embeddings"
        echo "  drope_from_rope - DroPE (RoPE -> NoPE at 70%)"
        echo "  drope_from_pope - DroPE (PoPE -> NoPE at 70%)"
        echo "  baselines       - Run nope and rope baselines"
        echo "  all             - Run all experiments"
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
