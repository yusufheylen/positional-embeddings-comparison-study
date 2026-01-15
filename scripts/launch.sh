#!/bin/bash
# Launch script for distributed training with accelerate
#
# Usage:
#   ./scripts/launch.sh <config> [accelerate_config] [extra_args...]
#
# Examples:
#   ./scripts/launch.sh configs/rope.yaml                     # Single GPU
#   ./scripts/launch.sh configs/rope.yaml zero1               # Multi-GPU with ZeRO-1
#   ./scripts/launch.sh configs/rope.yaml zero3 --seed 43     # Multi-GPU with ZeRO-3
#   ./scripts/launch.sh configs/pope.yaml single_gpu          # Explicit single GPU

set -e

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: ./scripts/launch.sh <config> [accelerate_config] [extra_args...]"
    echo ""
    echo "Accelerate configs:"
    echo "  single_gpu  - Single GPU, no DeepSpeed (default)"
    echo "  zero1       - Multi-GPU with ZeRO Stage 1"
    echo "  zero3       - Multi-GPU with ZeRO Stage 3"
    echo "  zero3_offload - ZeRO-3 with CPU offloading"
    echo ""
    echo "Examples:"
    echo "  ./scripts/launch.sh configs/rope.yaml"
    echo "  ./scripts/launch.sh configs/rope.yaml zero1"
    echo "  ./scripts/launch.sh configs/pope.yaml zero3 --seed 43"
    exit 1
fi

CONFIG=$1
ACCEL_CONFIG=${2:-single_gpu}
shift 2 2>/dev/null || shift 1

# Resolve accelerate config path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ACCEL_CONFIG_PATH="$PROJECT_DIR/configs/accelerate/${ACCEL_CONFIG}.yaml"

if [ ! -f "$ACCEL_CONFIG_PATH" ]; then
    echo "Error: Accelerate config not found: $ACCEL_CONFIG_PATH"
    echo "Available configs:"
    ls -1 "$PROJECT_DIR/configs/accelerate/"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: Training config not found: $CONFIG"
    exit 1
fi

echo "========================================"
echo "PE Comparison Study - Training Launch"
echo "========================================"
echo "Training config: $CONFIG"
echo "Accelerate config: $ACCEL_CONFIG_PATH"
echo "Extra args: $@"
echo "========================================"

# Launch with accelerate
if [ "$ACCEL_CONFIG" = "single_gpu" ]; then
    # Single GPU - run directly without accelerate
    echo "Running single GPU training..."
    python "$PROJECT_DIR/scripts/train.py" --config "$CONFIG" "$@"
else
    # Multi-GPU - use accelerate
    echo "Running distributed training with accelerate..."
    accelerate launch --config_file "$ACCEL_CONFIG_PATH" \
        "$PROJECT_DIR/scripts/train.py" --config "$CONFIG" "$@"
fi

echo "Training complete!"
