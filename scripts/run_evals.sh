#!/bin/bash
# Run evaluations for PE comparison study
#
# Usage:
#   ./scripts/run_evals.sh                          # Evaluate all checkpoints
#   ./scripts/run_evals.sh outputs/rope             # Evaluate single checkpoint
#   ./scripts/run_evals.sh --eval perplexity        # Run only perplexity eval
#   ./scripts/run_evals.sh --wandb                  # Log to W&B

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUTS_DIR="$PROJECT_DIR/outputs"

# Default settings
EVAL_TYPE="all"
USE_WANDB=false
CONTEXT_LENGTHS="2048 4096 8192 16384"
CHECKPOINT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --eval)
            EVAL_TYPE="$2"
            shift 2
            ;;
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --context-lengths)
            CONTEXT_LENGTHS="$2"
            shift 2
            ;;
        outputs/*|/*)
            CHECKPOINT="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to run evaluation on a single checkpoint
run_eval() {
    local checkpoint=$1
    local pe_type=$2

    if [ ! -d "$checkpoint" ]; then
        echo "Checkpoint not found: $checkpoint"
        return 1
    fi

    echo "========================================"
    echo "Evaluating: $checkpoint"
    echo "PE Type: $pe_type"
    echo "========================================"

    local cmd="python $SCRIPT_DIR/evaluate.py --checkpoint $checkpoint --pe-type $pe_type --eval $EVAL_TYPE --context-lengths $CONTEXT_LENGTHS"

    if [ "$USE_WANDB" = true ]; then
        cmd="$cmd --wandb"
    fi

    eval $cmd
}

# Function to get PE type for a checkpoint name
get_pe_type() {
    local name=$1
    case "$name" in
        nope) echo "nope" ;;
        rope) echo "rope" ;;
        rope_yarn) echo "yarn" ;;
        pope) echo "pope" ;;
        drope_from_rope) echo "nope" ;;  # After switch
        drope_from_pope) echo "nope" ;;  # After switch
        *) echo "rope" ;;  # Default
    esac
}

# Main logic
if [ -n "$CHECKPOINT" ]; then
    # Single checkpoint evaluation
    # Try to detect PE type from directory name
    basename=$(basename "$CHECKPOINT")
    PE_TYPE=$(get_pe_type "$basename")
    run_eval "$CHECKPOINT" "$PE_TYPE"
else
    # Evaluate all checkpoints
    echo "Scanning for checkpoints in $OUTPUTS_DIR..."

    CHECKPOINT_NAMES="nope rope rope_yarn pope drope_from_rope drope_from_pope"

    found=0
    for name in $CHECKPOINT_NAMES; do
        checkpoint="$OUTPUTS_DIR/$name"
        if [ -d "$checkpoint" ]; then
            pe_type=$(get_pe_type "$name")
            run_eval "$checkpoint" "$pe_type"
            found=$((found + 1))
        fi
    done

    if [ $found -eq 0 ]; then
        echo "No checkpoints found in $OUTPUTS_DIR"
        echo "Expected directories: nope, rope, rope_yarn, pope, drope_from_rope, drope_from_pope"
        exit 1
    fi

    echo ""
    echo "========================================"
    echo "Evaluated $found checkpoint(s)"
    echo "========================================"
fi
