#!/bin/bash
# Run evaluations for PE comparison study
#
# Usage:
#   ./scripts/run_evals.sh                          # Evaluate all checkpoints found
#   ./scripts/run_evals.sh --baselines              # Evaluate only baseline runs
#   ./scripts/run_evals.sh --scaffold               # Evaluate only scaffold runs
#   ./scripts/run_evals.sh outputs/scaffold_rope_10k # Evaluate single checkpoint
#   ./scripts/run_evals.sh --eval perplexity        # Run only perplexity eval
#   ./scripts/run_evals.sh --wandb                  # Log to W&B

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUTS_DIR="$PROJECT_DIR/outputs"
BASELINE_DIR="$PROJECT_DIR/../initial-run-outputs/outputs"

# Default settings
EVAL_TYPE="all"
USE_WANDB=false
CONTEXT_LENGTHS=(1024 2048 4096 8192 16384)
CHECKPOINT=""
RUN_SET="all"  # all, baselines, scaffold

usage_error() {
    echo "Error: $1" >&2
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --eval)
            if [[ $# -lt 2 ]] || [[ "$2" == --* ]]; then
                usage_error "--eval requires a value (perplexity|needle|passkey|all)"
            fi
            EVAL_TYPE="$2"
            shift 2
            ;;
        --wandb)
            USE_WANDB=true
            shift
            ;;
        --context-lengths)
            shift
            CONTEXT_LENGTHS=()
            while [[ $# -gt 0 ]]; do
                if [[ "$1" == --* ]]; then
                    break
                fi
                if [[ "$1" =~ ^[0-9]+$ ]]; then
                    CONTEXT_LENGTHS+=("$1")
                    shift
                else
                    break
                fi
            done
            if [[ ${#CONTEXT_LENGTHS[@]} -eq 0 ]]; then
                usage_error "--context-lengths requires one or more integer values"
            fi
            ;;
        --baselines)
            RUN_SET="baselines"
            shift
            ;;
        --scaffold)
            RUN_SET="scaffold"
            shift
            ;;
        *)
            if [[ "$1" == --* ]]; then
                usage_error "Unknown option: $1"
            fi
            if [[ -n "$CHECKPOINT" ]]; then
                usage_error "Multiple checkpoint paths provided: '$CHECKPOINT' and '$1'"
            fi
            CHECKPOINT="$1"
            shift
            ;;
    esac
done

# Function to run evaluation on a single checkpoint
run_eval() {
    local checkpoint="$1"
    local pe_type="$2"

    if [ ! -d "$checkpoint" ]; then
        echo "Checkpoint not found: $checkpoint"
        return 1
    fi

    echo "========================================"
    echo "Evaluating: $checkpoint"
    echo "PE Type: $pe_type"
    echo "========================================"

    local cmd=(
        python
        "$SCRIPT_DIR/evaluate.py"
        --checkpoint "$checkpoint"
        --pe-type "$pe_type"
        --eval "$EVAL_TYPE"
        --context-lengths "${CONTEXT_LENGTHS[@]}"
    )

    if [ "$USE_WANDB" = true ]; then
        cmd+=(--wandb)
    fi

    "${cmd[@]}"
}

# PE type lookup by directory name
get_pe_type() {
    local name=$1
    case "$name" in
        run0_nope)          echo "nope" ;;
        run1_rope)          echo "rope" ;;
        run2_yarn)          echo "yarn" ;;
        run3_pope)          echo "pope" ;;
        scaffold_rope_10k)  echo "nope" ;;
        scaffold_rope_15k)  echo "nope" ;;
        scaffold_yarn_10k)  echo "nope" ;;
        scaffold_yarn_15k)  echo "nope" ;;
        scaffold_pope_10k)  echo "nope" ;;
        scaffold_pope_15k)  echo "nope" ;;
        *)
            echo "unknown"
            echo "WARNING: Could not detect PE type for '$name'" >&2
            ;;
    esac
}

# Checkpoint definitions
# Baselines live in ../initial-run-outputs/outputs/
# Scaffold outputs live in ./outputs/
BASELINE_RUNS="run0_nope run1_rope run2_yarn run3_pope"
SCAFFOLD_RUNS="scaffold_rope_10k scaffold_rope_15k scaffold_yarn_10k scaffold_yarn_15k scaffold_pope_10k scaffold_pope_15k"

# Main logic
if [ -n "$CHECKPOINT" ]; then
    # Single checkpoint evaluation
    dir_name=$(basename "$CHECKPOINT")
    PE_TYPE=$(get_pe_type "$dir_name")
    if [ "$PE_TYPE" = "unknown" ]; then
        echo "ERROR: Cannot detect PE type for '$dir_name'. Pass --pe-type manually via evaluate.py."
        exit 1
    fi
    run_eval "$CHECKPOINT" "$PE_TYPE"
else
    echo "=== PE Study: Evaluation ==="
    echo ""

    found=0

    # Baseline runs
    if [ "$RUN_SET" = "all" ] || [ "$RUN_SET" = "baselines" ]; then
        echo "--- Baseline runs (from $BASELINE_DIR) ---"
        for name in $BASELINE_RUNS; do
            checkpoint="$BASELINE_DIR/$name"
            if [ -d "$checkpoint" ]; then
                pe_type=$(get_pe_type "$name")
                run_eval "$checkpoint" "$pe_type"
                found=$((found + 1))
            else
                echo "Skipping $name (not found at $checkpoint)"
            fi
        done
        echo ""
    fi

    # Scaffold runs
    if [ "$RUN_SET" = "all" ] || [ "$RUN_SET" = "scaffold" ]; then
        echo "--- Scaffold runs (from $OUTPUTS_DIR) ---"
        for name in $SCAFFOLD_RUNS; do
            checkpoint="$OUTPUTS_DIR/$name"
            if [ -d "$checkpoint" ]; then
                pe_type=$(get_pe_type "$name")
                run_eval "$checkpoint" "$pe_type"
                found=$((found + 1))
            else
                echo "Skipping $name (not found at $checkpoint)"
            fi
        done
        echo ""
    fi

    if [ $found -eq 0 ]; then
        echo "No checkpoints found."
        echo ""
        echo "Expected baselines in: $BASELINE_DIR/{run0_nope,run1_rope,run2_yarn,run3_pope}"
        echo "Expected scaffold in:  $OUTPUTS_DIR/{scaffold_rope_10k,...,scaffold_pope_15k}"
        exit 1
    fi

    echo "========================================"
    echo "Evaluated $found checkpoint(s)"
    echo "========================================"
fi
