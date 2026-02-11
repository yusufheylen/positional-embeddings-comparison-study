#!/bin/bash
# Server setup script
# Usage: ./scripts/setup_server.sh

set -e

cd "$(dirname "$0")/.."

echo "=== PE Study: Server Setup ==="

# Create conda env if needed
if ! conda env list | grep -q "pe-study"; then
    echo "Creating conda environment..."
    conda create -n pe-study python=3.11 -y
fi

echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pe-study

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Installing GPU packages..."
./scripts/install.sh

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. huggingface-cli login"
echo "  2. wandb login"
echo "  3. ./scripts/run_scaffold_experiments.sh"
