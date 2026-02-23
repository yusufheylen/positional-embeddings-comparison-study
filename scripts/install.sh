#!/bin/bash
# Install GPU-specific packages for PE comparison study
# Run after: pip install -r requirements.txt
#
# Supports:
#   - Apple Silicon (M1/M2/M3) - uses MPS backend, no Flash Attention
#   - NVIDIA GPUs - installs Flash Attention and Triton

set -e

echo "Installing GPU-specific packages..."

# Detect platform
PLATFORM=$(uname -m)
OS=$(uname -s)

if [[ "$OS" == "Darwin" && ("$PLATFORM" == "arm64" || "$PLATFORM" == "aarch64") ]]; then
    echo "Detected Apple Silicon Mac"
    DEVICE="mps"

    # Check PyTorch MPS support
    echo "Checking PyTorch MPS support..."
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('✓ MPS backend ready for local testing')
else:
    print('⚠ MPS not available - will use CPU')
"

    echo ""
    echo "=== Apple Silicon Setup Complete ==="
    echo "Note: Flash Attention is not available on Apple Silicon."
    echo "For local testing, models will use 'sdpa' or 'eager' attention."
    echo "For full training, use a server with NVIDIA GPUs."
    echo ""

elif command -v nvcc &> /dev/null || [[ -n "$CUDA_HOME" ]]; then
    echo "Detected NVIDIA CUDA environment"
    DEVICE="cuda"

    # Get CUDA version
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
        echo "CUDA version: $CUDA_VERSION"
    fi

    # Ensure CUDA-enabled PyTorch is installed (requirements.txt installs CPU-only torch)
    echo "Checking PyTorch CUDA support..."
    TORCH_CUDA=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    if [[ "$TORCH_CUDA" != "True" ]]; then
        echo "CUDA torch not found — installing PyTorch with CUDA support..."
        CUDA_TAG="cu$(echo "$CUDA_VERSION" | tr -d '.')"
        pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
    else
        echo "CUDA torch already installed."
    fi

    # Install Flash Attention 2
    # --no-build-isolation is required: flash-attn's setup.py imports torch to detect CUDA,
    # so pip must use the current environment rather than creating an isolated build env.
    # NOTE: this compiles CUDA kernels from source and takes ~20-30 minutes. Do not interrupt.
    echo "Installing Flash Attention 2 (compiling from source, ~20-30 min)..."
    pip install flash-attn --no-build-isolation

    # Install Triton (for custom kernels)
    echo "Installing Triton..."
    pip install triton

    # Verify CUDA installation
    echo ""
    echo "=== NVIDIA GPU Setup Complete ==="
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

try:
    import flash_attn
    print(f'Flash Attention: {flash_attn.__version__}')
except ImportError:
    print('Flash Attention: Not installed')

try:
    import triton
    print(f'Triton: {triton.__version__}')
except ImportError:
    print('Triton: Not installed')
"

else
    echo "No GPU detected (no CUDA, no Apple Silicon)"
    DEVICE="cpu"

    echo ""
    echo "=== CPU-only Setup ==="
    echo "Warning: Training will be very slow without GPU acceleration."
    echo "For Apple Silicon: ensure you have arm64 PyTorch installed."
    echo "For NVIDIA: ensure CUDA toolkit is installed and nvcc is in PATH."
    echo ""
fi

echo ""
echo "Installation complete! Device: $DEVICE"
