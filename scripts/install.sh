#!/bin/bash
# Install GPU-specific packages for PE comparison study
# Run after: pip install -r requirements.txt

set -e

echo "Installing GPU-specific packages..."

# Detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    echo "Detected CUDA version: $CUDA_VERSION"
else
    echo "Warning: nvcc not found. Assuming CUDA 12.x"
    CUDA_VERSION="12.1"
fi

# Install PyTorch with CUDA support (if not already installed)
echo "Checking PyTorch installation..."
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" || {
    echo "Installing PyTorch with CUDA support..."
    pip install torch --index-url https://download.pytorch.org/whl/cu121
}

# Install Flash Attention 2
echo "Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation

# Install Triton (for custom kernels)
echo "Installing Triton..."
pip install triton

# Verify installation
echo ""
echo "=== Installation Summary ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

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

echo ""
echo "Installation complete!"
