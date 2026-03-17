#!/bin/bash
# Build wheel distribution for OneBitLLM

set -e

echo "Building OneBitLLM Python wheel..."
echo

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
fi

# Create dist directory
mkdir -p dist

# Build wheel in release mode
echo "Building release wheel..."
maturin build --release --out dist

echo
echo "✓ Wheel built successfully!"
echo
ls -lh dist/*.whl

# Optional: Build source distribution
if command -v build &> /dev/null; then
    echo
    echo "Building source distribution..."
    python -m build --sdist --outdir dist
    echo "✓ Source distribution built!"
    ls -lh dist/*.tar.gz
fi

echo
echo "=== Build Summary ==="
ls -lh dist/
echo
echo "Install with: pip install dist/onebitllm*.whl"
