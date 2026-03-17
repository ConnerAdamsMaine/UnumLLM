#!/bin/bash
# Install OneBitLLM in development mode

set -e

echo "Installing OneBitLLM in development mode..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required"
    exit 1
fi

# Check if Rust is available
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust/Cargo is required"
    echo "Install from: https://rustup.rs/"
    exit 1
fi

# Install maturin if needed
if ! command -v maturin &> /dev/null; then
    echo "Installing maturin..."
    python3 -m pip install maturin
fi

# Build and install in development mode
echo "Building and installing in development mode..."
python3 -m maturin develop --release

echo
echo "✓ Installation complete!"
echo
python3 -c "import onebitllm; print(f'OnebitLLM {onebitllm.__version__} installed successfully')"
echo

# Run quick test
echo "Running quick test..."
python3 -c "
import onebitllm
print('Available classes:', dir(onebitllm))
"

echo
echo "=== Installation Summary ==="
echo "Package: onebitllm"
echo "Version: 0.1.0"
echo "Location: $(python3 -c 'import onebitllm; print(onebitllm.__file__)')"
echo
echo "Try the examples:"
echo "  python3 -c 'from onebitllm.examples import example_basic_usage; example_basic_usage()'"
