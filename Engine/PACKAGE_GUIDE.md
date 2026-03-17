# OneBitLLM Python Package Build Guide

## Quick Start

### Build Release Binary
```bash
cd /home/_404connernotfound/Desktop/OneBitLLM/Engine
cargo build --release
```

The compiled extension module is available at:
- `/target/release/libonebitllm.so` (Linux)
- `/target/release/libonebitllm.dylib` (macOS)  
- `/target/release/onebitllm.pyd` (Windows)

### Install Package

#### Option 1: Development Install (editable)
```bash
pip install -e .
```

#### Option 2: Using Maturin (recommended)
```bash
pip install maturin
maturin develop --release
```

#### Option 3: Build Wheel
```bash
pip install maturin wheel
maturin build --release
pip install target/wheels/onebitllm-0.1.0-*.whl
```

## Project Structure

```
Engine/
├── crates/
│   ├── onebitllm-core/        # Core Rust library
│   │   └── src/
│   │       ├── tensor.rs      # Tensor operations
│   │       ├── quant.rs       # 1-bit quantization
│   │       └── nn/            # Neural network modules
│   │
│   ├── onebitllm-cli/         # Command-line interface
│   │
│   └── onebitllm-python/      # PyO3 bindings
│       └── src/
│           ├── lib.rs         # Module root
│           ├── config.rs      # Config wrapper
│           ├── model.rs       # Model wrapper
│           ├── generate.rs    # Generation wrapper
│           └── tokenizer.rs   # Tokenizer wrapper
│
├── python/
│   └── onebitllm/             # Python package (helper module)
│       └── __init__.py        # Package init
│
├── setup.py                   # Legacy setuptools config
├── pyproject.toml             # Modern Python config (maturin)
└── Cargo.toml                 # Workspace config
```

## Building for Distribution

### Create Source Distribution
```bash
pip install build
python -m build --sdist
# Output: dist/onebitllm-0.1.0.tar.gz
```

### Create Wheels (requires maturin)
```bash
pip install maturin
maturin build --release
# Output: target/wheels/onebitllm-0.1.0-*.whl
```

### Cross-platform Compilation
```bash
# For multiple Python versions (requires cibuildwheel)
pip install cibuildwheel
cibuildwheel --output-dir dist/
```

## Python API

After installation, use in Python:

```python
import onebitllm

# Load configuration
config = onebitllm.ModelConfig(
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
)

# Create model
model = onebitllm.OneBitModel(config)

# Generate text
output = model.generate(
    prompt="Hello, world!",
    max_tokens=100
)

print(output)
```

## Publishing to PyPI

1. Build wheels:
   ```bash
   maturin build --release
   ```

2. Install twine:
   ```bash
   pip install twine
   ```

3. Upload to TestPyPI first:
   ```bash
   twine upload --repository testpypi dist/*
   ```

4. Upload to PyPI:
   ```bash
   twine upload dist/*
   ```

## Troubleshooting

### ImportError on first use
```bash
# Ensure Rust is installed
rustc --version

# Rebuild from scratch
cargo clean
cargo build --release

# Reinstall Python package
pip install -e .
```

### Version mismatch errors
```bash
# Update cargo and rust
rustup update

# Update dependencies
cargo update
```

### Binary compatibility issues
```bash
# View compiled module info (Linux)
ldd target/release/libonebitllm.so

# Check symbols (Linux)
nm -D target/release/libonebitllm.so

# View architecture (macOS)
lipo -info target/release/libonebitllm.dylib
```

## Development Workflow

### Edit Rust code
```bash
# Edit crates/onebitllm-python/src/*.rs or crates/onebitllm-core/src/**
vim crates/onebitllm-python/src/model.rs

# Rebuild
maturin develop --release
```

### Test changes
```bash
# Run Python tests
python -m pytest tests/

# Run Rust tests
cargo test --all
```

### Format code
```bash
# Rust formatting
cargo fmt --all

# Check for lints
cargo clippy --all
```

## Performance Optimization

For maximum performance when distributing:

1. **Build with SIMD**: Enabled by default in `onebitllm-core`
2. **Use release mode**: Always build with `--release`
3. **Enable LTO**: Add to `Cargo.toml`:
   ```toml
   [profile.release]
   lto = true
   codegen-units = 1
   ```

## License

Licensed under either of:
- Apache License, Version 2.0
- MIT license

at your option.
