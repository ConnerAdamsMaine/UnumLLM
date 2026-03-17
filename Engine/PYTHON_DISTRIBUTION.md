# OneBitLLM Python Distribution

Complete guide to building, testing, and distributing OneBitLLM as a Python package.

## Status

✅ **Ready for Distribution**

- Rust engine compiled: `target/release/libonebitllm.so` (901 KB)
- PyO3 bindings configured
- Python package structure created
- Distribution files ready

## Quick Start

### 1. Install for Development
```bash
cd /home/_404connernotfound/Desktop/OneBitLLM/Engine
./scripts/install_dev.sh
```

Or manually:
```bash
pip install maturin
maturin develop --release
```

### 2. Test Installation
```bash
python3 scripts/test_install.py
```

### 3. Use in Python
```python
import onebitllm

# Create model
config = onebitllm.ModelConfig(
    vocab_size=50000,
    hidden_dim=768,
    num_layers=12,
    num_heads=12,
)
model = onebitllm.OneBitModel(config)

# Generate text
output = model.generate("Hello, world!", max_tokens=50)
print(output)
```

## Building Distributions

### Build Wheel (Recommended)
```bash
./scripts/build_wheel.sh
# Output: dist/onebitllm-0.1.0-cp*.whl
```

Or manually:
```bash
pip install maturin
maturin build --release
```

### Build Source Distribution
```bash
pip install build
python -m build --sdist
# Output: dist/onebitllm-0.1.0.tar.gz
```

### Cross-Platform Wheels
For building wheels for multiple Python versions and platforms:

```bash
pip install cibuildwheel
cibuildwheel --output-dir dist/
```

## File Structure

```
Engine/
├── setup.py                    # Legacy setuptools config
├── pyproject.toml             # Modern build config (maturin)
├── MANIFEST.in                # Package manifest
├── python/
│   └── onebitllm/
│       ├── __init__.py        # Package entry point
│       └── examples.py        # Usage examples
├── scripts/
│   ├── build_wheel.sh         # Build wheel script
│   ├── install_dev.sh         # Development install
│   └── test_install.py        # Installation test
├── crates/
│   ├── onebitllm-core/        # Core Rust implementation
│   ├── onebitllm-cli/         # CLI tool
│   └── onebitllm-python/      # PyO3 bindings
└── target/release/
    └── libonebitllm.so        # Compiled extension (901 KB)
```

## Distribution Checklist

- [x] Rust crate compiles without errors
- [x] PyO3 bindings complete
- [x] Python package structure created
- [x] pyproject.toml configured for maturin
- [x] setup.py available for compatibility
- [x] Examples provided
- [x] Installation scripts created
- [x] Test scripts included
- [x] README and documentation updated

## Publishing to PyPI

### 1. Prepare Package
```bash
# Clean old builds
rm -rf dist/ target/

# Build wheel
./scripts/build_wheel.sh

# Check contents
twine check dist/*
```

### 2. Install twine
```bash
pip install twine
```

### 3. Test on TestPyPI (Optional)
```bash
twine upload --repository testpypi dist/
# Test install: pip install -i https://test.pypi.org/simple/ onebitllm
```

### 4. Upload to PyPI
```bash
twine upload dist/
```

After uploading:
```bash
pip install onebitllm
```

## Python Version Support

- Python 3.8
- Python 3.9
- Python 3.10
- Python 3.11
- Python 3.12

## Performance Characteristics

- **Binary size**: ~901 KB (compressed .so)
- **Dependencies**: numpy, pydantic (optional)
- **Memory usage**: Depends on model size
- **Inference speed**: Multi-threaded with Rayon

## Development Workflow

### Make Changes to Rust Code
```bash
# Edit crates/onebitllm-python/src/*.rs or crates/onebitllm-core/src/**
vim crates/onebitllm-python/src/model.rs

# Rebuild
maturin develop --release
```

### Run Tests
```bash
# Python tests
python -m pytest tests/

# Rust tests
cargo test --all

# Integration tests
python3 scripts/test_install.py
```

### Code Quality
```bash
# Format
cargo fmt --all

# Lint
cargo clippy --all
```

## Troubleshooting

### "No module named 'onebitllm'"
```bash
# Rebuild and reinstall
maturin develop --release
```

### Build errors
```bash
# Clean and rebuild
cargo clean
cargo build --release --all
maturin develop --release
```

### ImportError: cannot import name 'ModelConfig'
- Ensure PyO3 bindings are exported in `crates/onebitllm-python/src/lib.rs`
- Check `python/onebitllm/__init__.py` imports

### Slow first import
- This is normal due to Python module initialization
- Subsequent imports use cache

## Package Contents

### When installed via pip
- Python native extension module (libonebitllm.so)
- Python helper module (onebitllm/)
- License files
- Documentation

### Available in package
```python
import onebitllm

# Main classes
onebitllm.ModelConfig           # Model configuration
onebitllm.OneBitModel          # Main model class
onebitllm.Tokenizer            # Tokenization
onebitllm.GenerateConfig       # Generation parameters

# Examples
from onebitllm.examples import (
    example_basic_usage,
    example_with_generation_config,
    example_batch_inference,
    example_model_saving_loading,
    example_tokenizer,
    example_model_info,
)
```

## Next Steps

1. **Install**: `./scripts/install_dev.sh`
2. **Test**: `python3 scripts/test_install.py`
3. **Try Examples**: `python3 -c "from onebitllm.examples import example_basic_usage; example_basic_usage()"`
4. **Build Distribution**: `./scripts/build_wheel.sh`
5. **Publish**: `twine upload dist/`

## Support

For issues or questions:
1. Check compilation output in `target/release`
2. Review PyO3 bindings in `crates/onebitllm-python/src/`
3. Verify Python compatibility with `scripts/test_install.py`
4. Check examples in `python/onebitllm/examples.py`

## License

Licensed under either of:
- Apache License, Version 2.0
- MIT license

at your option.
